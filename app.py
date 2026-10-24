"""PanGuide 3D: research CT segmentation with genuine live and cached results."""
import os
from pathlib import Path
# Own cache, bounded retention, no scan content in logs.
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
os.environ.setdefault('GRADIO_TEMP_DIR', str(Path(__file__).parent / '.runtime-cache'))
import json, time, tempfile, threading, shutil
from runtime import infer  # initialize checkpoint once at startup, before any GPU request
import gradio as gr
import numpy as np
import SimpleITK as sitk
from webcore import ROOT, DISCLAIMER, load_upload, restore, slices, mesh, report_text, export_pdf, reliability_text, CANCER_CONTEXT

CACHE = ROOT / 'demo-cache'
RUNTIME = Path(os.environ['GRADIO_TEMP_DIR']); RUNTIME.mkdir(exist_ok=True)
# Every request receives private Gradio state. No cross-user inference cache.
RATE = {}; RATE_LOCK = threading.Lock()
import hashlib, secrets
RATE_SALT = secrets.token_bytes(32)
def cleanup_reports():
    while True:
        for folder in RUNTIME.glob('report-*'):
            if folder.is_dir() and time.time()-folder.stat().st_mtime>1800:
                shutil.rmtree(folder,ignore_errors=True)
        time.sleep(60)
threading.Thread(target=cleanup_reports,daemon=True).start()

def rate_limit(request):
    identity = request.client.host if request and request.client else 'local'
    key = hashlib.sha256(RATE_SALT+identity.encode()).hexdigest()
    now=time.monotonic()
    with RATE_LOCK:
        for old in list(RATE):
            RATE[old]=[t for t in RATE[old] if now-t<300]
            if not RATE[old]: del RATE[old]
        times=RATE.setdefault(key,[])
        if len(times)>=3: raise gr.Error('Please wait: limit of three analyses per five minutes per session.')
        times.append(now)

from webcore import data_from

def focus(data):
    coordinates=np.argwhere(data['tumor'] if data['tumor'].any() else data['pan'])
    return coordinates.mean(axis=0).astype(int) if len(coordinates) else np.asarray(data['ct'].shape)//2

def present(data):
    z,y,x=focus(data)
    imgs=slices(data,z,y,x,400,50,.45,True,True)
    if not data.get('inferred',True):
        return [data,*imgs,mesh(data),report_text(data),None,
            gr.update(maximum=data['ct'].shape[0]-1,value=int(z)),
            gr.update(maximum=data['ct'].shape[1]-1,value=int(y)),
            gr.update(maximum=data['ct'].shape[2]-1,value=int(x)),
            f"Validated CT preview · {data['original_size']} voxels · original spacing {data['original_spacing']} mm · no inference yet",None]
    folder=tempfile.mkdtemp(prefix='report-',dir=RUNTIME)
    pdf=export_pdf(data,imgs,folder)
    for name,img in zip(('axial','coronal','sagittal'),imgs): img.save(Path(folder)/f'{name}.png')
    return [data,*imgs,mesh(data),report_text(data),pdf,
        gr.update(maximum=data['ct'].shape[0]-1,value=int(z)),
        gr.update(maximum=data['ct'].shape[1]-1,value=int(y)),
        gr.update(maximum=data['ct'].shape[2]-1,value=int(x)),
        f"Ready · {data['provenance']} · {data['ct'].shape[::-1]} voxels · spacing {data['spacing']} mm",
        [str(Path(folder)/f'{name}.png') for name in ('axial','coronal','sagittal')]]

def sample(case, consent, cached, request: gr.Request=None):
    if not consent: raise gr.Error('Read and accept the research and privacy notice first.')
    if cached:
        path=CACHE/f'{case}.npz'
        if not path.exists(): raise gr.Error('No real cached masks exist for this case yet. Run scripts/build_demo_cache.py.')
        with np.load(path,allow_pickle=False) as f: data={k:f[k] for k in f.files}
        if not np.allclose(np.asarray(data['direction']).reshape(3,3),np.eye(3)):
            for field in ('ct','pan','tumor'):
                image=sitk.GetImageFromArray(data[field]); image.SetSpacing(data['spacing']); image.SetOrigin(data['origin']); image.SetDirection(data['direction'])
                oriented=sitk.DICOMOrient(image,'LPS'); data[field]=sitk.GetArrayFromImage(oriented)
            data['spacing']=oriented.GetSpacing(); data['origin']=oriented.GetOrigin(); data['direction']=oriented.GetDirection()
        data['provenance']='Recorded Demo Results • cached from a completed real model run'
        data['elapsed']=float(data['elapsed'])
        return present(data)
    path=(ROOT/'public-samples'/case/'ct.nii.gz') if case.startswith('MSD_') else ROOT/'samples'/case/'input'/'ct.nii.gz'
    return run_scan(str(path),consent,request)

def run_scan(path, consent, request: gr.Request=None, progress=gr.Progress()):
    if not consent: raise gr.Error('Read and accept the research and privacy notice first.')
    if not path: raise gr.Error('Select a CT scan first.')
    rate_limit(request)
    try:
        progress(.05,desc="Validating CT and preprocessing on CPU")
        original,volume=load_upload(path)
        progress(.25,desc="Running actual checkpoint; waiting for GPU if hosted")
        pred,elapsed,peak=infer(volume)
        progress(.85,desc="Restoring physical geometry and building findings")
        data=data_from(original,volume,pred,elapsed)
        data['gpu_peak_bytes']=peak
        return present(data)
    except (ValueError,FileNotFoundError) as exc:
        raise gr.Error(str(exc)) from None
    except Exception:
        raise gr.Error('Analysis did not complete. GPU capacity, quota, queue limits, or memory may be unavailable. Try Recorded Demo Results or retry later. No new prediction is shown.') from None

def preview(path, consent):
    if not consent: raise gr.Error('Accept the research and privacy notice before upload.')
    if not path: raise gr.Error('Select a CT file first.')
    try:
        original, volume=load_upload(path)
        display=sitk.DICOMOrient(original,'LPS')
        ct=sitk.GetArrayFromImage(display).astype(np.float32)
        data=dict(ct=ct,pan=np.zeros(ct.shape,dtype=np.uint8),tumor=np.zeros(ct.shape,dtype=np.uint8),
            spacing=display.GetSpacing(),origin=display.GetOrigin(),direction=display.GetDirection(),
            original_size=original.GetSize(),original_spacing=original.GetSpacing(),inferred=False,
            provenance='CT preview only; no model results')
        return present(data)
    except ValueError as exc: raise gr.Error(str(exc)) from None
    except Exception: raise gr.Error('Unable to read this CT scan.') from None

CSS='''
.gradio-container {max-width:1600px!important;background:#0a1320;color:#e6edf5}
#hero {padding:32px;background:linear-gradient(120deg,#122a43,#102033);border:1px solid #28506a;border-radius:18px;margin-bottom:18px}
#hero h1 {font-size:42px;color:#e6f6ff;letter-spacing:-1.5px}
#hero small {color:#6dcde0;letter-spacing:2px}
#notice {color:#ffdca8}
.gradio-container .prose {color:#c5d5e5}
#hero p {color:#a9c2d7;font-size:17px}
#notice {padding:14px;border-left:4px solid #ffb454;background:#30271e;border-radius:8px}
#provenance {position:sticky;top:0;z-index:100;background:#102437;padding:10px;border:1px solid #28627c;border-radius:8px}
footer {display:none!important}
'''
with gr.Blocks(title='PanGuide 3D',theme=gr.themes.Soft(primary_hue='cyan'),js="() => { document.documentElement.classList.add('dark'); }",css=CSS,delete_cache=(60,1800)) as demo:
    gr.HTML('<div id="hero"><small>COMPUTATIONAL IMAGING / RESEARCH PLATFORM</small><h1>PanGuide 3D</h1><p>AI-Powered 3D Pancreatic CT Analysis</p><p>Localize candidate tumors. Inspect pancreas overlap. Support informed research review.</p></div>')
    gr.HTML(f'<div id="notice"><b>{DISCLAIMER}</b></div>')
    gr.Markdown('Upload **only fully de-identified research CT**. Metadata checks cannot guarantee de-identification, including burned-in annotations. Files are temporarily written to the host, expire within approximately 31 minutes, and are not retained as a dataset. This system is not assessed as HIPAA-compliant. Anonymous cloud GPU use is quota-limited; no application login is required.')
    consent=gr.Checkbox(label='I understand the limitations and confirm that the scan is de-identified research CT.',value=False)
    state=gr.State(value=None,time_to_live=1800)
    source_label=gr.Markdown('**Result provenance:** no model run loaded.',elem_id='provenance')
    with gr.Tabs():
        with gr.Tab('CT Analysis'):
            with gr.Row():
                with gr.Column(scale=2):
                    upload=gr.File(label='Drop a 3D CT scan',file_types=['.nii','.gz','.zip'],type='filepath')
                    inspect=gr.Button('Validate & preview CT')
                    analyze=gr.Button('Analyze CT • Live Inference',variant='primary')
                with gr.Column(scale=2):
                    case=gr.Dropdown(['MSD_pancreas_045','PanTS_00009473','PanTS_00009005','PanTS_00009001','PanTS_00009160'] if (ROOT/'samples').exists() else ['MSD_pancreas_045'],value='MSD_pancreas_045',label='Research example (MSD public / PanTS local only)')
                    cached=gr.Checkbox(value=True,label='Recorded Demo Results (previous real model run)')
                    load=gr.Button('Try Sample CT / Demo Mode')
            status=gr.Textbox(label='Processing status',value='Awaiting scan. No prediction has been computed.',interactive=False)
            gr.Markdown('**Overlay legend** · 🟢 predicted pancreas · 🟠 predicted tumor · 🟣 pancreas–tumor overlap. Slice indices share one physical volume; use the three controls to set the intersection. Views use lossless LPS axis reorientation. Oblique scans are rejected.')
            with gr.Row():
                width=gr.Slider(50,1500,value=400,label='Window width (HU)')
                level=gr.Slider(-500,500,value=50,label='Window level (HU)')
                alpha=gr.Slider(0,1,value=.45,label='Overlay opacity')
                pan_on=gr.Checkbox(value=True,label='Pancreas'); tumor_on=gr.Checkbox(value=True,label='Tumor')
            with gr.Row():
                with gr.Column(): axial=gr.Image(label='Axial',interactive=False,elem_id='axial-view'); z=gr.Slider(0,1,value=0,step=1,label='Z slice')
                with gr.Column(): coronal=gr.Image(label='Coronal',interactive=False,elem_id='coronal-view'); y=gr.Slider(0,1,value=0,step=1,label='Y slice')
                with gr.Column(): sagittal=gr.Image(label='Sagittal',interactive=False,elem_id='sagittal-view'); x=gr.Slider(0,1,value=0,step=1,label='X slice')
        with gr.Tab('3D Visualization') as tab3d:
            gr.Markdown('Drag to rotate · scroll to zoom · shift-drag to pan · Plotly reset-camera control restores the preset. Physical coordinates are LPS millimeters. Use legend to toggle surfaces. With both predictions enabled, green and orange show mask-only regions; magenta shows their intersection, so the colors partition the predicted tissue.')
            plot=gr.Plot(label='Predicted anatomy in physical space',elem_id='anatomy3d')
            mesh_alpha=gr.Slider(.1,1,value=.55,label='Pancreas surface opacity')
            reset=gr.Button('Reset camera / refresh surfaces')
            gr.Markdown('For a smooth recorded rotation, drag continuously across the 3D view; the optional recording script captures a guided orbit.')
        with gr.Tab('AI Findings'):
            report=gr.Markdown('Load or analyze a CT to see tumor segmentation confidence, the scan-specific explanation, overlap findings, and reliability evidence.')
            pdf=gr.File(label='Download educational PDF report')
            screenshots=gr.File(label='Download selected CT views',file_count='multiple')
        with gr.Tab('About the Research'):
            gr.Markdown('''### Research foundation
PanGuide 3D is based on the research paper:

**[PanGuide3D: Cohort-Robust Pancreas Tumor Segmentation via Probabilistic Pancreas Conditioning and a Transformer Bottleneck](https://arxiv.org/abs/2604.20981)**

This web app brings the paper's pancreas-conditioned tumor segmentation architecture into an interactive CT visualization and research-review workflow. Study results and the deployed checkpoint's validation limits are distinguished below.
''')
            gr.Markdown(reliability_text(include_intro=False))
            gr.Markdown('### About pancreatic cancer\n'+CANCER_CONTEXT)
            gr.Markdown('''### Segmentation, with evidence and limits
Shared 3D encoder, two decoders and a Transformer bottleneck. The tumor decoder is conditioned on pancreas probabilities. One-channel input; two independent sigmoid channels (pancreas and tumor), threshold 0.5. Training preprocessing: 1.5 mm isotropic B-spline resampling, −125 to 275 HU window, scale to [0,1]. Gaussian sliding windows: 96×160×160 in Z,Y,X, overlap 0.5.

This particular 52.3M parameter checkpoint was trained on **MSD Task07 fold 0**, according to its training metadata. Architecture benchmark tables in the original project describe other training/evaluation settings and do not establish this file's accuracy. No cancer classifier is present. Head/body/tail subdivisions in the original package are geometric approximations; this viewer displays the model's whole-pancreas mask.

Public example: MSD Task07 pancreas_045, CC BY-SA 4.0; see public-samples/ATTRIBUTION.md. Local examples: PanTS research dataset, attributed in samples/README.md. Selected cases are demonstrations, not a performance estimate. Redistribution of derivative PanTS assets needs license review before public packaging. The research foundation and linked paper are cited above.
''')
    state.change(lambda d: '**Result provenance:** '+str(d['provenance']) if d else '**Result provenance:** no model run loaded.',state,source_label,api_name=False)
    outputs=[state,axial,coronal,sagittal,plot,report,pdf,z,y,x,status,screenshots]
    load.click(sample,[case,consent,cached],outputs,concurrency_limit=1,concurrency_id='inference',api_name=False)
    inspect.click(preview,[upload,consent],outputs,concurrency_limit=1,api_name=False)
    analyze.click(run_scan,[upload,consent],outputs,concurrency_limit=1,concurrency_id='inference',api_name=False)
    controls=[state,z,y,x,width,level,alpha,pan_on,tumor_on]
    for control in [z,y,x,width,level,alpha,pan_on,tumor_on]: control.change(slices,controls,[axial,coronal,sagittal],show_progress='hidden',api_name=False)
    tab3d.select(mesh,[state,mesh_alpha,pan_on,tumor_on],plot,api_name=False)
    for control in [mesh_alpha,reset]:
        (control.click if control==reset else control.change)(mesh,[state,mesh_alpha,pan_on,tumor_on],plot,api_name=False)
    demo.queue(max_size=8,default_concurrency_limit=1)

if __name__=='__main__':
    demo.launch(server_name=os.getenv('PANGUIDE_HOST','127.0.0.1'),server_port=int(os.getenv('PORT','7860')),max_file_size='256mb',show_error=False)
