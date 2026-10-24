"""Validated uploads, physical geometry, deterministic reporting and rendering."""
from pathlib import Path
import tempfile, zipfile, time, json, uuid
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import plotly.graph_objects as go
from skimage.measure import marching_cubes
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from panguide3d.preprocess import preprocess, looks_like_ct

ROOT = Path(__file__).parent
MAX_BYTES = 256 * 1024**2
MAX_VOXELS = 40_000_000
DISCLAIMER = 'Research & Educational Use Only — Not for Clinical Diagnosis'

def validate_image(image):
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError('A scalar 3D CT volume is required.')
    if np.prod(image.GetSize()) > MAX_VOXELS or min(image.GetSize()) < 2:
        raise ValueError('Unsupported volume size (maximum 40 million voxels).')
    direction = np.abs(np.asarray(image.GetDirection()).reshape(3,3))
    if not np.allclose(direction.max(axis=0),1,atol=1e-4) or not np.allclose(direction.sum(axis=0),1,atol=1e-4):
        raise ValueError('Oblique CT geometry is not supported by this version. Supply an axis-aligned research CT.')
    spacing = np.asarray(image.GetSpacing())
    if not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Invalid voxel spacing.')
    if not np.isfinite(sitk.GetArrayViewFromImage(image)).all():
        raise ValueError('Scan contains nonfinite intensities.')
    if not looks_like_ct(image):
        raise ValueError('CT intensity check failed. MRI is unsupported; NIfTI modality cannot be guaranteed.')
    if np.prod(np.asarray(image.GetSize()) * spacing / 1.5) > MAX_VOXELS:
        raise ValueError('Resampled volume would exceed memory limits.')

def load_upload(path):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('Maximum upload size is 256 MB.')
    with tempfile.TemporaryDirectory(prefix='panguide-') as tmp:
        if path.name.lower().endswith('.zip'):
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                if len(members) > 1500 or sum(m.file_size for m in members) > 512*1024**2:
                    raise ValueError('ZIP exceeds decompression limits.')
                files = []
                for member in members:
                    if member.is_dir(): continue
                    if member.flag_bits & 1 or Path(member.filename).is_absolute() or '..' in Path(member.filename).parts:
                        raise ValueError('Unsafe ZIP entry.')
                    target = Path(tmp) / f'{len(files):05d}.dcm'
                    target.write_bytes(archive.read(member))
                    files.append(target)
            image = dicom_image(files)
            clean = Path(tmp) / 'scan.nii.gz'
            sitk.WriteImage(image, str(clean))
        elif path.name.lower().endswith(('.nii', '.nii.gz')):
            reader = sitk.ImageFileReader(); reader.SetFileName(str(path)); reader.ReadImageInformation()
            if reader.GetDimension()!=3 or np.prod(reader.GetSize()) > MAX_VOXELS:
                raise ValueError('Expected 3D CT, maximum 40 million voxels.')
            if path.name.lower().endswith('.gz'):
                import gzip
                count=0
                with gzip.open(path,'rb') as stream:
                    while True:
                        block=stream.read(1024**2)
                        if not block: break
                        count+=len(block)
                        if count>512*1024**2: raise ValueError('NIfTI decompression limit exceeded.')
            image = reader.Execute(); clean = path
        else:
            raise ValueError('Upload .nii, .nii.gz, or a ZIP containing one CT DICOM series.')
        validate_image(image)
        volume = preprocess(clean, case_id='research-scan')
        volume.meta = {}
        return image, volume

def dicom_image(files):
    import pydicom
    records = [pydicom.dcmread(str(p), stop_before_pixels=True) for p in files]
    if len(records) < 2: raise ValueError('At least two DICOM slices are required.')
    first = records[0]
    if int(first.Rows)*int(first.Columns)*len(records)>MAX_VOXELS:
        raise ValueError('DICOM series exceeds voxel limit.')
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    normal = np.cross(orientation[:3], orientation[3:])
    positions = []
    for ds in records:
        if ds.file_meta.TransferSyntaxUID.is_compressed:
            raise ValueError('Compressed DICOM transfer syntax is unsupported; use uncompressed CT or NIfTI.')
        if ds.Modality != 'CT' or ds.SeriesInstanceUID != first.SeriesInstanceUID or getattr(ds,'FrameOfReferenceUID',None) != getattr(first,'FrameOfReferenceUID',None):
            raise ValueError('ZIP must contain exactly one coherent CT series.')
        if int(getattr(ds, 'NumberOfFrames', 1)) != 1:
            raise ValueError('Enhanced multiframe DICOM is unsupported.')
        if (ds.Rows, ds.Columns) != (first.Rows, first.Columns) or not np.allclose(ds.ImageOrientationPatient, orientation) or not np.allclose(ds.PixelSpacing, first.PixelSpacing):
            raise ValueError('Inconsistent DICOM geometry.')
        if getattr(ds, 'BurnedInAnnotation', 'NO') == 'YES':
            raise ValueError('DICOM declares burned-in annotations. Use a de-identified research case.')
        positions.append(np.asarray(ds.ImagePositionPatient, dtype=float))
        # Inspect common identifiers without logging or retaining their values.
        for tag in ('PatientName','PatientID','PatientBirthDate','PatientAddress','AccessionNumber','InstitutionName','ReferringPhysicianName','OtherPatientIDs'):
            if tag in ds: del ds[tag]
        ds.remove_private_tags()
    order = np.argsort(np.asarray(positions) @ normal)
    ordered = np.asarray(positions)[order]
    steps = np.diff(ordered, axis=0)
    distances = steps @ normal
    if (distances <= 0).any() or not np.allclose(distances, distances[0], rtol=.01, atol=.01) or not np.allclose(steps, distances[:,None]*normal, atol=.05):
        raise ValueError('Duplicate, irregular, or tilted DICOM slices are unsupported.')
    # GDCM applies per-slice rescale slope/intercept. Do not copy patient tags.
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(files[i]) for i in order])
    return reader.Execute()

def restore(mask, volume, original):
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.SetSpacing(volume.spacing); image.SetOrigin(volume.origin); image.SetDirection(volume.direction)
    return sitk.Resample(image, original, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

def statistics(pan, tumor, spacing):
    pan, tumor = pan.astype(bool), tumor.astype(bool)
    voxel = float(np.prod(spacing))/1000
    intersection = int((pan & tumor).sum())
    return dict(pancreas_cm3=float(pan.sum())*voxel, tumor_cm3=float(tumor.sum())*voxel,
        regions=int(ndimage.label(tumor)[1]), intersection_cm3=intersection*voxel,
        tumor_inside_pancreas_percent=100*intersection/max(1,int(tumor.sum())),
        pancreas_intersected_percent=100*intersection/max(1,int(pan.sum())))

def confidence_summary(mask, probabilities):
    """Mean actual sigmoid output in each six-connected original-grid region."""
    mask=np.asarray(mask,dtype=bool); probabilities=np.asarray(probabilities)
    if mask.shape!=probabilities.shape:
        raise ValueError('Probability map must align with the tumor mask')
    if not np.isfinite(probabilities).all() or (probabilities<0).any() or (probabilities>1).any():
        raise ValueError('Invalid tumor probabilities')
    labels,n=ndimage.label(mask)
    counts=np.bincount(labels.ravel(),minlength=n+1)[1:]
    totals=np.bincount(labels.ravel(),weights=probabilities.ravel(),minlength=n+1)[1:]
    order=np.argsort(-counts,kind='stable')
    return dict(tumor_confidence_means=(totals/counts)[order] if n else np.array([],dtype=float),
        tumor_confidence_voxels=counts[order])


def confidence_text(data):
    title='### Tumor segmentation confidence'
    note='Uncalibrated model score; not the probability that the region is cancerous. High scores can still be wrong.'
    if not np.any(data['tumor']):
        return title+'\nNot applicable: no tumor region was segmented. This does not rule out cancer.\n\n'+note
    if 'tumor_confidence_means' not in data or 'tumor_confidence_voxels' not in data:
        return title+'\nUnavailable for this saved result: voxel probabilities were not retained. Run Live Inference to compute the score.\n\n'+note
    means=np.asarray(data['tumor_confidence_means'],dtype=float)
    counts=np.asarray(data['tumor_confidence_voxels'],dtype=int)
    if len(means)!=len(counts) or not len(means) or counts.sum()!=np.count_nonzero(data['tumor']) or (counts<=0).any() or not np.isfinite(means).all() or (means<0).any() or (means>1).any():
        raise ValueError('Confidence summary does not match the tumor segmentation')
    overall=float(np.average(means,weights=counts))*100
    rows='\n'.join(f'| {i+1} | {m*100:.1f}% | {int(n):,} |' for i,(m,n) in enumerate(zip(means,counts)))
    return f"""{title}
**Mean model score across the predicted tumor mask: {overall:.1f}%**

{note}

| Region (largest first) | Mean model score | Voxels |
|---|---|---|
{rows}

Scores average the network's tumor-channel sigmoid outputs, linearly resampled onto the original CT grid, within each displayed six-connected region. They describe the model's label preference, not diagnostic accuracy or a calibrated clinical risk.
"""


def reliability_text(include_intro=True):
    """Cohort evidence is not an individual diagnosis probability."""
    evidence=json.loads((ROOT/'reference'/'benchmarks.json').read_text())
    rows=[]
    for key,name in [('pants','PanTS'),('msd','MSD Task07')]:
        r=evidence['results'][key]
        rows.append(f"| {name} | {r['tumor_dice']:.3f} ± {r['tumor_dice_sd']:.3f} | {r['pancreas_dice']:.3f} ± {r['pancreas_dice_sd']:.3f} |")
    intro="""### Accuracy and reliability evidence
**Cancer diagnosis reliability: not established.** No validated patient-level probability of cancer is available for this deployed checkpoint. Accuracy for this scan cannot be measured without an expert reference annotation; pancreas–tumor overlap is not an accuracy score.

"""
    return (intro if include_intro else '')+"""Published architecture results (research preprint; mean ± standard deviation across folds):

| Evaluation cohort | Tumor Dice | Pancreas Dice |
|---|---|---|
"""+'\n'.join(rows)+"""

Dice measures agreement with reference masks on a 0–1 scale; it is not the probability that a patient has cancer. These are study results for the architecture, not validated performance of this MSD-trained checkpoint or this scan. The paper reports high-score overconfidence in its calibration analysis. A reliable diagnosis score would require patient-level outcome validation and calibration of this exact checkpoint in the intended population.

Source: https://arxiv.org/html/2604.20981v1 (Table 2; Figure 9).
"""


def findings_summary(data):
    s=statistics(data['pan'],data['tumor'],data['spacing'])
    if not s['tumor_cm3']:
        return 'No tumor region was segmented by the model in this scan. This does not establish the absence of pancreatic cancer.'
    if not s['pancreas_cm3']:
        return 'Candidate tumor voxels were predicted, but no pancreas was segmented. Anatomical localization is unreliable and requires expert review.'
    return (f"The model marked {s['regions']} candidate tumor region(s), totaling {s['tumor_cm3']:.2f} cm³ of predicted mask volume. "
        f"Their mathematical intersection with the predicted pancreas is {s['intersection_cm3']:.2f} cm³ "
        f"({s['tumor_inside_pancreas_percent']:.1f}% of the candidate tumor mask). "
        'These views support inspection of location and spatial relationships during research review. '
        'The highlighted tissue is not confirmed cancer; this output cannot determine tumor type, stage, invasion, or treatment.')


CANCER_CONTEXT = ("Pancreatic cancer is a malignancy arising in pancreatic tissue. "
    "CT can help clinicians examine the pancreas and nearby structures, but a highlighted CT region alone cannot confirm cancer. "
    "Diagnosis integrates clinical findings and further evaluation, including tissue examination when appropriate. "
    "This app visualizes experimental segmentation predictions for review; it does not replace that diagnostic process. "
    "Source: https://www.cancer.gov/types/pancreatic/diagnosis")


def report_text(data):
    if not data.get('inferred', True):
        return '## CT preview only\nScan validated. Run Live Inference to compute model masks and an educational report. No prediction has been generated.'
    s = statistics(data['pan'], data['tumor'], data['spacing'])
    interpretation = findings_summary(data)
    return f'''## Tumor localization and review support
**{data['provenance']}**

**Examination overview** · CT · original dimensions (X,Y,Z): {tuple(int(v) for v in data.get('original_size', data['ct'].shape[::-1]))} · {data['ct'].shape[0]} slices · spacing {tuple(round(float(x),3) for x in data['spacing'])} mm.

**Pancreas localization** · predicted mask volume {s['pancreas_cm3']:.2f} cm³.

**Candidate lesion segmentation** · {s['tumor_cm3']:.2f} cm³ · {s['regions']} connected predicted regions.

**Geometric intersection** · {s['intersection_cm3']:.2f} cm³ · {s['tumor_inside_pancreas_percent']:.1f}% of tumor mask inside pancreas mask · {s['pancreas_intersected_percent']:.1f}% of pancreas mask intersected. Mathematical overlap is not evidence of malignancy.

{confidence_text(data)}

### Scan-specific explanation
{interpretation}


Model: PanGuide3D, 52,328,834 parameters · checkpoint 917acfa35774 · web software 0.3.0. Runtime: {data.get('elapsed', 'not measured')} seconds. Peak CUDA allocated memory: {str(round(float(data['gpu_peak_bytes'])/1024**3,3))+' GiB' if data.get('gpu_peak_bytes') is not None else 'not measured'}.
'''

def slices(data, z, y, x, width, level, opacity, show_pan, show_tumor):
    if data is None: return [None]*3
    images=[]
    for axis, position in enumerate((z,y,x)):
        idx=min(data['ct'].shape[axis]-1,max(0,int(position)))
        gray=np.clip((np.take(data['ct'],idx,axis=axis)-level+width/2)/width,0,1)
        rgb=np.repeat(gray[...,None],3,axis=-1)
        for key, enabled, color in [('pan',show_pan,[.1,.9,.65]),('tumor',show_tumor,[1,.3,.16])]:
            mask=np.take(data[key],idx,axis=axis).astype(bool)
            if enabled: rgb[mask]=rgb[mask]*(1-opacity)+np.array(color)*opacity
        if show_pan and show_tumor:
            overlap=np.take(data['pan'].astype(bool) & data['tumor'].astype(bool),idx,axis=axis)
            rgb[overlap]=rgb[overlap]*(1-opacity)+np.array([.95,.25,.9])*opacity
        intersection=np.delete([int(z),int(y),int(x)],axis)
        row=min(rgb.shape[0]-1,max(0,intersection[0])); col=min(rgb.shape[1]-1,max(0,intersection[1]))
        rgb[row,:,:]=rgb[row,:,:]*.6+np.array([.1,.6,.8])*.4
        rgb[:,col,:]=rgb[:,col,:]*.6+np.array([.1,.6,.8])*.4
        # Physical aspect ratio, with a consistent array-axis display convention.
        spacing=np.asarray(data['spacing'])[::-1]; plane=np.delete(spacing,axis)
        image=Image.fromarray(((rgb if axis==0 else np.flipud(rgb))*255).astype('uint8'))
        size=np.asarray(image.size)*plane[::-1]/min(plane)
        image=image.resize(tuple(np.maximum(1,size.astype(int))))
        image.thumbnail((700,700))
        from PIL import ImageDraw
        draw=ImageDraw.Draw(image); left,right,top,bottom=[('R','L','A','P'),('R','L','S','I'),('A','P','S','I')][axis]
        draw.text((5,image.height//2),left,fill='#7ee7ef',stroke_width=1,stroke_fill='black')
        draw.text((image.width-15,image.height//2),right,fill='#7ee7ef',stroke_width=1,stroke_fill='black')
        draw.text((image.width//2,5),top,fill='#7ee7ef',stroke_width=1,stroke_fill='black')
        draw.text((image.width//2,image.height-15),bottom,fill='#7ee7ef',stroke_width=1,stroke_fill='black')
        images.append(image)
    return images

def mesh(data, opacity=.55, show_pan=True, show_tumor=True):
    fig=go.Figure()
    if data and data.get('inferred',True):
        for key,color,name,enabled in [('pan','#27d7a2','Predicted pancreas',show_pan),('tumor','#ff613d','Predicted tumor',show_tumor),('overlap','#f340df','Mask intersection (not diagnostic)',show_pan and show_tumor)]:
            intersection=data['pan'].astype(bool) & data['tumor'].astype(bool)
            mask=intersection if key=='overlap' else data[key].astype(bool)
            # Partition visible categories to avoid coincident tumor/intersection surfaces.
            if show_pan and show_tumor and key!='overlap':
                mask=mask & ~intersection
                name='Pancreas-only mask' if key=='pan' else 'Tumor-only mask'
            if not mask.any() or not enabled: continue
            # Pad to close surfaces touching volume boundaries. Coordinates in original LPS mm.
            coords=np.argwhere(mask); low=coords.min(axis=0); high=coords.max(axis=0)+1
            cropped=mask[tuple(slice(int(a),int(b)) for a,b in zip(low,high))]
            vertices,faces,_,_=marching_cubes(np.pad(cropped,1),.5,spacing=data['spacing'][::-1],step_size=1)
            xyz=vertices[:,::-1]+(low[::-1]-1)*np.asarray(data['spacing'])
            xyz=np.einsum('ij,kj->ik',xyz,np.asarray(data['direction']).reshape(3,3))+data['origin']
            if not np.isfinite(xyz).all(): raise ValueError('Nonfinite mesh geometry')
            fig.add_trace(go.Mesh3d(x=xyz[:,0].tolist(),y=xyz[:,1].tolist(),z=xyz[:,2].tolist(),i=faces[:,0].tolist(),j=faces[:,1].tolist(),k=faces[:,2].tolist(),color=color,name=name,opacity=opacity if key=='pan' else (.65 if key=='tumor' else 1),showlegend=True))
    fig.update_layout(template='plotly_dark',paper_bgcolor='#101e30',margin=dict(l=0,r=0,t=20,b=0),height=530,autosize=True,scene=dict(aspectmode='data',xaxis_title='L (mm)',yaxis_title='P (mm)',zaxis_title='S (mm)',camera=dict(eye=dict(x=.15,y=-2,z=.8))))
    return fig

def export_pdf(data, images, directory):
    path=Path(directory)/'educational-report.pdf'; c=canvas.Canvas(str(path))
    c.setTitle('PanGuide 3D educational report')
    c.setFillColorRGB(.04,.12,.21); c.rect(0,730,600,112,fill=1,stroke=0)
    c.setFillColorRGB(.9,.97,1); c.setFont('Helvetica-Bold',22); c.drawString(40,792,'PanGuide 3D')
    c.setFont('Helvetica',10); c.drawString(40,767,'Experimental segmentation | research and education only')
    c.setFillColorRGB(0,0,0); c.setFont('Helvetica-Bold',12); c.drawString(40,705,'Segmentation summary (mask geometry only)')
    stats=statistics(data['pan'],data['tumor'],data['spacing'])
    rows=[('Predicted pancreas',f"{stats['pancreas_cm3']:.2f} cm3"),('Predicted tumor',f"{stats['tumor_cm3']:.2f} cm3"),('Intersection',f"{stats['intersection_cm3']:.2f} cm3"),('Connected tumor regions',str(stats['regions']))]
    y=683
    for label,value in rows:
        c.setFont('Helvetica',10); c.drawString(45,y,label); c.drawString(320,y,value)
        c.setStrokeColorRGB(.8,.85,.9); c.line(40,y-6,550,y-6); y-=23
    y-=15
    import textwrap
    for line in report_text(data).replace('**','').replace('##','').splitlines():
        for part in textwrap.wrap(line,95):
            if y<55: c.showPage(); y=800
            c.setFont('Helvetica',10); c.drawString(40,y,part.translate(str.maketrans({'±':'+/-','³':'3','–':'-','—':'-','≤':'<=','·':' | ','•':' | '})).encode('ascii','replace').decode()); y-=15
    for image in images:
        c.showPage(); c.drawString(40,800,'PanGuide 3D | predicted overlays | educational use only')
        c.drawImage(ImageReader(image),40,180,width=510,height=570,preserveAspectRatio=True,anchor='c')
    c.save(); return str(path)


def data_from(original, volume, pred, elapsed):
    pan_original = restore(pred.pancreas_mask,volume,original)
    tumor_original = restore(pred.tumor_mask,volume,original)
    # A lossless axis permutation/flip for display. Restored masks share original physical geometry.
    display = sitk.DICOMOrient(original,'LPS')
    pan = sitk.DICOMOrient(pan_original,'LPS')
    tumor = sitk.DICOMOrient(tumor_original,'LPS')
    probability=sitk.GetImageFromArray(np.asarray(pred.tumor_prob,dtype=np.float32))
    probability.SetSpacing(volume.spacing); probability.SetOrigin(volume.origin); probability.SetDirection(volume.direction)
    probability=sitk.Resample(probability,original,sitk.Transform(),sitk.sitkLinear,0,sitk.sitkFloat32)
    probability=sitk.DICOMOrient(probability,'LPS')
    scores=confidence_summary(sitk.GetArrayFromImage(tumor),sitk.GetArrayViewFromImage(probability))
    return dict(ct=sitk.GetArrayFromImage(display).astype(np.float32),
        pan=sitk.GetArrayFromImage(pan),tumor=sitk.GetArrayFromImage(tumor),
        spacing=display.GetSpacing(),origin=display.GetOrigin(),direction=display.GetDirection(),
        original_size=original.GetSize(),original_spacing=original.GetSpacing(),
        provenance='Live inference • actual trained checkpoint',elapsed=elapsed,**scores)
