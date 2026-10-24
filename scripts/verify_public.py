"""Anonymous HTTPS browser smoke test with real cached and fresh cloud inference."""
from pathlib import Path
import time,json,requests,argparse
parser=argparse.ArgumentParser();parser.add_argument('--upload',action='store_true');parser.add_argument('--authenticated',action='store_true');args=parser.parse_args()
from playwright.sync_api import sync_playwright
out=Path('deliverables');out.mkdir(exist_ok=True)
result={}
for attempt in range(30):
    try:
        ready=requests.get('https://ilmuuu81-panguide3d.hf.space/config',timeout=10)
        if ready.status_code==200 and 'components' in ready.json(): break
    except Exception: pass
    if attempt==29: raise RuntimeError('Public app did not become ready')
    time.sleep(10)
with sync_playwright() as p:
    browser=p.chromium.launch()
    page=browser.new_page(viewport={'width':1600,'height':1000})
    if args.authenticated:
        from huggingface_hub import get_token
        token=get_token()
        # Authenticate only to the known Space origin; never forward credentials to assets on other domains.
        page.route('https://ilmuuu81-panguide3d.hf.space/**',lambda route: route.continue_(headers={**route.request.headers,'authorization':'Bearer '+token}))
        result['provider_authenticated']=True
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    response=page.goto('https://ilmuuu81-panguide3d.hf.space',wait_until='domcontentloaded',timeout=60000)
    result['http_status']=response.status
    page.get_by_role('button',name='Try Sample CT / Demo Mode').wait_for(timeout=90000)
    page.get_by_label('I understand the limitations',exact=False).check()
    page.get_by_role('button',name='Try Sample CT / Demo Mode').click()
    page.wait_for_function("() => [...document.querySelectorAll('textarea,input')].some(e => e.value && e.value.includes('Ready ·'))",timeout=90000)
    result['cached_results']=True
    print('Public cached result verified',flush=True)
    page.get_by_role('tab',name='3D Visualization',exact=True).click();page.wait_for_timeout(4000)
    result['mesh_traces']=page.locator('#anatomy3d .js-plotly-plot').evaluate('(el) => el.data.length')
    page.screenshot(path=str(out/'public-3d.png'),full_page=True)
    page.get_by_role('tab',name='CT Analysis',exact=True).click()
    page.get_by_label('Recorded Demo Results',exact=False).uncheck()
    if args.upload:
        with page.expect_response(lambda r:'/upload' in r.url and r.request.method=='POST',timeout=60000) as uploaded:
            page.locator('input[type=file]').first.set_input_files('public-samples/MSD_pancreas_045/ct.nii.gz')
        uploaded.value.body();page.wait_for_timeout(2000)
        result['uploaded_research_nifti']=True
        print('Research CT upload completed',flush=True)
    started=time.monotonic()
    page.get_by_role('button',name='Analyze CT • Live Inference' if args.upload else 'Try Sample CT / Demo Mode',exact=True).click()
    print('Awaiting actual live inference',flush=True)
    try:
        page.wait_for_function("() => [...document.querySelectorAll('textarea,input')].some(e => e.value && e.value.includes('Ready · Live inference'))",timeout=180000)
        result['live_inference']=True
        result['live_wall_seconds']=round(time.monotonic()-started,3)
        result['status']=page.get_by_label('Processing status',exact=True).input_value()
        page.get_by_role('tab',name='AI Findings',exact=True).click()
        result['findings']=page.locator('.prose').all_text_contents()
        page.screenshot(path=str(out/'public-live-findings.png'),full_page=True)
    except Exception as e:
        result['live_inference']=False;result['live_error']=type(e).__name__
        result['visible_text']=page.locator('body').inner_text()[-4500:]
    result['browser_errors']=errors
    (out/'public-verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('findings','visible_text')},indent=2),flush=True)
    browser.close()
