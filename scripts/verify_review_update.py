"""Verify review-support copy, real overlap, and export on the public app."""
import time,json,requests
from pathlib import Path
from playwright.sync_api import sync_playwright
out=Path('deliverables')
for attempt in range(36):
    try:
        r=requests.get('https://ilmuuu81-panguide3d.hf.space/config',timeout=10)
        if r.status_code==200 and 'Cancer diagnosis reliability: not established' in r.text and 'colors partition' in r.text and 'research preprint' in r.text: break
    except requests.RequestException: pass
    if attempt==35: raise RuntimeError('Updated app did not become ready')
    time.sleep(10)
with sync_playwright() as p:
    browser=p.chromium.launch();page=browser.new_page(viewport={'width':1600,'height':1000})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('https://ilmuuu81-panguide3d.hf.space',wait_until='domcontentloaded')
    page.get_by_role('button',name='Try Sample CT / Demo Mode').wait_for(timeout=60000)
    page.get_by_label('I understand the limitations',exact=False).check()
    page.get_by_role('button',name='Try Sample CT / Demo Mode').click()
    page.wait_for_function("() => [...document.querySelectorAll('textarea,input')].some(e => e.value && e.value.includes('Ready ·'))",timeout=60000)
    page.get_by_role('tab',name='AI Findings',exact=True).click()
    page.get_by_text('Cancer diagnosis reliability: not established.',exact=True).first.wait_for()
    page.get_by_role('heading',name='Accuracy and reliability evidence',exact=True).first.scroll_into_view_if_needed()
    assert '0.460' in page.locator('body').inner_text()
    page.screenshot(path=str(out/'review-reliability-update.png'),full_page=True)
    link=page.locator('a[href*="educational-report.pdf"]').first
    with page.expect_download(timeout=15000) as download: link.click()
    download.value.save_as(out/'PanGuide3D-educational-report.pdf')
    page.get_by_role('tab',name='3D Visualization',exact=True).click();page.wait_for_timeout(3000)
    names=page.locator('#anatomy3d .js-plotly-plot').evaluate('(el) => el.data.map(t=>t.name)')
    assert 'Mask intersection (not diagnostic)' in names
    page.screenshot(path=str(out/'review-overlap-update.png'),full_page=True)
    result=dict(cached_real_results=True,diagnosis_reliability_labeled_unestablished=True,published_metrics_displayed=True,mesh_layers=names,pdf_download=True,browser_errors=errors)
    (out/'review-update-verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    browser.close()
