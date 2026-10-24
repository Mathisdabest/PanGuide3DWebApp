"""Verify actual cached confidence on the hosted UI without spending GPU quota."""
import time,json,requests,re
from pathlib import Path
from playwright.sync_api import sync_playwright
for attempt in range(48):
    try:
        r=requests.get('https://ilmuuu81-panguide3d.hf.space/config',timeout=10)
        if r.status_code==200 and 'see tumor segmentation confidence' in r.text:break
    except requests.RequestException:pass
    if attempt==47:raise RuntimeError('New app was not ready')
    time.sleep(5)
with sync_playwright() as p:
    browser=p.chromium.launch();page=browser.new_page(viewport={'width':1600,'height':1000})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('https://ilmuuu81-panguide3d.hf.space',wait_until='domcontentloaded')
    page.get_by_role('button',name='Try Sample CT / Demo Mode').wait_for(timeout=60000)
    page.get_by_label('I understand the limitations',exact=False).check()
    page.get_by_role('button',name='Try Sample CT / Demo Mode').click()
    page.wait_for_function("() => [...document.querySelectorAll('textarea,input')].some(e=>e.value && e.value.includes('Ready ·'))",timeout=90000)
    page.get_by_role('tab',name='AI Findings',exact=True).click()
    heading=page.get_by_role('heading',name='Tumor segmentation confidence',exact=True)
    heading.wait_for();heading.scroll_into_view_if_needed()
    text=page.locator('body').inner_text()
    match=re.search(r'Mean model score across the predicted tumor mask: ([\d.]+)%',text)
    assert match,'No genuine score shown'
    assert 'Uncalibrated model score; not the probability that the region is cancerous.' in text
    page.screenshot(path='deliverables/confidence-verification.png',full_page=True)
    with page.expect_download() as download:page.locator('a[href*="educational-report.pdf"]').first.click()
    download.value.save_as('deliverables/PanGuide3D-educational-report.pdf')
    result=dict(score_percent=float(match.group(1)),uncalibrated_explanation=True,pdf_download=True,browser_errors=errors)
    Path('deliverables/confidence-verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    browser.close()
