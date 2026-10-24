"""Optional real cached-results browser recording; never substitutes predictions."""
from pathlib import Path
from playwright.sync_api import sync_playwright
out=Path('recordings'); out.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch()
    context=browser.new_context(viewport={'width':1920,'height':1080},record_video_dir=str(out))
    page=context.new_page(); page.goto('http://127.0.0.1:7860',wait_until='domcontentloaded')
    page.get_by_label('I understand the limitations',exact=False).check()
    page.get_by_role('button',name='Try Sample CT / Demo Mode').click()
    page.wait_for_function("() => [...document.querySelectorAll('textarea,input')].some(e => e.value && e.value.includes('Ready ·'))",timeout=120000)
    page.screenshot(path=str(out/'01-analysis.png'),full_page=True)
    page.get_by_role('tab',name='3D Visualization',exact=True).click()
    page.wait_for_timeout(2000)
    canvas=page.locator('canvas').first; box=canvas.bounding_box()
    if box:
        page.mouse.move(box['x']+box['width']*.4,box['y']+box['height']*.5)
        page.mouse.down(); page.mouse.move(box['x']+box['width']*.8,box['y']+box['height']*.6,steps=100); page.mouse.up()
    page.screenshot(path=str(out/'02-3d.png'),full_page=True)
    page.get_by_role('tab',name='AI Findings',exact=True).click()
    page.wait_for_timeout(2000); page.screenshot(path=str(out/'03-findings.png'),full_page=True)
    context.close(); browser.close()
print(out.resolve())
