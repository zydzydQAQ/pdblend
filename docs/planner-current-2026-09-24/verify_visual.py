"""Optional browser check for the accompanying inline diagram (requires Playwright)."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

root = Path('/root/.codex/visualizations/2026/09/24/01a0d0f0-d65f-7af3-93dd-96ba71373380')
results = []
with sync_playwright() as p:
    browser = p.chromium.launch(args=['--no-sandbox'])
    for width in (360, 736):
        for theme in ('light', 'dark'):
            page = browser.new_page(viewport={'width': width + 32, 'height': 1100}, color_scheme=theme)
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.goto((root / 'planner-preview.html').as_uri())
            frame = page.frames[1]
            frame.wait_for_selector('#pdblend-planner-decision svg text')
            for output, count in [('64', 40), ('128', 17), ('256', 0)]:
                frame.select_option('#pd-output', output)
                frame.wait_for_timeout(80)
                assert f'{count} 个严格可行候选' in frame.inner_text('#pdblend-planner-decision')
                bounds = frame.evaluate('''() => {
                  const s=document.querySelector('.pd-flow'), b=s.getBoundingClientRect();
                  const labels=[...s.querySelectorAll('text')].map(t=>({text:t.textContent,b:t.getBoundingClientRect()}));
                  return {overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth,
                  outside:labels.filter(t=>t.b.left < b.left-1 || t.b.right>b.right+1 || t.b.bottom>b.bottom+1).map(t=>t.text),
                  overlap:labels.flatMap((a,i)=>labels.slice(i+1).filter(c=>a.b.left<c.b.right && c.b.left<a.b.right && a.b.top<c.b.bottom && c.b.top<a.b.bottom).map(c=>[a.text,c.text]))};
                }''')
                assert not bounds['overflow'] and not bounds['outside'] and not bounds['overlap'], bounds
                results.append(dict(width=width, theme=theme, output=output, errors=errors, **bounds))
            frame.select_option('#pd-output', '128')
            frame.locator('#pdblend-planner-decision').screenshot(path=str(root / f'preview-{width}-{theme}.png'))
            assert not errors, errors
            page.close()
    browser.close()
(root / 'validation.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
print('12 viewport/theme/state checks passed; no JS errors, overflow, clipped or overlapping SVG labels.')
