"""Build and inspect the 12-page bilingual PDblend method note. No GPU use.

Install requirements.txt in a document-only Python environment, then run:
    python build_pdf.py
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PDF = ROOT / 'pdblend-method-bilingual.pdf'


def package_sources():
    """Bundle editable content and its font license; omit generated previews."""
    source_files = [p for p in ROOT.iterdir()
                    if p.is_file() and p.suffix in {'.py', '.css', '.html', '.md', '.json', '.txt'}]
    for directory in ('chapters', 'figures', 'assets'):
        source_files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file())
    target = ROOT / 'pdblend-method-source.zip'
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(source_files):
            bundle.write(path, Path('pdblend-method')/path.relative_to(ROOT))
    return target


def capture_evidence():
    groups = {
        'modeling': ['src/pdblend/profile/model.py', 'src/pdblend/profile/profiler.py',
                     'src/pdblend/profile/calibration.py', 'src/pdblend/profile/power_table.py',
                     'src/pdblend/profile/power_calibration.md'],
        'planning': ['src/pdblend/control/planner.py', 'src/pdblend/control/forecast.py',
                     'src/pdblend/control/policies/__init__.py'],
        'online': ['src/pdblend/control/controller.py', 'src/pdblend/control/shield.py',
                   'src/pdblend/proxy/router.py', 'src/pdblend/proxy/server.py',
                   'src/pdblend/engine/carry.py', 'src/pdblend/engine/client.py',
                   'src/pdblend/engine/handoff_timing.py', 'src/pdblend/bench/run.py',
                   'src/pdblend/bench/metering.py'],
        'examples': ['tests/pdblend/synthetic.py', 'tests/pdblend/test_model.py',
                     'tests/pdblend/test_carry_protocol.py'],
    }
    paths = [p for group in groups.values() for p in group]
    out = {
        'captured_utc': datetime.now(timezone.utc).isoformat(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'working_tree_status': subprocess.check_output(['git', 'status', '--short', '--', *paths], cwd=REPO, text=True).splitlines(),
        'source_groups': groups,
        'sha256': {p: sha256((REPO/p).read_bytes()).hexdigest() for p in paths},
        'document_scope': 'Target selective-disaggregation design, with chapter-end implementation notes.',
        'example_status': 'Illustrative, not measured GPU performance or an accepted energy comparison.',
        'current_vs_target': {
            'modeling': 'Current code has performance/power queries, bounded overrides, handoff timing and partial switch costs. Uniform incremental handoff energy remains a design target.',
            'planning': 'Current default search retains min(4, slots) M instances. Target planning removes the fixed floor and separately models visible TTFT and continuation cost.',
            'online': 'Current main controller drains proxy request counts. Full native KV/transfer acknowledgement and the asynchronous transaction presentation are target design.',
        },
    }
    (ROOT/'evidence.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def build():
    from weasyprint import HTML
    import fitz
    from PIL import Image, ImageDraw

    subprocess.run([sys.executable, str(ROOT/'build_figures.py')], check=True)
    capture_evidence()
    chapters = sorted((ROOT/'chapters').glob('*.html'))
    if len(chapters) != 4:
        raise RuntimeError('Expected exactly four chapter source files')
    body = '\n'.join(p.read_text(encoding='utf-8') for p in chapters)
    if body.count('<section class="page"') != 12:
        raise RuntimeError('Expected exactly 12 authored pages')
    html = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>PDblend · 选择性 P/D 分离的方法说明</title>
<meta name="author" content="PDblend method documentation">
<meta name="description" content="Bilingual target design: overview, offline modeling, planning and online coordination.">
<link rel="stylesheet" href="style.css"></head><body>'''+body+'</body></html>'
    (ROOT/'pdblend-method-bilingual.html').write_text(html, encoding='utf-8')
    draft = ROOT/'preview'/'draft.pdf'
    draft.parent.mkdir(exist_ok=True)
    HTML(string=html, base_url=str(ROOT)).write_pdf(draft)
    doc = fitz.open(draft)
    # Resolve chapter starts from rendered text rather than hard-coding them.
    starts = []
    for needle in ('选择性 P/D 分离', 'Offline Performance and Energy Modeling',
                   'Cost-Aware Planning for Selective Disaggregation',
                   'Online Coordination and Selective Routing'):
        hits = [i+1 for i,p in enumerate(doc) if needle in p.get_text()]
        if not hits:
            raise RuntimeError(f'Cannot find chapter heading: {needle}')
        starts.append(hits[0])
    titles = ['1  Overview · 系统概览', '2  Offline Modeling · 离线性能与能耗建模',
              '3  Planning · 选择性分离规划', '4  Online Coordination · 在线协同与选择性分流']
    doc.set_toc([[1,t,p] for t,p in zip(titles,starts)])
    doc.save(PDF, garbage=4, deflate=True)
    doc.close()
    doc = fitz.open(PDF)
    for old in (ROOT/'preview').glob('page-*.png'):
        old.unlink()
    records=[]
    thumbnails=[]
    for i,page in enumerate(doc):
        pix=page.get_pixmap(matrix=fitz.Matrix(1.35,1.35), alpha=False)
        filename=ROOT/'preview'/f'page-{i+1:02d}.png'
        pix.save(filename)
        im=Image.open(filename).convert('RGB')
        im.thumbnail((280,396))
        thumb=Image.new('RGB',(304,426),'#e6ebef')
        thumb.paste(im,((304-im.width)//2,8))
        ImageDraw.Draw(thumb).text((12,406), f'Page {i+1:02d}',fill='#163047')
        thumbnails.append(thumb)
        text=page.get_text()
        blocks=[b for b in page.get_text('blocks') if b[6]==0]
        outside=[list(b[:4]) for b in blocks if b[0]<0 or b[1]<0 or b[2]>page.rect.width+.5 or b[3]>page.rect.height+.5]
        records.append({'page':i+1,'characters':len(text),'outside_page':outside,
                        'head':text[:110].replace('\n',' / ')})
    rows=(len(thumbnails)+2)//3
    sheet=Image.new('RGB',(304*3,426*rows),'#e6ebef')
    for i,thumb in enumerate(thumbnails):
        sheet.paste(thumb,((i%3)*304,(i//3)*426))
    sheet.save(ROOT/'preview'/'contact-sheet.png')
    fulltext='\n\f\n'.join(p.get_text() for p in doc)
    (ROOT/'preview'/'extracted-text.txt').write_text(fulltext, encoding='utf-8')
    checks={
        'page_count':len(doc),'target_pages':12,'max_pages':14,'chapter_start_pages':starts,
        'has_chinese_text':'选择性' in fulltext,
        'has_planning_algorithm':'SELECTIVE-PD-PLAN' in fulltext,
        'has_online_algorithm':'ONLINE-COORDINATE' in fulltext,
        'figure_count':len(list((ROOT/'figures').glob('*.svg'))),
        'normalized_example_correct':78+8+4==90 and 90<100<103,
        'carry_budget_correct':1+127==128 and 2048+1==2049,
        'energy_example_correct':1200*10/1000==12,
        'replacement_characters':fulltext.count('\ufffd'),
        'pages':records,'pdf_sha256':sha256(PDF.read_bytes()).hexdigest(),
        'pdf_bytes':PDF.stat().st_size,
    }
    (ROOT/'validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in checks.items() if k!='pages'},ensure_ascii=False,indent=2))
    if len(doc)>14 or any(x['outside_page'] for x in records):
        raise RuntimeError('PDF failed pagination or page-boundary checks')
    draft.unlink(missing_ok=True)
    package_sources()
    return checks


if __name__=='__main__':
    build()
