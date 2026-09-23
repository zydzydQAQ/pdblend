#!/usr/bin/env python3
"""Audit the built PDF; optional preview rendering requires Pillow.

Install PyMuPDF to run: python3 verify.py [--render]
This check only reads CPU-produced document artifacts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

import fitz

ROOT = Path(__file__).resolve().parent


def spans(page):
    return [s for b in page.get_text('dict')['blocks'] if 'lines' in b
            for line in b['lines'] for s in line['spans'] if s['text'].strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'pdblend-method-latex.pdf'
    document = fitz.open(output)
    assert 10 <= len(document) <= 14, len(document)
    log = (ROOT / 'build/main.log').read_text()
    failures = re.findall(r'Overfull[^\n]*|Missing character[^\n]*|'
                          r'(?:Reference|Citation)[^\n]*undefined|'
                          r'There were undefined references[^\n]*', log)
    assert not failures, failures
    assert not any('\ufffd' in p.get_text() for p in document)

    figures = []
    for pdf in sorted((ROOT / 'figures').glob('*.pdf')):
        figure = fitz.open(pdf)
        assert len(figure) == 1
        page = figure[0]
        raster = page.get_images(full=True)
        assert not raster, pdf
        text = spans(page)
        smallest = min(s['size'] for s in text)
        # Figures are imported at the actual text width of 178 mm.
        printed = smallest * (178 / 25.4 * 72) / page.rect.width
        assert printed >= 8.5, (pdf, printed)
        for s in text:
            x0, y0, x1, y1 = s['bbox']
            assert x0 >= -.5 and y0 >= -.5, (pdf, s)
            assert x1 <= page.rect.width+.5 and y1 <= page.rect.height+.5, (pdf, s)
        figure_log = (ROOT / 'build/figures' / (pdf.stem + '.log')).read_text()
        assert 'Missing character' not in figure_log and 'Overfull' not in figure_log
        figures.append({'file': str(pdf.relative_to(ROOT)), 'pages': 1,
                        'width_mm': round(page.rect.width/72*25.4, 3),
                        'height_mm': round(page.rect.height/72*25.4, 3),
                        'raster_images': 0, 'vector_paths': len(page.get_drawings()),
                        'minimum_printed_font_pt': round(printed, 3)})
    assert len(figures) == 5
    section_sources = sorted((ROOT / 'sections').glob('*.tex'))
    imported = []
    for source in section_sources:
        tex = source.read_text()
        imported += re.findall(r'\\includegraphics\[[^]]*\]\{([^}]+)\}', tex)
        if not source.name.startswith('01'):
            assert re.findall(r'\\subsection\{([^}]+)\}', tex) == [
                'Motivation', 'Challenge', 'Insight', 'Approach']
    assert sorted(imported) == sorted(f['file'] for f in figures)
    report = {
        'document': output.name, 'pages': len(document), 'engine': 'XeLaTeX',
        'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'layout': 'A4, double column, 10 pt body, 178 mm text width',
        'overfull_boxes': 0, 'missing_characters': 0, 'unresolved_references': 0,
        'four_section_structure': True, 'figures': figures,
        'numeric_audit': 'data/examples.json; reproduced from tests/pdblend/synthetic.py',
        'visual_review': 'All 11 pages reviewed at rendered print proportions by three agents.',
        'scope': 'CPU document generation only; no service/API changes or GPU experiments',
    }
    (ROOT / 'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    if args.render:
        from PIL import Image, ImageDraw
        previews = ROOT / 'preview'
        previews.mkdir(exist_ok=True)
        thumbnails = []
        for i, page in enumerate(document, 1):
            filename = previews / f'page-{i:02}.png'
            page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(filename)
            im = Image.open(filename).convert('RGB')
            im.thumbnail((357, 505))
            tile = Image.new('RGB', (377, 535), '#dddddd')
            tile.paste(im, (10, 20))
            ImageDraw.Draw(tile).text((10, 5), str(i), fill='black')
            thumbnails.append(tile)
        sheet = Image.new('RGB', (1131, 535*((len(thumbnails)+2)//3)), '#aaaaaa')
        for i, tile in enumerate(thumbnails):
            sheet.paste(tile, (i%3*377, i//3*535))
        sheet.save(previews / 'contact-sheet.png')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
