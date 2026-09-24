"""Render a source-grounded PDBlend scheduling figure (CPU only).

Python packages: cairosvg, Pillow, PyMuPDF.  No serving code is imported.
"""
from pathlib import Path
from html import escape
import json
import subprocess

ROOT = Path(__file__).resolve().parent
W, H = 1800, 1490
INK, MUTED = '#161616', '#535353'
GREEN, BLUE, GREY, GOLD = '#a2cc83', '#4b78bf', '#d2d2d2', '#ffe080'
PALE, RED = '#f6f6f6', '#b43831'


class Figure:
    def __init__(self):
        self.texts = []
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
            '<title>PDBlend: request routing and per-instance scheduling</title>',
            '<desc>Default single-pool PDBlend with vLLM V1 continuous batching. The router uses input length and lexicographic load selection. Local waiting queues default to FCFS; running requests are scheduled first. P sends KV and the first token to continue on D. Mixed batches can contain prefill and decode together. Numerical loads and timelines are illustrative, not benchmark measurements.</desc>',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" fill="#161616"/></marker></defs>',
            f'<rect width="{W}" height="{H}" fill="white"/>']

    def text(self, x, y, value, size=24, bold=False, color=INK, anchor='start', italic=False):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Liberation Serif, Times New Roman, serif" font-size="{size}" font-weight="{"bold" if bold else "normal"}" font-style="{"italic" if italic else "normal"}" fill="{color}" text-anchor="{anchor}">{escape(str(value))}</text>')
        self.texts.append((x, y, str(value), size, bold, italic, anchor))

    def rect(self, x, y, w, h, fill='white', stroke=INK, sw=1.5, dash=False):
        ds = ' stroke-dasharray="7 5"' if dash else ''
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{ds}/>')

    def line(self, x1, y1, x2, y2, color=INK, sw=1.5, dash=False, arrow=False, both=False):
        self.path([(x1, y1), (x2, y2)], color, sw, dash, arrow, both)

    def path(self, pts, color=INK, sw=1.5, dash=False, arrow=False, both=False):
        d = 'M' + ' L'.join(f'{x},{y}' for x, y in pts)
        ds = ' stroke-dasharray="6 5"' if dash else ''
        ar = ' marker-end="url(#arrow)"' if arrow else ''
        ar += ' marker-start="url(#arrow)"' if both else ''
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{ds}{ar}/>')

    def label(self, x, y, w, h, value, fill=PALE, size=24, bold=False, color=INK):
        self.rect(x, y, w, h, fill)
        self.text(x+w/2, y+h/2+size*.34, value, size, bold, color, 'middle')

    def section(self, y, title):
        self.line(40, y-31, 1760, y-31, '#777', 1)
        self.text(40, y, title, 27, True)


def build():
    f = Figure()
    f.text(900, 43, 'PDBlend: request routing and per-instance scheduling', 34, True, anchor='middle')
    for x, c, label in [(450, GREEN, 'Prefill'), (660, BLUE, 'Decode'), (880, GREY, 'Wait / KV transfer'), (1220, GOLD, 'Output token')]:
        f.rect(x, 68, 28, 20, c)
        f.text(x+40, 86, label, 23)
    f.text(900, 116, 'Default single-pool path  ·  TP1 / PP1 illustration  ·  vLLM V1', 21, color=MUTED, anchor='middle')

    f.section(164, '(a) Global routing: length threshold, then lexicographic instance load')
    f.rect(40, 194, 300, 144)
    f.text(190, 224, 'Incoming requests', 26, True, anchor='middle')
    f.label(57, 239, 266, 37, 'R1: L = 512 tokens', size=24)
    f.label(57, 286, 266, 37, 'R2: L = 2048, K = 64', size=24)
    f.line(340, 266, 405, 266, arrow=True)

    f.rect(405, 211, 337, 220, '#fafafa', sw=2)
    f.text(573, 247, 'PDBlend Router', 29, True, anchor='middle')
    f.text(573, 280, 'Example threshold: τ = 1024', 23, anchor='middle')
    f.line(425, 295, 722, 295, '#777', 1)
    f.text(430, 327, 'L < τ    →    Mixed path', 25)
    f.text(430, 367, 'L ≥ τ    →    P + D path', 25)
    f.text(573, 409, 'Bind endpoints at dispatch', 22, color=MUTED, anchor='middle')

    f.rect(40, 371, 300, 86, '#fafafa', dash=True)
    f.text(190, 402, 'Planner + Controller', 24, True, anchor='middle')
    f.text(190, 433, 'Publish roles, admission and τ', 21, anchor='middle')
    f.line(340, 411, 405, 411, dash=True, arrow=True)

    f.path([(742, 326), (779, 326), (779, 251), (815, 251)], arrow=True)
    f.rect(815, 191, 945, 133)
    f.text(837, 224, 'Mixed: choose min (N_M, U_M)', 27, True)
    f.text(837, 258, 'M0 (3, 2048)    M1 (1, 1024)    M2 (1, 0)    M3 (2, 0)', 24)
    f.text(837, 294, 'R1 → M2: fewer unfinished requests; then less prefill load', 24)

    f.line(742, 367, 815, 367, arrow=True)
    f.rect(815, 348, 945, 155)
    f.text(837, 383, 'PD: choose min (U_P, N_D, U_D, pair IDs)', 27, True)
    f.text(837, 417, 'P0: U = 3072    P1: U = 1024     |     D0: N = 4    D1: N = 2', 24)
    f.text(837, 451, 'R2 → P1 → D1; both D instances have U = 0', 24)
    f.text(837, 484, 'P generates y1; D generates the remaining 63 tokens', 24)

    f.text(40, 533, 'N = unfinished sequences; U = prompt tokens awaiting first output. Counts include engine waiting and running work.', 22, color=MUTED)
    f.text(40, 563, 'Only accepting instances; PD endpoints must be compatible. If a path is unavailable, use the other; if neither is available, reject.', 21, color=MUTED)

    f.section(617, '(b) Inside each engine: continuous batching; FCFS applies to the waiting queue')
    f.rect(40, 644, 505, 104)
    f.text(292, 676, '1. RUNNING first', 27, True, anchor='middle')
    f.text(292, 706, 'Active decode + unfinished prefill chunks', 23, anchor='middle')
    f.text(292, 733, 'Advance eligible requests within the budget', 21, color=MUTED, anchor='middle')
    f.line(545, 696, 595, 696, arrow=True)
    f.rect(595, 644, 450, 104)
    f.text(820, 676, '2. WAITING next', 27, True, anchor='middle')
    f.text(820, 706, 'Default order: FCFS', 24, anchor='middle')
    f.text(820, 733, 'Admit when token / sequence / KV limits allow', 20, color=MUTED, anchor='middle')
    f.line(1045, 696, 1095, 696, arrow=True)
    f.rect(1095, 644, 665, 104)
    f.text(1427, 676, '3. One iteration may mix P and D', 26, True, anchor='middle')
    f.label(1120, 696, 78, 37, 'D(A)', BLUE, 22, color='white')
    f.label(1198, 696, 78, 37, 'D(B)', BLUE, 22, color='white')
    f.label(1276, 696, 460, 37, 'Prefill chunk of C', GREEN)
    f.text(40, 782, 'Launcher defaults: 8192 scheduled tokens per iteration; 256 running requests; chunked prefill enabled.', 23)
    f.text(40, 812, 'This is not request-by-request execution to completion. KV pressure may cause preemption and recomputation.', 22, color=MUTED)

    f.section(867, '(c) Example execution: separate P/D engines overlap; a Mixed engine can co-batch P and D')
    f.text(1760, 899, 'Time (schematic, not to scale)', 22, color=MUTED, anchor='end', italic=True)
    f.line(350, 911, 1760, 911, arrow=True)
    f.line(350, 904, 350, 1383, '#999', 1, dash=True)
    f.text(338, 928, 'arrival', 20, color=MUTED, anchor='end')

    f.text(40, 970, 'P1 instance', 27, True)
    f.text(40, 999, 'Prefill + first token', 22, color=MUTED)
    f.rect(350, 940, 95, 48, GREY)
    f.text(397, 971, 'wait', 21, anchor='middle')
    f.label(445, 940, 400, 48, 'Prefill R2 → y1', GREEN, 27)
    f.label(865, 940, 340, 48, 'Prefill R4 → first token', GREEN, 25)
    f.label(1225, 940, 380, 48, 'More prefill requests …', GREEN, 25)
    f.line(1605, 964, 1758, 964, arrow=True)

    f.text(40, 1042, 'P2P KV transfer', 24, True)
    f.text(350, 1042, 'May overlap prefill', 22, color=MUTED)
    f.rect(650, 1015, 175, 36, GREY)
    f.text(737, 1041, 'Layer-wise KV', 23, True, anchor='middle')
    f.path([(670, 987), (670, 1014)], arrow=True)
    f.path([(825, 1033), (1008, 1033), (1008, 1088)], arrow=True)
    f.text(1043, 1043, 'D receives X + y1; reuses KV(X)', 23)

    f.text(40, 1122, 'D1 instance', 27, True)
    f.text(40, 1151, 'Batched token steps', 22, color=MUTED)
    f.label(350, 1089, 495, 48, 'Decode other requests …', BLUE, 26, color='white')
    for x in range(405, 845, 55):
        f.line(x, 1090, x, 1100, 'white', 1)
        f.line(x, 1127, x, 1136, 'white', 1)
    f.label(845, 1089, 163, 48, 'handoff / wait', GREY, 23)
    for i, x in enumerate(range(1008, 1533, 75)):
        f.label(x, 1089, 75, 48, f'y{i+2}', BLUE, 25, color='white')
    f.label(1533, 1089, 110, 48, '… y64', BLUE, 24, color='white')
    f.line(1643, 1113, 1758, 1113, arrow=True)
    f.text(1015, 1170, 'First D step consumes y1 to produce y2', 23)

    f.text(40, 1245, 'M2 instance', 27, True)
    f.text(40, 1274, 'Local prefill + decode', 22, color=MUTED)
    f.label(350, 1210, 200, 64, 'Prefill R1', GREEN, 26)
    for x in range(550, 750, 50):
        f.label(x, 1210, 50, 64, 'D', BLUE, 23, color='white')
    for x in (750, 850):
        f.label(x, 1210, 100, 32, 'P: R3', GREEN, 21)
        f.label(x, 1242, 100, 32, 'D: R1', BLUE, 21, color='white')
    f.text(850, 1198, 'R3 joins the batch', 22, color=RED, anchor='middle')
    for x in range(950, 1550, 100):
        f.rect(x, 1210, 100, 64, BLUE)
    f.text(1250, 1250, 'Decode R1 + R3 (continuous batches)', 24, color='white', anchor='middle')
    f.line(1550, 1242, 1758, 1242, arrow=True)
    f.text(350, 1306, 'Stacked green / blue blocks = requests in one batch, not independent parallel GPU streams.', 22, color=MUTED)

    f.text(40, 1380, 'Client: R2 tokens', 25, True)
    f.line(350, 1384, 1760, 1384, arrow=True)
    token_xs = [845, 1083, 1158, 1233, 1308, 1383, 1458, 1533]
    for i, x in enumerate(token_xs):
        f.rect(x-8, 1358, 17, 26, GOLD)
        f.text(x, 1410, f'y{i+1}', 20, anchor='middle')
    f.text(1650, 1378, '… y64', 23)
    f.line(350, 1342, 845, 1342, both=True, arrow=True, dash=True)
    f.text(597, 1334, 'TTFT', 23, color='#487c36', anchor='middle', italic=True)
    f.line(845, 1342, 1083, 1342, both=True, arrow=True, dash=True)
    f.text(964, 1334, 'handoff + first D step', 19, color=MUTED, anchor='middle')
    f.line(1158, 1342, 1233, 1342, both=True, arrow=True, dash=True)
    f.text(1195, 1334, 'TPOT', 22, color=BLUE, anchor='middle', italic=True)

    f.text(900, 1452, 'Figure. Default PDBlend scheduling. Panel (a) is a load snapshot; panel (c) is a separate illustrative schedule, not a measured trace.', 22, anchor='middle')
    f.text(900, 1478, 'Stable roles shown. Optional resident-pool / energy routing is outside this figure. FCFS does not imply global completion order.', 20, color=MUTED, anchor='middle')
    f.parts.append('</svg>')
    return f


def render(f):
    import cairosvg
    import fitz
    from PIL import ImageFont

    svg = ROOT / 'pdblend-scheduling.svg'
    svg.write_text('\n'.join(f.parts), encoding='utf-8')
    raw = svg.read_bytes()
    cairosvg.svg2pdf(bytestring=raw, write_to=str(ROOT/'pdblend-scheduling.pdf'))
    cairosvg.svg2png(bytestring=raw, write_to=str(ROOT/'pdblend-scheduling.png'), scale=1.6)
    fonts, boxes, outside = {}, [], []
    for x,y,text,size,bold,italic,anchor in f.texts:
        style = ('Bold Italic' if italic else 'Bold') if bold else ('Italic' if italic else 'Regular')
        key = (size, style)
        if key not in fonts:
            path = subprocess.check_output(['fc-match', '-f', '%{file}', f'Liberation Serif:style={style}'], text=True)
            fonts[key] = ImageFont.truetype(path, size)
        font = fonts[key]
        width = font.getlength(text)
        left = x - (width/2 if anchor=='middle' else width if anchor=='end' else 0)
        bbox = font.getbbox(text, anchor='ls')
        box = (left+bbox[0], y+bbox[1], left+bbox[2], y+bbox[3])
        boxes.append((box, text))
        if box[0] < 0 or box[2] > W or box[1] < 0 or box[3] > H:
            outside.append(text)
    overlaps = []
    for i, (a, ta) in enumerate(boxes):
        for b,tb in boxes[i+1:]:
            if min(a[2], b[2])-max(a[0], b[0])>2 and min(a[3], b[3])-max(a[1], b[1])>2:
                overlaps.append([ta, tb])
    pdf = fitz.open(ROOT/'pdblend-scheduling.pdf')
    report = dict(width=W, height=H, pdf_pages=len(pdf), text_items=len(f.texts),
                  searchable_characters=len(pdf[0].get_text()), outside_canvas=outside,
                  overlapping_text=overlaps, timing_is_measured=False)
    (ROOT/'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if outside or overlaps:
        raise RuntimeError('Figure text needs layout adjustment')


if __name__ == '__main__':
    render(build())
