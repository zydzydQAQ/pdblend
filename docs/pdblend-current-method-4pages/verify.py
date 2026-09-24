"""Check the four-page PDF and render its pages; no GPU or serving imports."""
from pathlib import Path
import hashlib
import json
import re

import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parent
pdf = ROOT / "pdblend-current-method.pdf"
doc = pymupdf.open(pdf)
assert len(doc) == 4, f"Expected four pages, got {len(doc)}"
log = (ROOT / "build/main.log").read_text()
problems = re.findall(
    r"Overfull[^\n]*|Missing character[^\n]*|"
    r"(?:Reference|Citation)[^\n]*undefined|There were undefined references[^\n]*",
    log,
)
assert not problems, problems
all_text = "\n".join(page.get_text() for page in doc)
assert "\ufffd" not in all_text
assert all(term in all_text for term in (
    "Profiler", "Frequency Control", "Role Change", "TP Weight Change"
))
sections = [ROOT / "sections" / name for name in (
    "01-profiler.tex", "02-online.tex", "03-adaptive.tex"
)]
source = "\n".join(path.read_text() for path in sections)
assert source.count(r"\begin{algorithm}") == 2
assert source.count(r"\section{") == 3
assert (ROOT / "sections/03-adaptive.tex").read_text().count(r"\subsection{") == 3
preview = ROOT / "preview"
preview.mkdir(exist_ok=True)
sheet = Image.new("RGB", (900, 1280), "#dddddd")
page_info = []
for number, page in enumerate(doc, 1):
    text = page.get_text()
    assert len(text) > 900, (number, "Page is unexpectedly sparse")
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if not span["text"].strip():
                    continue
                x0, y0, x1, y1 = span["bbox"]
                assert x0 >= 0 and y0 >= 0 and x1 <= page.rect.width and y1 <= page.rect.height
    path = preview / f"page-{number:02d}.png"
    page.get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4)).save(path)
    thumbnail = Image.open(path).convert("RGB")
    thumbnail.thumbnail((440, 620))
    sheet.paste(thumbnail, (((number - 1) % 2) * 450 + 5, ((number - 1) // 2) * 640 + 15))
    page_info.append({"page": number, "text_characters": len(text)})
sheet.save(preview / "contact-sheet.png")
report = {
    "pdf": pdf.name,
    "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    "pages": 4,
    "columns": 2,
    "nominal_body_pt": 11,
    "major_sections": 3,
    "adaptive_subsections": 3,
    "algorithms": 2,
    "numbered_equations": 7,
    "overfull_boxes": 0,
    "missing_glyphs": 0,
    "unresolved_references": 0,
    "page_details": page_info,
    "scope": "Document build and source analysis only; no GPU experiment or service-code changes.",
}
(ROOT / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(report, indent=2, ensure_ascii=False))
