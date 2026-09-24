"""Produce a standalone LaTeX file and a portable source archive."""
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent
main = (ROOT / "main.tex").read_text()
standalone = re.sub(
    r"\\input\{(sections/[^}]+)\}",
    lambda m: (ROOT / (m[1] + ".tex")).read_text(),
    main,
)
(ROOT / "pdblend-method.tex").write_text(standalone)
files = [
    "main.tex", "pdblend-method.tex", "build.sh", "package.py", "verify.py",
    "README.md", "source-manifest.json",
    "sections/01-profiler.tex", "sections/02-online.tex", "sections/03-adaptive.tex",
    "notes/profiler.md", "notes/online.md", "notes/adaptive.md",
]
with zipfile.ZipFile(ROOT / "pdblend-method-source.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for name in files:
        z.write(ROOT / name, name)
print("Created pdblend-method.tex and pdblend-method-source.zip")
