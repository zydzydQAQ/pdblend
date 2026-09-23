#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
for executable in xelatex latexmk; do
  if ! command -v "$executable" >/dev/null; then
    echo "Missing $executable. See README.md for TeX Live dependencies." >&2
    exit 1
  fi
done
mkdir -p build/figures
python3 - <<'PY'
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil, subprocess
def compile_figure(source):
    log = Path('build/figures') / (source.stem + '.console.log')
    with log.open('w') as stream:
        result = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error',
            '-file-line-error', '-output-directory=build/figures', str(source)],
            stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Figure failed: {source}; inspect {log}')
    shutil.copy2(Path('build/figures') / (source.stem + '.pdf'),
                 Path('figures') / (source.stem + '.pdf'))
    print(f'Compiled {source.stem}', flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(compile_figure, sorted(Path('figures/src').glob('[0-9][0-9]-*.tex'))))
PY
latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir=build main.tex > build/main.console.log 2>&1 || {
    tail -n 60 build/main.console.log >&2
    exit 1
  }
cp build/main.pdf pdblend-method-latex.pdf
echo "Built $(pwd)/pdblend-method-latex.pdf"
