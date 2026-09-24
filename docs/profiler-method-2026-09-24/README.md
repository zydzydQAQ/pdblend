# PDblend profiler method note

`pdblend-profiler.pdf` is a bilingual four-subsection note with formal cost
interfaces, the Mixed/Prefill/Decode models, a vector process diagram, and a
reproducible example from the existing 7B TP4 2100 MHz calibration component.

The example separates frozen-model queries, independent timing observations,
and a hypothetical Mixed workload. It does not claim a new GPU experiment or
full-system energy qualification. Mixed's algebraic example closes Little's
law at batch 12; its energy covers one TP4 instance for 60 seconds.

## Rebuild

From the repository, using its existing NumPy environment:

```sh
PYTHONDONTWRITEBYTECODE=1 /home/pdblend/.venv/bin/python -B \
  scripts/2026-09-24_build_profiler_note.py
```

The script verifies and explicitly loads the calibration version, recomputes
all example values, writes the numbers and evidence excerpts, and compiles
the diagram and PDF with XeLaTeX. This step requires the original local
calibration registry and its checksum-bound files.

For a typesetting-only rebuild using the saved numbers:

```sh
cd docs/profiler-method-2026-09-24
mkdir -p build
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build figures/profiler-process.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
cp build/main.pdf pdblend-profiler.pdf
```

`main.tex` and `figures/profiler-process.tex` are editable sources.
`data/example.json` records identities, exact arithmetic and qualification;
`data/profile-snapshot.json` is the unchanged candidate used for queries;
`data/holdout-excerpts.json` contains the selected original summaries.
`validation.json` records the final PDF and rendering checks.
