# PDblend — bilingual LaTeX method note

The final document is `pdblend-method-latex.pdf`. This is a four-section,
double-column XeLaTeX document with five independently compiled vector PDF
figures. Chinese prose explains the method; English paragraphs can serve as a
starting point for paper writing. All example performance numbers are synthetic,
not measurements or experimental claims.

## Build (CPU only)

Run from this directory, or invoke the script by its absolute path:

```bash
bash build.sh
```

Requirements: Python 3, XeLaTeX, latexmk, and a TeX Live installation containing
ctex/Fandol, standalone, TikZ/PGFPlots, algorithms, booktabs, and fontspec.
On Ubuntu these are provided by:

```bash
apt-get install texlive-xetex texlive-latex-extra texlive-lang-chinese texlive-science latexmk
```

The build uses three CPU workers for the figures and latexmk for automatic
cross-reference resolution. No serving code, public API, CUDA, or GPU experiment
is used or modified. TeX Gyre Termes/Heros are used when available, with Latin
Modern fallbacks; Chinese uses the Fandol fonts supplied by TeX Live.

## Editable sources

- `main.tex`: typography, macros, title, and four-section composition.
- `sections/*.tex`: bilingual prose, native tables, formulas, and algorithms.
- `figures/src/01-overview.tex` through `05-control.tex`: standalone TikZ figures.
- `figures/src/common.tex`: figure typography, colors, and common styles.
- `figures/*.pdf`: vector figures actually imported with `includegraphics`.
- `data/reproduce.py` and `data/examples.json`: independent numerical audit.
- `validation.json`: final document and figure checks.
- `verify.py`: optional PDF audit and preview generation (requires PyMuPDF;
  `--render` also requires Pillow).

To reproduce the numeric audit against the PDblend repository, use its CPU Python
environment with NumPy installed:

```bash
python3 data/reproduce.py --repo /path/to/pdblend
```

The document build itself does not require the repository or model dependencies.
The checked JSON values and complete TeX/figure sources are included in the
source archive. Recomputing model outputs requires the source revision/hashes
recorded by the audit script.

## Reading the examples

All examples use eight TP1 instances and output budget 128. The model illustration
uses the repository's `tests/pdblend/synthetic.py`. The planning figure compares
only three named candidates at 12 requests/s over 60 seconds; it does not claim
the complete search optimum. Its current model latency proxy is not measured
client TTFT. Unknown extra handoff energy is left unknown and bounded below by
zero. The routing and control figures start from a separate effective
P1+D2+M4+L1 configuration, not the output of that candidate comparison.

Each technical section marks the distinction between the target design and
current implementation, including energy accounting, versioned readiness, and
complete drain confirmation. The previous document under
`../pdblend-method/` is retained.
