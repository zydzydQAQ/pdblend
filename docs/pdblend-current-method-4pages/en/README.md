# PDblend method: English edition

The standalone pdblend-method.tex is an English adaptation of the Chinese
four-page method note in the parent directory. It retains three major sections,
two algorithms, and seven numbered equations. The PDF uses A4 paper,
two columns, and an 11 pt body, and is four pages long.

The English edition incorporates the subsequent source review: default
ResidentRouter scoring has no explicit transfer term; its optional SLO/energy
context does. It also states the measured endpoint component's limited domain
and clarifies FCFS admission versus continuous batching and TP execution.
TP weight change means traffic-share adjustment across resident TP pools.

Build with:

    bash build.sh

The source requires pdfLaTeX, latexmk, mathptmx, geometry, amsmath,
algorithms/algorithmicx, microtype, titlesec, flushend, and hyperref. It has no
external figures, bibliography, Chinese-font dependency, or other input files.
The final PDF is pdblend-method.pdf.

Source evidence remains in the parent directory's notes and source-manifest.json.
This is a document translation and clarification, not a new GPU measurement.
