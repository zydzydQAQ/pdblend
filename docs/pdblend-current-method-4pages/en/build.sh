#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p build
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build pdblend-method.tex
cp build/pdblend-method.pdf pdblend-method.pdf
