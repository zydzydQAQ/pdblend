#!/usr/bin/env bash
# Create this checkout's Python 3.10 controller environment from the v3 lock.
set -euo pipefail
PDBLEND_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PDBLEND_PYTHON="${PDBLEND_PYTHON:-python3.10}"
PDBLEND_VENV="${PDBLEND_VENV:-$PDBLEND_PROJECT/.venv}"
"$PDBLEND_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,10), "Use Python 3.10"'
if [[ ! -e "$PDBLEND_VENV" ]]; then
    "$PDBLEND_PYTHON" -m venv "$PDBLEND_VENV"
fi
"$PDBLEND_VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,10), "Existing venv is not Python 3.10"'
"$PDBLEND_VENV/bin/python" -m pip install --no-deps -r "$PDBLEND_PROJECT/requirements/pdblend4-v3-host.lock"
"$PDBLEND_VENV/bin/python" -m pip install --no-deps --no-build-isolation -e "$PDBLEND_PROJECT"
"$PDBLEND_VENV/bin/python" -m pip check
"$PDBLEND_VENV/bin/python" -c 'import aiohttp, msgpack, numpy, pandas, psutil, pulp, scipy, simpy, torch, transformers, zmq; print("Host controller dependencies ready:", torch.__version__, transformers.__version__)'
