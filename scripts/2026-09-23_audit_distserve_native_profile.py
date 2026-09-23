#!/usr/bin/env python3
"""CPU-only independent DistServe native receipt/search integration audit."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from pdblend_baselines.distserve.native_audit import main

if __name__ == '__main__':
    raise SystemExit(main())
