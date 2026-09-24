#!/usr/bin/env python3
"""Stable entrance to the currently bound campaign implementation."""
from pathlib import Path
import runpy
if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1] / '2026-09-22_gpu_campaign_queue.py'), run_name='__main__')
