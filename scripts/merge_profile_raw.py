#!/usr/bin/env python3
"""Compatibility entrance; maintained implementation: scripts/profile/merge_raw.py."""
from pathlib import Path
import runpy
if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parent / 'profile/merge_raw.py'), run_name='__main__')
