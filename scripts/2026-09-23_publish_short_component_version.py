#!/usr/bin/env python3
"""Compatibility entrance; maintained implementation: scripts/profile/publish_short_component.py."""
from pathlib import Path
import runpy
if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parent / 'profile/publish_short_component.py'), run_name='__main__')
