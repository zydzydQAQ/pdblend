"""Explicit import-only placeholder when aiohttp is unavailable on the CPU host.

The checked constructors/planner/early rejection do not use HTTP. This module
does not emulate HTTP and any attempt to use it raises AttributeError.
"""
import runpy
import sys
import types
from pathlib import Path

assert 'aiohttp' not in sys.modules
stub=types.ModuleType('aiohttp')
stub.CPU_IMPORT_STUB=True
stub.web=types.ModuleType('aiohttp.web')
sys.modules['aiohttp']=stub
sys.modules['aiohttp.web']=stub.web
runpy.run_path(str(Path(__file__).with_name('verify.py')),run_name='__main__')
