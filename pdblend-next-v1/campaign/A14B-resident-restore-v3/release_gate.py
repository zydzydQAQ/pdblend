"""A-only adapter for the user's explicitly authorized model150 scale release."""
import hashlib
import importlib.util
from pathlib import Path

CAMPAIGN=Path(__file__).resolve().parent.parent
SOURCE=CAMPAIGN/'model-main-release-v1/release.py'
SOURCE_SHA='33e5e9288e7c97a2717960eb2817cb958ce0d661f807be5d3093148a06e66dbf'

def verifier():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise RuntimeError('frozen model release verifier changed')
    spec=importlib.util.spec_from_file_location('a_restore_v3_model_release',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.package_check()
    return module

def verify_release(path,expected_sha256,**kwargs):
    if kwargs.pop('expected_model','14b')!='14b':
        raise RuntimeError('A restore accepts only the14b model release')
    return verifier().verify_release(path,expected_sha256,expected_model='14b',**kwargs)

def process_scan(*args,**kwargs):
    return verifier().v1.process_scan(*args,**kwargs)
