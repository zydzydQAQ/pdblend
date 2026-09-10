"""Binding-reference entrypoint for the independent retained native audit."""
import importlib.util
from pathlib import Path


def verify(reference):
    path = Path(__file__).resolve().with_name('qualify_retained.py')
    spec = importlib.util.spec_from_file_location('_uniform_v2_retained_binding_verifier', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit_binding(reference)
