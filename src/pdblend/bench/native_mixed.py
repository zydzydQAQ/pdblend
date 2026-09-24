"""Compatibility alias for :mod:`pdblend_baselines.mixed.run_native`."""
from importlib import import_module as _import_module
import sys as _sys
_impl = _import_module('pdblend_baselines.mixed.run_native')
_sys.modules[__name__] = _impl
