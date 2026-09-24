"""Compatibility alias for :mod:`pdblend_baselines.mixed.policy`."""
from importlib import import_module as _import_module
import sys as _sys
_impl = _import_module('pdblend_baselines.mixed.policy')
_sys.modules[__name__] = _impl
