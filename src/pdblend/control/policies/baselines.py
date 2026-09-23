"""Compatibility alias for :mod:`pdblend.legacy.baselines`."""
from importlib import import_module as _import_module
import sys as _sys
_impl = _import_module('pdblend.legacy.baselines')
if __name__ == "__main__":
    _impl.main()
else:
    _sys.modules[__name__] = _impl
