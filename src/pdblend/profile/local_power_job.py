"""Compatibility alias for :mod:`pdblend.profile.collection.local_power_job`."""
from importlib import import_module as _import_module
import sys as _sys
_impl = _import_module('pdblend.profile.collection.local_power_job')
if __name__ == "__main__":
    _impl.main()
else:
    _sys.modules[__name__] = _impl
