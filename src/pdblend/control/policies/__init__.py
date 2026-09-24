"""Compatibility alias for :mod:`pdblend.online.policies`."""
from importlib import import_module as _import_module
import sys as _sys
_impl = _import_module('pdblend.online.policies')
from pathlib import Path as _Path
_impl.__path__ = [str(_Path(__file__).parent)]
if __name__ == "__main__":
    _impl.main()
else:
    _sys.modules[__name__] = _impl
