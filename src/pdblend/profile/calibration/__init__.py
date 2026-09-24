"""Compatibility alias for :mod:`pdblend.profile.calibration.core`."""
from importlib import import_module as _import_module
import sys as _sys
_impl = _import_module('pdblend.profile.calibration.core')
from pathlib import Path as _Path
_impl.__path__ = [str(_Path(__file__).parent)]
_impl.core = _impl
# ``from calibration import child`` uses the parent's __name__ to import a
# missing child. The facade must retain the package name, or import order can
# create both calibration.short_domain and calibration.core.short_domain with
# distinct exception/classes even though they come from the same file.
_impl.__name__ = __name__
_impl.__package__ = __name__
if __name__ == "__main__":
    _impl.main()
else:
    _sys.modules[__name__] = _impl
