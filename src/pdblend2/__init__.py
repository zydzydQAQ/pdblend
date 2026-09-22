"""Deprecated compatibility namespace for the pre-rename ``pdblend2`` CLI.

New code must import from :mod:`pdblend`. The aliases keep a queued worker
that still imports ``pdblend2.bench`` (or another public subpackage) loadable
while its matrix run is being drained.
"""
from importlib import import_module
import sys

from pdblend import __version__

for _name in ("bench", "control", "engine", "profile", "proxy"):
    _module = import_module(f"pdblend.{_name}")
    sys.modules.setdefault(f"{__name__}.{_name}", _module)
    setattr(sys.modules[__name__], _name, _module)

__all__ = ["__version__"]
