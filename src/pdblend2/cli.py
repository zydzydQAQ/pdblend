"""Deprecated ``pdblend2`` CLI shim; use ``pdblend`` instead."""
from pdblend.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
