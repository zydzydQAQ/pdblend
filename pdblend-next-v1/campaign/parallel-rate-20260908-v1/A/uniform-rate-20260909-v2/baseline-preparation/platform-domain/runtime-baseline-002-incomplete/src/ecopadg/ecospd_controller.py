#!/usr/bin/env python3
"""Compatible entry point; explicit configs use the modular three-pool runtime.

The old import API remains available for historical experiments and their
regression tests. New runs pass --config or use ecopadg.serving.controller.
"""
import os
import sys

# Direct file execution must not shadow the standard-library types module.
_package = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _package]
sys.path.insert(0, os.path.dirname(_package))

if __name__ == "__main__":
    if any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]):
        from ecopadg.serving.controller import main
        main()
    else:
        import asyncio
        from ecopadg.legacy_controller import main
        asyncio.run(main())
else:
    # Preserve module identity so existing callers' monkeypatches affect the
    # historical implementation, instead of a detached copy of its globals.
    from ecopadg import legacy_controller as _legacy
    sys.modules[__name__] = _legacy
