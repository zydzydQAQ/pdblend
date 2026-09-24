"""Run blocking clock operations outside asyncio, serialized by physical GPU."""
from __future__ import annotations

import asyncio
import contextlib
import fcntl
import os
import threading
from pathlib import Path

from pdblend.measure.backends import physical_gpu

_locks: dict[str, threading.RLock] = {}
_guard = threading.Lock()


def _run(gpus, function):
    ids = sorted({str(physical_gpu(gpu)) for gpu in gpus})
    root = Path(os.environ.get('PDBLEND_CLOCK_LOCK_DIR', '/tmp/pdblend-physical-clock-owners'))
    root.mkdir(parents=True, exist_ok=True)
    with _guard:
        locks = [_locks.setdefault(gpu, threading.RLock()) for gpu in ids]
    with contextlib.ExitStack() as stack:
        for gpu, lock in zip(ids, locks):
            stack.enter_context(lock)
            safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in gpu)
            handle = stack.enter_context((root / (safe + '.action.lock')).open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX)
        return function()


async def gpu_action(gpus, function):
    # Shield the worker from coroutine cancellation: it must finish and release
    # its physical locks before a caller can start recovery on the same GPUs.
    task = asyncio.create_task(asyncio.to_thread(_run, tuple(gpus), function))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise
