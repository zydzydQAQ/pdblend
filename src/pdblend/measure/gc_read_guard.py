"""Defer automatic cyclic GC during one unchanged eight-GPU power read.

This process-wide guard does not block explicit gc.collect(), OS scheduling,
or NVML delays. It never retries, removes samples, or relaxes their validation.
Concurrent guarded reads restore the prior GC state only after the last exit.
Other code must not independently toggle GC while these scoped reads overlap.
"""
from contextlib import contextmanager
import gc
import threading

_state_lock=threading.Lock()
_readers=0
_restore_enabled=False


@contextmanager
def gc_read_guard():
    global _readers,_restore_enabled
    with _state_lock:
        if _readers==0:
            _restore_enabled=gc.isenabled()
            gc.disable()
        _readers+=1
    try:
        yield
    finally:
        with _state_lock:
            _readers-=1
            if _readers==0 and _restore_enabled:
                gc.enable()
