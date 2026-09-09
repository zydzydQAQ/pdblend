"""Passive child-only evidence around unchanged ClockOwner/hardware methods.

Recording errors never suppress a clock operation or its cleanup. The later
evidence gate fails if the journal/status is missing, incomplete, or errored.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time


@contextmanager
def capture_clock_commands(operation):
    from ecopadg.serving import backend, runtime
    original_backend, original_runtime = backend.ClockOwner, runtime.ClockOwner
    if original_backend is not original_runtime:
        raise RuntimeError('ClockOwner aliases differ; refuse incomplete observation')
    operation = Path(operation)
    errors, owners = [], []
    lock = threading.Lock()
    log = (operation / 'clock-commands.jsonl').open('x', buffering=1)
    sequence = 0

    def record(kind, **fields):
        nonlocal sequence
        # Worker-thread hardware writes and event-loop intent share one order.
        with lock:
            sequence += 1
            assigned = sequence
            try:
                log.write(json.dumps(dict(sequence=sequence, kind=kind, pid=os.getpid(),
                    observed_s=time.time(), **fields), allow_nan=False)+'\n')
                log.flush()
            except Exception as exc:
                errors.append(repr(exc))
        return assigned

    class HardwareObserver:
        def __init__(self, target, owner):
            self.target, self.owner = target, owner

        def __getattr__(self, name):
            return getattr(self.target, name)

        def invoke(self, method, gpu, *args):
            extra = {'frequency_mhz': args[0]} if method == 'set_clock' else {}
            call = record('hardware_begin', owner=self.owner, method=method, gpu=gpu, **extra)
            try:
                result = getattr(self.target, method)(gpu, *args)
            except BaseException as exc:
                record('hardware_end', owner=self.owner, call=call, complete=False, error=repr(exc))
                raise
            record('hardware_end', owner=self.owner, call=call, complete=True)
            return result

        def set_clock(self, gpu, frequency):
            return self.invoke('set_clock', gpu, frequency)

        def reset_clock(self, gpu):
            return self.invoke('reset_clock', gpu)

    class ObservedOwner(original_backend):
        def __init__(self, hardware, gpus, *args, **kwargs):
            owner = len(owners)+1
            self._observation_owner = owner
            super().__init__(HardwareObserver(hardware, owner), gpus, *args, **kwargs)
            owners.append(owner)
            record('owner_acquired', owner=owner, gpus=list(self.gpus))

        async def set(self, gpus, frequency, *, verify_rise=True):
            gpus = tuple(gpus)
            call = record('owner_set', owner=self._observation_owner, gpus=list(gpus),
                          frequency_mhz=frequency, verify_rise=verify_rise)
            try:
                result = await super().set(gpus, frequency, verify_rise=verify_rise)
            except BaseException as exc:
                record('owner_set_end', call=call, complete=False, error=repr(exc))
                raise
            record('owner_set_end', call=call, complete=True)
            return result

        async def park(self, gpus, expected_epochs=None):
            gpus = tuple(gpus)
            record('owner_park', owner=self._observation_owner, gpus=list(gpus))
            return await super().park(gpus, expected_epochs=expected_epochs)

        async def close(self):
            record('owner_close', owner=self._observation_owner)
            try:
                result = await super().close()
            except BaseException as exc:
                record('owner_close_end', owner=self._observation_owner, complete=False, error=repr(exc))
                raise
            record('owner_close_end', owner=self._observation_owner, complete=True)
            return result

    backend.ClockOwner = runtime.ClockOwner = ObservedOwner
    try:
        yield
    finally:
        backend.ClockOwner, runtime.ClockOwner = original_backend, original_runtime
        try:
            log.close()
        except Exception as exc:
            errors.append(repr(exc))
        status = dict(schema=1, pid=os.getpid(), finished_s=time.time(), owners=owners,
                      records=sequence, recording_errors=errors, complete=not errors,
                      aliases_restored=True, observation_only=True)
        # A failed final write also leaves the required status absent: the
        # parent refuses a checkpoint while keeping all primary/outer energy.
        with (operation / 'clock-commands-status.json').open('x') as handle:
            json.dump(status, handle, indent=2, allow_nan=False)
            handle.write('\n')
