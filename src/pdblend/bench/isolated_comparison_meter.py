"""Run the unchanged comparison sampler in an owned spawned process.

Raw power samples, physical UUID validation, NVML source, polling cadence,
integration and the one-second gap limit stay in ComparisonMeteringSession.
The wrapper also enables PowerSampler's existing read-only clock observations
and appends those observations to snapshots. Method evidence is separate.
Snapshot/stop RPCs serialize data and must occur outside service AND drain-tail
measurement. Optional local begin_window/end_window guards enforce this rule.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import math
import multiprocessing
import os
from pathlib import Path
import time


SCHEMA = 'isolated-comparison-meter/v1'


def _implementation(factory):
    path = inspect.getsourcefile(factory)
    return dict(module=factory.__module__, name=factory.__qualname__,
                path=str(Path(path).resolve()) if path else None,
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None)


def _snapshot(session):
    value = session.snapshot()
    # As with the original power metadata snapshot, only copy the matching
    # prefix if the sampler appends another row during this operation.
    value['frequency_samples'] = list(session.sampler.frequency_samples[:len(value['samples'])])
    return value


def _liveness(session):
    sampler = session.sampler
    thread = getattr(sampler, '_thread', None)
    return dict(observed_s=time.time(), child_pid=os.getpid(),
                sampler_thread_alive=thread is not None and thread.is_alive(),
                sampler_error=getattr(sampler, 'error', None),
                sampler_error_at_s=getattr(sampler, 'error_at_s', None))


def _worker(connection, parent_pid, gpus, gpu_uuids, interval_s, max_gap_s, session_factory):
    session = None
    stopped = False
    operation, sequence = 'start', 0
    try:
        # Import and instantiate the production sampler only in the child.
        from .comparison_metering import ComparisonMeteringSession
        from pdblend.measure.power import PowerSampler
        from pdblend.measure.backends import PynvmlBackend
        factory = session_factory or ComparisonMeteringSession
        session = factory(gpus, gpu_uuids, interval_s=interval_s, max_gap_s=max_gap_s)
        session.sampler.sample_clocks = True
        session.start()
        connection.send(dict(operation=operation, sequence=sequence, ok=True,
            liveness=_liveness(session),
            payload=dict(child_pid=os.getpid(), started_s=time.time(),
                sample_clocks=True, sampler=_implementation(type(session)), factory=_implementation(factory),
                public_sampler=_implementation(ComparisonMeteringSession),
                power_sampler=_implementation(PowerSampler), backend=_implementation(PynvmlBackend))))
        while True:
            if not connection.poll(.25):
                if os.getppid() != parent_pid:
                    break
                continue
            command = connection.recv()
            operation, sequence = command['operation'], command['sequence']
            if operation == 'snapshot':
                payload = _snapshot(session)
            elif operation == 'stop':
                session.stop(**command['arguments'])
                stopped = True
                payload = _snapshot(session)
            else:
                raise ValueError('unsupported isolated meter operation')
            connection.send(dict(operation=operation, sequence=sequence, ok=True,
                                 payload=payload, liveness=_liveness(session)))
            if operation == 'stop':
                break
    except (EOFError, BrokenPipeError):
        pass  # Owner disappeared; finish the owned sampler below.
    except BaseException as exc:
        try:
            connection.send(dict(operation=operation, sequence=sequence, ok=False,
                                 error=f'{type(exc).__name__}: {exc}'))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if session is not None and not stopped:
            try:
                session.stop()
            except BaseException:
                pass
        connection.close()


class IsolatedComparisonMeter:
    """Drop-in lifecycle wrapper; no automatic restart after a sampling failure.

    ``session_factory`` exists for CPU tests and is recorded explicitly. Real
    measurements leave it unset. Factory arguments contain no live NVML object.
    Call ``method_receipt`` and bind its output beside, not inside, power.json.
    ``begin_window``/``end_window`` are local guards, not sampling timestamps.
    Their scope must include tail draining. All public timestamps remain the
    original sampler's per-device acquisition times.
    """
    def __init__(self, gpus, gpu_uuids, *, interval_s=.1, max_gap_s=1.,
                 startup_timeout_s=30., rpc_timeout_s=30., session_factory=None):
        for name, value in (('startup_timeout_s', startup_timeout_s), ('rpc_timeout_s', rpc_timeout_s)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('positive finite '+name+' required')
        self.gpus, self.gpu_uuids = tuple(gpus), tuple(gpu_uuids)
        self.interval_s, self.max_gap_s = interval_s, max_gap_s
        self.startup_timeout_s, self.rpc_timeout_s = startup_timeout_s, rpc_timeout_s
        self._factory, self._process, self._connection = session_factory, None, None
        self._status, self._sequence, self._window_active = 'new', 0, False
        self._final_snapshot = None
        self._receipt = dict(schema=SCHEMA, formal_eligible=False, hardware_preflight_qualified=False,
            method_difference='original comparison sampler in a separate Python process; existing PowerSampler frequency observations enabled',
            rationale='isolate uncertain interference from client work and in-process scheduling',
            wrapper=_implementation(type(self)),
            process_start_method='spawn', parent_pid=os.getpid(), child_pid=None, test_factory=session_factory is not None,
            gpu_ids=list(self.gpus), gpu_uuids=list(self.gpu_uuids), polling_interval_s=interval_s,
            maximum_interpolation_gap_s=max_gap_s, raw_snapshot_modified=True,
            public_snapshot_fields_modified=False, additional_snapshot_fields=['frequency_samples'],
            sample_clocks=True, additional_frequency_observation=True,
            frequency_observation_method='PowerSampler.sample_clocks; backend.current_freq for each bound GPU',
            frequency_timestamp_method='original PowerSampler power-row timestamp',
            rpc_scope='outside_service_and_drain_tail', commands=[], local_window_guards=[],
            liveness_observations=[])

    def _abort(self, reason):
        self._status = 'failed'
        self._receipt['error'] = reason
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._process is not None and self._process.pid is not None:
            self._process.join(timeout=.25)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1.)
            self._receipt['child_exitcode'] = self._process.exitcode
        self._receipt['finished_s'] = time.time()

    def _receive(self, operation, sequence, timeout):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if self._connection.poll(min(.05, max(0., deadline-time.monotonic()))):
                try:
                    response = self._connection.recv()
                except (EOFError, OSError) as exc:
                    raise RuntimeError('isolated meter child closed without a response') from exc
                if (response.get('operation') != operation or response.get('sequence') != sequence):
                    raise RuntimeError('isolated meter response identity differs')
                if response.get('ok') is not True:
                    raise RuntimeError('isolated meter '+operation+' failed: '+response.get('error', 'unknown error'))
                self._receipt['liveness_observations'].append(dict(
                    response['liveness'], operation=operation, sequence=sequence))
                return response['payload']
            if not self._process.is_alive():
                raise RuntimeError('isolated meter child exited: '+str(self._process.exitcode))
        raise TimeoutError('isolated meter '+operation+' timed out')

    def start(self):
        if self._status != 'new':
            raise RuntimeError('isolated meter cannot be restarted; create a new session')
        context = multiprocessing.get_context('spawn')
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(target=_worker, name='pdblend-comparison-meter',
            args=(child, os.getpid(), self.gpus, self.gpu_uuids, self.interval_s, self.max_gap_s, self._factory),
            daemon=True)
        self._receipt['start_requested_s'] = time.time()
        try:
            self._process.start()
            self._receipt['child_pid'] = self._process.pid
            child.close()
            ready = self._receive('start', 0, self.startup_timeout_s)
            if ready['child_pid'] != self._process.pid:
                raise RuntimeError('isolated meter child PID differs')
            self._receipt.update(ready)
            self._status = 'running'
        except BaseException as exc:
            child.close()
            self._abort(f'{type(exc).__name__}: {exc}')
            raise
        return self

    def begin_window(self):
        if self._status != 'running' or self._window_active:
            raise RuntimeError('isolated meter needs a running, unguarded session')
        self._window_active = True
        self._receipt['local_window_guards'].append(dict(begin_s=time.time(), end_s=None))

    def end_window(self):
        if not self._window_active:
            raise RuntimeError('isolated meter has no active window guard')
        self._receipt['local_window_guards'][-1]['end_s'] = time.time()
        self._window_active = False

    def _rpc(self, operation, arguments=None):
        if self._window_active:
            raise RuntimeError('meter RPC is forbidden inside service/drain-tail measurement')
        if self._status != 'running':
            raise RuntimeError('isolated meter is not running: '+self._status)
        self._sequence += 1
        command = dict(operation=operation, sequence=self._sequence, arguments=arguments or {})
        evidence = dict(operation=operation, sequence=self._sequence, requested_s=time.time())
        self._receipt['commands'].append(evidence)
        try:
            self._connection.send(command)
            result = self._receive(operation, self._sequence, self.rpc_timeout_s)
            evidence.update(finished_s=time.time(), passed=True)
            return result
        except BaseException as exc:
            evidence.update(finished_s=time.time(), passed=False, error=f'{type(exc).__name__}: {exc}')
            self._abort(evidence['error'])
            raise

    def snapshot(self):
        if self._status == 'stopped':
            return deepcopy(self._final_snapshot)
        return self._rpc('snapshot')

    def stop(self, *, after_s=None, timeout_s=2.):
        if self._status == 'stopped':
            return
        arguments = dict(after_s=after_s, timeout_s=timeout_s)
        snapshot = self._rpc('stop', arguments)
        self._process.join(timeout=2.)
        if self._process.is_alive() or self._process.exitcode != 0:
            self._abort('isolated meter child did not exit cleanly after stop')
            raise RuntimeError(self._receipt['error'])
        self._connection.close()
        self._connection = None
        self._final_snapshot = snapshot
        self._status = 'stopped'
        self._receipt.update(finished_s=time.time(), child_exitcode=self._process.exitcode)

    def method_receipt(self):
        alive = self._process is not None and self._process.pid is not None and self._process.is_alive()
        return deepcopy(dict(self._receipt, status=self._status, window_guard_active=self._window_active,
                             child_alive=alive, receipt_observed_s=time.time()))

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        if self._status == 'running':
            if self._window_active:
                self._abort('context exited during guarded measurement; no snapshot retained')
                if exc_type is None:
                    raise RuntimeError(self._receipt['error'])
            elif exc_type is None:
                self.stop()
            else:
                try:
                    self.stop()
                except BaseException:
                    pass  # Preserve the original exception; stop recorded its failure.
        return False
