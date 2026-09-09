"""Opt-in, bounded control/telemetry I/O; worker threads never read a scheduler.

Control refresh and telemetry publication have separate workers so a slow disk
write cannot prevent control validation. Only scheduler observations advance
``ts``; an I/O heartbeat is deliberately a different timestamp.
"""
from __future__ import annotations

import atexit
import copy
import json
import logging
import math
import os
import threading
import time
import weakref

logger = logging.getLogger(__name__)
_LIVE = weakref.WeakSet()
_MAX_CONTROL_BYTES = 65536


def enabled():
    return os.environ.get("PDBLEND_ASYNC_IO") == "1"


def _validate(payload, kind):
    if not isinstance(payload, dict):
        raise ValueError("control must be a JSON object")
    if type(payload.get("generation")) is not int or payload["generation"] < 0:
        raise ValueError("invalid generation")
    if payload.get("mode") not in ("continuous", "temporal"):
        raise ValueError("invalid mode")
    if kind == "runtime":
        if payload.get("role") not in ("mixed", "prefill", "decode"):
            raise ValueError("invalid role")
        if type(payload.get("admit_prefill")) is not bool:
            raise ValueError("invalid admission flag")
        if type(payload.get("admit_decode", True)) is not bool:
            raise ValueError("invalid decode admission flag")
    elif type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported control schema_version")
    # Reject NaN and non-JSON objects, including in otherwise unused fields.
    json.dumps(payload, allow_nan=False)


class ControlCache:
    """One latest validated version and one polling worker, with bounded input."""

    def __init__(self, path, kind, *, interval_s=.02, max_age_s=1.):
        if not (math.isfinite(interval_s) and math.isfinite(max_age_s)
                and 0 < interval_s < max_age_s):
            raise ValueError("control polling interval must be below max age")
        self.path, self.kind = path, kind
        self.interval_s, self.max_age_s = interval_s, max_age_s
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._stop = threading.Event()
        self._payload = None
        self._signature = None
        self._error = "control not observed"
        self._checked_mono = 0.
        self._checked_s = None
        self._closed = False
        # Initial read is startup work, before Scheduler enters its hot path.
        self._refresh()
        self._thread = threading.Thread(target=self._run,
            name="pdblend-control-" + kind, daemon=True)
        self._thread.start()

    @staticmethod
    def _stat_signature(st):
        return (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

    def _refresh(self):
        # Explicit administrative refresh and polling serialize here. Ordinary
        # scheduler reads take only _lock, never this I/O lock.
        with self._refresh_lock:
            if self._stop.is_set():
                return
            self._refresh_serial()

    def _refresh_serial(self):
        try:
            signature = self._stat_signature(os.stat(self.path))
            if signature != self._signature:
                with open(self.path, "rb") as handle:
                    before = self._stat_signature(os.fstat(handle.fileno()))
                    raw = handle.read(_MAX_CONTROL_BYTES + 1)
                    after = self._stat_signature(os.fstat(handle.fileno()))
                if len(raw) > _MAX_CONTROL_BYTES:
                    raise ValueError("control exceeds size limit")
                if before != after or after != signature:
                    raise ValueError("control changed while being read")
                payload = json.loads(raw)
                _validate(payload, self.kind)
                with self._lock:
                    previous = self._payload
                    if previous and payload["generation"] < previous["generation"]:
                        raise ValueError("stale generation")
                    if previous and payload["generation"] == previous["generation"] and payload != previous:
                        raise ValueError("conflicting generation")
                    self._payload = payload
                    self._signature = signature
                    self._error = None
            with self._lock:
                # An unchanged, readable file revalidates control availability,
                # never the age of scheduler/GPU state.
                self._checked_mono, self._checked_s = time.monotonic(), time.time()
                self._error = None
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
                self._checked_mono, self._checked_s = time.monotonic(), time.time()

    def _run(self):
        while not self._stop.wait(self.interval_s):
            self._refresh()

    def read(self):
        with self._lock:
            error = self._error
            if self._closed:
                error = "control cache closed"
            elif not self._thread.is_alive():
                error = "control worker stopped"
            elif time.monotonic() - self._checked_mono > self.max_age_s:
                error = "control observation stale"
            return copy.deepcopy(self._payload), error, self._checked_s

    def force_refresh(self):
        """Administrative ACK boundary only; may perform synchronous file I/O."""
        self._refresh()
        return self.read()

    def close(self, timeout_s=2.):
        with self._lock:
            self._closed = True
        self._stop.set()
        self._thread.join(max(0., timeout_s))
        return not self._thread.is_alive()


class SnapshotWriter:
    """A latest-only mailbox: one in-flight write and at most one pending view."""

    def __init__(self, path, *, heartbeat_s=.5):
        if not math.isfinite(heartbeat_s) or heartbeat_s <= 0:
            raise ValueError("heartbeat interval must be positive")
        self.path, self.heartbeat_s = path, heartbeat_s
        self._cv = threading.Condition()
        self._pending = self._last = None
        self._closing = False
        self._sequence = self._written_sequence = self._coalesced = 0
        self._failures = 0
        self._error = self._last_error = self._last_error_s = None
        self._thread = threading.Thread(target=self._run,
            name="pdblend-telemetry", daemon=True)
        self._thread.start()

    def publish(self, payload):
        # Detach all data before returning; workers never retain live scheduler
        # queues, block managers, mutable metadata, or references to the owner.
        snapshot = copy.deepcopy(payload)
        with self._cv:
            if self._closing:
                raise RuntimeError("telemetry writer closed")
            self._sequence += 1
            snapshot["snapshot_sequence"] = self._sequence
            if self._pending is not None:
                self._coalesced += 1
            self._pending = snapshot
            self._cv.notify()
        return self._sequence

    def status(self):
        with self._cv:
            return dict(published_sequence=self._sequence,
                written_sequence=self._written_sequence,
                pending_snapshots=int(self._pending is not None),
                coalesced_snapshots=self._coalesced, write_failures=self._failures,
                write_error=self._error, last_write_error=self._last_error,
                last_write_error_s=self._last_error_s,
                closing=self._closing, worker_alive=self._thread.is_alive())

    def _write(self, payload):
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        temporary = "%s.tmp.%d.%d" % (self.path, os.getpid(), id(self))
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
                handle.write("\n")
            os.replace(temporary, self.path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _run(self):
        while True:
            with self._cv:
                if self._pending is None and not self._closing:
                    self._cv.wait(self.heartbeat_s)
                if self._pending is not None:
                    self._last, self._pending = self._pending, None
                closing = self._closing
                if self._last is None:
                    if closing:
                        return
                    continue
                payload = copy.deepcopy(self._last)
            payload.update(io_heartbeat_s=time.time(), io_closed=closing,
                           telemetry_io=self.status())
            # ts and scheduler_observed_s are copied verbatim, including during
            # heartbeats and the final publication. Neither is a write time.
            try:
                self._write(payload)
                with self._cv:
                    self._written_sequence = payload["snapshot_sequence"]
                    self._error = None
                    self._cv.notify_all()
            except Exception as exc:
                with self._cv:
                    changed = self._error != str(exc)
                    self._error = self._last_error = str(exc)
                    self._last_error_s = time.time()
                    self._failures += 1
                    self._cv.notify_all()
                if changed:
                    logger.warning("PDBlend telemetry write failed: %s", exc)
            if closing:
                return

    def close(self, timeout_s=2.):
        with self._cv:
            self._closing = True
            self._cv.notify_all()
        self._thread.join(max(0., timeout_s))
        with self._cv:
            return (not self._thread.is_alive() and self._error is None
                    and self._written_sequence == self._sequence)


class EngineIO:
    def __init__(self, *, poll_interval_s=.02, max_control_age_s=1., heartbeat_s=.5):
        self.caches = {}
        self.writer = None
        try:
            for kind, variable in (("runtime", "PDBLEND_RUNTIME_PATH"),
                                   ("legacy", "PDBLEND_CONTROL_PATH")):
                path = os.environ.get(variable)
                if path:
                    self.caches[kind] = ControlCache(path, kind,
                        interval_s=poll_interval_s, max_age_s=max_control_age_s)
            if path := os.environ.get("PDBLEND_TELEMETRY_PATH"):
                self.writer = SnapshotWriter(path, heartbeat_s=heartbeat_s)
        except BaseException:
            self.close()
            raise
        _LIVE.add(self)

    def status(self):
        controls = {}
        for kind, cache in self.caches.items():
            payload, error, checked_s = cache.read()
            controls[kind] = dict(generation=(payload or {}).get("generation"),
                                  error=error, control_checked_s=checked_s)
        return dict(controls=controls,
                    telemetry=self.writer.status() if self.writer else None)

    def close(self, timeout_s=2.):
        deadline = time.monotonic() + max(0., timeout_s)
        ok = True
        for component in [*self.caches.values(), self.writer]:
            if component is not None:
                ok = component.close(max(0., deadline-time.monotonic())) and ok
        return ok


def _finalize(io):
    if not io.close():
        logger.warning("PDBlend I/O shutdown incomplete; inspect writer status")


def initialize_engine_io(scheduler, **options):
    if not enabled():
        return None
    io = getattr(scheduler, "_pdblend_async_io", None)
    if io is None:
        io = EngineIO(**options)
        scheduler._pdblend_async_io = io
        try:
            scheduler._pdblend_async_finalizer = weakref.finalize(scheduler, _finalize, io)
        except TypeError:
            # Tiny CPU-test doubles may not be weak-referenceable. Engine
            # lifecycle shutdown and the process fallback still apply.
            pass
    return io


def shutdown_engine_io(scheduler, timeout_s=2.):
    io = getattr(scheduler, "_pdblend_async_io", None)
    if io is None:
        return True
    ok = io.close(timeout_s)
    if ok and (finalizer := getattr(scheduler, "_pdblend_async_finalizer", None)):
        finalizer.detach()
    return ok


@atexit.register
def _shutdown_all():
    for io in list(_LIVE):
        _finalize(io)
