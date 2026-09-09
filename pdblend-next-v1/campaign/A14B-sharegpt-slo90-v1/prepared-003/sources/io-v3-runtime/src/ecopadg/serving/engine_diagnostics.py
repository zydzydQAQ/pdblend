"""Bounded CPU timing histograms with periodic, asynchronous publication.

Durations use a monotonic clock. Publication time is never an engine/GPU state
observation. Histogram percentiles are upper bounds, not exact samples.
"""
from contextlib import contextmanager
import copy
import json
import logging
import math
import os
import threading
import time

logger = logging.getLogger(__name__)
METRICS = ("owner_queue_wait", "engine_step", "stop_worker_loop",
           "transfer_rpc", "other_control_rpc", "allocation_scan",
           "event_write", "snapshot_total")
BOUNDS_S = tuple(2. ** exponent for exponent in range(-20, 9))


class EngineTimings:
    def __init__(self, *, enabled=False, interval_s=1., clock=time.perf_counter):
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("timing summary interval must be positive")
        self.enabled, self.interval_s, self.clock = enabled, interval_s, clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._closed = False
        self._active_spans = self._late_observations = 0
        self.path = None
        self._observed_s = None
        self._write_error = self._last_write_error = None
        self._write_failures = 0
        self._values = {name: dict(count=0, failed=0, total_s=0., max_s=0.,
                                  histogram=[0] * (len(BOUNDS_S) + 1)) for name in METRICS}

    def observe(self, name, duration_s, *, failed=False):
        if not self.enabled:
            return
        if name not in self._values or not math.isfinite(duration_s) or duration_s < 0:
            raise ValueError("invalid timing observation")
        bucket = next((i for i, bound in enumerate(BOUNDS_S) if duration_s <= bound), len(BOUNDS_S))
        with self._lock:
            if self._closed:
                self._late_observations += 1
            value = self._values[name]
            value["count"] += 1
            value["failed"] += bool(failed)
            value["total_s"] += duration_s
            value["max_s"] = max(value["max_s"], duration_s)
            value["histogram"][bucket] += 1
            self._observed_s = time.time()

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        started = self.clock()
        with self._lock:
            self._active_spans += 1
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            try:
                self.observe(name, max(0., self.clock()-started), failed=failed)
            finally:
                with self._lock:
                    self._active_spans -= 1

    def invoke(self, queued_at, function, args):
        self.observe("owner_queue_wait", max(0., self.clock()-queued_at))
        return function(*args)

    @staticmethod
    def _quantile(value, quantile):
        if not value["count"]:
            return None
        target = math.ceil(value["count"] * quantile)
        count = 0
        for index, number in enumerate(value["histogram"]):
            count += number
            if count >= target:
                return min(BOUNDS_S[index], value["max_s"]) if index < len(BOUNDS_S) else value["max_s"]

    def snapshot(self):
        with self._lock:
            values = copy.deepcopy(self._values)
            io = dict(write_error=self._write_error, last_write_error=self._last_write_error,
                      write_failures=self._write_failures, closing=self._closed,
                      worker_alive=bool(self._thread and self._thread.is_alive()),
                      inflight_spans=self._active_spans, late_observations=self._late_observations)
            observed = self._observed_s
        for value in values.values():
            value["mean_s"] = value["total_s"]/value["count"] if value["count"] else None
            value["p50_upper_s"] = self._quantile(value, .5)
            value["p95_upper_s"] = self._quantile(value, .95)
            value["p99_upper_s"] = self._quantile(value, .99)
        return dict(schema_version=1, enabled=self.enabled,
            timing_observed_s=observed, histogram_upper_bounds_s=list(BOUNDS_S) + [None],
            metrics=values, io=io,
            semantics="CPU wall durations; cumulative histogram percentile upper bounds; not GPU kernel time. "
                      "snapshot_total contains stop_worker_loop/transfer_rpc/allocation_scan; do not add overlaps. "
                      "owner_queue_wait includes only calls that actually started.")

    def start(self, path):
        if not self.enabled:
            return
        with self._lock:
            if self._closed or self._thread is not None:
                raise RuntimeError("timing publisher cannot restart")
            self.path = str(path)
            self._thread = threading.Thread(target=self._run, name="pdblend-engine-timing", daemon=True)
            self._thread.start()

    def _write(self, report):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        temporary = "%s.tmp.%d.%d" % (self.path, os.getpid(), id(self))
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(report, handle, separators=(",", ":"), allow_nan=False)
                handle.write("\n")
            os.replace(temporary, self.path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _run(self):
        previous = None
        while True:
            closing = self._stop.wait(self.interval_s)
            report = self.snapshot()
            report.update(summary_generated_s=time.time(), final=closing)
            report["interval"] = {name: {key: value[key] - (previous[name][key] if previous else 0)
                for key in ("count", "failed", "total_s")} for name, value in report["metrics"].items()}
            try:
                self._write(report)
                previous = report["metrics"]
                with self._lock:
                    self._write_error = None
            except Exception as exc:
                with self._lock:
                    changed = self._write_error != str(exc)
                    self._write_error = self._last_write_error = str(exc)[:512]
                    self._write_failures += 1
                if changed:
                    logger.warning("PDBlend timing summary write failed: %s", exc)
            if closing:
                return

    def close(self, timeout_s=2.):
        with self._lock:
            self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(max(0., timeout_s))
        with self._lock:
            return (not (self._thread and self._thread.is_alive()) and self._write_error is None
                    and not self._active_spans and not self._late_observations)
