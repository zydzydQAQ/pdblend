"""Actual synchronous PUT call counters; failures remain sticky evidence."""
import threading
import time


class SyncSendCounters:
    def __init__(self):
        self._lock = threading.Lock()
        self.started = self.inflight = self.completed = self.failed = 0
        self.last_error = self.last_error_s = None

    def begin(self):
        with self._lock:
            self.started += 1
            self.inflight += 1

    def finish(self, completed, error=None):
        with self._lock:
            if self.inflight <= 0:
                raise RuntimeError("send completion without in-flight operation")
            self.inflight -= 1
            if completed:
                self.completed += 1
            else:
                self.failed += 1
                self.last_error = str(error or "synchronous PUT did not acknowledge completion")[:512]
                self.last_error_s = time.time()

    def snapshot(self):
        with self._lock:
            return dict(send_counter_schema=1, send_counter_scope="remote synchronous PUT tensor calls",
                send_counters_observed=True, inflight_sends=self.inflight,
                send_started=self.started, send_completed=self.completed,
                send_failed=self.failed, send_healthy=self.failed == 0,
                send_last_error=self.last_error, send_last_error_s=self.last_error_s)
