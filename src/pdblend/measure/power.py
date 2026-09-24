# -*- coding: utf-8 -*-
"""梯形功率积分与 20ms PowerSampler。"""
from __future__ import annotations

import threading
import time
import math
from typing import List, Optional, Sequence, Tuple

from pdblend.measure.backends import GpuBackend, INSTANT_POWER_SOURCE_ID
from pdblend.measure.gc_read_guard import gc_read_guard


PowerRow = Tuple[float, Sequence[float]]


def instant_power_verified(summary):
    """A legacy average or an unlabeled result cannot become instant evidence."""
    return (summary.get('power_mode') == 'instant'
            and summary.get('power_source_id') == INSTANT_POWER_SOURCE_ID
            and summary.get('power_field_id') == 186
            and summary.get('power_source_verified') is True)


def trapezoid_energy(rows: Sequence[PowerRow]) -> float:
    """[(t_s, [每卡 W...])] 梯形积分,功率为各卡之和。"""
    if len(rows) < 2:
        return 0.0
    width = len(rows[0][1])
    if (not width or any(len(ws) != width or not math.isfinite(float(t))
            or any(not math.isfinite(float(w)) or float(w) < 0 for w in ws)
            for t, ws in rows)
            or any(float(b[0]) <= float(a[0]) for a, b in zip(rows, rows[1:]))):
        raise ValueError("invalid power samples: timestamps, GPU count or watts")
    total = 0.0
    prev_t, prev_ws = float(rows[0][0]), [float(x) for x in rows[0][1]]
    prev_p = sum(prev_ws)
    for t_raw, ws in rows[1:]:
        t = float(t_raw)
        p = sum(float(x) for x in ws)
        dt = t - prev_t
        if dt > 0:
            total += (prev_p + p) * 0.5 * dt
        prev_t, prev_p = t, p
    return float(total)


def trapezoid_mean_power(rows: Sequence[PowerRow]) -> float:
    if len(rows) < 2:
        if not rows:
            return 0.0
        return float(sum(float(x) for x in rows[0][1]))
    dur = float(rows[-1][0]) - float(rows[0][0])
    if dur <= 0:
        return float(sum(float(x) for x in rows[0][1]))
    return trapezoid_energy(rows) / dur


class PowerSampler:
    """后台按 interval 采各卡功率。start/stop 围一次测量窗。"""

    def __init__(self, gpus: Sequence[int], interval: float = 0.02,
                 backend: Optional[GpuBackend] = None, clock=time.time, sample_clocks=False):
        if backend is None:
            from pdblend.measure.backends import get_backend
            backend = get_backend("pynvml")
        self.gpus = [int(g) for g in gpus]
        self.interval = float(interval)
        self.backend = backend
        self._clock = clock
        self.sample_clocks=sample_clocks
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[PowerRow] = []
        self.utilization_samples: List[PowerRow] = []
        # Legacy complete rows stay numeric. New consumers use individual
        # device readings, including failed reads, rather than dropping peers.
        self.utilization_readings: list[dict] = []
        self.utilization_errors: list[dict] = []
        self.utilization_source = dict(getattr(backend, 'utilization_source',
            dict(source_id=type(backend).__name__ + ':utilization_pct', unit='percent',
                 sensor_period_s=None)))
        self.frequency_samples: List[PowerRow] = []
        self.power_source = dict(getattr(backend, 'power_source', dict(schema=1,
            mode='unspecified', source_id=type(backend).__name__, field_id=None, unit='W')))
        self.power_metadata = []
        self.error: Optional[str] = None
        self.error_at_s: Optional[float] = None

    def start(self) -> None:
        self.stop()
        self.samples = []
        self.utilization_samples = []
        self.utilization_readings = []
        self.utilization_errors = []
        self.frequency_samples = []
        self.power_metadata = []
        self.error = None
        self.error_at_s = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                sample = self._read()
                self.samples.append(sample)
                self._read_utilization()
                if self.sample_clocks:
                    self.frequency_samples.append((sample[0],[self.backend.current_freq(g) for g in self.gpus]))
            except Exception as exc:
                self.error = str(exc)
                self.error_at_s = float(self._clock())
                return
            self._stop.wait(self.interval)

    def _read_utilization(self) -> None:
        """A failed optional utilization read must not stop energy sampling."""
        read = getattr(self.backend, 'utilization_reading', None)
        scalar = getattr(self.backend, 'utilization_pct', None)
        if read is None and scalar is None:
            return
        rows = []
        for gpu in self.gpus:
            started = float(self._clock())
            try:
                if read is not None:
                    row = dict(read(gpu))
                else:
                    value = float(scalar(gpu))
                    finished = float(self._clock())
                    row = dict(gpu_util_pct=value, t_s=finished, read_started_s=started,
                               read_finished_s=finished,
                               source_id=self.utilization_source['source_id'], error=None)
                value = row.get('gpu_util_pct')
                a, b = row.get('read_started_s'), row.get('read_finished_s')
                if (row.get('error') or not isinstance(value, (float, int))
                        or not math.isfinite(value) or not 0 <= value <= 100
                        or not all(isinstance(t, (float, int)) and math.isfinite(t)
                                   for t in (a, b)) or b < a):
                    raise ValueError('invalid GPU utilization value/acquisition interval')
                row.update(gpu=gpu, t_s=b, error=None)
            except Exception as exc:
                finished = float(self._clock())
                row = dict(gpu=gpu, gpu_util_pct=None, t_s=finished,
                           read_started_s=started, read_finished_s=finished,
                           source_id=self.utilization_source['source_id'], error=str(exc))
                self.utilization_errors.append(dict(row))
            self.utilization_readings.append(row)
            rows.append(row)
        if rows and all(row['error'] is None for row in rows):
            self.utilization_samples.append((max(row['t_s'] for row in rows),
                                             [row['gpu_util_pct'] for row in rows]))

    def _read(self) -> PowerRow:
        with gc_read_guard():
            read = getattr(self.backend, 'power_reading', None)
            if read is None:
                started = float(self._clock())
                ws = [float(self.backend.power_w(g)) for g in self.gpus]
                finished = float(self._clock())
                readings = [dict(watts=w, mode=self.power_source['mode'],
                    source_id=self.power_source['source_id'], field_id=None, scope_id=None,
                    value_type=None, return_code=None, nvml_timestamp_us=None, nvml_latency_us=None,
                    read_started_s=started, read_finished_s=finished) for w in ws]
            else:
                readings = [read(g) for g in self.gpus]
                ws = [float(value['watts']) for value in readings]
            if any(not math.isfinite(w) or w < 0 for w in ws):
                raise ValueError('nonfinite or negative power sample')
            if any(value['mode'] != self.power_source['mode'] for value in readings):
                raise ValueError('power mode changed during sampling')
            # Anchor the row to the latest completed NVML read.  Using a
            # second wall-clock call after all reads makes a valid sample look
            # stale when the sampler thread is descheduled between the read
            # and metadata assembly (observed on long profiling runs).  The
            # per-GPU read timestamps remain in metadata and are still checked
            # for monotonicity, latency and source identity.
            t = max(float(self._clock()), max(float(value['read_finished_s']) for value in readings))
            metadata = dict(t_s=t, gpus=list(self.gpus))
            for key in ('mode', 'source_id', 'field_id', 'scope_id', 'value_type', 'return_code',
                        'nvml_timestamp_us', 'nvml_latency_us', 'read_started_s', 'read_finished_s'):
                metadata[key] = [value.get(key) for value in readings]
            self.power_metadata.append(metadata)
            return (t, ws)

    def mean_power_w(self) -> float:
        return trapezoid_mean_power(self.samples)

    def total_energy_j(self) -> float:
        return trapezoid_energy(self.samples)

    def temperature_c(self) -> float:
        if not self.gpus:
            return float("nan")
        return float(self.backend.temperature_c(self.gpus[0]))
