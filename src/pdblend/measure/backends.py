# -*- coding: utf-8 -*-
"""GPU 频率/功率后端:FakeBackend 单测,PynvmlBackend 上机。"""
from __future__ import annotations

import subprocess
import math
import time
import os
from typing import Dict, List, Optional, Protocol, Sequence


class BackendError(Exception):
    """锁频或读功率失败。"""


INSTANT_POWER_SOURCE_ID = 'nvml:field:186:scope:0:mW'


def physical_gpu(gpu: int) -> str:
    """Translate a container-local ordinal to its leased physical UUID."""
    leased = os.environ.get('PDBLEND_GPU_UUIDS', '')
    if not leased:
        return str(int(gpu))
    uuids = leased.split(',')
    index = int(gpu)
    if index < 0 or index >= len(uuids) or not uuids[index].startswith('GPU-'):
        raise BackendError(f'GPU {gpu} is outside this process lease')
    return uuids[index]


class GpuBackend(Protocol):
    def current_freq(self, gpu: int) -> int: ...
    def set_clock(self, gpu: int, freq: int) -> None: ...
    def reset_clock(self, gpu: int) -> None: ...
    def supported_freqs(self, gpu: int) -> List[int]: ...
    def power_w(self, gpu: int) -> float: ...
    def power_limit_w(self, gpu: int) -> float: ...
    def utilization_pct(self, gpu: int) -> float: ...
    def temperature_c(self, gpu: int) -> float: ...


class FakeBackend:
    """可注入的锁频假后端。未锁时 current 为 freqs 最高档。"""

    def __init__(self, gpu_count: int = 1,
                 freqs: Sequence[int] = (2520, 2100, 1800, 1500, 1200, 900)):
        self.gpu_count = int(gpu_count)
        self.freqs = tuple(int(f) for f in freqs)
        if not self.freqs:
            raise BackendError("freqs 不能为空")
        self._default = max(self.freqs)
        self.locked: Dict[int, int] = {}
        self._power_w = 140.0

    def current_freq(self, gpu: int) -> int:
        self._check_gpu(gpu)
        return int(self.locked.get(int(gpu), self._default))

    def set_clock(self, gpu: int, freq: int) -> None:
        self._check_gpu(gpu)
        f = int(freq)
        if f not in self.freqs:
            raise BackendError("非法频率 %s,候选 %s" % (f, self.freqs))
        self.locked[int(gpu)] = f

    def reset_clock(self, gpu: int) -> None:
        self._check_gpu(gpu)
        self.locked.pop(int(gpu), None)

    def supported_freqs(self, gpu: int) -> List[int]:
        self._check_gpu(gpu)
        return list(self.freqs)

    def power_w(self, gpu: int) -> float:
        self._check_gpu(gpu)
        return float(self._power_w)

    def temperature_c(self, gpu: int) -> float:
        self._check_gpu(gpu)
        return 40.0

    def power_limit_w(self, gpu: int) -> float:
        self._check_gpu(gpu)
        return 350.0

    def utilization_pct(self, gpu: int) -> float:
        self._check_gpu(gpu)
        return 0.0

    def _check_gpu(self, gpu: int) -> None:
        if int(gpu) < 0 or int(gpu) >= self.gpu_count:
            raise BackendError("GPU %s 不存在(count=%s)" % (gpu, self.gpu_count))


class PynvmlBackend:
    """Direct NVML control/counters; nvidia-smi only for unsupported bindings."""

    def __init__(self, power_mode='average', *, clock=time.time):
        if power_mode not in ('average', 'instant'):
            raise ValueError('power_mode must be average or instant')
        self._power_mode = power_mode
        self._power_clock = clock
        self._power_timestamps = {}
        self._nvml = None
        self._handles: Dict[int, object] = {}
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
        except Exception:
            self._nvml = None
        if power_mode == 'instant' and self._nvml is None:
            raise BackendError('instant power requires NVML field 186; no average fallback')

    @property
    def power_mode(self):
        return self._power_mode

    @property
    def power_source(self):
        instant = self.power_mode == 'instant'
        return dict(schema=1, mode=self.power_mode, unit='W',
            source_id=INSTANT_POWER_SOURCE_ID if instant else 'legacy:power-usage:average-on-L20',
            api='nvmlDeviceGetFieldValues' if instant else 'nvmlDeviceGetPowerUsage / nvidia-smi power.draw',
            field_id=186 if instant else None, scope_id=0 if instant else None,
            raw_unit='mW' if instant else 'mW or W as identified per reading',
            timestamp_semantics='NVML CPU epoch microseconds; not an independent GPU clock' if instant else 'host read interval',
            temporal_resolution_s=None if instant else 1,
            semantics='NVML current-power field; sensor update period is not inferred from polling' if instant
                else 'legacy behavior retained; one-second average on L20; architecture-dependent on other GPUs')

    def _handle(self, gpu: int):
        if self._nvml is None:
            return None
        if gpu not in self._handles:
            device = physical_gpu(gpu)
            self._handles[gpu] = (self._nvml.nvmlDeviceGetHandleByUUID(device)
                                  if device.startswith('GPU-') else
                                  self._nvml.nvmlDeviceGetHandleByIndex(int(device)))
        return self._handles[gpu]

    def current_freq(self, gpu: int) -> int:
        if self._nvml is not None:
            return int(self._nvml.nvmlDeviceGetClockInfo(
                self._handle(gpu), self._nvml.NVML_CLOCK_SM))
        out = _smi_query("clocks.current.graphics", gpu)
        return int(float(out))

    def clock_idle(self, gpu: int) -> bool:
        """An idle clock drop is not an active-workload frequency failure."""
        if self._nvml is None:
            return False
        try:
            reasons=self._nvml.nvmlDeviceGetCurrentClocksThrottleReasons(self._handle(gpu))
            return bool(reasons & 0x1) and not bool(reasons & ~0x1)
        except (AttributeError,self._nvml.NVMLError):
            return False

    def set_clock(self, gpu: int, freq: int) -> None:
        f = int(freq)
        if self._nvml is not None:
            try:
                self._nvml.nvmlDeviceSetGpuLockedClocks(self._handle(gpu),f,f)
                return
            except (AttributeError,self._nvml.NVMLError_NotSupported,
                    self._nvml.NVMLError_FunctionNotFound):
                pass
        _smi(["-i", physical_gpu(gpu), "-lgc", "%d,%d" % (f, f)])

    def reset_clock(self, gpu: int) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlDeviceResetGpuLockedClocks(self._handle(gpu))
                return
            except (AttributeError,self._nvml.NVMLError_NotSupported,
                    self._nvml.NVMLError_FunctionNotFound):
                pass
        _smi(["-i", physical_gpu(gpu), "-rgc"])

    def supported_freqs(self, gpu: int) -> List[int]:
        if self._nvml is not None:
            try:
                h = self._handle(gpu)
                mems = self._nvml.nvmlDeviceGetSupportedMemoryClocks(h)
                freqs = set()
                for mem in mems:
                    for sm in self._nvml.nvmlDeviceGetSupportedGraphicsClocks(h, mem):
                        freqs.add(int(sm))
                if freqs:
                    return sorted(freqs, reverse=True)
            except Exception:
                pass
        return [2520, 2100, 1800, 1500, 1200, 1050, 900, 600]

    def power_w(self, gpu: int) -> float:
        return self.power_reading(gpu)['watts']

    def power_reading(self, gpu: int) -> dict:
        """Return watts and its exact retrieval provenance; instant fails closed.

        A single transient NVML field-186 glitch is retried a bounded number of
        times before the read fails closed; every accepted reading still passes
        the full timestamp/latency validation and records exact provenance.
        """
        started = float(self._power_clock())
        if self.power_mode == 'instant':
            last = None
            for attempt in range(3):
                if attempt:
                    time.sleep(.05)
                attempt_started = float(self._power_clock())
                try:
                    values = self._nvml.nvmlDeviceGetFieldValues(self._handle(gpu), [(186, 0)])
                    finished = float(self._power_clock())
                    if len(values) != 1:
                        raise ValueError('expected one NVML field value')
                    value = values[0]
                    # The union is undefined on failure. Validate its tag and status
                    # before reading uiVal (L20 exposes power as an unsigned int).
                    if (value.nvmlReturn != 0 or value.valueType != 1
                            or value.fieldId != 186 or value.scopeId != 0):
                        raise ValueError('invalid field ID, scope, return code or unsigned-int value type')
                    timestamp = value.timestamp
                    latency = value.latencyUsec
                    if (type(timestamp) is not int or timestamp <= 0 or type(latency) is not int or latency < 0
                            or not all(math.isfinite(t) for t in (started, finished)) or finished < started
                            or not -.05 <= finished-timestamp/1e6 <= .25
                            or timestamp < self._power_timestamps.get(gpu, 0)):
                        raise ValueError('invalid, stale or regressed NVML timestamp/latency')
                    mw = value.value.uiVal
                    if type(mw) is not int or not 0 <= mw < 2**32-1:
                        raise ValueError('invalid milliwatt field value')
                    self._power_timestamps[gpu] = timestamp
                    return dict(watts=mw/1000., mode='instant', source_id=INSTANT_POWER_SOURCE_ID,
                        api='nvmlDeviceGetFieldValues', field_id=186, scope_id=0, raw_unit='mW',
                        value_type=1, return_code=0, nvml_timestamp_us=timestamp, nvml_latency_us=latency,
                        read_started_s=attempt_started, read_finished_s=finished)
                except Exception as exc:
                    last = exc
            raise BackendError('instant NVML power field 186 failed; no average fallback: '+str(last)) from last
        if self._nvml is not None:
            try:
                mw = self._nvml.nvmlDeviceGetPowerUsage(self._handle(gpu))
                watts = float(mw)/1000.
                if not math.isfinite(watts) or watts < 0: raise ValueError('invalid average power')
                return dict(watts=watts, mode='average', source_id='nvmlDeviceGetPowerUsage',
                    api='nvmlDeviceGetPowerUsage', field_id=None, scope_id=None, raw_unit='mW',
                    value_type=None, return_code=0, nvml_timestamp_us=None, nvml_latency_us=None,
                    read_started_s=started, read_finished_s=float(self._power_clock()))
            except Exception:
                pass
        out = _smi_query("power.draw", gpu)
        watts = float(out)
        if not math.isfinite(watts) or watts < 0: raise BackendError('invalid average power')
        return dict(watts=watts, mode='average', source_id='nvidia-smi:power.draw',
            api='nvidia-smi', field_id=None, scope_id=None, raw_unit='W',
            value_type=None, return_code=None, nvml_timestamp_us=None, nvml_latency_us=None,
            read_started_s=started, read_finished_s=float(self._power_clock()))

    def temperature_c(self, gpu: int) -> float:
        if self._nvml is not None:
            try:
                return float(self._nvml.nvmlDeviceGetTemperature(
                    self._handle(gpu), self._nvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
        out = _smi_query("temperature.gpu", gpu)
        return float(out)

    def power_limit_w(self, gpu: int) -> float:
        if self._nvml is not None:
            return self._nvml.nvmlDeviceGetEnforcedPowerLimit(self._handle(gpu))/1000.0
        return float(_smi_query('enforced.power.limit',gpu))

    def utilization_pct(self, gpu: int) -> float:
        if self._nvml is not None:
            return float(self._nvml.nvmlDeviceGetUtilizationRates(
                self._handle(gpu)).gpu)
        return float(_smi_query("utilization.gpu", gpu))


def get_backend(name: str = "pynvml") -> GpuBackend:
    key = str(name or "pynvml").strip().lower()
    if key in ("fake", "dummy"):
        return FakeBackend()
    if key in ("pynvml", "nvml", "nvidia", "smi", "nvidia-smi"):
        return PynvmlBackend()
    raise BackendError("未知 backend: %s" % name)


def _smi(args: Sequence[str]) -> str:
    cmd = ["nvidia-smi", *args]
    try:
        p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BackendError("nvidia-smi 失败: %s" % exc) from exc
    return p.stdout


def _smi_query(field: str, gpu: int) -> str:
    out = _smi([
        "--query-gpu=%s" % field, "--format=csv,noheader,nounits",
        "-i", physical_gpu(gpu),
    ])
    line = out.strip().splitlines()[0].strip() if out.strip() else ""
    if not line or line.upper() == "N/A":
        raise BackendError("nvidia-smi %s 空" % field)
    return line.split()[0]
