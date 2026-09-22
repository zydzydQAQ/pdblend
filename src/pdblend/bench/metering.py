"""GPU power metering and clock control, reusing pdblend.measure."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Sequence

from pdblend.measure.backends import PynvmlBackend
from pdblend.measure.power import PowerSampler

FREQUENCY_TIERS = (900, 1200, 1500, 1800, 2100, 2520)
MAX_FREQUENCY = 2520
PARK_MEM_MHZ = 405       # lowest GDDR6 clock: with a live CUDA context this alone takes idle from ~75 W to ~35 W
PARK_GR_MHZ = 210


class Gpus:
    def __init__(self, gpus: Sequence[int], power_mode: str = "average"):
        self.gpus = [int(g) for g in gpus]
        self.backend = PynvmlBackend(power_mode=power_mode)

    def set_clock(self, gpu: int, mhz: int) -> None:
        self.backend.set_clock(gpu, mhz)

    def reset_clock(self, gpu: int) -> None:
        self.backend.reset_clock(gpu)

    def reset_all(self) -> None:
        for g in self.gpus:
            self.unpark(g)
            self.backend.reset_clock(g)

    def park(self, gpu: int) -> None:
        """Weights stay resident; memory and SM clocks are pinned to their floor. Wake is reset_clock-fast."""
        nv, h = self.backend._nvml, self.backend._handle(gpu)
        nv.nvmlDeviceSetMemoryLockedClocks(h, PARK_MEM_MHZ, PARK_MEM_MHZ)
        nv.nvmlDeviceSetGpuLockedClocks(h, PARK_GR_MHZ, PARK_GR_MHZ)

    def unpark(self, gpu: int) -> None:
        nv, h = self.backend._nvml, self.backend._handle(gpu)
        nv.nvmlDeviceResetMemoryLockedClocks(h)

    def mem_freq(self, gpu: int) -> int:
        nv, h = self.backend._nvml, self.backend._handle(gpu)
        return int(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM))

    def mem_used_mb(self, gpu: int) -> float:
        nv, h = self.backend._nvml, self.backend._handle(gpu)
        return nv.nvmlDeviceGetMemoryInfo(h).used / 2**20

    def wait_released(self, gpus: Sequence[int], threshold_mb: float = 1024.0, timeout_s: float = 60.0,
                      power_threshold_w: float = 45.0) -> float:
        """Seconds until every GPU has freed its memory and its power decayed to no-process idle.

        Engine subprocesses release CUDA late, and after the context is gone the board stays ~65 W
        for several more seconds before dropping to ~34 W."""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if all(self.mem_used_mb(g) < threshold_mb and self.power_w(g) < power_threshold_w for g in gpus):
                break
            time.sleep(0.5)
        return time.time() - t0

    def current_freq(self, gpu: int) -> int:
        return self.backend.current_freq(gpu)

    def power_w(self, gpu: int) -> float:
        return self.backend.power_w(gpu)

    def sampler(self, gpus: Sequence[int] | None = None, interval_s: float = 0.05) -> PowerSampler:
        return PowerSampler(list(gpus) if gpus is not None else self.gpus, interval=interval_s,
                            backend=self.backend, sample_clocks=True)

    @contextmanager
    def measure(self, gpus: Sequence[int] | None = None, interval_s: float = 0.05):
        """Yields a dict filled with energy_j / mean_power_w / duration_s / per_gpu_mean_w on exit."""
        sampler = self.sampler(gpus, interval_s)
        result: dict = {}
        sampler.start()
        started = time.time()
        try:
            yield result
        finally:
            sampler.stop()
            duration = time.time() - started
            result.update(energy_j=sampler.total_energy_j(), mean_power_w=sampler.mean_power_w(),
                          duration_s=duration, samples=len(sampler.samples), error=sampler.error,
                          per_gpu_mean_w=per_gpu_mean(sampler.samples))
            if sampler.frequency_samples:
                result["mean_freq_mhz"] = per_gpu_mean(sampler.frequency_samples)

    def settle_and_measure(self, seconds: float, gpus: Sequence[int] | None = None,
                           settle_s: float = 3.0) -> dict:
        time.sleep(settle_s)
        with self.measure(gpus) as m:
            time.sleep(seconds)
        return m


def per_gpu_mean(samples) -> list[float]:
    if not samples:
        return []
    width = len(samples[0][1])
    sums = [0.0] * width
    for _, values in samples:
        for i, v in enumerate(values):
            sums[i] += float(v)
    return [s / len(samples) for s in sums]
