#!/usr/bin/env python3
"""Timestamped NVML sampling and GPU-energy integration."""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pynvml


def _safe(call: Any, default: int | float | str = -1) -> Any:
    try:
        return call()
    except pynvml.NVMLError:
        return default


@dataclass
class PowerSampler:
    gpu_indices: list[int]
    interval_s: float = 0.05
    samples: list[dict[str, int | float | str]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("power sampler is already running")
        pynvml.nvmlInit()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._sample_loop, name="nvml-power-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_s * 4))
        self._thread = None
        pynvml.nvmlShutdown()

    def _sample_loop(self) -> None:
        handles = {
            index: pynvml.nvmlDeviceGetHandleByIndex(index)
            for index in self.gpu_indices
        }
        next_sample = time.monotonic()
        while not self._stop.is_set():
            mono_ns = time.monotonic_ns()
            wall_ns = time.time_ns()
            for index, handle in handles.items():
                util = _safe(
                    lambda h=handle: pynvml.nvmlDeviceGetUtilizationRates(h), None
                )
                memory = _safe(
                    lambda h=handle: pynvml.nvmlDeviceGetMemoryInfo(h), None
                )
                self.samples.append(
                    {
                        "monotonic_ns": mono_ns,
                        "wall_time_ns": wall_ns,
                        "gpu_index": index,
                        "gpu_uuid": _safe(
                            lambda h=handle: pynvml.nvmlDeviceGetUUID(h), "unknown"
                        ),
                        "power_w": float(
                            _safe(
                                lambda h=handle: pynvml.nvmlDeviceGetPowerUsage(h),
                                -1000,
                            )
                        )
                        / 1000.0,
                        "gpu_util_pct": getattr(util, "gpu", -1),
                        "memory_util_pct": getattr(util, "memory", -1),
                        "memory_used_mb": (
                            float(getattr(memory, "used", -1)) / (1024 * 1024)
                        ),
                        "sm_clock_mhz": _safe(
                            lambda h=handle: pynvml.nvmlDeviceGetClockInfo(
                                h, pynvml.NVML_CLOCK_SM
                            )
                        ),
                        "memory_clock_mhz": _safe(
                            lambda h=handle: pynvml.nvmlDeviceGetClockInfo(
                                h, pynvml.NVML_CLOCK_MEM
                            )
                        ),
                    }
                )
            next_sample += self.interval_s
            self._stop.wait(max(0.0, next_sample - time.monotonic()))

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "monotonic_ns",
            "wall_time_ns",
            "gpu_index",
            "gpu_uuid",
            "power_w",
            "gpu_util_pct",
            "memory_util_pct",
            "memory_used_mb",
            "sm_clock_mhz",
            "memory_clock_mhz",
        ]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.samples)


def _interpolated_power(
    points: list[tuple[int, float]], timestamp_ns: int
) -> float:
    if timestamp_ns <= points[0][0]:
        return points[0][1]
    if timestamp_ns >= points[-1][0]:
        return points[-1][1]
    for (left_t, left_p), (right_t, right_p) in zip(points, points[1:]):
        if left_t <= timestamp_ns <= right_t:
            fraction = (timestamp_ns - left_t) / (right_t - left_t)
            return left_p + fraction * (right_p - left_p)
    raise AssertionError("timestamp interpolation fell through")


def integrate_energy(
    samples: list[dict[str, int | float | str]],
    start_ns: int,
    end_ns: int,
) -> dict[str, Any]:
    """Integrate power in [start_ns, end_ns] with trapezoids."""
    if end_ns <= start_ns:
        raise ValueError("end_ns must be greater than start_ns")

    grouped: dict[int, list[dict[str, int | float | str]]] = {}
    for sample in samples:
        grouped.setdefault(int(sample["gpu_index"]), []).append(sample)

    per_gpu: dict[str, dict[str, float | int | str]] = {}
    for gpu_index, gpu_samples in sorted(grouped.items()):
        ordered = sorted(gpu_samples, key=lambda item: int(item["monotonic_ns"]))
        points = [
            (int(item["monotonic_ns"]), float(item["power_w"]))
            for item in ordered
            if float(item["power_w"]) >= 0
        ]
        if len(points) < 2 or start_ns < points[0][0] or end_ns > points[-1][0]:
            raise ValueError(
                f"GPU {gpu_index} samples do not bracket the measurement window"
            )

        bounded = [(start_ns, _interpolated_power(points, start_ns))]
        bounded.extend(point for point in points if start_ns < point[0] < end_ns)
        bounded.append((end_ns, _interpolated_power(points, end_ns)))
        energy_j = sum(
            (left_p + right_p)
            * 0.5
            * ((right_t - left_t) / 1_000_000_000)
            for (left_t, left_p), (right_t, right_p) in zip(
                bounded, bounded[1:]
            )
        )

        window_samples = [
            item
            for item in ordered
            if start_ns <= int(item["monotonic_ns"]) <= end_ns
        ]
        busy_clocks = [
            float(item["sm_clock_mhz"])
            for item in window_samples
            if float(item["gpu_util_pct"]) > 10
            and float(item["sm_clock_mhz"]) >= 0
        ]
        per_gpu[str(gpu_index)] = {
            "gpu_uuid": str(ordered[0]["gpu_uuid"]),
            "energy_j": energy_j,
            "mean_power_w": energy_j
            / ((end_ns - start_ns) / 1_000_000_000),
            "peak_power_w": max(
                float(item["power_w"]) for item in window_samples
            ),
            "mean_gpu_util_pct": (
                sum(float(item["gpu_util_pct"]) for item in window_samples)
                / len(window_samples)
            ),
            "median_busy_sm_clock_mhz": (
                sorted(busy_clocks)[len(busy_clocks) // 2]
                if busy_clocks
                else -1
            ),
            "sample_count": len(window_samples),
        }

    return {
        "duration_s": (end_ns - start_ns) / 1_000_000_000,
        "total_energy_j": sum(
            float(metrics["energy_j"]) for metrics in per_gpu.values()
        ),
        "per_gpu": per_gpu,
    }
