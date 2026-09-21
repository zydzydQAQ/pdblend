# -*- coding: utf-8 -*-
"""多卡锁频/解锁与档位就近。"""
from __future__ import annotations

from typing import Sequence

from pdblend.measure.backends import BackendError, GpuBackend


def nearest_freq(target: int, levels: Sequence[int]) -> int:
    """在驱动档里取最接近 target 的频率。"""
    if not levels:
        raise BackendError("无可用频率档")
    want = int(target)
    return int(min((int(x) for x in levels), key=lambda x: (abs(x - want), -x)))


def check_gpu_present(backend: GpuBackend, gpu: int = 0) -> None:
    """后端读不到时钟则视为无卡。"""
    try:
        backend.current_freq(int(gpu))
    except Exception as exc:
        raise BackendError("GPU 不可用: %s" % exc) from exc


def current_sm_clock(gpu: int, backend: GpuBackend) -> int:
    return int(backend.current_freq(int(gpu)))


def set_all_gpus_clock(gpus: Sequence[int], freq: int,
                       backend: GpuBackend) -> None:
    for gpu in gpus:
        backend.set_clock(int(gpu), int(freq))


def reset_all_gpus(gpus: Sequence[int], backend: GpuBackend) -> None:
    for gpu in gpus:
        backend.reset_clock(int(gpu))
