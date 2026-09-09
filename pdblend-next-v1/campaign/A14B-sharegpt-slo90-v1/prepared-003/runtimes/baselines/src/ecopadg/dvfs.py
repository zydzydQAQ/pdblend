# -*- coding: utf-8 -*-
"""DVFS 控制器:多卡同步锁频、最小驻留(E5:≥3s)、强制复位。

纯状态机逻辑,后端经 GpuBackend 注入(FakeBackend 可单测);
硬件约束依据 E5 多卡切换实测:api 24-73ms、settle 0.4-0.7s、瞬态 ≤20J。
"""
from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

from ecopadg.measure.backends import GpuBackend


class DvfsController:
    """单控制器独占时钟所有权;切换间强制最小驻留;force_reset 永远放行。"""

    def __init__(self, backend: GpuBackend, min_dwell_s: float = 3.0,
                 settle_s: float = 0.7, clock=time.time):
        self.backend = backend
        self.min_dwell_s = float(min_dwell_s)
        self.settle_s = float(settle_s)
        self._clock = clock
        self.clock = clock  # 测试可推进的时钟引用
        self._hist: List[Tuple[float, int, int]] = []
        self._last_change: Dict[int, float] = {}
        self._applied: Dict[int, int] = {}

    def current(self, gpu: int) -> int:
        """当前生效频率(优先最近一次 apply;否则问后端)。"""
        if gpu in self._applied:
            return self._applied[gpu]
        return int(self.backend.current_freq(gpu))

    def _dwell_ok(self, gpu: int, now: float) -> bool:
        last = self._last_change.get(gpu)
        return last is None or (now - last) >= self.min_dwell_s

    def plan(self, desired: Dict[int, int], now: float) -> Dict[int, int]:
        """只返回"允许且与当前不同"的切换(gpu → 目标频率)。"""
        out: Dict[int, int] = {}
        for gpu, freq in sorted(desired.items()):
            if int(freq) == self.current(gpu):
                continue
            if self._dwell_ok(gpu, now):
                out[gpu] = int(freq)
        return out

    def apply(self, desired: Dict[int, int], now: float) -> List[Tuple[int, int, int]]:
        """执行允许的切换,返回 [(gpu, old_freq, new_freq)]。"""
        plan = self.plan(desired, now)
        out: List[Tuple[int, int, int]] = []
        for gpu, freq in sorted(plan.items()):
            old = self.current(gpu)
            self.backend.set_clock(gpu, freq)
            self._applied[gpu] = freq
            self._last_change[gpu] = now
            self._hist.append((now, gpu, freq))
            out.append((gpu, old, freq))
        return out

    def force_reset(self, gpus: Sequence[int]) -> None:
        """强制解锁(finally 语义):不受驻留窗限制,并清除驻留账本。"""
        now = float(self._clock())
        for gpu in gpus:
            self.backend.reset_clock(gpu)
            self._applied.pop(gpu, None)
            self._last_change.pop(gpu, None)
            self._hist.append((now, gpu, -1))

    def history(self) -> List[Tuple[float, int, int]]:
        return list(self._hist)
