# -*- coding: utf-8 -*-
"""pytest 公共配置:src 与仓库根加入 sys.path,共享 fixture。"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
_WS = os.path.dirname(_ROOT)
for _p in (_SRC, _WS, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest


@pytest.fixture(scope="session")
def e1b_tables():
    """E1b_32B 实测表路径(缺省跳过依赖该 fixture 的测试)。"""
    env = os.environ.get("PDBLEND_RESULTS")
    candidates = []
    if env:
        candidates.append(env)
    candidates.extend([
        os.path.join(_ROOT, "new-motivations", "results", "e1b_32b_tp2"),
        os.path.join(_ROOT, "motivations", "results", "E1b_32B"),
    ])
    for base in candidates:
        lat = os.path.join(base, "p1b_latency.csv")
        pw = os.path.join(base, "p1b_power.csv")
        if os.path.exists(lat) and os.path.exists(pw):
            return lat, pw
    pytest.skip("缺少 E1b_32B 表: %s" % candidates[-1])


class FakeClock:
    """可控时钟:手动推进,供调度器测试注入。"""

    def __init__(self, t: float = 0.0):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def fake_clock():
    return FakeClock()


class SynthOpModel:
    """合成 OpModel:iter 随 freq 线性变慢,能耗 U 型(1500 最优)。

    iter_ms(b, f, ctx) = (40.0 + 2.0*b) * (1.0 + (2520.0-f)/2520.0)
    dyn_j_per_token(b, f) = 1.0 + 0.001*b + 0.5*((f-1500)/510)**2   [J/tok, U 型]
    prefill_time_ms(tokens) = 50 + 0.05*tokens(与 f 无关,简化)
    """

    def __init__(self):
        self.freqs = (2520, 2100, 1800, 1500, 1200, 1050, 900, 600)

    def iter_time_ms(self, batch: int, freq_mhz: int, ctx_tokens: float = 272.0) -> float:
        base = 40.0 + 2.0 * float(batch)
        return base * (1.0 + max(0.0, 2520.0 - freq_mhz) / 2520.0)

    def dyn_j_per_token(self, batch: int, freq_mhz: int) -> float:
        # 返回 mJ/token(与 OpModel/PerfModel 单位一致;U 型,1500 最优)
        return 1000.0 * (1.0 + 0.001 * batch
                         + 0.5 * ((freq_mhz - 1500.0) / 510.0) ** 2)

    def prefill_time_ms(self, tokens: int, freq_mhz: int = 2520) -> float:
        return 50.0 + 0.05 * float(tokens)

    def prefill_dyn_j_per_token(self, freq_mhz: int) -> float:
        return 2.0e-3  # 2 mJ/token 固定

    def residency_w(self) -> float:
        return 140.0


@pytest.fixture
def synth_opmodel():
    return SynthOpModel()
