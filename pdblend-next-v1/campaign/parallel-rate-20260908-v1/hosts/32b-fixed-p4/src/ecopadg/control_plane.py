# -*- coding: utf-8 -*-
"""三层在线控制面策略(只看已发生的到达/队列/slack)。

L0:谁接 prefill、突发时拒新 PD。
L1:CV / rate_fast 扩环、SLO 违约满频。
L2:低载缩环;mixed↔PD 只给门控结论(8 卡互斥,热路径不能 restart)。
不读 dataset、文件名 rate、poisson/gamma、inherit 表。
"""
from __future__ import annotations

from typing import Sequence

# Gamma 网格 CV=2;Poisson≈1。高于此视为突发。
CV_BURST = 1.5
RATE_SPIKE = 2.0
RING_MIN = 2
FRAC_LONG_PD_MAX = 0.35
CV_PD_MAX = 1.2
RATE_STABLE_MAX = 1.4


def burst_wanted(cv: float, rate_fast: float, rate: float,
                 slo_trip: bool = False) -> bool:
    """L1:间隔 CV 高或短窗速率相对长窗翻倍 → 先扩环。"""
    if slo_trip:
        return True
    try:
        cv_f = float(cv)
        rf = float(rate_fast)
        rs = float(rate)
    except (TypeError, ValueError):
        return False
    if cv_f == cv_f and cv_f >= CV_BURST:
        return True
    if rs != rs or rs <= 1e-12:
        return rf == rf and rf > 0
    if rf != rf:
        return False
    return rf / rs >= RATE_SPIKE


def want_expand_ring(cv: float, rate_fast: float, rate: float,
                     slo_trip: bool = False) -> bool:
    return burst_wanted(cv, rate_fast, rate, slo_trip=slo_trip)


def want_shrink_ring(cv: float, rate_fast: float, rate: float,
                     queues_empty: bool, slo_trip: bool = False) -> bool:
    """L2:队列空、无违约、到达不突发 → 可缩到 RING_MIN。"""
    if slo_trip or not queues_empty:
        return False
    if burst_wanted(cv, rate_fast, rate, slo_trip=False):
        return False
    return True


def refuse_new_pd(burst: bool, has_mixed: bool) -> bool:
    """已在 PD 模式时,突发且还有 mixed → 拒新 PD。"""
    return bool(burst and has_mixed)


def want_pd_mode(frac_long_hat: float, cv: float,
                 rate_fast: float, rate: float,
                 slack_ok: bool) -> bool:
    """L2 拓扑门控:短上下文、到达稳、slack 够才考虑切空间 PD。

    8 卡满配 mixed 与满配 PD 互斥,热路径不能 rematerialize。
    本函数只给结论;n=80 短 trace 预期很少为真。
    """
    if not slack_ok:
        return False
    try:
        fl = float(frac_long_hat)
        cv_f = float(cv)
        rf = float(rate_fast)
        rs = float(rate)
    except (TypeError, ValueError):
        return False
    if fl != fl or fl >= FRAC_LONG_PD_MAX:
        return False
    if cv_f != cv_f or cv_f >= CV_PD_MAX:
        return False
    if rs != rs or rs <= 1e-12:
        return False
    if rf != rf or rf / rs >= RATE_STABLE_MAX:
        return False
    return True


def pick_prefill_role(active: Sequence[int], role: int,
                      inflight: Sequence[int], cap: int) -> int:
    """当前 prefill_role 未饱和则沿用,否则滚到下一个未满实例。"""
    act = [int(i) for i in active]
    if not act:
        return 0
    if role not in act:
        return act[0]
    try:
        load = int(inflight[role])
    except (IndexError, TypeError, ValueError):
        load = 0
    if load < int(cap):
        return role
    start = act.index(role)
    for k in range(1, len(act)):
        nxt = act[(start + k) % len(act)]
        try:
            if int(inflight[nxt]) < int(cap):
                return nxt
        except (IndexError, TypeError, ValueError):
            return nxt
    return min(act, key=lambda i: inflight[i] if i < len(inflight) else 0)


def next_prefill_role(active: Sequence[int], role: int) -> int:
    act = [int(i) for i in active]
    if not act:
        return 0
    if role not in act:
        return act[0]
    return act[(act.index(role) + 1) % len(act)]
