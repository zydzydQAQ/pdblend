# -*- coding: utf-8 -*-
"""在线容量崖护栏。不含 inherit/keep-mixed 查表。"""
from __future__ import annotations

# 不可信频率模型时 decode 地板,避免低载深 DVFS 削 att。
UNTRUSTED_DECODE_FLOOR_MHZ = 1500
CLIFF_RHO = 0.9


def cliff_lock_wanted(lam_hat: float, mu_mixed: float,
                      static_cliff: bool = False,
                      slo_trip: bool = False,
                      rho: float = CLIFF_RHO) -> bool:
    """高载/SLO 抢边 → 锁 f_max、禁 park。μ 缺测时只认 slo_trip。"""
    if static_cliff or slo_trip:
        return True
    try:
        lam = float(lam_hat)
        mu = float(mu_mixed)
    except (TypeError, ValueError):
        return False
    if lam != lam or mu != mu or mu <= 0:
        return False
    return lam + 1e-12 >= float(rho) * mu
