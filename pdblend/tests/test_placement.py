# -*- coding: utf-8 -*-
"""placement:DistServe 配置空间 + 事件仿真 + 二分搜索 + 选优。"""
from __future__ import annotations

import pytest

from ecopadg.placement import (
    Simulator, bisect_rate, decode_time_ms, distserve_configs,
    find_best_config, prefill_time_ms,
)


# 合成画像:prefill 批时延 = 10 + 0.05*Σtok (ms);decode 迭代 = 5 + 0.5*bs (ms)
PROFILE = {
    "test-model": {
        "1": {
            "decoding_large_small_bs_threshold": 1000,
            "prefill": [10.0, 0.05, 0.0],
            "decoding_smallbs": [5.0, 0.25, 0.25],
            "decoding_largebs": [5.0, 0.25, 0.25],
        },
        "2": {
            "decoding_large_small_bs_threshold": 1000,
            "prefill": [8.0, 0.03, 0.0],
            "decoding_smallbs": [4.0, 0.15, 0.15],
            "decoding_largebs": [4.0, 0.15, 0.15],
        },
    }
}


def test_cost_formulas_match_distserve():
    # prefill: a + b*Σtok + c*Σtok²
    # DistServe 原公式:delay = (a + b*Σtok + c*Σtok²)/pp + pp(pp_const=pp)
    assert prefill_time_ms(PROFILE, "test-model", tp=1, pp=1,
                           tokens_list=[100, 200]) == pytest.approx(26.0)
    # decode: a + b*Σgen + c*bs
    assert decode_time_ms(PROFILE, "test-model", tp=1, pp=1,
                          batch_size=4) == pytest.approx(7.0)  # a + b*bs + c*bs


def test_distserve_config_space():
    cfgs = distserve_configs(tps=[1, 2, 4], pps=[1, 2], total_gpus=8,
                              pp_cross_options=(1,))
    assert len(cfgs) > 0
    for (pp_cross, tp_p, pp_p, tp_d, pp_d) in cfgs:
        assert pp_cross * (tp_p * pp_p + tp_d * pp_d) <= 8
    assert len(set(cfgs)) == len(cfgs)  # 无重复
    # 含两阶段不对称配置(如 prefill TP2 + decode TP1)
    assert any(tp_p != tp_d for _, tp_p, _, tp_d, _ in cfgs)


def _workload(n=200, seed=0):
    import random
    rng = random.Random(seed)
    return [(100 + rng.randrange(0, 800), 100 + rng.randrange(0, 200))
            for _ in range(n)]


def test_simulator_low_load_all_served():
    sim = Simulator(profile=PROFILE, model="test-model",
                    workload_lengths=_workload(), tp_p=1, pp_p=1,
                    tp_d=1, pp_d=1, transfer_ms_per_kv_token=0.0, seed=0)
    r = sim.run(rate=0.2, N=30, ttft_target_ms=1000.0, tpot_target_ms=100.0)
    assert r["completed"] == 30
    assert r["p90_ttft_ms"] <= r["ttft_target_ms"]
    assert r["p90_tpot_ms"] <= r["tpot_target_ms"]
    # TTFT 下界 = prefill(100 tok)+ 首 decode 迭代,不可能为 0
    assert r["p50_ttft_ms"] > 10.0


def test_simulator_containment_flag():
    sim = Simulator(profile=PROFILE, model="test-model",
                    workload_lengths=_workload(), tp_p=1, pp_p=1,
                    tp_d=1, pp_d=1, transfer_ms_per_kv_token=0.0, seed=0)
    ok = sim.run(rate=0.2, N=30, ttft_target_ms=5000.0, tpot_target_ms=500.0)
    assert ok["containment"]
    bad = sim.run(rate=2.0, N=60, ttft_target_ms=20.0, tpot_target_ms=5.0)
    assert not bad["containment"]


def test_bisect_rate_monotone_by_slo():
    kw = dict(profile=PROFILE, model="test-model", workload_lengths=_workload(),
              tp_p=1, pp_p=1, tp_d=1, pp_d=1, transfer_ms_per_kv_token=0.0,
              seed=0, N=40, max_per_gpu_rate=4.0, esp=0.5)
    loose = bisect_rate(ttft_target_ms=2000.0, tpot_target_ms=500.0, **kw)
    tight = bisect_rate(ttft_target_ms=100.0, tpot_target_ms=50.0, **kw)
    assert loose >= tight  # SLO 越松,可达率越高
    assert 0.0 <= tight <= 4.0


def test_find_best_config():
    results = {
        (1, 1, 1, 1, 1): 2.0,
        (1, 2, 1, 2, 1): 3.5,   # 更优 rate
        (1, 2, 1, 1, 1): 3.5,   # 同 rate,GPU 更少(3 vs 4)
    }
    best = find_best_config(results)
    assert best == (1, 2, 1, 1, 1)
