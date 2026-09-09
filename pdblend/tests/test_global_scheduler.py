# -*- coding: utf-8 -*-
"""global_scheduler:分区选择、迟滞、最小驻留、迁移成本。"""
from __future__ import annotations

from ecopadg.global_scheduler import (
    GlobalScheduler, GlobalState, Prediction, build_partition_space,
)
from ecopadg.types import Partition, SystemConfig
from tests.conftest import FakeClock


def _space():
    return [
        Partition(n_mixed=2, n_prefill=0, n_decode=0, tp_mixed=2,
                  tp_prefill=2, tp_decode=2, gpus_total=8),  # A: 全 mixed
        Partition(n_mixed=0, n_prefill=2, n_decode=2, tp_mixed=2,
                  tp_prefill=2, tp_decode=2, gpus_total=8),  # B: 2P2D
    ]


class _Predictor:
    """合成预测器:分区 A 能耗 1000W/可行,分区 B 能耗 800W/可行。"""

    def __init__(self, attainable=None):
        self.attainable = attainable if attainable is not None else {0: True, 1: True}

    def predict(self, partition, lam_hat, state):
        power_w = 1000.0 if partition.n_mixed == 2 else 800.0
        att = self.attainable.get(partition.n_mixed, True)
        return Prediction(attainable=att, attainment=0.6 if att else 0.2,
                          power_w=power_w, energy_j_per_s=power_w)


def _cfg():
    return SystemConfig(model="syn", partition_min_dwell_s=120.0,
                        hysteresis=0.15, target_attainment=0.5,
                        attainment_tolerance_pp=0.03)


def test_migrates_to_lower_energy_when_clear():
    clock = FakeClock(0.0)
    pred = _Predictor()
    gs = GlobalScheduler(space=_space(), predictor=pred.predict, config=_cfg(),
                        clock=clock)
    st = GlobalState(partition=_space()[0], lam_hat=1.0,
                     baseline_attainment=0.6, last_switch=-999.0)
    act = gs.step(st, now=200.0)
    assert act.migrate
    assert act.partition.n_mixed == 0  # 迁到 B(800W)
    assert act.partition.n_prefill == 2


def test_no_migrate_within_min_dwell():
    clock = FakeClock(0.0)
    pred = _Predictor()
    gs = GlobalScheduler(space=_space(), predictor=pred.predict, config=_cfg(),
                        clock=clock)
    st = GlobalState(partition=_space()[0], lam_hat=1.0,
                     baseline_attainment=0.6, last_switch=150.0)
    act = gs.step(st, now=200.0)  # 距上次切换 50s < 120s
    assert not act.migrate
    assert "dwell" in act.reasons[0] or any("dwell" in r for r in act.reasons)


def test_no_migrate_when_gain_below_hysteresis():
    clock = FakeClock(0.0)
    pred = _Predictor()
    pred.predict = (lambda p, lam, st: Prediction(
        attainable=True, attainment=0.6,
        power_w=1000.0 if p.n_mixed == 2 else 950.0,  # 只省 5% < 15%
        energy_j_per_s=1000.0 if p.n_mixed == 2 else 950.0))
    gs = GlobalScheduler(space=_space(), predictor=pred.predict, config=_cfg(),
                        clock=clock)
    st = GlobalState(partition=_space()[0], lam_hat=1.0,
                     baseline_attainment=0.6, last_switch=-999.0)
    act = gs.step(st, now=200.0)
    assert not act.migrate


def test_unattainable_candidate_filtered():
    clock = FakeClock(0.0)
    pred = _Predictor(attainable={2: True, 0: False})  # B(n_mixed=0)不可行
    gs = GlobalScheduler(space=_space(), predictor=pred.predict, config=_cfg(),
                        clock=clock)
    st = GlobalState(partition=_space()[0], lam_hat=1.0,
                     baseline_attainment=0.6, last_switch=-999.0)
    act = gs.step(st, now=200.0)
    assert not act.migrate
    assert act.partition.n_mixed == 2  # 留在 A


def test_ranks_energy_proxy_not_power_w():
    # 低功率但高 e_req 的分区不得赢:L3 比 energy_j_per_s
    clock = FakeClock(0.0)

    def predict(p, lam, st):
        if p.n_mixed == 2:
            return Prediction(attainable=True, attainment=0.95,
                              power_w=900.0, energy_j_per_s=950.0)
        return Prediction(attainable=True, attainment=0.95,
                          power_w=700.0, energy_j_per_s=1400.0)

    gs = GlobalScheduler(space=_space(), predictor=predict, config=_cfg(),
                        clock=clock)
    st = GlobalState(partition=_space()[1], lam_hat=1.0,
                     baseline_attainment=0.6, last_switch=-999.0)
    act = gs.step(st, now=200.0)
    assert act.migrate
    assert act.partition.n_mixed == 2


def test_target_attainment_enforced():
    clock = FakeClock(0.0)
    # B 可行但 attainment 0.4 < target(0.6 - 0.03)→ 过滤
    pred = _Predictor()
    pred.predict = (lambda p, lam, st: Prediction(
        attainable=True, attainment=0.6 if p.n_mixed == 2 else 0.4,
        power_w=1000.0 if p.n_mixed == 2 else 700.0,
        energy_j_per_s=1000.0 if p.n_mixed == 2 else 700.0))
    gs = GlobalScheduler(space=_space(), predictor=pred.predict, config=_cfg(),
                        clock=clock)
    st = GlobalState(partition=_space()[0], lam_hat=1.0,
                     baseline_attainment=0.6, last_switch=-999.0)
    act = gs.step(st, now=200.0)
    assert not act.migrate


def test_partition_space_allows_asymmetric_pd():
    space = build_partition_space(
        {"mixed": [2], "prefill": [2], "decode": [2]}, 8)
    assert any(p.n_prefill == 1 and p.n_decode == 2 for p in space)
    assert any(p.n_prefill == 2 and p.n_decode == 1 for p in space)
    assert not any(p.n_prefill > 0 and p.n_decode == 0 for p in space)


def test_target_is_baseline_minus_delta_not_absolute_s95():
    # baseline 0.80 → 门槛 0.79;att 0.80 的更省电分区应入选
    clock = FakeClock(0.0)

    def predict(p, lam, st):
        if p.n_mixed == 2:
            return Prediction(attainable=True, attainment=0.80,
                              power_w=1000.0, energy_j_per_s=1000.0)
        return Prediction(attainable=True, attainment=0.80,
                          power_w=800.0, energy_j_per_s=800.0)

    cfg = SystemConfig(model="syn", partition_min_dwell_s=120.0,
                       hysteresis=0.15, slo_non_inferior_pp=0.01)
    gs = GlobalScheduler(space=_space(), predictor=predict, config=cfg,
                         clock=clock)
    st = GlobalState(partition=_space()[0], lam_hat=1.0,
                     baseline_attainment=0.80, last_switch=-999.0)
    act = gs.step(st, now=200.0)
    assert act.migrate
    assert act.partition.n_prefill == 2


def test_no_downscale_when_lambda_cold():
    """λ̂=0 时 1 段更省,但禁止 park 掉启动的 2×1P1D。"""
    space = [
        Partition(n_mixed=0, n_prefill=2, n_decode=2, tp_mixed=2,
                  tp_prefill=2, tp_decode=2, gpus_total=8),
        Partition(n_mixed=0, n_prefill=1, n_decode=1, tp_mixed=2,
                  tp_prefill=2, tp_decode=2, gpus_total=8),
    ]

    def predict(p, lam, st):
        e = 400.0 if p.pairs() == 1 else 800.0
        return Prediction(attainable=True, attainment=0.98,
                          power_w=e, energy_j_per_s=e)

    gs = GlobalScheduler(space=space, predictor=predict, config=_cfg())
    st = GlobalState(partition=space[0], lam_hat=0.0,
                     baseline_attainment=0.97, last_switch=-999.0)
    act = gs.step(st, now=200.0)
    assert not act.migrate
    assert act.partition.pairs() == 2
    assert "no-downscale-cold" in act.reasons


def test_no_feasible_asks_max_freq():
    pred = _Predictor(attainable={2: False, 0: False})
    gs = GlobalScheduler(space=_space(), predictor=pred.predict, config=_cfg())
    st = GlobalState(partition=_space()[0], lam_hat=1.0,
                     baseline_attainment=0.99, last_switch=-999.0)
    act = gs.step(st, now=200.0)
    assert not act.migrate
    assert "no-feasible-candidate" in act.reasons
    assert "max-freq" in act.reasons


def test_spatial_restart_is_not_charged_as_inplace_migration():
    pred = _Predictor()
    cfg = _cfg()
    cfg.migration_cost_j = 1e30
    gs = GlobalScheduler(space=_space(), predictor=pred.predict, config=cfg)
    st = GlobalState(
        partition=_space()[0],
        lam_hat=1.0,
        baseline_attainment=0.6,
        last_switch=-999.0,
    )
    action = gs.step(st, now=200.0)
    assert action.migrate
    assert action.restart_required
    assert action.projected_savings_w == 200.0
