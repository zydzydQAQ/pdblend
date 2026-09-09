# -*- coding: utf-8 -*-
"""periodic:周期调度闭环(LoadMonitor + GlobalScheduler + PoolManager)。

测试先行(M4):
  - 低负载 → 选更省能分区 → park 空闲实例(inplace 迁移);
  - 负载回升 → unpark 扩池;
  - 迁移三关(dwell/迟滞/成本)由 GlobalScheduler 把守(已有单测),此处测闭环;
  - 未排空实例:本周期跳过,下周期续迁(幂等)。
"""
from __future__ import annotations

import pytest

from ecopadg.global_scheduler import GlobalScheduler, Prediction, build_partition_space
from ecopadg.periodic import PeriodicCoordinator
from ecopadg.pool_manager import PoolManager, STATE_PARKED, STATE_READY
from ecopadg.types import Partition, SystemConfig


class FakeExecutor:
    def __init__(self):
        self.healthy_all = True
        self.stopped = []

    def start(self, name, spec, port, kv_pair=None):
        return "h-" + name

    def stop(self, name, handle):
        self.stopped.append(name)

    def is_healthy(self, name, handle, port):
        return self.healthy_all


def fake_predictor(p: Partition, lam: float, state=None) -> Prediction:
    """容量 2 req/s / mixed 实例,1 req/s / 段;能耗 = 全部实例×100 + 空卡×35。"""
    cap = 2.0 * p.n_mixed + 1.0 * p.pairs()
    attainable = cap > 0 and lam < 0.85 * cap
    power = 100.0 * (p.n_mixed + p.n_prefill + p.n_decode) \
        + 35.0 * (p.gpus_total - p.total_gpus())
    return Prediction(attainable=attainable,
                      attainment=0.98 if attainable else 0.0,
                      power_w=power, energy_j_per_s=power)


@pytest.fixture
def coord():
    ex = FakeExecutor()
    pm = PoolManager(model="/models/m", tp=2, gpus=range(8), executor=ex,
                     port_base=8100)
    pm.materialize(Partition(n_mixed=4, tp_mixed=2, gpus_total=8))
    pm.poll_health()
    cfg = SystemConfig(model="m", target_attainment=0.9)
    space = build_partition_space({"mixed": [2], "prefill": [2],
                                   "decode": [2]}, 8)
    sched = GlobalScheduler(space, fake_predictor, cfg)
    c = PeriodicCoordinator(pool_manager=pm, scheduler=sched, config=cfg,
                            initial_partition=Partition(n_mixed=4, tp_mixed=2,
                                                        gpus_total=8))
    return c, pm, ex


def test_low_load_shrinks_pool(coord):
    c, pm, ex = coord
    d = c.step(now=0.0, lam_hat=1.0)
    # λ=1:nm=1(能耗 100+6×35=310)最优 → park 3 个实例
    assert d.migrated
    assert d.target.n_mixed == 1
    parked = [i for i in pm.instances() if i.state == STATE_PARKED]
    assert len(parked) == 3
    assert ex.stopped == []  # park 不重启


def test_partition_state_updates_after_migration(coord):
    c, pm, ex = coord
    c.step(now=0.0, lam_hat=1.0)
    assert c.current_partition.n_mixed == 1
    assert c.last_switch == 0.0


def test_urgent_expansion_bypasses_dwell(coord):
    # SLO 紧急通道:当前分区不可行时容量优先,dwell/迟滞/成本三关不拦
    c, pm, ex = coord
    c.step(now=0.0, lam_hat=1.0)      # → nm=1
    d = c.step(now=10.0, lam_hat=6.0)  # nm=1 cap=2 远超载 → 紧急扩容
    assert d.migrated
    assert d.action_reasons == ["urgent-slo"]
    assert d.target.n_mixed == 4


def test_grow_unparks_after_dwell(coord):
    c, pm, ex = coord
    c.step(now=0.0, lam_hat=1.0)       # → nm=1(3 parked)
    d = c.step(now=200.0, lam_hat=6.0)  # λ=6:nm=4(cap 8)才可行
    assert d.migrated
    assert d.target.n_mixed == 4
    ready = [i for i in pm.instances() if i.state == STATE_READY]
    assert len(ready) == 4
    assert ex.stopped == []  # unpark 也不重启


def test_busy_instance_defers_until_drained(coord):
    c, pm, ex = coord
    # 全部实例有 inflight → park 需先 drain,本周期不完成
    for i in pm.instances():
        pm.acquire(i.name)
    d = c.step(now=0.0, lam_hat=1.0)
    assert not d.migrated          # 分区未生效(等待排空)
    assert c.current_partition.n_mixed == 4
    # 释放后下周期完成迁移
    for i in pm.instances():
        pm.release(i.name)
    d = c.step(now=1.0, lam_hat=1.0)
    assert d.migrated
    assert c.current_partition.n_mixed == 1


def test_no_feasible_no_migration(coord):
    c, pm, ex = coord
    d = c.step(now=0.0, lam_hat=100.0)  # 超一切容量
    assert not d.migrated
    assert c.current_partition.n_mixed == 4
