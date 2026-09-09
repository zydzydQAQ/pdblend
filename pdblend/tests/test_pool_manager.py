# -*- coding: utf-8 -*-
"""pool_manager:实例池运行时(测试先行,M2)。

语义:
  - materialize(partition):按分区启动 mixed 实例与 P/D 段(KV 连接器成对);
  - 池视图:ready_instances(role);inflight 计数 acquire/release;
  - drain:排空(不进新请求),inflight==0 即 drained;
  - plan_transition:分区 diff → 迁移步;mixed↔P/D 需重启(KV 配置在启动期),
    park/unpark(池收缩/扩张)为 inplace(秒级,只改路由+DVFS);
  - apply_transition:drain → stop/start 或 park,经注入 executor 执行。
"""
from __future__ import annotations

import pytest

from ecopadg.pool_manager import (
    MODE_INPLACE, MODE_RESTART, PoolManager, RuntimeInstance,
    STATE_DRAINING, STATE_PARKED, STATE_READY, STATE_STARTING, STATE_STOPPED,
)
from ecopadg.types import Partition


class FakeExecutor:
    """记录 start/stop 调用;health 可控。"""

    def __init__(self):
        self.started = []   # (name, spec, port, kv_pair)
        self.stopped = []   # name
        self.healthy = set()
        self._n = 0

    def start(self, name, spec, port, kv_pair=None):
        self._n += 1
        self.started.append((name, spec.role, port, kv_pair))
        return "h%d" % self._n

    def stop(self, name, handle):
        self.stopped.append(name)

    def is_healthy(self, name, handle, port):
        return name in self.healthy


@pytest.fixture
def ex():
    return FakeExecutor()


def make_pm(ex, tp=2, gpus=tuple(range(8))):
    return PoolManager(model="/models/m", tp=tp, gpus=gpus, executor=ex,
                       port_base=8100)


# ---------------------------------------------------------------------------
# materialize
# ---------------------------------------------------------------------------

def test_materialize_mixed_only(ex):
    pm = make_pm(ex)
    part = Partition(n_mixed=4, tp_mixed=2, gpus_total=8)
    insts = pm.materialize(part)
    assert len(insts) == 4
    assert all(i.spec.role == "mixed" for i in insts)
    assert all(i.state == STATE_STARTING for i in insts)
    # GPU 无重叠且全覆盖
    used = [g for i in insts for g in i.spec.gpus]
    assert sorted(used) == list(range(8))
    # 端口唯一
    ports = [i.port for i in insts]
    assert len(set(ports)) == 4
    # mixed 无 KV 配置
    assert all(kv is None for (_, _, _, kv) in ex.started)


def test_materialize_pd_segments_paired(ex):
    pm = make_pm(ex)
    part = Partition(n_prefill=2, n_decode=2, tp_prefill=2, tp_decode=2,
                     gpus_total=8)
    insts = pm.materialize(part)
    assert len(insts) == 4
    pre = [i for i in insts if i.spec.role == "prefill"]
    dec = [i for i in insts if i.spec.role == "decode"]
    assert len(pre) == 2 and len(dec) == 2
    # 成段配对:同段 producer rank=0 / consumer rank=1,parallel=2
    for p in pre:
        assert p.kv_pair == (0, 2)
        assert p.segment is not None
    for d in dec:
        assert d.kv_pair == (1, 2)
    segs = {p.segment for p in pre}
    assert segs == {d.segment for d in dec}
    assert len(segs) == 2


def test_materialize_hybrid(ex):
    pm = make_pm(ex)
    part = Partition(n_mixed=2, n_prefill=1, n_decode=1,
                     tp_mixed=2, tp_prefill=2, tp_decode=2, gpus_total=8)
    insts = pm.materialize(part)
    roles = sorted(i.spec.role for i in insts)
    assert roles == ["decode", "mixed", "mixed", "prefill"]


def test_materialize_rejects_overcommit(ex):
    pm = make_pm(ex)
    part = Partition(n_mixed=5, tp_mixed=2, gpus_total=8)
    with pytest.raises(ValueError):
        pm.materialize(part)


# ---------------------------------------------------------------------------
# 就绪 / 池视图 / inflight
# ---------------------------------------------------------------------------

def test_health_and_pools_view(ex):
    pm = make_pm(ex)
    pm.materialize(Partition(n_mixed=2, tp_mixed=2, gpus_total=8))
    assert pm.ready_instances("mixed") == []
    ex.healthy = {i.name for i in pm.instances()}
    pm.poll_health()
    ready = pm.ready_instances("mixed")
    assert len(ready) == 2
    assert all(i.state == STATE_READY for i in ready)


def test_acquire_release_inflight(ex):
    pm = make_pm(ex)
    pm.materialize(Partition(n_mixed=1, tp_mixed=2, gpus_total=8))
    name = pm.instances()[0].name
    ex.healthy = {name}
    pm.poll_health()
    pm.acquire(name)
    pm.acquire(name)
    assert pm.instances()[0].inflight == 2
    pm.release(name)
    assert pm.instances()[0].inflight == 1


def test_drain_blocks_new_and_completes(ex):
    pm = make_pm(ex)
    pm.materialize(Partition(n_mixed=1, tp_mixed=2, gpus_total=8))
    inst = pm.instances()[0]
    ex.healthy = {inst.name}
    pm.poll_health()
    pm.acquire(inst.name)
    pm.start_drain(inst.name)
    assert inst.state == STATE_DRAINING
    assert pm.ready_instances("mixed") == []  # 排空中不进新请求
    with pytest.raises(RuntimeError):
        pm.acquire(inst.name)
    assert not pm.is_drained(inst.name)
    pm.release(inst.name)
    assert pm.is_drained(inst.name)


# ---------------------------------------------------------------------------
# park / unpark(池收缩,inplace)
# ---------------------------------------------------------------------------

def test_park_unpark(ex):
    pm = make_pm(ex)
    pm.materialize(Partition(n_mixed=2, tp_mixed=2, gpus_total=8))
    ex.healthy = {i.name for i in pm.instances()}
    pm.poll_health()
    a = pm.instances()[0]
    pm.start_drain(a.name)
    pm.park(a.name)
    assert a.state == STATE_PARKED
    assert len(pm.ready_instances("mixed")) == 1
    assert ex.stopped == []  # park 不停实例(保权重,省驻留靠 DVFS)
    pm.unpark(a.name)
    assert a.state == STATE_READY


def test_park_requires_drained(ex):
    pm = make_pm(ex)
    pm.materialize(Partition(n_mixed=1, tp_mixed=2, gpus_total=8))
    inst = pm.instances()[0]
    ex.healthy = {inst.name}
    pm.poll_health()
    pm.acquire(inst.name)
    pm.start_drain(inst.name)
    with pytest.raises(RuntimeError):
        pm.park(inst.name)  # 还有 inflight


# ---------------------------------------------------------------------------
# plan_transition:分区 diff → 迁移步
# ---------------------------------------------------------------------------

def test_plan_transition_same_partition_empty(ex):
    pm = make_pm(ex)
    part = Partition(n_mixed=4, tp_mixed=2, gpus_total=8)
    pm.materialize(part)
    assert pm.plan_transition(part) == []


def test_plan_transition_shrink_is_inplace(ex):
    pm = make_pm(ex)
    pm.materialize(Partition(n_mixed=4, tp_mixed=2, gpus_total=8))
    steps = pm.plan_transition(Partition(n_mixed=3, tp_mixed=2, gpus_total=8))
    assert len(steps) == 1
    assert steps[0].mode == MODE_INPLACE
    assert steps[0].to_role == "parked"


def test_plan_transition_mixed_to_segment_is_restart(ex):
    pm = make_pm(ex)
    pm.materialize(Partition(n_mixed=4, tp_mixed=2, gpus_total=8))
    target = Partition(n_mixed=2, n_prefill=1, n_decode=1,
                       tp_mixed=2, tp_prefill=2, tp_decode=2, gpus_total=8)
    steps = pm.plan_transition(target)
    modes = sorted(s.mode for s in steps)
    to_roles = sorted(s.to_role for s in steps)
    assert modes == [MODE_RESTART, MODE_RESTART]
    assert to_roles == ["decode", "prefill"]


def test_plan_transition_prefers_idle_donors(ex):
    pm = make_pm(ex)
    pm.materialize(Partition(n_mixed=2, tp_mixed=2, gpus_total=8))
    ex.healthy = {i.name for i in pm.instances()}
    pm.poll_health()
    busy = pm.instances()[0]
    pm.acquire(busy.name)
    steps = pm.plan_transition(Partition(n_mixed=1, tp_mixed=2, gpus_total=8))
    assert len(steps) == 1
    assert steps[0].instance != busy.name  # 优先动空闲实例


# ---------------------------------------------------------------------------
# apply_transition
# ---------------------------------------------------------------------------

def test_apply_transition_park(ex):
    pm = make_pm(ex)
    pm.materialize(Partition(n_mixed=2, tp_mixed=2, gpus_total=8))
    ex.healthy = {i.name for i in pm.instances()}
    pm.poll_health()
    steps = pm.plan_transition(Partition(n_mixed=1, tp_mixed=2, gpus_total=8))
    done = pm.apply_transition(steps)
    assert len(done) == 1
    parked = [i for i in pm.instances() if i.state == STATE_PARKED]
    assert len(parked) == 1
    assert ex.stopped == []


def test_apply_transition_restart_stops_and_starts(ex):
    pm = make_pm(ex)
    pm.materialize(Partition(n_mixed=2, tp_mixed=2, gpus_total=8))
    ex.healthy = {i.name for i in pm.instances()}
    pm.poll_health()
    target = Partition(n_prefill=1, n_decode=1, tp_prefill=2, tp_decode=2,
                       gpus_total=8)
    steps = pm.plan_transition(target)
    n_started_before = len(ex.started)
    pm.apply_transition(steps)
    assert len(ex.stopped) == 2          # 两个 mixed 停止
    assert len(ex.started) == n_started_before + 2  # P/D 各启一个
    roles = sorted(i.spec.role for i in pm.instances()
                   if i.state != STATE_STOPPED)
    assert roles == ["decode", "prefill"]
    # 新 P/D 成段配对
    pre = [i for i in pm.instances() if i.spec.role == "prefill"
           and i.state != STATE_STOPPED][0]
    assert pre.kv_pair == (0, 2)


def test_apply_transition_blocked_by_inflight(ex):
    pm = make_pm(ex)
    pm.materialize(Partition(n_mixed=2, tp_mixed=2, gpus_total=8))
    ex.healthy = {i.name for i in pm.instances()}
    pm.poll_health()
    for i in pm.instances():
        pm.acquire(i.name)
    steps = pm.plan_transition(Partition(n_mixed=1, tp_mixed=2, gpus_total=8))
    done = pm.apply_transition(steps)
    assert done == []  # 未排空 → 本周期不执行,等下周期
    assert pm.instances()[0].state == STATE_DRAINING \
        or pm.instances()[1].state == STATE_DRAINING


def test_gpus_of_pool(ex):
    pm = make_pm(ex)
    pm.materialize(Partition(n_mixed=2, n_prefill=1, n_decode=1,
                             tp_mixed=2, tp_prefill=2, tp_decode=2,
                             gpus_total=8))
    gp = pm.pool_gpus()
    mixed, pre, dec = gp["mixed"], gp["prefill"], gp["decode"]
    flat = mixed + pre + dec
    assert len(flat) == 8 and len(set(flat)) == 8
    assert len(pre) == 2 and len(dec) == 2 and len(mixed) == 4
