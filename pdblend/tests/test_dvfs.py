# -*- coding: utf-8 -*-
"""dvfs:DVFS 控制器状态机(驻留、迟滞、强制复位)。"""
from __future__ import annotations

import pytest

from ecopadg.dvfs import DvfsController
from ecopadg.measure.backends import FakeBackend, BackendError
from tests.conftest import FakeClock


def _ctrl(clock=None, min_dwell_s=3.0):
    backend = FakeBackend(gpu_count=2, freqs=(2100, 1800, 1500, 1200, 900, 600))
    return DvfsController(backend=backend, min_dwell_s=min_dwell_s,
                          clock=clock or FakeClock())


def test_apply_first_transition():
    c = _ctrl()
    out = c.apply({0: 1500, 1: 1500}, now=10.0)
    assert set(out) == {(0, 2100, 1500), (1, 2100, 1500)}
    assert c.backend.locked == {0: 1500, 1: 1500}


def test_dwell_blocks_rapid_switch():
    c = _ctrl()
    c.apply({0: 1500}, now=10.0)
    c.clock.advance(1.0)  # 未满 3s 驻留
    out = c.apply({0: 900}, now=11.0)
    assert out == []  # 被驻留窗拦截
    assert c.backend.locked == {0: 1500}


def test_dwell_passes_after_window():
    c = _ctrl()
    c.apply({0: 1500}, now=10.0)
    c.clock.advance(3.0)
    out = c.apply({0: 900}, now=13.0)
    assert out == [(0, 1500, 900)]
    assert c.backend.locked == {0: 900}


def test_plan_returns_only_allowed():
    c = _ctrl()
    c.apply({0: 1500, 1: 1500}, now=10.0)
    c.clock.advance(3.5)
    plan = c.plan({0: 900, 1: 600}, now=13.5)
    assert plan == {0: 900, 1: 600}  # 两卡都已过驻留
    c.apply(plan, now=13.5)  # 实际执行切换,记录驻留账本
    c.clock.advance(0.5)
    plan2 = c.plan({0: 1200, 1: 900}, now=14.0)
    assert plan2 == {}  # 刚切完,全部拦截


def test_force_reset_always_allowed():
    c = _ctrl()
    c.apply({0: 1500}, now=10.0)
    c.clock.advance(0.1)
    c.force_reset([0])
    assert c.backend.locked == {}
    # 复位后可立即再设(不触发驻留拦截)
    out = c.apply({0: 900}, now=10.2)
    assert out == [(0, 2100, 900)]


def test_noop_transition_skipped():
    c = _ctrl()
    c.apply({0: 1500}, now=10.0)
    c.clock.advance(5.0)
    out = c.apply({0: 1500}, now=15.0)
    assert out == []  # 目标频率相同,不产生切换


def test_invalid_freq_raises():
    c = _ctrl()
    with pytest.raises(BackendError):
        c.apply({0: 1111}, now=0.0)


def test_history_recorded():
    c = _ctrl()
    c.apply({0: 1500}, now=10.0)
    c.clock.advance(4.0)
    c.apply({0: 900}, now=14.0)
    assert [(h[0], h[1]) for h in c.history()] == [(10.0, 0), (14.0, 0)]
