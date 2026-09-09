# -*- coding: utf-8 -*-
"""三层控制面策略:突发扩环、拒 PD、启动不查表。"""
from __future__ import annotations

from ecopadg.control_plane import (
    burst_wanted, next_prefill_role, pick_prefill_role,
    refuse_new_pd, want_pd_mode, want_shrink_ring,
)


def test_burst_cv_and_rate_spike():
    assert burst_wanted(2.0, 1.0, 1.0) is True
    assert burst_wanted(0.8, 6.0, 2.0) is True
    assert burst_wanted(0.8, 2.0, 2.0) is False
    assert burst_wanted(0.8, 1.0, 1.0, slo_trip=True) is True


def test_refuse_new_pd_only_when_mixed_exists():
    assert refuse_new_pd(True, True) is True
    assert refuse_new_pd(True, False) is False
    assert refuse_new_pd(False, True) is False


def test_want_pd_rejects_long_or_bursty():
    assert want_pd_mode(0.1, 0.8, 2.0, 2.0, slack_ok=True) is True
    assert want_pd_mode(0.8, 0.8, 2.0, 2.0, slack_ok=True) is False
    assert want_pd_mode(0.1, 2.0, 2.0, 2.0, slack_ok=True) is False
    assert want_pd_mode(0.1, 0.8, 6.0, 2.0, slack_ok=True) is False
    assert want_pd_mode(0.1, 0.8, 2.0, 2.0, slack_ok=False) is False


def test_shrink_ring_needs_quiet_empty_queue():
    assert want_shrink_ring(0.8, 1.0, 1.0, True) is True
    assert want_shrink_ring(0.8, 1.0, 1.0, False) is False
    assert want_shrink_ring(2.0, 1.0, 1.0, True) is False


def test_pick_and_rotate_prefill_role():
    inflight = [8, 1, 0, 0]
    assert pick_prefill_role([0, 1, 2, 3], 0, inflight, cap=8) == 1
    assert pick_prefill_role([0, 1], 0, [3, 1], cap=8) == 0
    assert next_prefill_role([0, 1, 2], 1) == 2
    assert next_prefill_role([0, 1, 2], 2) == 0
