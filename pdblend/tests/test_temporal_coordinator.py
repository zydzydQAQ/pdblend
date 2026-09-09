# -*- coding: utf-8 -*-
"""CPU-only tests for global KV-local temporal admission."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ecopadg.temporal_coordinator import (
    MODE_CONTINUOUS,
    MODE_TEMPORAL,
    GlobalTemporalCoordinator,
    TemporalConfig,
)


def _view(phase="decode", *, stale=False, n_prefill=0):
    return SimpleNamespace(
        phase=phase,
        stale=stale,
        n_prefill_groups=n_prefill,
    )


@pytest.mark.parametrize(
    ("n_prefill", "initial", "rotated"),
    [
        (1, (0,), (1,)),
        (2, (0, 1), (2, 3)),
        (4, (0, 1, 2, 3), (0, 1, 2, 3)),
    ],
)
def test_temporal_np_eligibility_and_round_robin_rotation(
        n_prefill, initial, rotated):
    coordinator = GlobalTemporalCoordinator(
        4,
        TemporalConfig(
            mode=MODE_TEMPORAL,
            n_prefill_active=n_prefill,
            window_s=1.5,
        ),
        clock=lambda: 0.0,
    )
    assert tuple(coordinator.eligible_indices([0, 1, 2, 3])) == initial

    before = coordinator.step(
        1.49, {index: _view() for index in initial})
    assert before.active_set == initial
    assert before.epoch == 0

    after = coordinator.step(
        1.5, {index: _view() for index in initial})
    assert after.active_set == rotated
    assert after.epoch == 1


def test_prefill_defers_expired_rotation_until_phase_clears():
    coordinator = GlobalTemporalCoordinator(
        4,
        TemporalConfig(
            mode=MODE_TEMPORAL,
            n_prefill_active=1,
            window_s=1.5,
        ),
        clock=lambda: 0.0,
    )
    deferred = coordinator.step(1.5, {0: _view("prefill", n_prefill=1)})
    assert deferred.active_set == (0,)
    assert deferred.epoch == 0
    assert deferred.deferred_rotations == 1

    # Repeated pump ticks in the same expired window do not inflate the count.
    repeated = coordinator.step(1.6, {0: _view("prefill", n_prefill=1)})
    assert repeated.deferred_rotations == 1

    rotated = coordinator.step(1.7, {0: _view("decode")})
    assert rotated.active_set == (1,)
    assert rotated.epoch == 1


@pytest.mark.parametrize(
    "phase_views",
    [
        {},
        {0: _view("decode", stale=True)},
        {0: _view("unknown")},
    ],
)
def test_missing_stale_or_unknown_telemetry_defers_rotation(phase_views):
    coordinator = GlobalTemporalCoordinator(
        4,
        TemporalConfig(
            mode=MODE_TEMPORAL,
            n_prefill_active=1,
            window_s=1.5,
        ),
        clock=lambda: 0.0,
    )
    state = coordinator.step(2.0, phase_views)
    assert state.active_set == (0,)
    assert state.epoch == 0
    assert state.deferred_rotations == 1
    assert coordinator.snapshot().mode == MODE_TEMPORAL


def test_continuous_mode_uses_all_active_engines():
    coordinator = GlobalTemporalCoordinator(
        4,
        TemporalConfig(mode=MODE_CONTINUOUS, n_prefill_active=1),
        active=[0, 2, 3],
        clock=lambda: 0.0,
    )
    assert coordinator.eligible_indices([0, 2, 3]) == [0, 2, 3]
    assert coordinator.choose_prefill_target(
        [0, 2, 3], [5, 99, 1, 2]) == 2


def test_temporal_np_is_bounded_and_recovers_when_engines_return():
    coordinator = GlobalTemporalCoordinator(
        4,
        TemporalConfig(
            mode=MODE_TEMPORAL,
            n_prefill_active=4,
            window_s=1.5,
        ),
        active=[0, 1],
        clock=lambda: 0.0,
    )
    assert coordinator.snapshot().n_prefill_active == 2
    assert coordinator.eligible_indices([0, 1]) == [0, 1]

    assert coordinator.eligible_indices([0, 1, 2, 3]) == [0, 1, 2, 3]
    assert coordinator.snapshot().n_prefill_active == 4


def test_emergency_fallback_opens_all_and_records_reason():
    coordinator = GlobalTemporalCoordinator(
        4,
        TemporalConfig(
            mode=MODE_TEMPORAL,
            n_prefill_active=1,
            window_s=1.5,
        ),
        clock=lambda: 0.0,
    )
    state = coordinator.emergency_continuous(
        reason="slo-trip:tpot", active=[0, 1, 2, 3], now=0.5)
    assert state.mode == MODE_CONTINUOUS
    assert state.n_prefill_active == 4
    assert state.active_set == (0, 1, 2, 3)
    assert state.fallback_reason == "slo-trip:tpot"
    assert coordinator.eligible_indices([0, 1, 2, 3]) == [0, 1, 2, 3]
    assert any(event.event == "fallback"
               for event in coordinator.drain_events())


def test_burst_fallback_is_temporary():
    coordinator = GlobalTemporalCoordinator(
        4,
        TemporalConfig(
            mode=MODE_TEMPORAL,
            n_prefill_active=1,
            window_s=1.5,
        ),
        clock=lambda: 0.0,
    )
    expanded = coordinator.emergency_continuous(
        reason="burst",
        active=[0, 1, 2, 3],
        now=0.5,
        temporary=True,
        hold_s=1.5,
    )
    assert expanded.mode == MODE_CONTINUOUS
    assert expanded.active_set == (0, 1, 2, 3)

    recovered = coordinator.step(
        2.0, {index: _view() for index in range(4)})
    assert recovered.mode == MODE_TEMPORAL
    assert recovered.active_set == (0,)
    assert recovered.fallback_reason == ""


def test_configure_is_atomic_and_carries_future_targets():
    coordinator = GlobalTemporalCoordinator(4, clock=lambda: 0.0)
    original = coordinator.snapshot()
    with pytest.raises(ValueError):
        coordinator.configure(
            mode=MODE_TEMPORAL,
            n_prefill_active=3,
            window_s=1.5,
            now=1.0,
        )
    assert coordinator.snapshot() == original

    state = coordinator.configure(
        mode=MODE_TEMPORAL,
        n_prefill_active=2,
        window_s=2.5,
        token_budget=4096,
        fP=2520,
        fD=1350,
        active=[0, 1, 2, 3],
        now=1.0,
    )
    assert state.generation == original.generation + 1
    assert state.n_prefill_active == 2
    assert state.window_s == 2.5
    assert state.token_budget == 4096
    assert state.fP == 2520
    assert state.fD == 1350
