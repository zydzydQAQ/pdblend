"""Toy fast/slow MPC searches use shielded gross-J objectives."""
from __future__ import annotations

from dataclasses import replace

import pytest

from ecopadg.mpc.fast_mpc import FastMPC
from ecopadg.mpc.slow_mpc import SlowMPC
from ecopadg.mpc.types import (
    MODE_CONTINUOUS,
    MODE_NODG,
    MODE_STRICT_PADG,
    MODE_TEMPORAL,
    ControlAction,
    FastAction,
    RolloutResult,
    SlowAction,
    TransitionCosts,
)
from ecopadg.safety_shield import SafetyShield
from ecopadg.telemetry import ControlState


def _state(**updates):
    state = ControlState(
        timestamp_s=1000.0,
        arrival_rate_rps=2.0,
        arrival_rate_fast_rps=2.0,
        queue_stable=True,
        ttft_p90_s=0.2,
        tpot_p90_s=0.05,
        ttft_slo_s=2.0,
        tpot_slo_s=0.2,
        kv_free_blocks=100,
        baseline_attainment=0.98,
        baseline_measured=True,
        capacity_trusted=True,
        frequency_trusted=True,
        strict_telemetry_available=True,
        strict_omega=0.0,
        active_replicas=3,
        full_replicas=4,
        fmax_mhz=2520,
        max_token_budget=8192,
        current_mode=MODE_NODG,
        current_frequency_mhz=2520,
        current_token_budget=8192,
        current_rolling_offset=0,
        last_fast_change_s=0.0,
        last_slow_change_s=0.0,
        temporal_generation=1,
        temporal_ack_generation=1,
        temporal_ack_complete=True,
    )
    return replace(state, **updates)


def _result(gross, *, queue_growth=0.0):
    return RolloutResult(
        gross_j=float(gross),
        ttft_upper_s=0.5,
        tpot_upper_s=0.1,
        kv_blocks_required=10,
        queue_end=max(float(queue_growth), 0.0),
        queue_growth=float(queue_growth),
        capacity_rps=10.0,
    )


def _zero_costs():
    return TransitionCosts(
        mode_switch_j=0.0,
        frequency_switch_j=0.0,
        token_budget_switch_j=0.0,
        rolling_offset_switch_j=0.0,
        replica_switch_j=0.0,
    )


def test_fast_mpc_selects_lowest_feasible_gross_j():
    def rollout(state, action, loads):
        del state, loads
        gross = 40.0 if (
            action.fast.mode == MODE_STRICT_PADG
            and action.fast.frequency_mhz == 1800
        ) else 100.0
        return _result(gross)

    mpc = FastMPC(
        rollout,
        SafetyShield(),
        modes=(MODE_NODG, MODE_STRICT_PADG),
        frequencies_mhz=(2520, 1800),
        token_budgets=(8192,),
        rolling_offsets=(0,),
        transition_costs=_zero_costs(),
    )
    decision = mpc.decide(_state(), [2.0, 2.0])
    assert decision.fallback is False
    assert decision.chosen.fast.mode == MODE_STRICT_PADG
    assert decision.chosen.fast.frequency_mhz == 1800


def test_fast_mpc_adds_transition_cost_to_gross_j():
    def rollout(state, action, loads):
        del state, loads
        return _result(
            95.0 if action.fast.frequency_mhz == 1800 else 100.0
        )

    mpc = FastMPC(
        rollout,
        SafetyShield(),
        actions=[
            FastAction(MODE_NODG, 2520, 8192, 0),
            FastAction(MODE_NODG, 1800, 8192, 0),
        ],
        transition_costs=TransitionCosts(
            mode_switch_j=0.0,
            frequency_switch_j=10.0,
            token_budget_switch_j=0.0,
            rolling_offset_switch_j=0.0,
            replica_switch_j=0.0,
        ),
    )
    decision = mpc.decide(_state(), [2.0])
    assert decision.chosen.fast.frequency_mhz == 2520


def test_slow_mpc_filters_unstable_small_ring():
    def rollout(state, action, loads):
        del state, loads
        count = action.slow.active_replicas
        if count == 1:
            return _result(30.0, queue_growth=1.0)
        return _result(60.0 if count == 2 else 90.0)

    mpc = SlowMPC(
        rollout,
        SafetyShield(),
        replica_counts=(1, 2, 3),
        transition_costs=_zero_costs(),
    )
    decision = mpc.decide(_state(), [2.0])
    assert decision.fallback is False
    assert decision.chosen.slow.active_replicas == 2
    assert decision.candidates[0].feasible is False
    assert "queue-unstable-predicted" in (
        decision.candidates[0].shield_reasons
    )


def test_no_feasible_candidate_returns_emergency_action():
    mpc = FastMPC(
        lambda state, action, loads: _result(10.0),
        SafetyShield(),
        actions=[FastAction(MODE_STRICT_PADG, 1800, 4096, 1)],
        transition_costs=_zero_costs(),
    )
    decision = mpc.decide(
        _state(baseline_measured=False, strict_telemetry_available=False),
        [2.0],
    )
    assert decision.fallback is True
    assert decision.chosen.fast.mode == MODE_NODG
    assert decision.chosen.fast.frequency_mhz == 2520
    assert decision.chosen.slow.active_replicas == 4


def test_fast_action_normalizes_joint_fields_and_mode_aliases():
    legacy = FastAction("nodg", 1800, 4096)
    assert legacy.mode == MODE_CONTINUOUS
    assert legacy.prefill_freq_mhz == 1800
    assert legacy.decode_freq_mhz == 1800
    assert legacy.n_prefill_active == 4
    assert legacy.window_s == 3.0

    temporal = FastAction(
        mode="strict-padg",
        frequency_mhz=2520,
        token_budget=8192,
        rolling_offset=1,
        n_prefill_active=2,
        window_s=6.0,
        prefill_freq_mhz=2100,
        decode_freq_mhz=1800,
    )
    payload = temporal.to_dict()
    assert payload == {
        "mode": MODE_TEMPORAL,
        "frequency_mhz": 2520,
        "n_prefill_active": 2,
        "window_s": 6.0,
        "token_budget": 8192,
        "prefill_freq_mhz": 2100,
        "decode_freq_mhz": 1800,
        "rolling_offset": 1,
    }
    assert FastAction("temporal", 2520, 8192).mode == MODE_TEMPORAL
    with pytest.raises(ValueError):
        FastAction("padg", 2520, 8192)


def test_transition_costs_charge_joint_dimensions_separately():
    current = ControlAction(
        FastAction(
            "continuous",
            2520,
            8192,
            0,
            4,
            3.0,
            2520,
            2520,
        ),
        SlowAction(4),
    )
    target = ControlAction(
        FastAction(
            "temporal",
            2100,
            8192,
            0,
            2,
            6.0,
            1800,
            2100,
        ),
        SlowAction(4),
    )
    costs = TransitionCosts(
        mode_switch_j=1.0,
        frequency_switch_j=100.0,
        token_budget_switch_j=0.0,
        rolling_offset_switch_j=0.0,
        replica_switch_j=0.0,
        window_switch_j=2.0,
        prefill_count_switch_j=3.0,
        prefill_frequency_switch_j=5.0,
        decode_frequency_switch_j=7.0,
    )
    assert costs.cost(current, target) == 18.0
    legacy_current = ControlAction(
        FastAction("nodg", 2520, 8192), SlowAction(4)
    )
    legacy_target = ControlAction(
        FastAction("nodg", 1800, 8192), SlowAction(4)
    )
    assert costs.cost(legacy_current, legacy_target) == 100.0


def test_fast_mpc_enumerates_and_prunes_declared_joint_product():
    kwargs = dict(
        modes=("continuous", "temporal"),
        frequencies_mhz=(2520,),
        token_budgets=(8192,),
        rolling_offsets=(0, 1, 2, 3),
        prefill_counts=(1, 2, 4),
        window_seconds=(3.0, 6.0),
        prefill_frequencies_mhz=(1800, 2520),
        decode_frequencies_mhz=(1200, 2520),
    )
    first = FastMPC(
        lambda state, action, loads: _result(1.0),
        SafetyShield(),
        **kwargs,
    )
    second = FastMPC(
        lambda state, action, loads: _result(1.0),
        SafetyShield(),
        **kwargs,
    )

    # Per window/fP/fD: one canonical continuous action plus
    # 1 + 2 + 4 valid temporal offsets.
    assert len(first.actions) == 2 * 2 * 2 * (1 + 1 + 2 + 4)
    assert [item.to_dict() for item in first.actions] == [
        item.to_dict() for item in second.actions
    ]
    assert all(
        action.rolling_offset < action.n_prefill_active
        for action in first.actions
    )
    continuous = [
        action
        for action in first.actions
        if action.mode == MODE_CONTINUOUS
    ]
    assert continuous
    assert {
        (action.n_prefill_active, action.rolling_offset)
        for action in continuous
    } == {(4, 0)}

