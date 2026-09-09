"""Conservative rollout coverage for joint temporal actions."""
from __future__ import annotations

import pytest

from ecopadg.mpc.rollout import ConservativeRollout
from ecopadg.mpc.types import ControlAction, FastAction, SlowAction
from ecopadg.telemetry import ControlState


def _joint_state(**updates):
    values = dict(
        queue_depth=0.0,
        ttft_p90_s=0.2,
        tpot_p90_s=0.05,
        ttft_slo_s=20.0,
        tpot_slo_s=2.0,
        active_replicas=4,
        full_replicas=4,
        fmax_mhz=2520,
        max_token_budget=8192,
        capacity_per_replica_rps=5.0,
        mean_prompt_tokens=1024.0,
        mean_output_tokens=256.0,
        prompt_queue_tokens=4096.0,
        decode_backlog_per_instance=(2.0, 1.0, 0.0, 0.0),
        kv_free_blocks_per_instance=(100, 100, 100, 100),
    )
    values.update(updates)
    return ControlState(**values)


def _action(
    *,
    mode="continuous",
    n_prefill=4,
    window=3.0,
    fP=2520,
    fD=2520,
):
    return ControlAction(
        FastAction(
            mode=mode,
            frequency_mhz=fD,
            token_budget=8192,
            rolling_offset=0,
            n_prefill_active=n_prefill,
            window_s=window,
            prefill_freq_mhz=fP,
            decode_freq_mhz=fD,
        ),
        SlowAction(4),
    )


def test_legacy_rollout_retains_single_frequency_behavior():
    state = ControlState(
        queue_depth=0.0,
        ttft_p90_s=0.2,
        tpot_p90_s=0.05,
        ttft_slo_s=2.0,
        tpot_slo_s=0.2,
        active_replicas=2,
        full_replicas=2,
        fmax_mhz=2520,
        max_token_budget=8192,
        capacity_per_replica_rps=5.0,
    )
    action = ControlAction(
        FastAction("nodg", 1260, 8192),
        SlowAction(2),
    )
    result = ConservativeRollout()(state, action, [2.0, 2.0])

    assert result.capacity_rps == pytest.approx(5.0)
    assert result.gross_j == pytest.approx(504.0)
    assert result.ttft_upper_s == pytest.approx(0.2)
    assert result.tpot_upper_s == pytest.approx(0.1)


def test_joint_rollout_uses_prefill_and_decode_frequencies_separately():
    rollout = ConservativeRollout()
    state = _joint_state()

    fast_prefill = rollout(
        state, _action(fP=2520, fD=2520), [2.0, 2.0]
    )
    slow_prefill = rollout(
        state, _action(fP=1260, fD=2520), [2.0, 2.0]
    )
    slow_decode = rollout(
        state, _action(fP=2520, fD=1260), [2.0, 2.0]
    )

    assert slow_prefill.capacity_rps < fast_prefill.capacity_rps
    assert slow_prefill.ttft_upper_s > fast_prefill.ttft_upper_s
    assert (
        slow_prefill.details["prefill_energy_j"]
        < fast_prefill.details["prefill_energy_j"]
    )
    assert slow_decode.tpot_upper_s > fast_prefill.tpot_upper_s
    assert (
        slow_decode.details["decode_frequency_ratio"]
        < fast_prefill.details["decode_frequency_ratio"]
    )


def test_temporal_rollout_accounts_for_np_and_window_waiting():
    rollout = ConservativeRollout()
    state = _joint_state()
    all_prefill = rollout(
        state,
        _action(mode="temporal", n_prefill=4, window=3.0),
        [2.0, 2.0],
    )
    one_prefill = rollout(
        state,
        _action(mode="temporal", n_prefill=1, window=3.0),
        [2.0, 2.0],
    )
    long_window = rollout(
        state,
        _action(mode="temporal", n_prefill=4, window=6.0),
        [2.0, 2.0],
    )

    assert one_prefill.capacity_rps < all_prefill.capacity_rps
    assert one_prefill.ttft_upper_s > all_prefill.ttft_upper_s
    assert long_window.ttft_upper_s > all_prefill.ttft_upper_s
    assert all_prefill.details["window_wait_s"] == 3.0
    assert one_prefill.details["prefill_engine_fraction"] == 0.25
