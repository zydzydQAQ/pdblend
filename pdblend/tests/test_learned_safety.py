"""Hard safety gates and fail-closed control-state aggregation."""
from __future__ import annotations

from dataclasses import replace

from ecopadg.mpc.types import (
    MODE_NODG,
    MODE_STRICT_PADG,
    ControlAction,
    FastAction,
    RolloutResult,
    SlowAction,
)
from ecopadg.safety_shield import SafetyShield, ShieldConfig
from ecopadg.telemetry import ControlState


def _state(**updates):
    state = ControlState(
        timestamp_s=200.0,
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
        active_replicas=2,
        full_replicas=4,
        fmax_mhz=2520,
        max_token_budget=8192,
        current_mode=MODE_NODG,
        current_frequency_mhz=2520,
        current_token_budget=8192,
        last_fast_change_s=0.0,
        last_slow_change_s=0.0,
        temporal_generation=4,
        temporal_ack_generation=4,
        temporal_ack_complete=True,
    )
    return replace(state, **updates)


def _action(mode=MODE_STRICT_PADG, freq=1800, replicas=2):
    return ControlAction(
        FastAction(mode, freq, 4096, 1),
        SlowAction(replicas),
    )


def _rollout(**updates):
    values = dict(
        gross_j=100.0,
        ttft_upper_s=0.5,
        tpot_upper_s=0.1,
        kv_blocks_required=10,
        queue_end=0.0,
        queue_growth=0.0,
        capacity_rps=10.0,
    )
    values.update(updates)
    return RolloutResult(**values)


def test_control_state_aggregates_complete_strict_telemetry():
    state = ControlState.aggregate(
        timestamp_s=10.0,
        engine={
            "expected": 2,
            "available": 2,
            "strict": True,
            "omega": 0.0,
            "free_gpu_blocks": {"mixed-0": 20, "mixed-1": 12},
        },
        baseline_attainment=0.97,
        baseline_measured=True,
        queue_observed=True,
    )
    assert state.strict_telemetry_available is True
    assert state.strict_omega == 0.0
    assert state.kv_free_blocks == 12
    assert state.has_measured_baseline is True


def test_shield_accepts_safe_strict_candidate():
    shield = SafetyShield()
    assert shield.reasons(_state(), _action(), _rollout()) == ()


def test_shield_rejects_latency_kv_queue_and_dwell():
    shield = SafetyShield(ShieldConfig(min_fast_dwell_s=50.0))
    state = _state(
        timestamp_s=20.0,
        last_fast_change_s=0.0,
        queue_stable=False,
        kv_free_blocks=5,
    )
    reasons = shield.reasons(
        state,
        _action(),
        _rollout(
            ttft_upper_s=2.0,
            tpot_upper_s=0.2,
            kv_blocks_required=10,
            queue_growth=1.0,
        ),
    )
    assert "ttft-headroom-predicted" in reasons
    assert "tpot-headroom-predicted" in reasons
    assert "kv-headroom" in reasons
    assert "queue-unstable-observed" in reasons
    assert "queue-unstable-predicted" in reasons
    assert "fast-dwell" in reasons


def test_strict_mode_requires_complete_zero_omega_telemetry():
    shield = SafetyShield()
    state = _state(
        current_mode=MODE_STRICT_PADG,
        strict_telemetry_available=False,
        strict_omega=0.1,
    )
    reasons = shield.reasons(state, _action(), _rollout())
    assert "strict-telemetry-missing" in reasons
    assert "strict-omega-nonzero" in reasons
    # Continuous admission does not depend on temporal phase telemetry.
    reasons_nodg = shield.reasons(
        state, _action(mode=MODE_NODG), _rollout()
    )
    assert "strict-telemetry-missing" not in reasons_nodg
    assert "strict-omega-nonzero" not in reasons_nodg


def test_emergency_fallback_is_deterministic_full_ring_nodg_fmax():
    fallback = SafetyShield.fallback(
        _state(
            baseline_measured=False,
            active_replicas=2,
            full_replicas=4,
        )
    )
    assert fallback.fast.mode == MODE_NODG
    assert fallback.fast.frequency_mhz == 2520
    assert fallback.fast.prefill_freq_mhz == 2520
    assert fallback.fast.decode_freq_mhz == 2520
    assert fallback.fast.n_prefill_active == 4
    assert fallback.fast.window_s >= 3.0
    assert fallback.fast.token_budget == 8192
    assert fallback.fast.rolling_offset == 0
    assert fallback.slow.active_replicas == 4
    configured = SafetyShield(
        ShieldConfig(min_fast_dwell_s=7.0)
    ).fallback(_state())
    assert configured.fast.window_s >= 7.0


def test_temporal_candidate_requires_generation_ack():
    state = _state(
        temporal_generation=8,
        temporal_ack_generation=7,
        temporal_ack_complete=False,
    )
    reasons = SafetyShield().reasons(
        state, _action(), _rollout()
    )
    assert "temporal-ack-incomplete" in reasons
    assert "temporal-generation-unacked" in reasons

    continuous = SafetyShield().reasons(
        state, _action(mode=MODE_NODG), _rollout()
    )
    assert "temporal-ack-incomplete" not in continuous
    assert "temporal-generation-unacked" not in continuous


def test_shield_validates_joint_action_ranges_and_window():
    action = ControlAction(
        FastAction(
            mode="temporal",
            frequency_mhz=2520,
            token_budget=4096,
            rolling_offset=3,
            n_prefill_active=3,
            window_s=1.0,
            prefill_freq_mhz=3000,
            decode_freq_mhz=600,
        ),
        SlowAction(4),
    )
    reasons = SafetyShield().reasons(
        _state(fmin_mhz=900), action, _rollout()
    )
    assert "prefill-count-out-of-range" in reasons
    assert "rolling-offset-out-of-range" in reasons
    assert "window-below-dwell" in reasons
    assert "prefill-frequency-out-of-range" in reasons
    assert "decode-frequency-out-of-range" in reasons


def test_prompt_age_must_preserve_ttft_headroom():
    state = _state(
        prompt_queue_tokens=4096.0,
        oldest_prompt_age_s=1.5,
    )
    reasons = SafetyShield().reasons(
        state, _action(mode=MODE_NODG), _rollout()
    )
    assert "prompt-age-ttft-headroom" in reasons


def test_control_state_aggregates_joint_observations_and_ack():
    state = ControlState.aggregate(
        timestamp_s=20.0,
        engine={
            "expected": 2,
            "available": 2,
            "strict": True,
            "omega": 0.0,
            "free_gpu_blocks": {"mixed-1": 12, "mixed-0": 20},
            "decode_backlog": {"mixed-1": 3, "mixed-0": 1},
            "applied_generations": {"mixed-1": 9, "mixed-0": 9},
            "requested_generation": 9,
        },
        prompt_queue_tokens=2048,
        oldest_prompt_age_s=0.25,
    )
    assert state.prompt_queue_tokens == 2048
    assert state.oldest_prompt_age_s == 0.25
    assert state.decode_backlog_per_instance == (1.0, 3.0)
    assert state.kv_free_blocks_per_instance == (20, 12)
    assert state.temporal_generation == 9
    assert state.temporal_ack_generation == 9
    assert state.temporal_ack_complete is True

