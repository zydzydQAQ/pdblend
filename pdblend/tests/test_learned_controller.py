"""Learned controller orchestration, fallback, logging, and shadow isolation."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ecopadg.learned_controller import LearnedController, ModelBundle
from ecopadg.logging_schema import DecisionJSONLLogger
from ecopadg.mpc.types import (
    MODE_NODG,
    RolloutResult,
    TransitionCosts,
)
from ecopadg.shadow_runner import ShadowRunner
from ecopadg.strict_padg_executor import InMemoryStrictPaDGExecutor
from ecopadg.telemetry import ControlState


def _state(**updates):
    state = ControlState(
        timestamp_s=1000.0,
        arrival_rate_rps=1.0,
        arrival_rate_fast_rps=1.0,
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
        capacity_per_replica_rps=5.0,
    )
    return replace(state, **updates)


def _bundle():
    return ModelBundle(
        version="test-model-v1",
        frequencies_mhz=(2520,),
        token_budgets=(8192,),
        rolling_offsets=(0,),
        replica_counts=(2,),
        modes=(MODE_NODG,),
        horizon_steps=2,
        transition_costs=TransitionCosts(
            mode_switch_j=0.0,
            frequency_switch_j=0.0,
            token_budget_switch_j=0.0,
            rolling_offset_switch_j=0.0,
            replica_switch_j=0.0,
        ),
    )


def _rollout(state, action, loads):
    del state, action, loads
    return RolloutResult(
        gross_j=100.0,
        ttft_upper_s=0.5,
        tpot_upper_s=0.1,
        kv_blocks_required=5,
        queue_end=0.0,
        queue_growth=0.0,
        capacity_rps=10.0,
    )


def test_learned_controller_falls_back_without_measured_baseline():
    controller = LearnedController(bundle=_bundle(), rollout=_rollout)
    decision = controller.decide(_state(baseline_measured=False))
    assert decision.fallback is True
    assert decision.chosen.fast.mode == MODE_NODG
    assert decision.chosen.fast.frequency_mhz == 2520
    assert decision.chosen.slow.active_replicas == 4
    assert "baseline-unmeasured" in decision.reasons


def test_shadow_runner_logs_but_never_calls_executor(tmp_path):
    executor = InMemoryStrictPaDGExecutor()
    controller = LearnedController(
        bundle=_bundle(), rollout=_rollout, executor=executor
    )
    logger = DecisionJSONLLogger(str(tmp_path / "decisions.jsonl"))
    shadow = ShadowRunner(controller, logger)
    decision = shadow.evaluate(_state())
    assert decision.fallback is False
    assert executor.history == []
    payload = json.loads(
        (tmp_path / "decisions.jsonl").read_text(encoding="utf-8")
    )
    assert payload["model_version"] == "test-model-v1"
    assert payload["model_config_hash"] == controller.model_config_hash
    assert payload["candidates"]
    assert payload["chosen_action"] is None
    assert payload["shadow_action"]["fast"]["mode"] == MODE_NODG
    assert "shield_reasons" in payload


def test_active_run_uses_explicit_executor_boundary():
    executor = InMemoryStrictPaDGExecutor()
    controller = LearnedController(
        bundle=_bundle(), rollout=_rollout, executor=executor
    )
    decision = controller.run(_state(), actuate=True)
    assert executor.history == [decision.chosen]


def test_model_bundle_loads_typed_json(tmp_path):
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps({
        "version": "bundle-v2",
        "frequencies_mhz": [2520, 1800],
        "token_budgets": [4096, 8192],
        "rolling_offsets": [0, 1],
        "replica_counts": [1, 2],
        "modes": ["nodg", "strict-padg"],
        "horizon_steps": 4,
    }), encoding="utf-8")
    bundle = ModelBundle.load(str(path))
    assert bundle.version == "bundle-v2"
    assert bundle.frequencies_mhz == (2520, 1800)
    assert bundle.prefill_frequencies_mhz == ()
    assert bundle.decode_frequencies_mhz == ()
    assert bundle.modes == ("continuous", "temporal")
    assert bundle.horizon_steps == 4


def test_model_bundle_joint_round_trip_and_hash_is_deterministic():
    bundle = ModelBundle(
        version="joint-v1",
        frequencies_mhz=(2520,),
        token_budgets=(4096, 8192),
        rolling_offsets=(0, 1, 2, 3),
        prefill_counts=(1, 2, 4),
        window_seconds=(3.0, 6.0),
        prefill_frequencies_mhz=(1800, 2520),
        decode_frequencies_mhz=(1200, 1800, 2520),
        replica_counts=(4,),
        modes=("nodg", "strict-padg"),
    )
    payload = bundle.to_dict()
    restored = ModelBundle.from_dict(
        json.loads(json.dumps(payload, sort_keys=True))
    )

    assert restored == bundle
    assert restored.config_hash == bundle.config_hash
    assert restored.version_id.startswith("joint-v1+")
    assert payload["modes"] == ["continuous", "temporal"]
    assert payload["prefill_counts"] == [1, 2, 4]

    changed = ModelBundle.from_dict({
        **payload,
        "config_hash": "",
        "window_seconds": [3.0, 9.0],
    })
    assert changed.config_hash != bundle.config_hash


def test_model_bundle_rejects_tampered_hashed_payload():
    payload = ModelBundle(
        version="hashed-v1",
        prefill_counts=(4,),
        window_seconds=(3.0,),
    ).to_dict()
    payload["token_budgets"] = [4096]
    with pytest.raises(ValueError, match="config_hash"):
        ModelBundle.from_dict(payload)

