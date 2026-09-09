"""Conservative pure-Python/numpy rollout used by both MPC timescales."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

from ecopadg.forecast import ResidualRegressor
from ecopadg.mpc.types import (
    MODE_TEMPORAL,
    ControlAction,
    RolloutResult,
)
from ecopadg.telemetry import ControlState


class RolloutPredictor(Protocol):
    def __call__(
        self,
        state: ControlState,
        action: ControlAction,
        load_upper_rps: Sequence[float],
    ) -> RolloutResult:
        ...


@dataclass(frozen=True)
class RolloutConfig:
    step_s: float = 1.0
    replica_residency_w: float = 76.0
    request_energy_j: float = 100.0
    strict_capacity_factor: float = 0.95
    strict_energy_factor: float = 0.9
    kv_blocks_per_request: float = 1.0
    minimum_capacity_rps: float = 1e-6

    def __post_init__(self) -> None:
        if float(self.step_s) <= 0.0:
            raise ValueError("step_s must be positive")
        if float(self.minimum_capacity_rps) <= 0.0:
            raise ValueError("minimum_capacity_rps must be positive")
        for name in (
            "replica_residency_w",
            "request_energy_j",
            "strict_capacity_factor",
            "strict_energy_factor",
            "kv_blocks_per_request",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("%s must be finite and non-negative" % name)


def action_features(
    state: ControlState,
    action: ControlAction,
    max_load_rps: float,
) -> np.ndarray:
    """Stable feature vector for passively fitted residual models."""

    if action.fast.joint_fields_explicit:
        prompt = float(state.mean_prompt_tokens)
        output = float(state.mean_output_tokens)
        prompt = prompt if math.isfinite(prompt) and prompt >= 0.0 else 0.0
        output = output if math.isfinite(output) and output >= 0.0 else 0.0
        total = prompt + output
        prompt_share = prompt / total if total > 0.0 else 0.5
        output_share = output / total if total > 0.0 else 0.5
        fmax = max(float(state.fmax_mhz), 1.0)
        prefill_ratio = max(
            float(action.fast.prefill_freq_mhz) / fmax, 1e-6
        )
        decode_ratio = max(
            float(action.fast.decode_freq_mhz) / fmax, 1e-6
        )
        frequency_ratio = 1.0 / (
            prompt_share / prefill_ratio
            + output_share / decode_ratio
        )
    else:
        frequency_ratio = (
            float(action.fast.frequency_mhz)
            / max(float(state.fmax_mhz), 1.0)
        )
    return np.asarray(
        [[
            float(max_load_rps),
            float(state.queue_depth),
            float(action.slow.active_replicas),
            float(frequency_ratio),
            float(action.fast.token_budget)
            / max(float(state.max_token_budget), 1.0),
            float(action.fast.mode == MODE_TEMPORAL),
        ]],
        dtype=float,
    )


class ConservativeRollout:
    """Small analytic queue/energy model with optional learned residuals."""

    def __init__(
        self,
        config: Optional[RolloutConfig] = None,
        energy_residual: Optional[ResidualRegressor] = None,
        ttft_residual: Optional[ResidualRegressor] = None,
        tpot_residual: Optional[ResidualRegressor] = None,
    ):
        self.config = config or RolloutConfig()
        self.energy_residual = energy_residual
        self.ttft_residual = ttft_residual
        self.tpot_residual = tpot_residual

    @staticmethod
    def _upper(
        model: Optional[ResidualRegressor],
        features: np.ndarray,
        baseline: float,
    ) -> float:
        if model is None or not model.fitted:
            return float(baseline)
        value = model.predict_upper(features, [baseline])
        return max(float(value[0]), 0.0)

    @staticmethod
    def _frequency_ratio(value: int, fmax_mhz: int) -> float:
        return min(
            max(float(value) / max(float(fmax_mhz), 1.0), 0.0),
            1.0,
        )

    @staticmethod
    def _finite_nonnegative(value: object, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(parsed) or parsed < 0.0:
            return float(default)
        return parsed

    @classmethod
    def _sum_nonnegative(cls, values: Sequence[object]) -> float:
        return float(sum(cls._finite_nonnegative(value) for value in values))

    @staticmethod
    def _invalid_result() -> RolloutResult:
        return RolloutResult(
            gross_j=float("inf"),
            ttft_upper_s=float("inf"),
            tpot_upper_s=float("inf"),
            kv_blocks_required=2**31 - 1,
            queue_end=float("inf"),
            queue_growth=float("inf"),
            capacity_rps=0.0,
            details={"reason": "invalid-load-forecast"},
        )

    def __call__(
        self,
        state: ControlState,
        action: ControlAction,
        load_upper_rps: Sequence[float],
    ) -> RolloutResult:
        loads = tuple(max(float(value), 0.0) for value in load_upper_rps)
        if not loads or not all(math.isfinite(value) for value in loads):
            return self._invalid_result()
        has_joint_observations = bool(
            self._finite_nonnegative(state.prompt_queue_tokens) > 0.0
            or state.oldest_prompt_age_s is not None
            or state.decode_backlog_per_instance
            or state.kv_free_blocks_per_instance
            or state.kv_required_blocks_per_instance
        )
        if (
            not action.fast.joint_fields_explicit
            and not has_joint_observations
        ):
            return self._legacy_rollout(state, action, loads)
        return self._joint_rollout(state, action, loads)

    def _legacy_rollout(
        self,
        state: ControlState,
        action: ControlAction,
        loads: Sequence[float],
    ) -> RolloutResult:
        """The original single-frequency model for legacy action bundles."""

        cfg = self.config
        frequency_ratio = min(
            max(
                float(action.fast.frequency_mhz)
                / max(float(state.fmax_mhz), 1.0),
                0.0,
            ),
            1.0,
        )
        mode_capacity = (
            cfg.strict_capacity_factor
            if action.fast.mode == MODE_TEMPORAL
            else 1.0
        )
        capacity = (
            max(float(state.capacity_per_replica_rps), 0.0)
            * action.slow.active_replicas
            * frequency_ratio
            * mode_capacity
        )
        capacity = max(capacity, cfg.minimum_capacity_rps)
        queue = max(float(state.queue_depth), 0.0)
        queue_start = queue
        gross = 0.0
        peak_queue = queue
        mode_energy = (
            cfg.strict_energy_factor
            if action.fast.mode == MODE_TEMPORAL
            else 1.0
        )
        for load in loads:
            queue = max(0.0, queue + (load - capacity) * cfg.step_s)
            peak_queue = max(peak_queue, queue)
            residency = (
                action.slow.active_replicas
                * cfg.replica_residency_w
                * cfg.step_s
            )
            dynamic = (
                min(load, capacity)
                * cfg.request_energy_j
                * max(frequency_ratio, 0.1)
                * mode_energy
                * cfg.step_s
            )
            gross += residency + dynamic

        current_ttft = (
            float(state.ttft_p90_s)
            if state.ttft_p90_s is not None
            else 0.5 * max(float(state.ttft_slo_s), 0.0)
        )
        current_tpot = (
            float(state.tpot_p90_s)
            if state.tpot_p90_s is not None
            else 0.5 * max(float(state.tpot_slo_s), 0.0)
        )
        budget_ratio = (
            float(action.fast.token_budget)
            / max(float(state.max_token_budget), 1.0)
        )
        ttft = max(current_ttft, 0.0) + peak_queue / capacity
        ttft *= max(budget_ratio, 0.1)
        tpot = max(current_tpot, 0.0) / max(frequency_ratio, 1e-3)
        max_load = max(loads)
        features = action_features(state, action, max_load)
        gross = self._upper(self.energy_residual, features, gross)
        ttft = self._upper(self.ttft_residual, features, ttft)
        tpot = self._upper(self.tpot_residual, features, tpot)
        required = int(
            math.ceil(
                (
                    peak_queue
                    + max_load * cfg.step_s
                    + max(float(state.kv_required_blocks), 0.0)
                )
                * cfg.kv_blocks_per_request
            )
        )
        return RolloutResult(
            gross_j=float(gross),
            ttft_upper_s=float(ttft),
            tpot_upper_s=float(tpot),
            kv_blocks_required=max(required, 0),
            queue_end=float(queue),
            queue_growth=float(queue - queue_start),
            capacity_rps=float(capacity),
            details={
                "peak_queue": float(peak_queue),
                "max_load_rps": float(max_load),
            },
        )

    def _joint_rollout(
        self,
        state: ControlState,
        action: ControlAction,
        loads: Sequence[float],
    ) -> RolloutResult:
        """Conservative phase-aware model for a declared joint action."""

        cfg = self.config
        temporal = action.fast.mode == MODE_TEMPORAL
        physical_engines = max(int(action.slow.active_replicas), 1)
        full_engines = max(int(state.full_replicas), physical_engines, 1)
        n_prefill = min(
            max(int(action.fast.n_prefill_active), 1),
            full_engines,
        )
        prefill_ratio = self._frequency_ratio(
            int(action.fast.prefill_freq_mhz), int(state.fmax_mhz)
        )
        decode_ratio = self._frequency_ratio(
            int(action.fast.decode_freq_mhz), int(state.fmax_mhz)
        )

        prompt_tokens = self._finite_nonnegative(
            state.mean_prompt_tokens
        )
        output_tokens = self._finite_nonnegative(
            state.mean_output_tokens
        )
        token_total = prompt_tokens + output_tokens
        if token_total > 0.0:
            prefill_share = prompt_tokens / token_total
            decode_share = output_tokens / token_total
        else:
            prefill_share = 0.5
            decode_share = 0.5

        prefill_engine_fraction = (
            min(float(n_prefill) / float(full_engines), 1.0)
            if temporal
            else 1.0
        )
        prefill_service_ratio = max(
            prefill_ratio * prefill_engine_fraction,
            cfg.minimum_capacity_rps,
        )
        decode_service_ratio = max(
            decode_ratio, cfg.minimum_capacity_rps
        )
        effective_frequency_ratio = 1.0 / max(
            prefill_share / prefill_service_ratio
            + decode_share / decode_service_ratio,
            cfg.minimum_capacity_rps,
        )
        temporal_capacity_factor = (
            cfg.strict_capacity_factor if temporal else 1.0
        )
        capacity = (
            self._finite_nonnegative(state.capacity_per_replica_rps)
            * physical_engines
            * effective_frequency_ratio
            * temporal_capacity_factor
        )
        capacity = max(capacity, cfg.minimum_capacity_rps)

        prompt_queue_tokens = self._finite_nonnegative(
            state.prompt_queue_tokens
        )
        prompt_request_denominator = (
            prompt_tokens
            if prompt_tokens > 0.0
            else max(float(action.fast.token_budget), 1.0)
        )
        prompt_queue_requests = (
            prompt_queue_tokens / prompt_request_denominator
        )
        decode_backlog = self._sum_nonnegative(
            state.decode_backlog_per_instance
        )
        max_decode_backlog = max(
            (
                self._finite_nonnegative(value)
                for value in state.decode_backlog_per_instance
            ),
            default=0.0,
        )
        per_instance_kv_required = self._sum_nonnegative(
            state.kv_required_blocks_per_instance
        )
        aggregate_kv_required = self._finite_nonnegative(
            state.kv_required_blocks
        )
        kv_required = max(
            aggregate_kv_required, per_instance_kv_required
        )
        kv_request_equivalent = (
            kv_required / max(cfg.kv_blocks_per_request, 1e-6)
        )
        active_decode_requests = max(
            decode_backlog, kv_request_equivalent
        )
        queue = max(
            self._finite_nonnegative(state.queue_depth),
            prompt_queue_requests,
        ) + active_decode_requests
        queue_start = queue
        peak_queue = queue
        gross = 0.0
        prefill_energy = 0.0
        decode_energy = 0.0
        mode_energy = cfg.strict_energy_factor if temporal else 1.0
        for load in loads:
            available = queue + float(load) * cfg.step_s
            completed = min(
                available, capacity * cfg.step_s
            )
            queue = max(available - completed, 0.0)
            peak_queue = max(peak_queue, queue)
            residency = (
                physical_engines
                * cfg.replica_residency_w
                * cfg.step_s
            )
            step_prefill_energy = (
                completed
                * cfg.request_energy_j
                * prefill_share
                * max(prefill_ratio, 0.1)
                * mode_energy
            )
            step_decode_energy = (
                completed
                * cfg.request_energy_j
                * decode_share
                * max(decode_ratio, 0.1)
                * mode_energy
            )
            prefill_energy += step_prefill_energy
            decode_energy += step_decode_energy
            gross += (
                residency
                + step_prefill_energy
                + step_decode_energy
            )

        current_ttft = (
            float(state.ttft_p90_s)
            if state.ttft_p90_s is not None
            else 0.5 * max(float(state.ttft_slo_s), 0.0)
        )
        current_tpot = (
            float(state.tpot_p90_s)
            if state.tpot_p90_s is not None
            else 0.5 * max(float(state.tpot_slo_s), 0.0)
        )
        budget_ratio = (
            float(action.fast.token_budget)
            / max(float(state.max_token_budget), 1.0)
        )
        window_wait = (
            max(float(action.fast.window_s), 0.0)
            if temporal
            else 0.0
        )
        ttft = (
            max(current_ttft, 0.0)
            + peak_queue / capacity
            + window_wait
        )
        ttft *= max(budget_ratio, 0.1)

        per_engine_decode_capacity = max(
            self._finite_nonnegative(state.capacity_per_replica_rps)
            * max(decode_ratio, cfg.minimum_capacity_rps),
            cfg.minimum_capacity_rps,
        )
        decode_backlog_delay = (
            max_decode_backlog / per_engine_decode_capacity
        )
        tpot = (
            max(current_tpot, 0.0)
            / max(decode_ratio, 1e-3)
            + decode_backlog_delay
        )
        usable_kv = self._sum_nonnegative(
            tuple(
                value
                for value in state.kv_free_blocks_per_instance
                if value is not None
            )
        )
        if usable_kv > 0.0 and kv_required > 0.0:
            tpot *= 1.0 + min(kv_required / usable_kv, 1.0)

        max_load = max(loads)
        features = action_features(state, action, max_load)
        gross = self._upper(self.energy_residual, features, gross)
        ttft = self._upper(self.ttft_residual, features, ttft)
        tpot = self._upper(self.tpot_residual, features, tpot)
        required = int(math.ceil(
            (
                peak_queue + max_load * cfg.step_s
            ) * cfg.kv_blocks_per_request
            + kv_required
        ))
        prefill_demand_tokens = (
            prompt_queue_tokens
            + sum(loads) * cfg.step_s * prompt_tokens
        )
        decode_demand_tokens = (
            (
                sum(loads) * cfg.step_s
                + active_decode_requests
            )
            * output_tokens
        )
        return RolloutResult(
            gross_j=float(gross),
            ttft_upper_s=float(ttft),
            tpot_upper_s=float(tpot),
            kv_blocks_required=max(required, 0),
            queue_end=float(queue),
            queue_growth=float(queue - queue_start),
            capacity_rps=float(capacity),
            details={
                "peak_queue": float(peak_queue),
                "max_load_rps": float(max_load),
                "prefill_demand_tokens": float(
                    prefill_demand_tokens
                ),
                "decode_demand_tokens": float(
                    decode_demand_tokens
                ),
                "prefill_energy_j": float(prefill_energy),
                "decode_energy_j": float(decode_energy),
                "prefill_frequency_ratio": float(prefill_ratio),
                "decode_frequency_ratio": float(decode_ratio),
                "prefill_engine_fraction": float(
                    prefill_engine_fraction
                ),
                "temporal_capacity_factor": float(
                    temporal_capacity_factor
                ),
                "window_wait_s": float(window_wait),
                "active_decode_requests": float(
                    active_decode_requests
                ),
            },
        )


class CallableRollout:
    """Typed adapter for compact test and deployment-specific rollouts."""

    def __init__(
        self,
        function: Callable[
            [ControlState, ControlAction, Sequence[float]], RolloutResult
        ],
    ):
        self.function = function

    def __call__(
        self,
        state: ControlState,
        action: ControlAction,
        load_upper_rps: Sequence[float],
    ) -> RolloutResult:
        return self.function(state, action, load_upper_rps)

