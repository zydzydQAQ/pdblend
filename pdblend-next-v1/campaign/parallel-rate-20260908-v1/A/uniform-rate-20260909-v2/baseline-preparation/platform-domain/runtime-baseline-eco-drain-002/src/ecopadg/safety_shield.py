"""Deterministic hard constraints for learned controller proposals."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from ecopadg.mpc.types import (
    DEFAULT_TEMPORAL_WINDOW_S,
    MODE_CONTINUOUS,
    MODE_TEMPORAL,
    PREFILL_COUNTS,
    CandidateEvaluation,
    ControlAction,
    FastAction,
    RolloutResult,
    SlowAction,
)
from ecopadg.telemetry import ControlState


@dataclass(frozen=True)
class ShieldConfig:
    ttft_headroom_fraction: float = 0.05
    tpot_headroom_fraction: float = 0.05
    kv_reserve_blocks: int = 1
    max_queue_growth: float = 0.0
    min_fast_dwell_s: float = 3.0
    min_dvfs_dwell_s: float = 1.5
    min_slow_dwell_s: float = 120.0
    omega_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.ttft_headroom_fraction < 1.0:
            raise ValueError("ttft_headroom_fraction must be in [0, 1)")
        if not 0.0 <= self.tpot_headroom_fraction < 1.0:
            raise ValueError("tpot_headroom_fraction must be in [0, 1)")
        if int(self.kv_reserve_blocks) < 0:
            raise ValueError("kv_reserve_blocks must be non-negative")
        for name in (
            "max_queue_growth",
            "min_fast_dwell_s",
            "min_dvfs_dwell_s",
            "min_slow_dwell_s",
            "omega_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "%s must be finite and non-negative" % name
                )


class SafetyShield:
    """Reject unsafe candidates and provide one conservative fallback."""

    def __init__(self, config: ShieldConfig | None = None):
        self.config = config or ShieldConfig()

    def reasons(
        self,
        state: ControlState,
        action: ControlAction,
        rollout: RolloutResult,
    ) -> Tuple[str, ...]:
        """Return stable, exhaustive reason codes for a candidate."""

        cfg = self.config
        reasons: List[str] = []
        if not math.isfinite(float(state.timestamp_s)):
            reasons.append("state-time-invalid")
        if (
            not math.isfinite(float(state.queue_depth))
            or float(state.queue_depth) < 0.0
            or not math.isfinite(float(state.queue_growth))
        ):
            reasons.append("queue-telemetry-invalid")
        if (
            not math.isfinite(float(state.prompt_queue_tokens))
            or float(state.prompt_queue_tokens) < 0.0
        ):
            reasons.append("prompt-queue-telemetry-invalid")
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in state.decode_backlog_per_instance
        ):
            reasons.append("decode-backlog-telemetry-invalid")
        if any(
            value is None or int(value) < 0
            for value in state.kv_free_blocks_per_instance
        ):
            reasons.append("per-instance-kv-telemetry-missing")
        if any(
            int(value) < 0
            for value in state.kv_required_blocks_per_instance
        ):
            reasons.append("per-instance-kv-telemetry-invalid")
        if not state.has_measured_baseline:
            reasons.append("baseline-unmeasured")
        if not state.capacity_trusted:
            reasons.append("capacity-untrusted")
        if not state.frequency_trusted:
            reasons.append("frequency-untrusted")
        invalid_prediction = (
            not rollout.finite
            or rollout.gross_j < 0.0
            or int(rollout.kv_blocks_required) < 0
            or rollout.queue_end < 0.0
            or rollout.capacity_rps <= 0.0
        )
        if invalid_prediction:
            reasons.append("prediction-invalid")
        fmin = max(int(getattr(state, "fmin_mhz", 1)), 1)
        fmax = max(int(state.fmax_mhz), 0)
        prefill_frequency_invalid = not (
            fmin <= int(action.fast.prefill_freq_mhz) <= fmax
        )
        decode_frequency_invalid = not (
            fmin <= int(action.fast.decode_freq_mhz) <= fmax
        )
        if prefill_frequency_invalid:
            reasons.append("prefill-frequency-out-of-range")
        if decode_frequency_invalid:
            reasons.append("decode-frequency-out-of-range")
        if prefill_frequency_invalid or decode_frequency_invalid:
            reasons.append("frequency-out-of-range")
        if action.fast.token_budget > max(int(state.max_token_budget), 0):
            reasons.append("token-budget-out-of-range")
        if action.slow.active_replicas > max(int(state.full_replicas), 0):
            reasons.append("replica-count-out-of-range")
        if (
            int(action.fast.n_prefill_active) not in PREFILL_COUNTS
            or int(action.fast.n_prefill_active)
            > max(int(state.full_replicas), 0)
        ):
            reasons.append("prefill-count-out-of-range")
        minimum_window = max(
            float(cfg.min_fast_dwell_s),
            float(cfg.min_dvfs_dwell_s),
        )
        if (
            not math.isfinite(float(action.fast.window_s))
            or float(action.fast.window_s) < minimum_window
        ):
            reasons.append("window-below-dwell")
        if (
            action.fast.rolling_offset
            >= int(action.fast.n_prefill_active)
        ):
            reasons.append("rolling-offset-out-of-range")

        ttft_limit = float(state.ttft_slo_s) * (
            1.0 - cfg.ttft_headroom_fraction
        )
        if not math.isfinite(ttft_limit) or ttft_limit <= 0.0:
            reasons.append("ttft-slo-missing")
        else:
            if state.ttft_p90_s is None:
                reasons.append("ttft-telemetry-missing")
            elif (
                not math.isfinite(float(state.ttft_p90_s))
                or float(state.ttft_p90_s) > ttft_limit
            ):
                reasons.append("ttft-headroom-observed")
            if (
                not math.isfinite(float(rollout.ttft_upper_s))
                or float(rollout.ttft_upper_s) > ttft_limit
            ):
                reasons.append("ttft-headroom-predicted")
            if float(state.prompt_queue_tokens) > 0.0:
                if state.oldest_prompt_age_s is None:
                    reasons.append("prompt-age-telemetry-missing")
                elif (
                    not math.isfinite(float(state.oldest_prompt_age_s))
                    or float(state.oldest_prompt_age_s) < 0.0
                ):
                    reasons.append("prompt-age-telemetry-invalid")
                elif (
                    float(state.oldest_prompt_age_s)
                    + float(rollout.ttft_upper_s)
                    > ttft_limit
                ):
                    reasons.append("prompt-age-ttft-headroom")

        tpot_limit = float(state.tpot_slo_s) * (
            1.0 - cfg.tpot_headroom_fraction
        )
        if not math.isfinite(tpot_limit) or tpot_limit <= 0.0:
            reasons.append("tpot-slo-missing")
        else:
            if state.tpot_p90_s is None:
                reasons.append("tpot-telemetry-missing")
            elif (
                not math.isfinite(float(state.tpot_p90_s))
                or float(state.tpot_p90_s) > tpot_limit
            ):
                reasons.append("tpot-headroom-observed")
            if (
                not math.isfinite(float(rollout.tpot_upper_s))
                or float(rollout.tpot_upper_s) > tpot_limit
            ):
                reasons.append("tpot-headroom-predicted")

        if state.kv_free_blocks is None or int(state.kv_free_blocks) < 0:
            reasons.append("kv-telemetry-missing")
        elif (
            int(rollout.kv_blocks_required) + int(cfg.kv_reserve_blocks)
            > int(state.kv_free_blocks)
        ):
            reasons.append("kv-headroom")
        if state.kv_required_blocks_per_instance:
            if (
                len(state.kv_free_blocks_per_instance)
                != len(state.kv_required_blocks_per_instance)
            ):
                reasons.append("per-instance-kv-telemetry-incomplete")
            elif any(
                free is None
                or int(required) + int(cfg.kv_reserve_blocks)
                > int(free)
                for required, free in zip(
                    state.kv_required_blocks_per_instance,
                    state.kv_free_blocks_per_instance,
                )
            ):
                reasons.append("per-instance-kv-headroom")

        if not state.queue_stable:
            reasons.append("queue-unstable-observed")
        if (
            not math.isfinite(float(rollout.queue_growth))
            or float(rollout.queue_growth) > float(cfg.max_queue_growth)
        ):
            reasons.append("queue-unstable-predicted")

        current = state.current_action
        fast_changed = action.fast != current.fast
        slow_changed = action.slow != current.slow
        if fast_changed and (
            float(state.timestamp_s) - float(state.last_fast_change_s)
            < cfg.min_fast_dwell_s
        ):
            reasons.append("fast-dwell")
        if slow_changed and (
            float(state.timestamp_s) - float(state.last_slow_change_s)
            < cfg.min_slow_dwell_s
        ):
            reasons.append("slow-dwell")

        temporal_changed = any((
            action.fast.mode != current.fast.mode,
            (
                action.fast.n_prefill_active
                != current.fast.n_prefill_active
            ),
            action.fast.window_s != current.fast.window_s,
            action.fast.rolling_offset != current.fast.rolling_offset,
        ))
        if temporal_changed and (
            float(state.timestamp_s)
            - float(state.last_temporal_change_s)
            < cfg.min_fast_dwell_s
        ):
            reasons.append("temporal-dwell")

        if action.fast.mode == MODE_TEMPORAL:
            # While already temporal, current phase purity is a hard invariant.
            # A continuous→temporal candidate cannot require Ω=0 *before* the
            # acknowledged scheduler mode switch; it instead requires a fresh,
            # complete control-generation ACK and is checked on the next tick.
            if current.fast.mode == MODE_TEMPORAL:
                if not state.strict_telemetry_available:
                    reasons.append("temporal-telemetry-missing")
                    reasons.append("strict-telemetry-missing")
                omega = state.strict_omega
                if (
                    omega is None
                    or not math.isfinite(float(omega))
                    or abs(float(omega)) > cfg.omega_tolerance
                ):
                    reasons.append("temporal-omega-nonzero")
                    reasons.append("strict-omega-nonzero")
            generation = state.temporal_generation
            try:
                generation_value = (
                    None if generation is None else int(generation)
                )
            except (TypeError, ValueError, OverflowError):
                generation_value = None
            if generation_value is None or generation_value < 0:
                reasons.append("temporal-generation-missing")
            if not state.temporal_ack_complete:
                reasons.append("temporal-ack-incomplete")
            acknowledged = state.temporal_ack_generation
            try:
                acknowledged_value = (
                    None
                    if acknowledged is None
                    else int(acknowledged)
                )
            except (TypeError, ValueError, OverflowError):
                acknowledged_value = None
            if (
                generation_value is not None
                and acknowledged is not None
                and acknowledged_value != generation_value
            ):
                reasons.append("temporal-generation-unacked")
        return tuple(reasons)

    def allows(
        self,
        state: ControlState,
        action: ControlAction,
        rollout: RolloutResult,
    ) -> bool:
        return not self.reasons(state, action, rollout)

    def filter(
        self,
        state: ControlState,
        candidates: Iterable[CandidateEvaluation],
    ) -> Tuple[CandidateEvaluation, ...]:
        """Re-evaluate candidate safety rather than trusting caller flags."""

        output = []
        for candidate in candidates:
            reasons = list(
                self.reasons(state, candidate.action, candidate.rollout)
            )
            if not math.isfinite(float(candidate.objective_j)):
                reasons.append("objective-invalid")
            for reason in candidate.shield_reasons:
                if reason not in reasons:
                    reasons.append(reason)
            output.append(
                CandidateEvaluation(
                    action=candidate.action,
                    rollout=candidate.rollout,
                    transition_cost_j=candidate.transition_cost_j,
                    objective_j=candidate.objective_j,
                    feasible=not reasons,
                    shield_reasons=tuple(reasons),
                )
            )
        return tuple(output)

    def fallback(
        self,
        state: ControlState | ShieldConfig | None = None,
        config: ShieldConfig | None = None,
    ) -> ControlAction:
        """Continuous full-ring/fmax action; never learned.

        Calling ``SafetyShield.fallback(state)`` remains supported for the
        historical class-level API.  A bound call uses that shield's dwell
        configuration.
        """

        if isinstance(self, SafetyShield):
            observed = state
            cfg = config or self.config
        else:
            observed = self
            cfg = (
                state if isinstance(state, ShieldConfig) else config
            ) or ShieldConfig()
        if not isinstance(observed, ControlState):
            raise TypeError("fallback requires a ControlState")
        full_engines = max(
            int(observed.full_replicas),
            int(observed.active_replicas),
            1,
        )
        fmax = max(int(observed.fmax_mhz), 1)
        return ControlAction(
            fast=FastAction(
                mode=MODE_CONTINUOUS,
                frequency_mhz=fmax,
                token_budget=max(int(observed.max_token_budget), 1),
                rolling_offset=0,
                n_prefill_active=full_engines,
                window_s=max(
                    DEFAULT_TEMPORAL_WINDOW_S,
                    float(cfg.min_fast_dwell_s),
                    float(cfg.min_dvfs_dwell_s),
                ),
                prefill_freq_mhz=fmax,
                decode_freq_mhz=fmax,
            ),
            slow=SlowAction(
                active_replicas=full_engines
            ),
        )

