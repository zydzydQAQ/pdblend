"""Deterministic discrete search for fast execution controls."""
from __future__ import annotations

import math
from itertools import product
from typing import Iterable, Optional, Sequence, Tuple

from ecopadg.forecast import Forecast
from ecopadg.mpc.rollout import RolloutPredictor
from ecopadg.mpc.types import (
    EXECUTION_MODES,
    MODE_CONTINUOUS,
    MODE_NODG,
    CandidateEvaluation,
    ControlAction,
    FastAction,
    MPCDecision,
    SlowAction,
    TransitionCosts,
)
from ecopadg.safety_shield import SafetyShield
from ecopadg.telemetry import ControlState


def _loads(values: Forecast | Sequence[float]) -> Tuple[float, ...]:
    if isinstance(values, Forecast):
        return tuple(float(value) for value in values.upper)
    return tuple(float(value) for value in values)


def _safe_tie_key(action: ControlAction) -> tuple:
    """Conservative deterministic order for numerically equal objectives."""

    return (
        action.fast.mode != MODE_NODG,
        -int(action.fast.n_prefill_active),
        float(action.fast.window_s),
        -int(action.fast.prefill_freq_mhz),
        -int(action.fast.decode_freq_mhz),
        -int(action.fast.token_budget),
        int(action.fast.rolling_offset),
        -int(action.slow.active_replicas),
    )


class FastMPC:
    def __init__(
        self,
        rollout: RolloutPredictor,
        shield: SafetyShield,
        *,
        actions: Optional[Iterable[FastAction]] = None,
        modes: Sequence[str] = EXECUTION_MODES,
        frequencies_mhz: Sequence[int] = (2520,),
        token_budgets: Sequence[int] = (8192,),
        rolling_offsets: Sequence[int] = (0,),
        prefill_counts: Optional[Sequence[int]] = None,
        window_seconds: Optional[Sequence[float]] = None,
        prefill_frequencies_mhz: Optional[Sequence[int]] = None,
        decode_frequencies_mhz: Optional[Sequence[int]] = None,
        transition_costs: Optional[TransitionCosts] = None,
        model_version: str = "builtin-v1",
        model_config_hash: str = "",
    ):
        if actions is None:
            joint_declared = any(
                values is not None
                for values in (
                    prefill_counts,
                    window_seconds,
                    prefill_frequencies_mhz,
                    decode_frequencies_mhz,
                )
            )
            if joint_declared:
                counts = tuple(
                    int(value)
                    for value in (
                        prefill_counts
                        if prefill_counts is not None
                        else (4,)
                    )
                )
                windows = tuple(
                    float(value)
                    for value in (
                        window_seconds
                        if window_seconds is not None
                        else (3.0,)
                    )
                )
                if (
                    prefill_frequencies_mhz is None
                    and decode_frequencies_mhz is None
                ):
                    frequency_pairs = tuple(
                        (int(value), int(value))
                        for value in frequencies_mhz
                    )
                else:
                    prefill_frequencies = tuple(
                        int(value)
                        for value in (
                            prefill_frequencies_mhz
                            if prefill_frequencies_mhz is not None
                            else frequencies_mhz
                        )
                    )
                    decode_frequencies = tuple(
                        int(value)
                        for value in (
                            decode_frequencies_mhz
                            if decode_frequencies_mhz is not None
                            else frequencies_mhz
                        )
                    )
                    frequency_pairs = tuple(
                        product(prefill_frequencies, decode_frequencies)
                    )
                actions = (
                    FastAction(
                        mode=mode,
                        frequency_mhz=int(decode_frequency),
                        token_budget=int(budget),
                        rolling_offset=int(offset),
                        n_prefill_active=int(count),
                        window_s=float(window),
                        prefill_freq_mhz=int(prefill_frequency),
                        decode_freq_mhz=int(decode_frequency),
                    )
                    for (
                        mode,
                        count,
                        window,
                        budget,
                        frequency_pair,
                        offset,
                    ) in product(
                        modes,
                        counts,
                        windows,
                        token_budgets,
                        frequency_pairs,
                        rolling_offsets,
                    )
                    for prefill_frequency, decode_frequency in (
                        frequency_pair,
                    )
                )
            else:
                actions = (
                    FastAction(mode, int(freq), int(budget), int(offset))
                    for mode, freq, budget, offset in product(
                        modes,
                        frequencies_mhz,
                        token_budgets,
                        rolling_offsets,
                    )
                )
        actions = (
            action
            for action in actions
            if action.rolling_offset < int(action.n_prefill_active)
            and (
                action.mode != MODE_CONTINUOUS
                or (
                    int(action.n_prefill_active) == 4
                    and int(action.rolling_offset) == 0
                )
            )
        )
        unique = {(
            action.mode,
            action.frequency_mhz,
            action.token_budget,
            action.rolling_offset,
            action.n_prefill_active,
            action.window_s,
            action.prefill_freq_mhz,
            action.decode_freq_mhz,
        ): action for action in actions}
        self.actions = tuple(
            sorted(
                unique.values(),
                key=lambda action: (
                    action.mode,
                    action.frequency_mhz,
                    action.token_budget,
                    action.rolling_offset,
                    action.n_prefill_active,
                    action.window_s,
                    action.prefill_freq_mhz,
                    action.decode_freq_mhz,
                ),
            )
        )
        if not self.actions:
            raise ValueError("FastMPC requires at least one action")
        self.rollout = rollout
        self.shield = shield
        self.transition_costs = transition_costs or TransitionCosts()
        self.model_version = str(model_version)
        self.model_config_hash = str(model_config_hash)

    def decide(
        self,
        state: ControlState,
        forecast: Forecast | Sequence[float],
        *,
        slow_action: Optional[SlowAction] = None,
    ) -> MPCDecision:
        try:
            loads = _loads(forecast)
        except (TypeError, ValueError):
            loads = ()
        if not loads or any(
            not math.isfinite(value) or value < 0.0 for value in loads
        ):
            return MPCDecision(
                chosen=self.shield.fallback(state),
                candidates=(),
                fallback=True,
                reasons=("forecast-invalid", "no-feasible-candidate"),
                model_version=self.model_version,
                model_config_hash=self.model_config_hash,
            )
        current = state.current_action
        slow = slow_action or current.slow
        raw = []
        for fast in self.actions:
            action = ControlAction(fast=fast, slow=slow)
            prediction = self.rollout(state, action, loads)
            transition = self.transition_costs.cost(current, action)
            objective = float(prediction.gross_j) + transition
            reasons = list(self.shield.reasons(state, action, prediction))
            if not math.isfinite(objective):
                if "objective-invalid" not in reasons:
                    reasons.append("objective-invalid")
            raw.append(
                CandidateEvaluation(
                    action=action,
                    rollout=prediction,
                    transition_cost_j=transition,
                    objective_j=objective,
                    feasible=not reasons,
                    shield_reasons=tuple(reasons),
                )
            )
        candidates = tuple(raw)
        feasible = [candidate for candidate in candidates if candidate.feasible]
        if not feasible:
            reasons = ["no-feasible-candidate"]
            for candidate in candidates:
                for reason in candidate.shield_reasons:
                    if reason not in reasons:
                        reasons.append(reason)
            return MPCDecision(
                chosen=self.shield.fallback(state),
                candidates=candidates,
                fallback=True,
                reasons=tuple(reasons),
                model_version=self.model_version,
                model_config_hash=self.model_config_hash,
            )
        best = min(
            feasible,
            key=lambda candidate: (
                candidate.objective_j,
                _safe_tie_key(candidate.action),
            ),
        )
        return MPCDecision(
            chosen=best.action,
            candidates=candidates,
            fallback=False,
            reasons=("minimum-predicted-gross-j",),
            model_version=self.model_version,
            model_config_hash=self.model_config_hash,
        )

