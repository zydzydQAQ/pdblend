"""Deterministic discrete search for active replica count."""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

from ecopadg.forecast import Forecast
from ecopadg.mpc.rollout import RolloutPredictor
from ecopadg.mpc.types import (
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


class SlowMPC:
    def __init__(
        self,
        rollout: RolloutPredictor,
        shield: SafetyShield,
        *,
        replica_counts: Optional[Iterable[int]] = None,
        transition_costs: Optional[TransitionCosts] = None,
        model_version: str = "builtin-v1",
        model_config_hash: str = "",
    ):
        counts = tuple(
            sorted({int(value) for value in (replica_counts or ()) if int(value) > 0})
        )
        self.replica_counts = counts
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
        fast_action: Optional[FastAction] = None,
        replica_counts: Optional[Iterable[int]] = None,
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
        fast = fast_action or current.fast
        supplied = replica_counts if replica_counts is not None else self.replica_counts
        counts = tuple(
            sorted(
                {
                    int(value)
                    for value in supplied
                    if 0 < int(value) <= max(int(state.full_replicas), 1)
                }
            )
        )
        if not counts:
            counts = (max(int(state.full_replicas), 1),)
        raw = []
        for count in counts:
            action = ControlAction(
                fast=fast, slow=SlowAction(active_replicas=count)
            )
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
                candidate.action.fast.mode != MODE_NODG,
                -candidate.action.slow.active_replicas,
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

