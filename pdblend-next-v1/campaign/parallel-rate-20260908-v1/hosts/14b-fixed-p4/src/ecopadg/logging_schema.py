"""Versioned JSONL schema for active and shadow controller decisions."""
from __future__ import annotations

import json
import math
import numbers
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from ecopadg.mpc.types import (
    CandidateEvaluation,
    ControlAction,
    MPCDecision,
)
from ecopadg.telemetry import ControlState

SCHEMA_VERSION = 1


def _json_safe(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class CandidateLog:
    action: Mapping[str, Any]
    predicted_gross_j: Optional[float]
    transition_cost_j: Optional[float]
    objective_j: Optional[float]
    feasible: bool
    shield_reasons: Tuple[str, ...]
    rollout: Mapping[str, Any]

    @classmethod
    def from_evaluation(cls, value: CandidateEvaluation) -> "CandidateLog":
        return cls(
            action=value.action.to_dict(),
            predicted_gross_j=float(value.rollout.gross_j),
            transition_cost_j=float(value.transition_cost_j),
            objective_j=float(value.objective_j),
            feasible=bool(value.feasible),
            shield_reasons=tuple(value.shield_reasons),
            rollout=value.rollout.to_dict(),
        )

    def to_dict(self) -> dict:
        return _json_safe(
            {
                "action": dict(self.action),
                "predicted_gross_j": self.predicted_gross_j,
                "transition_cost_j": self.transition_cost_j,
                "objective_j": self.objective_j,
                "feasible": self.feasible,
                "shield_reasons": list(self.shield_reasons),
                "rollout": dict(self.rollout),
            }
        )


@dataclass(frozen=True)
class DecisionLog:
    timestamp_s: float
    controller: str
    shadow: bool
    model_version: str
    state: Mapping[str, Any]
    candidates: Tuple[CandidateLog, ...]
    shield_reasons: Tuple[str, ...]
    chosen_action: Optional[Mapping[str, Any]]
    shadow_action: Optional[Mapping[str, Any]]
    fallback: bool
    reasons: Tuple[str, ...]
    model_config_hash: str = ""
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_decision(
        cls,
        state: ControlState,
        decision: MPCDecision,
        *,
        shadow: bool,
        controller: str = "mpc",
        active_action: Optional[ControlAction] = None,
    ) -> "DecisionLog":
        shield_reasons = []
        for candidate in decision.candidates:
            for reason in candidate.shield_reasons:
                if reason not in shield_reasons:
                    shield_reasons.append(reason)
        state_payload: Dict[str, Any] = {
            "arrival_rate_rps": state.arrival_rate_rps,
            "arrival_rate_fast_rps": state.arrival_rate_fast_rps,
            "queue_depth": state.queue_depth,
            "queue_growth": state.queue_growth,
            "prompt_queue_tokens": state.prompt_queue_tokens,
            "oldest_prompt_age_s": state.oldest_prompt_age_s,
            "decode_backlog_per_instance": (
                state.decode_backlog_per_instance
            ),
            "active_replicas": state.active_replicas,
            "full_replicas": state.full_replicas,
            "baseline_attainment": state.baseline_attainment,
            "baseline_measured": state.baseline_measured,
            "capacity_trusted": state.capacity_trusted,
            "frequency_trusted": state.frequency_trusted,
            "strict_telemetry_available": state.strict_telemetry_available,
            "strict_omega": state.strict_omega,
            "kv_free_blocks": state.kv_free_blocks,
            "kv_free_blocks_per_instance": (
                state.kv_free_blocks_per_instance
            ),
            "kv_required_blocks_per_instance": (
                state.kv_required_blocks_per_instance
            ),
            "temporal_generation": state.temporal_generation,
            "temporal_ack_generation": state.temporal_ack_generation,
            "temporal_ack_complete": state.temporal_ack_complete,
            "current_action": state.current_action.to_dict(),
        }
        chosen = None if shadow else decision.chosen.to_dict()
        shadow_action = decision.chosen.to_dict() if shadow else None
        if active_action is not None:
            state_payload["active_action"] = active_action.to_dict()
        return cls(
            timestamp_s=float(state.timestamp_s),
            controller=str(controller),
            shadow=bool(shadow),
            model_version=str(decision.model_version),
            state=state_payload,
            candidates=tuple(
                CandidateLog.from_evaluation(value)
                for value in decision.candidates
            ),
            shield_reasons=tuple(shield_reasons),
            chosen_action=chosen,
            shadow_action=shadow_action,
            fallback=bool(decision.fallback),
            reasons=tuple(decision.reasons),
            model_config_hash=str(decision.model_config_hash),
        )

    def to_dict(self) -> dict:
        return _json_safe(
            {
                "schema_version": int(self.schema_version),
                "timestamp_s": float(self.timestamp_s),
                "controller": self.controller,
                "shadow": self.shadow,
                "model_version": self.model_version,
                "model_config_hash": self.model_config_hash,
                "state": dict(self.state),
                "candidates": [
                    candidate.to_dict() for candidate in self.candidates
                ],
                "shield_reasons": list(self.shield_reasons),
                "chosen_action": self.chosen_action,
                "shadow_action": self.shadow_action,
                "fallback": self.fallback,
                "reasons": list(self.reasons),
            }
        )


class DecisionJSONLLogger:
    """Thread-safe append-only writer; one complete JSON object per line."""

    def __init__(self, path: str):
        self.path = str(path)
        self._lock = threading.Lock()

    def write(self, record: DecisionLog) -> None:
        if not self.path:
            return
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        line = json.dumps(
            record.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")

    log = write


JsonlDecisionLogger = DecisionJSONLLogger

