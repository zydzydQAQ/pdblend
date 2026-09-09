"""Typed actions and results shared by the learned control layers."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

MODE_CONTINUOUS = "continuous"
MODE_TEMPORAL = "temporal"

# Keep the old names as source-compatible aliases.  The canonical values are
# deliberately the admission modes used by the temporal coordinator; generic
# spatial P/D placement is a different control dimension.
MODE_NODG = MODE_CONTINUOUS
MODE_STRICT_PADG = MODE_TEMPORAL
EXECUTION_MODES = (MODE_CONTINUOUS, MODE_TEMPORAL)
PREFILL_COUNTS = (1, 2, 4)
DEFAULT_TEMPORAL_WINDOW_S = 3.0


def normalize_execution_mode(value: str) -> str:
    key = str(value).strip().lower().replace("_", "-")
    if key in ("continuous", "nodg", "no-dg", "none"):
        return MODE_CONTINUOUS
    if key in ("temporal", "strict-padg", "strictpadg"):
        return MODE_TEMPORAL
    raise ValueError("unknown execution mode: %r" % value)


@dataclass(frozen=True)
class FastAction:
    """A discrete fast-timescale execution choice."""

    mode: str
    frequency_mhz: int
    token_budget: int
    rolling_offset: int = 0
    n_prefill_active: Optional[int] = None
    window_s: Optional[float] = None
    prefill_freq_mhz: Optional[int] = None
    decode_freq_mhz: Optional[int] = None

    def __post_init__(self) -> None:
        joint_fields_explicit = any(
            value is not None
            for value in (
                self.n_prefill_active,
                self.window_s,
                self.prefill_freq_mhz,
                self.decode_freq_mhz,
            )
        )
        mode = normalize_execution_mode(self.mode)
        frequency = int(self.frequency_mhz)
        budget = int(self.token_budget)
        offset = int(self.rolling_offset)
        n_prefill = (
            max(PREFILL_COUNTS)
            if self.n_prefill_active is None
            else int(self.n_prefill_active)
        )
        window = (
            DEFAULT_TEMPORAL_WINDOW_S
            if self.window_s is None
            else float(self.window_s)
        )
        prefill_frequency = (
            frequency
            if self.prefill_freq_mhz is None
            else int(self.prefill_freq_mhz)
        )
        decode_frequency = (
            frequency
            if self.decode_freq_mhz is None
            else int(self.decode_freq_mhz)
        )
        if frequency <= 0:
            raise ValueError("frequency_mhz must be positive")
        if budget <= 0:
            raise ValueError("token_budget must be positive")
        if offset < 0:
            raise ValueError("rolling_offset must be non-negative")
        if n_prefill <= 0:
            raise ValueError("n_prefill_active must be positive")
        if not math.isfinite(window) or window <= 0.0:
            raise ValueError("window_s must be finite and positive")
        if prefill_frequency <= 0:
            raise ValueError("prefill_freq_mhz must be positive")
        if decode_frequency <= 0:
            raise ValueError("decode_freq_mhz must be positive")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "frequency_mhz", frequency)
        object.__setattr__(self, "token_budget", budget)
        object.__setattr__(self, "rolling_offset", offset)
        object.__setattr__(self, "n_prefill_active", n_prefill)
        object.__setattr__(self, "window_s", window)
        object.__setattr__(
            self, "prefill_freq_mhz", prefill_frequency
        )
        object.__setattr__(
            self, "decode_freq_mhz", decode_frequency
        )
        object.__setattr__(
            self, "_joint_fields_explicit", joint_fields_explicit
        )

    @property
    def freq_mhz(self) -> int:
        """Compatibility alias used by existing DVFS code."""
        return int(self.frequency_mhz)

    @property
    def fP(self) -> int:
        return int(self.prefill_freq_mhz)

    @property
    def fD(self) -> int:
        return int(self.decode_freq_mhz)

    @property
    def joint_fields_explicit(self) -> bool:
        """Whether the caller supplied at least one joint-action field."""

        return bool(self._joint_fields_explicit)

    @property
    def execution_mode(self) -> str:
        return self.mode

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "frequency_mhz": int(self.frequency_mhz),
            "n_prefill_active": int(self.n_prefill_active),
            "window_s": float(self.window_s),
            "token_budget": int(self.token_budget),
            "prefill_freq_mhz": int(self.prefill_freq_mhz),
            "decode_freq_mhz": int(self.decode_freq_mhz),
            "rolling_offset": int(self.rolling_offset),
        }


@dataclass(frozen=True)
class SlowAction:
    """A slow-timescale active-ring choice."""

    active_replicas: int

    def __post_init__(self) -> None:
        value = int(self.active_replicas)
        if value <= 0:
            raise ValueError("active_replicas must be positive")
        object.__setattr__(self, "active_replicas", value)

    def to_dict(self) -> dict:
        return {"active_replicas": int(self.active_replicas)}


@dataclass(frozen=True)
class ControlAction:
    fast: FastAction
    slow: SlowAction

    def to_dict(self) -> dict:
        return {"fast": self.fast.to_dict(), "slow": self.slow.to_dict()}


@dataclass(frozen=True)
class RolloutResult:
    """Conservative horizon prediction for one joint action."""

    gross_j: float
    ttft_upper_s: float
    tpot_upper_s: float
    kv_blocks_required: int
    queue_end: float
    queue_growth: float
    capacity_rps: float = 0.0
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def finite(self) -> bool:
        values = (
            self.gross_j,
            self.ttft_upper_s,
            self.tpot_upper_s,
            self.queue_end,
            self.queue_growth,
            self.capacity_rps,
        )
        return all(math.isfinite(float(value)) for value in values)

    def to_dict(self) -> dict:
        return {
            "gross_j": float(self.gross_j),
            "ttft_upper_s": float(self.ttft_upper_s),
            "tpot_upper_s": float(self.tpot_upper_s),
            "kv_blocks_required": int(self.kv_blocks_required),
            "queue_end": float(self.queue_end),
            "queue_growth": float(self.queue_growth),
            "capacity_rps": float(self.capacity_rps),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class TransitionCosts:
    """Fixed, measured-at-deployment transition penalties."""

    mode_switch_j: float = 20.0
    frequency_switch_j: float = 10.0
    token_budget_switch_j: float = 2.0
    rolling_offset_switch_j: float = 1.0
    replica_switch_j: float = 2000.0
    window_switch_j: float = 2.0
    prefill_count_switch_j: float = 2.0
    prefill_frequency_switch_j: Optional[float] = None
    decode_frequency_switch_j: Optional[float] = None

    def __post_init__(self) -> None:
        prefill_frequency = (
            self.frequency_switch_j
            if self.prefill_frequency_switch_j is None
            else self.prefill_frequency_switch_j
        )
        decode_frequency = (
            self.frequency_switch_j
            if self.decode_frequency_switch_j is None
            else self.decode_frequency_switch_j
        )
        object.__setattr__(
            self, "prefill_frequency_switch_j", prefill_frequency
        )
        object.__setattr__(
            self, "decode_frequency_switch_j", decode_frequency
        )
        for name in (
            "mode_switch_j",
            "frequency_switch_j",
            "token_budget_switch_j",
            "rolling_offset_switch_j",
            "replica_switch_j",
            "window_switch_j",
            "prefill_count_switch_j",
            "prefill_frequency_switch_j",
            "decode_frequency_switch_j",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("%s must be finite and non-negative" % name)
            object.__setattr__(self, name, value)

    @property
    def n_prefill_switch_j(self) -> float:
        return float(self.prefill_count_switch_j)

    @property
    def prefill_freq_switch_j(self) -> float:
        return float(self.prefill_frequency_switch_j)

    @property
    def decode_freq_switch_j(self) -> float:
        return float(self.decode_frequency_switch_j)

    def cost(self, current: ControlAction, target: ControlAction) -> float:
        value = 0.0
        if current.fast.mode != target.fast.mode:
            value += self.mode_switch_j
        if current.fast.window_s != target.fast.window_s:
            value += self.window_switch_j
        if (
            current.fast.n_prefill_active
            != target.fast.n_prefill_active
        ):
            value += self.prefill_count_switch_j
        joint_frequency_action = bool(
            current.fast.joint_fields_explicit
            or target.fast.joint_fields_explicit
        )
        if joint_frequency_action:
            if (
                current.fast.prefill_freq_mhz
                != target.fast.prefill_freq_mhz
            ):
                value += float(self.prefill_frequency_switch_j)
            if (
                current.fast.decode_freq_mhz
                != target.fast.decode_freq_mhz
            ):
                value += float(self.decode_frequency_switch_j)
        elif current.fast.frequency_mhz != target.fast.frequency_mhz:
            value += self.frequency_switch_j
        if current.fast.token_budget != target.fast.token_budget:
            value += self.token_budget_switch_j
        if current.fast.rolling_offset != target.fast.rolling_offset:
            value += self.rolling_offset_switch_j
        value += (
            abs(current.slow.active_replicas - target.slow.active_replicas)
            * self.replica_switch_j
        )
        return float(value)


@dataclass(frozen=True)
class CandidateEvaluation:
    action: ControlAction
    rollout: RolloutResult
    transition_cost_j: float
    objective_j: float
    feasible: bool
    shield_reasons: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "action": self.action.to_dict(),
            "rollout": self.rollout.to_dict(),
            "transition_cost_j": float(self.transition_cost_j),
            "objective_j": float(self.objective_j),
            "feasible": bool(self.feasible),
            "shield_reasons": list(self.shield_reasons),
        }


@dataclass(frozen=True)
class MPCDecision:
    chosen: ControlAction
    candidates: Tuple[CandidateEvaluation, ...]
    fallback: bool
    reasons: Tuple[str, ...]
    model_version: str
    model_config_hash: str = ""

    @property
    def chosen_action(self) -> ControlAction:
        return self.chosen

