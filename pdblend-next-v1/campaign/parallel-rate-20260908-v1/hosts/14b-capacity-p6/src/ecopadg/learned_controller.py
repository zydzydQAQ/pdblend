"""Orchestration for passive forecasting, shielded MPC, and execution."""
from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional, Tuple, Type, TypeVar

from ecopadg.forecast import LoadForecaster
from ecopadg.logging_schema import DecisionJSONLLogger, DecisionLog
from ecopadg.mpc.fast_mpc import FastMPC
from ecopadg.mpc.rollout import (
    ConservativeRollout, RolloutConfig, RolloutPredictor,
)
from ecopadg.mpc.slow_mpc import SlowMPC
from ecopadg.mpc.types import (
    EXECUTION_MODES,
    MPCDecision,
    PREFILL_COUNTS,
    TransitionCosts,
    normalize_execution_mode,
)
from ecopadg.safety_shield import SafetyShield, ShieldConfig
from ecopadg.strict_padg_executor import (
    ExecutionResult,
    NoOpStrictPaDGExecutor,
    StrictPaDGExecutor,
)
from ecopadg.telemetry import ControlState

_T = TypeVar("_T")


def _dataclass_from_mapping(
    cls: Type[_T], values: Optional[Mapping[str, Any]]
) -> _T:
    payload = dict(values or {})
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            "%s has unknown fields: %s" % (cls.__name__, ", ".join(unknown))
        )
    return cls(**payload)


@dataclass(frozen=True)
class ModelBundle:
    """Validated, JSON-serializable controller configuration."""

    version: str = "builtin-v1"
    frequencies_mhz: Tuple[int, ...] = (2520,)
    token_budgets: Tuple[int, ...] = (8192,)
    rolling_offsets: Tuple[int, ...] = (0,)
    prefill_counts: Tuple[int, ...] = ()
    window_seconds: Tuple[float, ...] = ()
    prefill_frequencies_mhz: Tuple[int, ...] = ()
    decode_frequencies_mhz: Tuple[int, ...] = ()
    replica_counts: Tuple[int, ...] = ()
    modes: Tuple[str, ...] = EXECUTION_MODES
    horizon_steps: int = 3
    shield: ShieldConfig = ShieldConfig()
    rollout: RolloutConfig = RolloutConfig()
    transition_costs: TransitionCosts = TransitionCosts()

    def __post_init__(self) -> None:
        frequencies = tuple(int(value) for value in self.frequencies_mhz)
        budgets = tuple(int(value) for value in self.token_budgets)
        offsets = tuple(int(value) for value in self.rolling_offsets)
        prefill_counts = tuple(int(value) for value in self.prefill_counts)
        windows = tuple(float(value) for value in self.window_seconds)
        prefill_frequencies = tuple(
            int(value) for value in self.prefill_frequencies_mhz
        )
        decode_frequencies = tuple(
            int(value) for value in self.decode_frequencies_mhz
        )
        replicas = tuple(int(value) for value in self.replica_counts)
        modes = tuple(normalize_execution_mode(mode) for mode in self.modes)
        if not str(self.version):
            raise ValueError("model bundle version is required")
        if int(self.horizon_steps) <= 0:
            raise ValueError("horizon_steps must be positive")
        if not frequencies or any(value <= 0 for value in frequencies):
            raise ValueError("frequencies_mhz must be positive")
        if not budgets or any(value <= 0 for value in budgets):
            raise ValueError("token_budgets must be positive")
        if not offsets or any(value < 0 for value in offsets):
            raise ValueError("rolling_offsets must be non-empty/non-negative")
        if any(value not in PREFILL_COUNTS for value in prefill_counts):
            raise ValueError("prefill_counts must contain only 1, 2, or 4")
        if any(
            not math.isfinite(value) or value <= 0.0 for value in windows
        ):
            raise ValueError("window_seconds must be finite and positive")
        if any(value <= 0 for value in prefill_frequencies):
            raise ValueError(
                "prefill_frequencies_mhz must be positive"
            )
        if any(value <= 0 for value in decode_frequencies):
            raise ValueError(
                "decode_frequencies_mhz must be positive"
            )
        if not modes:
            raise ValueError("modes must be non-empty")
        if any(value <= 0 for value in replicas):
            raise ValueError("replica_counts must be positive")
        object.__setattr__(self, "version", str(self.version))
        object.__setattr__(self, "frequencies_mhz", frequencies)
        object.__setattr__(self, "token_budgets", budgets)
        object.__setattr__(self, "rolling_offsets", offsets)
        object.__setattr__(self, "prefill_counts", prefill_counts)
        object.__setattr__(self, "window_seconds", windows)
        object.__setattr__(
            self, "prefill_frequencies_mhz", prefill_frequencies
        )
        object.__setattr__(
            self, "decode_frequencies_mhz", decode_frequencies
        )
        object.__setattr__(self, "replica_counts", replicas)
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "horizon_steps", int(self.horizon_steps))

    @staticmethod
    def _dataclass_dict(value: Any) -> dict:
        return {
            item.name: getattr(value, item.name)
            for item in fields(value)
            if item.init
        }

    def _config_dict(self) -> dict:
        return {
            "version": self.version,
            "frequencies_mhz": list(self.frequencies_mhz),
            "token_budgets": list(self.token_budgets),
            "rolling_offsets": list(self.rolling_offsets),
            "prefill_counts": list(self.prefill_counts),
            "window_seconds": list(self.window_seconds),
            "prefill_frequencies_mhz": list(
                self.prefill_frequencies_mhz
            ),
            "decode_frequencies_mhz": list(
                self.decode_frequencies_mhz
            ),
            "replica_counts": list(self.replica_counts),
            "modes": list(self.modes),
            "horizon_steps": int(self.horizon_steps),
            "shield": self._dataclass_dict(self.shield),
            "rollout": self._dataclass_dict(self.rollout),
            "transition_costs": self._dataclass_dict(
                self.transition_costs
            ),
        }

    @property
    def config_hash(self) -> str:
        encoded = json.dumps(
            self._config_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def version_id(self) -> str:
        return "%s+%s" % (self.version, self.config_hash[:12])

    @property
    def effective_prefill_frequencies_mhz(self) -> Tuple[int, ...]:
        return self.prefill_frequencies_mhz or self.frequencies_mhz

    @property
    def effective_decode_frequencies_mhz(self) -> Tuple[int, ...]:
        return self.decode_frequencies_mhz or self.frequencies_mhz

    def to_dict(self, *, include_config_hash: bool = True) -> dict:
        payload = self._config_dict()
        if include_config_hash:
            payload["config_hash"] = self.config_hash
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelBundle":
        values = dict(payload)
        allowed = {
            item.name for item in fields(cls) if item.init
        } | {"model_version", "config_hash", "model_config_hash"}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(
                "ModelBundle has unknown fields: %s" % ", ".join(unknown)
            )
        config_hash = values.pop("config_hash", "")
        model_config_hash = values.pop("model_config_hash", "")
        expected_hash = str(config_hash or model_config_hash or "")
        values["version"] = str(
            values.get("version")
            or values.get("model_version")
            or cls.version
        )
        values.pop("model_version", None)
        for key in (
            "frequencies_mhz",
            "token_budgets",
            "rolling_offsets",
            "prefill_counts",
            "prefill_frequencies_mhz",
            "decode_frequencies_mhz",
            "replica_counts",
        ):
            if key in values:
                values[key] = tuple(int(item) for item in values[key])
        if "window_seconds" in values:
            values["window_seconds"] = tuple(
                float(item) for item in values["window_seconds"]
            )
        if "modes" in values:
            values["modes"] = tuple(str(item) for item in values["modes"])
        values["shield"] = _dataclass_from_mapping(
            ShieldConfig, values.get("shield")
        )
        values["rollout"] = _dataclass_from_mapping(
            RolloutConfig, values.get("rollout")
        )
        values["transition_costs"] = _dataclass_from_mapping(
            TransitionCosts, values.get("transition_costs")
        )
        bundle = cls(**values)
        if expected_hash and expected_hash != bundle.config_hash:
            raise ValueError("ModelBundle config_hash does not match payload")
        return bundle

    @classmethod
    def load(cls, path: str) -> "ModelBundle":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("model bundle root must be a JSON object")
        return cls.from_dict(payload)


def load_model_bundle(path: str = "") -> ModelBundle:
    return ModelBundle.load(path) if path else ModelBundle()


class LearnedController:
    """Two-timescale MPC with a single explicit executor boundary."""

    def __init__(
        self,
        bundle: Optional[ModelBundle] = None,
        *,
        forecaster: Optional[LoadForecaster] = None,
        rollout: Optional[RolloutPredictor] = None,
        executor: Optional[StrictPaDGExecutor] = None,
        logger: Optional[DecisionJSONLLogger] = None,
    ):
        self.bundle = bundle or ModelBundle()
        self.forecaster = forecaster or LoadForecaster(
            model_version="%s-load" % self.bundle.version
        )
        self.rollout = rollout or ConservativeRollout(self.bundle.rollout)
        self.shield = SafetyShield(self.bundle.shield)
        self.executor = executor or NoOpStrictPaDGExecutor()
        self.logger = logger
        self.fast_mpc = FastMPC(
            self.rollout,
            self.shield,
            modes=self.bundle.modes,
            frequencies_mhz=self.bundle.frequencies_mhz,
            token_budgets=self.bundle.token_budgets,
            rolling_offsets=self.bundle.rolling_offsets,
            prefill_counts=(
                self.bundle.prefill_counts or None
            ),
            window_seconds=(
                self.bundle.window_seconds or None
            ),
            prefill_frequencies_mhz=(
                self.bundle.prefill_frequencies_mhz or None
            ),
            decode_frequencies_mhz=(
                self.bundle.decode_frequencies_mhz or None
            ),
            transition_costs=self.bundle.transition_costs,
            model_version=self.bundle.version,
            model_config_hash=self.bundle.config_hash,
        )
        self.slow_mpc = SlowMPC(
            self.rollout,
            self.shield,
            replica_counts=self.bundle.replica_counts,
            transition_costs=self.bundle.transition_costs,
            model_version=self.bundle.version,
            model_config_hash=self.bundle.config_hash,
        )

    @property
    def model_version(self) -> str:
        return self.bundle.version

    @property
    def model_config_hash(self) -> str:
        return self.bundle.config_hash

    def decide(self, state: ControlState) -> MPCDecision:
        observed_values = (
            float(state.arrival_rate_fast_rps),
            float(state.arrival_rate_rps),
        )
        finite_observed = [
            value for value in observed_values
            if math.isfinite(value) and value >= 0.0
        ]
        observed = max(finite_observed or [0.0])
        self.forecaster.observe(observed)
        forecast = self.forecaster.forecast(self.bundle.horizon_steps)
        counts = self.bundle.replica_counts or tuple(
            range(1, max(int(state.full_replicas), 1) + 1)
        )
        slow = self.slow_mpc.decide(
            state, forecast, replica_counts=counts
        )
        if slow.fallback:
            return slow
        fast = self.fast_mpc.decide(
            state, forecast, slow_action=slow.chosen.slow
        )
        return MPCDecision(
            chosen=fast.chosen,
            candidates=slow.candidates + fast.candidates,
            fallback=fast.fallback,
            reasons=(
                tuple(fast.reasons)
                if fast.fallback
                else ("slow-and-fast-minimum-predicted-gross-j",)
            ),
            model_version=self.bundle.version,
            model_config_hash=self.bundle.config_hash,
        )

    def log_decision(
        self,
        state: ControlState,
        decision: MPCDecision,
        *,
        shadow: bool,
    ) -> None:
        if self.logger is not None:
            self.logger.write(
                DecisionLog.from_decision(
                    state, decision, shadow=shadow, controller="mpc"
                )
            )

    def apply(
        self,
        state: ControlState,
        decision: MPCDecision,
    ) -> ExecutionResult:
        return self.executor.apply(decision.chosen, state)

    def run(
        self,
        state: ControlState,
        *,
        shadow: bool = False,
        actuate: bool = True,
    ) -> MPCDecision:
        decision = self.decide(state)
        self.log_decision(state, decision, shadow=shadow)
        if actuate and not shadow:
            self.apply(state, decision)
        return decision

