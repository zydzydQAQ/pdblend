"""Model-predictive control primitives.

Heavy modules are loaded lazily so ``ecopadg.telemetry`` can depend on the
action dataclasses without creating an import cycle.
"""

from ecopadg.mpc.types import (
    DEFAULT_TEMPORAL_WINDOW_S,
    MODE_CONTINUOUS,
    MODE_NODG,
    MODE_STRICT_PADG,
    MODE_TEMPORAL,
    PREFILL_COUNTS,
    CandidateEvaluation,
    ControlAction,
    FastAction,
    MPCDecision,
    RolloutResult,
    SlowAction,
    TransitionCosts,
    normalize_execution_mode,
)


def __getattr__(name):
    if name == "FastMPC":
        from ecopadg.mpc.fast_mpc import FastMPC
        return FastMPC
    if name == "SlowMPC":
        from ecopadg.mpc.slow_mpc import SlowMPC
        return SlowMPC
    if name in ("ConservativeRollout", "RolloutConfig"):
        from ecopadg.mpc.rollout import ConservativeRollout, RolloutConfig
        return {
            "ConservativeRollout": ConservativeRollout,
            "RolloutConfig": RolloutConfig,
        }[name]
    raise AttributeError(name)


__all__ = [
    "DEFAULT_TEMPORAL_WINDOW_S",
    "MODE_CONTINUOUS",
    "MODE_NODG",
    "MODE_STRICT_PADG",
    "MODE_TEMPORAL",
    "PREFILL_COUNTS",
    "CandidateEvaluation",
    "ConservativeRollout",
    "ControlAction",
    "FastAction",
    "FastMPC",
    "MPCDecision",
    "RolloutConfig",
    "RolloutResult",
    "SlowAction",
    "SlowMPC",
    "TransitionCosts",
    "normalize_execution_mode",
]

