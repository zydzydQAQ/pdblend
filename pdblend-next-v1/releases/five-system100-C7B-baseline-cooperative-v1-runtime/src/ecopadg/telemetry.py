"""Fail-closed aggregation of observations used by learned control."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence, Tuple

from ecopadg.mpc.types import (
    DEFAULT_TEMPORAL_WINDOW_S,
    MODE_NODG,
    ControlAction,
    FastAction,
    SlowAction,
)
from ecopadg.types import TOKEN_BUDGET


def _finite_optional(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _minimum_optional_int(values: Any) -> Optional[int]:
    if isinstance(values, Mapping):
        values = values.values()
    if not isinstance(values, Sequence) and not hasattr(values, "__iter__"):
        return None
    parsed = []
    for value in values:
        try:
            if value is not None:
                parsed.append(int(value))
        except (TypeError, ValueError):
            continue
    return min(parsed) if parsed else None


def _ordered_values(values: Any) -> Tuple[Any, ...]:
    if isinstance(values, Mapping):
        return tuple(
            values[key] for key in sorted(values, key=lambda item: str(item))
        )
    if isinstance(values, (str, bytes)) or values is None:
        return ()
    try:
        return tuple(values)
    except TypeError:
        return (values,)


def _float_tuple(values: Any) -> Tuple[float, ...]:
    parsed = []
    for value in _ordered_values(values):
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            parsed.append(float("nan"))
    return tuple(parsed)


def _optional_int_tuple(values: Any) -> Tuple[Optional[int], ...]:
    parsed = []
    for value in _ordered_values(values):
        try:
            parsed.append(None if value is None else int(value))
        except (TypeError, ValueError):
            parsed.append(None)
    return tuple(parsed)


def _int_tuple(values: Any) -> Tuple[int, ...]:
    parsed = []
    for value in _ordered_values(values):
        try:
            parsed.append(int(value))
        except (TypeError, ValueError):
            parsed.append(-1)
    return tuple(parsed)


@dataclass(frozen=True)
class ControlState:
    """One atomic controller observation.

    Defaults intentionally lack trust and measurements. A state assembled from
    partial telemetry therefore cannot authorize a learned action.
    """

    timestamp_s: float = 0.0
    arrival_rate_rps: float = 0.0
    arrival_rate_fast_rps: float = 0.0
    queue_depth: float = 0.0
    queue_growth: float = 0.0
    queue_stable: bool = False
    ttft_p90_s: Optional[float] = None
    tpot_p90_s: Optional[float] = None
    ttft_slo_s: float = 0.0
    tpot_slo_s: float = 0.0
    kv_free_blocks: Optional[int] = None
    kv_required_blocks: int = 0
    baseline_attainment: Optional[float] = None
    baseline_measured: bool = False
    capacity_trusted: bool = False
    frequency_trusted: bool = False
    strict_telemetry_available: bool = False
    strict_omega: Optional[float] = None
    active_replicas: int = 1
    full_replicas: int = 1
    fmin_mhz: int = 1
    fmax_mhz: int = 2520
    max_token_budget: int = TOKEN_BUDGET
    current_mode: str = MODE_NODG
    current_frequency_mhz: int = 2520
    current_token_budget: int = TOKEN_BUDGET
    current_rolling_offset: int = 0
    last_fast_change_s: float = float("-inf")
    last_slow_change_s: float = float("-inf")
    capacity_per_replica_rps: float = 0.0
    mean_prompt_tokens: float = 0.0
    mean_output_tokens: float = 0.0
    prompt_queue_tokens: float = 0.0
    oldest_prompt_age_s: Optional[float] = None
    decode_backlog_per_instance: Tuple[float, ...] = ()
    kv_free_blocks_per_instance: Tuple[Optional[int], ...] = ()
    kv_required_blocks_per_instance: Tuple[int, ...] = ()
    current_n_prefill_active: int = 4
    current_window_s: float = DEFAULT_TEMPORAL_WINDOW_S
    current_prefill_freq_mhz: Optional[int] = None
    current_decode_freq_mhz: Optional[int] = None
    last_temporal_change_s: float = float("-inf")
    temporal_generation: Optional[int] = None
    temporal_ack_generation: Optional[int] = None
    temporal_applied_generations: Tuple[Optional[int], ...] = ()
    temporal_ack_complete: bool = False
    observed_power_w: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def current_action(self) -> ControlAction:
        fast_values = {
            "mode": self.current_mode,
            "frequency_mhz": max(int(self.current_frequency_mhz), 1),
            "token_budget": max(int(self.current_token_budget), 1),
            "rolling_offset": max(int(self.current_rolling_offset), 0),
        }
        joint_observed = bool(
            self.current_prefill_freq_mhz is not None
            or self.current_decode_freq_mhz is not None
            or int(self.current_n_prefill_active) != 4
            or float(self.current_window_s)
            != DEFAULT_TEMPORAL_WINDOW_S
            or self.temporal_generation is not None
        )
        if joint_observed:
            window = float(self.current_window_s)
            fast_values.update({
                "n_prefill_active": max(
                    int(self.current_n_prefill_active), 1
                ),
                "window_s": (
                    window
                    if math.isfinite(window) and window > 0.0
                    else DEFAULT_TEMPORAL_WINDOW_S
                ),
                "prefill_freq_mhz": max(
                    int(
                        self.current_prefill_freq_mhz
                        if self.current_prefill_freq_mhz is not None
                        else self.current_frequency_mhz
                    ),
                    1,
                ),
                "decode_freq_mhz": max(
                    int(
                        self.current_decode_freq_mhz
                        if self.current_decode_freq_mhz is not None
                        else self.current_frequency_mhz
                    ),
                    1,
                ),
            })
        return ControlAction(
            fast=FastAction(**fast_values),
            slow=SlowAction(active_replicas=max(int(self.active_replicas), 1)),
        )

    @property
    def has_measured_baseline(self) -> bool:
        value = _finite_optional(self.baseline_attainment)
        return bool(
            self.baseline_measured
            and value is not None
            and 0.0 <= value <= 1.0
        )

    def with_action(self, action: ControlAction) -> "ControlState":
        return replace(
            self,
            active_replicas=action.slow.active_replicas,
            current_mode=action.fast.mode,
            current_frequency_mhz=action.fast.frequency_mhz,
            current_token_budget=action.fast.token_budget,
            current_rolling_offset=action.fast.rolling_offset,
            current_n_prefill_active=action.fast.n_prefill_active,
            current_window_s=action.fast.window_s,
            current_prefill_freq_mhz=action.fast.prefill_freq_mhz,
            current_decode_freq_mhz=action.fast.decode_freq_mhz,
        )

    @classmethod
    def aggregate(
        cls,
        *,
        timestamp_s: float,
        engine: Optional[Mapping[str, Any]] = None,
        baseline_attainment: Optional[float] = None,
        baseline_measured: bool = False,
        capacity_trusted: bool = False,
        frequency_trusted: bool = False,
        queue_depth: float = 0.0,
        queue_growth: float = 0.0,
        queue_observed: bool = True,
        queue_stable: Optional[bool] = None,
        ttft_p90_s: Optional[float] = None,
        tpot_p90_s: Optional[float] = None,
        kv_free_blocks: Optional[int] = None,
        strict_telemetry_available: Optional[bool] = None,
        strict_omega: Optional[float] = None,
        prompt_queue_tokens: float = 0.0,
        oldest_prompt_age_s: Optional[float] = None,
        decode_backlog_per_instance: Optional[Sequence[float]] = None,
        kv_free_blocks_per_instance: Optional[
            Sequence[Optional[int]]
        ] = None,
        kv_required_blocks_per_instance: Optional[Sequence[int]] = None,
        temporal_generation: Optional[int] = None,
        temporal_ack_generation: Optional[int] = None,
        temporal_applied_generations: Optional[
            Sequence[Optional[int]]
        ] = None,
        temporal_ack_complete: Optional[bool] = None,
        **values: Any,
    ) -> "ControlState":
        """Build a state from the existing engine aggregate and pool metrics."""

        aggregate = dict(engine or {})
        expected = int(aggregate.get("expected") or 0)
        available = int(aggregate.get("available") or 0)
        omega = _finite_optional(
            aggregate.get("omega")
            if strict_omega is None else strict_omega
        )
        aggregated_strict_available = bool(
            expected > 0
            and available >= expected
            and bool(aggregate.get("strict"))
            and omega is not None
        )
        strict_available = (
            aggregated_strict_available
            if strict_telemetry_available is None
            else bool(strict_telemetry_available)
        )
        if kv_free_blocks is None:
            free_blocks = _minimum_optional_int(
                aggregate.get("free_gpu_blocks") or ()
            )
        else:
            try:
                free_blocks = int(kv_free_blocks)
            except (TypeError, ValueError):
                free_blocks = None
        free_per_instance = _optional_int_tuple(
            (
                aggregate.get("free_gpu_blocks") or ()
                if kv_free_blocks_per_instance is None
                else kv_free_blocks_per_instance
            )
        )
        decode_per_instance = _float_tuple(
            (
                aggregate.get("decode_backlog")
                or aggregate.get("decode_backlogs")
                or aggregate.get("n_decode_groups")
                or ()
                if decode_backlog_per_instance is None
                else decode_backlog_per_instance
            )
        )
        required_per_instance = _int_tuple(
            (
                aggregate.get("required_gpu_blocks")
                or aggregate.get("kv_required_blocks")
                or ()
                if kv_required_blocks_per_instance is None
                else kv_required_blocks_per_instance
            )
        )
        applied_generations = _optional_int_tuple(
            (
                aggregate.get("applied_generations")
                or aggregate.get("applied_generation")
                or ()
                if temporal_applied_generations is None
                else temporal_applied_generations
            )
        )
        requested_generation = temporal_generation
        if requested_generation is None:
            raw_generation = (
                aggregate.get("temporal_generation")
                if aggregate.get("temporal_generation") is not None
                else (
                    aggregate.get("requested_generations")
                    if aggregate.get("requested_generations") is not None
                    else aggregate.get("requested_generation")
                )
            )
            if isinstance(raw_generation, Mapping) or (
                isinstance(raw_generation, Sequence)
                and not isinstance(raw_generation, (str, bytes))
            ):
                requested = {
                    value
                    for value in _optional_int_tuple(raw_generation)
                    if value is not None
                }
                requested_generation = (
                    next(iter(requested))
                    if len(requested) == 1
                    else None
                )
            else:
                try:
                    requested_generation = (
                        None
                        if raw_generation is None
                        else int(raw_generation)
                    )
                except (TypeError, ValueError):
                    requested_generation = None
        ack_generation = temporal_ack_generation
        if ack_generation is None:
            acknowledged = {
                value
                for value in applied_generations
                if value is not None
            }
            if len(acknowledged) == 1:
                ack_generation = next(iter(acknowledged))
        if temporal_ack_complete is None:
            ack_complete = bool(
                expected > 0
                and available >= expected
                and requested_generation is not None
                and len(applied_generations) >= expected
                and all(
                    value == requested_generation
                    for value in applied_generations[:expected]
                )
            )
        else:
            ack_complete = bool(temporal_ack_complete)
        stable = (
            bool(queue_observed and float(queue_growth) <= 0.0)
            if queue_stable is None
            else bool(queue_stable)
        )
        return cls(
            timestamp_s=float(timestamp_s),
            baseline_attainment=_finite_optional(baseline_attainment),
            baseline_measured=bool(baseline_measured),
            capacity_trusted=bool(capacity_trusted),
            frequency_trusted=bool(frequency_trusted),
            queue_depth=max(float(queue_depth), 0.0),
            queue_growth=float(queue_growth),
            queue_stable=stable,
            ttft_p90_s=_finite_optional(ttft_p90_s),
            tpot_p90_s=_finite_optional(tpot_p90_s),
            kv_free_blocks=free_blocks,
            prompt_queue_tokens=float(prompt_queue_tokens),
            oldest_prompt_age_s=_finite_optional(oldest_prompt_age_s),
            decode_backlog_per_instance=decode_per_instance,
            kv_free_blocks_per_instance=free_per_instance,
            kv_required_blocks_per_instance=required_per_instance,
            strict_telemetry_available=strict_available,
            strict_omega=omega,
            temporal_generation=requested_generation,
            temporal_ack_generation=ack_generation,
            temporal_applied_generations=applied_generations,
            temporal_ack_complete=ack_complete,
            **values,
        )


def aggregate_control_state(**values: Any) -> ControlState:
    """Functional alias convenient for sampling loops."""

    return ControlState.aggregate(**values)

