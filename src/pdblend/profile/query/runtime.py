"""Planner-facing profile contract; component qualification stays explicit."""
from typing import Protocol


class RuntimeQualificationError(ValueError):
    """The requested runtime component has not been independently qualified."""


class RuntimeProfile(Protocol):
    freqs: tuple[int, ...]
    model: str
    system: str
    tp: int
    pp: int
    profile_key: dict
    bounded_coverage: dict
    decode_power_overrides: dict
    kv_capacity_tokens: int
    kv_bytes_per_token: int
    freq_switch_s: float

    def prefill_seconds(self, n, f) -> float: ...
    def prefill_power_w(self, n, f) -> float: ...
    def prefill_marginal_seconds(self, n, f) -> float: ...
    def step_seconds(self, batch, ctx, f) -> float: ...
    def decode_power_w(self, batch, f, *, ctx=None) -> float: ...
    def decode_supported(self, batch, ctx, f) -> bool: ...
    def decode_power_supported(self, batch, ctx, f) -> bool: ...
    def static_power_w(self, state, f=None) -> float: ...
    def wake_seconds(self, state) -> float: ...
    def transfer_seconds(self, tokens) -> float: ...


def require_planner_components(model, *, allow_pd, allow_dvfs):
    """Historical PerfModel semantics remain; calibrated adapters fail closed."""
    gate = getattr(model, 'require_runtime_components', None)
    if gate is not None:
        components = ['capacity', 'static']
        if allow_pd: components.append('transfer')
        if allow_dvfs: components.append('clock_transition')
        gate(*components)
