"""Immutable controller interfaces. Seconds, joules, watts and tokens only."""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RequestBudget:
    request_id: str
    arrival_s: float
    input_tokens: int
    predicted_output: int
    ttft_s: float
    tpot_s: float
    output_limit: int | None = None
    # Prediction uses history, never the trace's eventual generated length.
    emitted: int = 0
    first_token_s: float | None = None
    last_token_s: float | None = None
    # Measured decode-side KV import already promised at admission. It stops
    # blocking other requests after this request emits its first token.
    pending_import_s: float = 0.
    # Absolute predicted end of the admitted queue/prefill/transfer work. The
    # estimate is fixed at admission and ignored after the first output token.
    pending_ready_s: float | None = None
    pending_frequency_mhz: int | None = None

    def ttft_remaining(self, now):
        return self.arrival_s + self.ttft_s - now

    def next_token_remaining(self, now):
        # TPOT is mean time per output token after the first. Enforce it for
        # every observed prefix: fast earlier tokens create one shared credit.
        if self.first_token_s is not None and self.emitted > 0:
            return self.first_token_s + self.emitted*self.tpot_s - now
        return ((self.last_token_s+self.tpot_s-now) if self.last_token_s is not None
                else self.ttft_remaining(now))


@dataclass(frozen=True)
class InstanceState:
    instance_id: str
    role: str
    tp: int
    gpus: tuple[int, ...]
    timestamp_s: float
    generation: int
    frequency_mhz: int
    free_kv_tokens: int
    running: int
    waiting: int
    requests: tuple[RequestBudget, ...] = ()
    accepting: bool = True
    reserved_kv_tokens: int = 0
    kv_allocations: tuple[tuple[str, int], ...] = ()
    parked: bool = False
    free_transfer_bytes: int = 0
    transfer_bytes_per_token: int = 0
    reserved_transfer_bytes: int = 0
    transfer_allocations: tuple[tuple[str,int], ...] = ()
    mode: str = 'continuous'
    admit_prefill: bool = True
    dvfs_allowed: bool = True


@dataclass(frozen=True)
class RuntimeSnapshot:
    version: int
    timestamp_s: float
    instances: tuple[InstanceState, ...]


@dataclass(frozen=True)
class FrequencyAction:
    instance_id: str
    frequency_mhz: int


@dataclass(frozen=True)
class RouteAction:
    request_id: str
    prefill_id: str
    decode_id: str
    reserve_tokens: int
    predicted_ttft_s: float
    predicted_tpot_s: float
    incremental_j: float
    prefill_reserve_tokens: int = 0
    transfer_reserve_bytes: int = 0
    import_block_s: float = 0.


@dataclass(frozen=True)
class RoleAction:
    instance_id: str
    expected_generation: int
    role: str
    savings_lower_j: float
    switching_upper_j: float


@dataclass(frozen=True)
class WindowAction:
    instance_id: str
    expected_generation: int
    admit_prefill: bool


@dataclass(frozen=True)
class ControlPlan:
    snapshot_version: int
    created_s: float
    expires_s: float
    routes: tuple[RouteAction, ...] = ()
    frequencies: tuple[FrequencyAction, ...] = ()
    roles: tuple[RoleAction, ...] = ()
    reason: str = ""
    feasible: bool = True
    windows: tuple[WindowAction,...] = ()


class EngineBackend(Protocol):
    async def read_state(self) -> RuntimeSnapshot: ...
    async def execute(self, plan: ControlPlan) -> None: ...
    async def confirm(self, plan: ControlPlan) -> bool: ...
    async def cancel(self, request_id: str) -> dict[str,str]: ...
