"""Receipt-checked slow TP changes, separate from online P/D KV transfer.

The engine adapter owns native operations and physical GPU leases. This module
never manufactures rank ACKs or treats process launch as a completed transfer.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from threading import Lock
from typing import Callable, ContextManager, Protocol


class TransitionState(str, Enum):
    PREPARE = "prepare"
    DRAIN = "drain"
    TRANSFER = "transfer"
    VERIFY = "verify"
    ACTIVATE = "activate"
    RETIRE = "retire"
    ROLLBACK = "rollback"
    QUARANTINE = "quarantine"
    COMPLETE = "complete"
    ABORTED = "aborted"


class QuarantineRequired(RuntimeError):
    """Lease contexts must retain ownership on this exception until recovery."""


@dataclass(frozen=True)
class TopologyTarget:
    model_id: str
    tp: int
    pp: int = 1
    gpu_ids: tuple[int, ...] = ()
    pool_id: str = ""


@dataclass
class TransitionRecord:
    transaction_id: str
    generation: int
    source: TopologyTarget
    target: TopologyTarget
    state: str = TransitionState.PREPARE.value
    started_s: float = field(default_factory=time.time)
    finished_s: float | None = None
    events: list[dict] = field(default_factory=list)
    error: str | None = None
    formal_eligible: bool = False

    def event(self, state: TransitionState, **details) -> None:
        self.state = state.value
        self.events.append(dict(state=state.value, t_s=time.time(), **details))

    def json(self) -> dict:
        return asdict(self)


class ReshardBackend(Protocol):
    def gpu_lock(self, gpu_ids: tuple[int, ...], transaction_id: str) -> ContextManager[dict]: ...
    def execute(self, operation: str, *, source: TopologyTarget, target: TopologyTarget,
                generation: int, source_generation: int, transaction_id: str,
                timeout_s: float) -> dict: ...


class NativeReshardBackend:
    """Bridge explicit native RPC and shared-lease implementations.

    ``dispatch`` returns native, transaction-bound results unchanged; ``claim``
    yields a lease receipt and retains ownership if its context exits with
    QuarantineRequired. No default shell actions or inferred all-rank success.
    """
    def __init__(self, dispatch: Callable[..., dict],
                 claim: Callable[[tuple[int, ...], str], ContextManager[dict]]):
        self.dispatch, self.claim = dispatch, claim

    def gpu_lock(self, gpu_ids: tuple[int, ...], transaction_id: str) -> ContextManager[dict]:
        return self.claim(gpu_ids, transaction_id)

    def execute(self, operation: str, **kwargs) -> dict:
        receipt = self.dispatch(operation, **kwargs)
        if not isinstance(receipt, dict):
            raise RuntimeError(f"native {operation} did not return a receipt")
        return receipt


def _check_ack(receipt: dict, operation: str, target: TopologyTarget | None, generation: int,
               transaction_id: str, *, ranks_key: str = "rank_acks") -> None:
    expected = {"operation": operation, "generation": generation, "transaction_id": transaction_id}
    if not isinstance(receipt, dict) or any(receipt.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f"{operation} transaction/generation ACK missing or stale")
    if target is None:
        return
    ranks = receipt.get(ranks_key)
    if not isinstance(ranks, list) or len(ranks) != len(target.gpu_ids):
        raise RuntimeError(f"{operation} requires one native ACK per physical rank")
    by_rank = {r.get("rank"): r for r in ranks if isinstance(r, dict)}
    if set(by_rank) != set(range(len(target.gpu_ids))):
        raise RuntimeError(f"{operation} contains duplicate or missing rank ACKs")
    for rank, gpu_id in enumerate(target.gpu_ids):
        ack = by_rank[rank]
        if (ack.get("ok") is not True or ack.get("gpu_id") != gpu_id
                or any(ack.get(k) != v for k, v in expected.items())):
            raise RuntimeError(f"{operation} rank {rank} identity/ACK mismatch")


class ReshardCoordinator:
    def __init__(self, backend: ReshardBackend, *, drain_timeout_s: float = 30.0):
        if drain_timeout_s <= 0:
            raise ValueError("drain timeout must be positive")
        self.backend = backend
        self.drain_timeout_s = drain_timeout_s
        self.active_generation = 0
        self.active: TopologyTarget | None = None
        self.records: list[TransitionRecord] = []
        self.quarantined_gpus: set[int] = set()
        self._transition_lock = Lock()

    @staticmethod
    def _validate_target(target: TopologyTarget) -> None:
        if target.pp != 1 or target.tp not in (1, 2, 4):
            raise ValueError("PDBlend reshard requires TP1/2/4 and PP1")
        if (not target.model_id or len(target.gpu_ids) != target.tp
                or len(set(target.gpu_ids)) != len(target.gpu_ids)
                or any(not isinstance(g, int) or g < 0 for g in target.gpu_ids)):
            raise ValueError("topology must declare exactly TP unique physical GPUs")

    def transition(self, source: TopologyTarget, target: TopologyTarget, *,
                   generation: int | None = None) -> TransitionRecord:
        self._validate_target(source)
        self._validate_target(target)
        if source == target:
            raise ValueError("source and target topology are identical")
        if source.model_id != target.model_id:
            raise ValueError("reshard cannot change model identity")
        if self.quarantined_gpus:
            raise RuntimeError("native recovery is required before leaving quarantine")
        if self.active is not None and self.active != source:
            raise RuntimeError("source topology is not the active generation")
        if generation is not None and generation <= self.active_generation:
            raise ValueError("target generation must be newer than the active generation")
        if not self._transition_lock.acquire(blocking=False):
            raise RuntimeError("another topology transaction is in progress")
        try:
            return self._transition(source, target, generation or self.active_generation + 1)
        finally:
            self._transition_lock.release()

    def _transition(self, source: TopologyTarget, target: TopologyTarget, generation: int) -> TransitionRecord:
        record = TransitionRecord(str(uuid.uuid4()), generation, source, target)
        self.records.append(record)
        source_generation = self.active_generation
        gpus = tuple(sorted(set(source.gpu_ids) | set(target.gpu_ids)))
        common = dict(source=source, target=target, generation=generation,
                      source_generation=source_generation, transaction_id=record.transaction_id,
                      timeout_s=self.drain_timeout_s)

        def execute(state: TransitionState, ack_target: TopologyTarget | None) -> dict:
            result = self.backend.execute(state.value, **common)
            record.event(state, receipt=result)
            _check_ack(result, state.value, ack_target, generation, record.transaction_id)
            return result

        try:
            with self.backend.gpu_lock(gpus, record.transaction_id) as lease:
                if (not isinstance(lease, dict) or lease.get("locked") is not True
                        or lease.get("transaction_id") != record.transaction_id
                        or set(lease.get("gpu_ids", ())) != set(gpus) or not lease.get("lease_id")):
                    raise RuntimeError("physical GPU lock receipt missing or mismatched")
                try:
                    # CPU metadata preparation can precede creation of the new
                    # TP workers. Native target rank proof starts at transfer.
                    prepared = execute(TransitionState.PREPARE, None)
                    if prepared.get("prepared") is not True:
                        raise RuntimeError("prepare ACK missing")
                    if (set(source.gpu_ids) & set(target.gpu_ids)
                            and prepared.get("preparation") != "metadata_only"):
                        raise RuntimeError("overlapping GPUs require metadata-only prepare before drain")
                    drained = execute(TransitionState.DRAIN, source)
                    if (drained.get("drained") is not True or drained.get("inflight") != 0
                            or drained.get("live_kv_blocks") != 0 or drained.get("pending_transfers") != 0
                            or drained.get("source_generation") != source_generation):
                        raise RuntimeError("drain must ACK zero inflight, live KV, and pending transfers")
                    transfer = execute(TransitionState.TRANSFER, target)
                    if (transfer.get("weights_verified") is not True or transfer.get("live_kv_transferred") is not False
                            or transfer.get("method") not in ("reload", "weight_transfer", "resident")):
                        raise RuntimeError("transfer lacks verified weights or attempted live KV migration")
                    verified = execute(TransitionState.VERIFY, target)
                    golden, observed = verified.get("golden_output_sha256"), verified.get("output_sha256")
                    if (verified.get("output_ok") is not True or verified.get("kv_ok") is not True
                            or verified.get("cancel_ok") is not True or verified.get("live_kv_blocks") != 0
                            or not isinstance(golden, str) or len(golden) != 64 or golden != observed):
                        raise RuntimeError("verification requires golden output, empty KV, and cancel receipts")
                    activated = execute(TransitionState.ACTIVATE, target)
                    if activated.get("active") is not True:
                        raise RuntimeError("target activation ACK missing")
                    retired = execute(TransitionState.RETIRE, source)
                    if retired.get("retired") is not True:
                        raise RuntimeError("source retirement ACK missing")
                    self.active, self.active_generation = target, generation
                    record.event(TransitionState.COMPLETE)
                except Exception as exc:
                    record.error = f"{type(exc).__name__}: {exc}"
                    try:
                        rollback = execute(TransitionState.ROLLBACK, source)
                        _check_ack(rollback, "rollback", target, generation, record.transaction_id,
                                   ranks_key="target_rank_acks")
                        if (rollback.get("rolled_back") is not True or rollback.get("source_restored") is not True
                                or rollback.get("target_quarantined") is not True
                                or rollback.get("source_generation") != source_generation):
                            raise RuntimeError("rollback lacks source restoration/target quarantine ACK")
                        self.active, self.active_generation = source, source_generation
                    except Exception as rollback_exc:
                        self.quarantined_gpus.update(gpus)
                        self.active = None
                        record.event(TransitionState.QUARANTINE, error=record.error,
                                     rollback_error=f"{type(rollback_exc).__name__}: {rollback_exc}", gpu_ids=gpus)
                        raise QuarantineRequired(record.error) from rollback_exc
        except QuarantineRequired:
            pass
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            record.event(TransitionState.ABORTED, error=record.error)
        record.finished_s = time.time()
        return record
