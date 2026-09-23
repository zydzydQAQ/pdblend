"""Independent measurement adapters for the vLLM 0.10.1.1 new stack.

This module owns measurement receipts and serialization only.  It never turns
an HTTP TTFT into a stage time, divides an end-to-end time by PP, or fabricates
GPU samples when the V1 worker hooks are unavailable.  The adapters are usable
by the baseline runners without importing ``pdblend`` or its performance model.

GPU methods require a real CUDA device and an actual callable/worker.  CPU tests
exercise validation, serialization, and unsupported receipts; those artifacts
are explicitly not hardware-qualified measurements.
"""
from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterable, Mapping, Sequence


TARGET_ENGINE = "vllm-0.10.1.1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class UnsupportedMeasurement(RuntimeError):
    """Raised when a requested receipt cannot be produced on this runtime."""

    def __init__(self, receipt: "CapabilityReceipt"):
        self.receipt = receipt
        super().__init__(f"{receipt.adapter}: {receipt.reason}")


@dataclass(frozen=True)
class CapabilityReceipt:
    adapter: str
    supported: bool
    reason: str
    engine_revision: str = TARGET_ENGINE
    hardware_qualified: bool = False
    checked_at_s: float = field(default_factory=time.time)
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _module_symbols(module: Any) -> tuple[str, ...]:
    return tuple(sorted(name for name in ("__version__", "distributed", "v1")
                        if hasattr(module, name)))


def detect_vllm_v1_capabilities(*, module: Any | None = None,
                                cuda_available: bool | None = None) -> CapabilityReceipt:
    """Detect the exact capabilities needed by the independent V1 adapters.

    Import and CUDA failures are returned as structured receipts so campaign
    code can mark a point ``unsupported_engine`` instead of silently changing
    its measurement meaning.  ``module`` and ``cuda_available`` are injectable
    solely for testing the detector; normal callers omit both.
    """
    details: dict[str, Any] = {}
    if module is None:
        if importlib.util.find_spec("vllm") is None:
            return CapabilityReceipt("vllm_v1_stage", False, "vllm_not_installed",
                                     details={"required": TARGET_ENGINE})
        try:
            module = importlib.import_module("vllm")
        except Exception as exc:
            return CapabilityReceipt("vllm_v1_stage", False, "vllm_import_failed",
                                     details={"error": repr(exc)})
    version = str(getattr(module, "__version__", ""))
    details["vllm_version"] = version or None
    details["module_symbols"] = _module_symbols(module)
    if version and version != "0.10.1.1":
        return CapabilityReceipt("vllm_v1_stage", False, "engine_version_mismatch",
                                 details={**details, "required": "0.10.1.1"})
    try:
        distributed = importlib.import_module("vllm.distributed")
        pp = getattr(distributed, "get_pp_group")
        tp = getattr(distributed, "get_tp_group")
        details["distributed_api"] = {"get_pp_group": callable(pp), "get_tp_group": callable(tp)}
    except Exception as exc:
        return CapabilityReceipt("vllm_v1_stage", False, "distributed_group_api_missing",
                                 details={**details, "error": repr(exc)})
    worker_api: dict[str, Any] = {}
    for name in ("vllm.v1.worker.gpu_worker", "vllm.v1.worker.gpu_model_runner"):
        try:
            candidate = importlib.import_module(name)
        except Exception as exc:
            worker_api[name] = {"imported": False, "error": repr(exc)}
            continue
        worker_api[name] = {"imported": True,
                            "execute_model": any(callable(getattr(candidate, n, None))
                                                  for n in ("execute_model", "GPUWorker", "GPUModelRunner"))}
    details["worker_api"] = worker_api
    if not (callable(pp) and callable(tp)):
        return CapabilityReceipt("vllm_v1_stage", False, "distributed_group_api_missing", details=details)
    if not any(value.get("execute_model") for value in worker_api.values()):
        return CapabilityReceipt("vllm_v1_stage", False, "v1_execute_model_api_missing", details=details)
    if cuda_available is None:
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:
            details["cuda_error"] = repr(exc)
            cuda_available = False
    details["cuda_available"] = bool(cuda_available)
    if not cuda_available:
        return CapabilityReceipt("vllm_v1_stage", False, "cuda_unavailable", details=details)
    return CapabilityReceipt("vllm_v1_stage", True, "ready", details=details)


def model_input_metadata(model_input: Any) -> dict[str, Any]:
    """Extract only native V1 shape fields; missing fields remain missing."""
    def sequence(name: str) -> list[int] | None:
        value = getattr(model_input, name, None)
        if value is None:
            return None
        try:
            return [int(x) for x in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid native model input {name}") from exc

    tokens = getattr(model_input, "input_tokens", None)
    token_count = None if tokens is None else int(tokens.numel())
    seq_lens, query_lens = sequence("seq_lens"), sequence("query_lens")
    return {"seq_lens": seq_lens, "query_lens": query_lens,
            "is_prompt": getattr(model_input, "is_prompt", None),
            "input_tokens": token_count,
            "virtual_engine": getattr(model_input, "virtual_engine", None)}


@dataclass(frozen=True)
class NativeStageSample:
    model_id: str
    role: str
    tp: int
    pp: int
    rank: int
    tp_rank: int
    pp_rank: int
    batch: int
    input_tokens: int
    max_context_tokens: int
    gpu_elapsed_ms: float
    source: str = "vllm_v1_model_runner.execute_model"
    measurement_scope: str = "cuda_event_model_runner_including_tp_collectives"
    hardware_qualified: bool = True

    def __post_init__(self) -> None:
        if self.role not in ("prefill", "decode"):
            raise ValueError("native stage role must be prefill or decode")
        if any(type(value) is not int or value < 1 for value in
               (self.tp, self.pp, self.batch, self.input_tokens, self.max_context_tokens)) or self.rank < 0:
            raise ValueError("native stage identity and shape must be positive integers")
        if self.tp_rank < 0 or self.tp_rank >= self.tp or self.pp_rank < 0 or self.pp_rank >= self.pp:
            raise ValueError("native TP/PP rank outside declared topology")
        if not math.isfinite(self.gpu_elapsed_ms) or self.gpu_elapsed_ms <= 0:
            raise ValueError("positive CUDA event elapsed time required")


class NativeStageCollector:
    """Wrap one V1 worker's native model runner with CUDA events.

    ``collect`` is the only place that synchronizes end events.  The wrapped
    execution path never substitutes wall-clock or HTTP measurements.
    """

    def __init__(self, *, model_id: str, tp: int, pp: int, rank: int,
                 tp_rank: int, pp_rank: int, event_factory: Callable[[], Any] | None = None):
        self.model_id, self.tp, self.pp = model_id, tp, pp
        self.rank, self.tp_rank, self.pp_rank = rank, tp_rank, pp_rank
        self._event_factory = event_factory
        self._pending: list[tuple[dict[str, Any], Any, Any]] = []
        self._installed = False

    @staticmethod
    def _torch_event() -> Any:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; native stage timing cannot be collected")
        return torch.cuda.Event(enable_timing=True)

    def install(self, worker: Any) -> Any:
        if self._installed:
            raise RuntimeError("native stage collector already installed")
        worker = getattr(worker, "worker", worker)
        runner = getattr(worker, "model_runner", None)
        original = getattr(runner, "execute_model", None)
        if runner is None or not callable(original):
            raise RuntimeError("V1 worker has no model_runner.execute_model")
        factory = self._event_factory or self._torch_event

        def measured(_runner: Any, *args: Any, **kwargs: Any) -> Any:
            model_input = args[0] if args else kwargs.get("model_input")
            if model_input is None:
                raise RuntimeError("execute_model call did not expose model_input")
            metadata = model_input_metadata(model_input)
            if metadata["is_prompt"] not in (True, False):
                raise RuntimeError("native is_prompt field is required; cannot infer P/D role")
            start, end = factory(), factory()
            row = {"metadata": metadata, "started_s": time.time(), "succeeded": False}
            start.record()
            try:
                result = original(*args, **kwargs)
                row["succeeded"] = True
                return result
            except BaseException as exc:
                row["error"] = repr(exc)
                raise
            finally:
                end.record()
                row["finished_s"] = time.time()
                self._pending.append((row, start, end))

        runner.execute_model = MethodType(measured, runner)
        self._installed = True
        return runner

    def collect(self) -> list[NativeStageSample]:
        pending, self._pending = self._pending, []
        samples: list[NativeStageSample] = []
        for row, start, end in pending:
            end.synchronize()
            if not row.get("succeeded"):
                raise RuntimeError("failed native execute_model call cannot become a profile sample")
            metadata = row["metadata"]
            seq = metadata.get("seq_lens") or []
            query = metadata.get("query_lens") or []
            batch = max(len(seq), len(query))
            if batch < 1 or metadata.get("input_tokens") is None:
                raise RuntimeError("native batch and input token receipt are incomplete")
            contexts = [max(a, b) for a, b in zip(seq, query)] or seq or query
            elapsed = float(start.elapsed_time(end))
            role = "prefill" if metadata["is_prompt"] else "decode"
            samples.append(NativeStageSample(
                model_id=self.model_id, role=role, tp=self.tp, pp=self.pp,
                rank=self.rank, tp_rank=self.tp_rank, pp_rank=self.pp_rank,
                batch=batch, input_tokens=int(metadata["input_tokens"]),
                max_context_tokens=max(contexts, default=int(metadata["input_tokens"])),
                gpu_elapsed_ms=elapsed))
        return samples


def install_native_stage_collector(worker: Any, *, model_id: str, tp: int, pp: int,
                                   rank: int | None = None, tp_rank: int | None = None,
                                   pp_rank: int | None = None,
                                   event_factory: Callable[[], Any] | None = None) -> NativeStageCollector:
    """Install a collector only after capability and physical rank checks."""
    receipt = detect_vllm_v1_capabilities()
    if not receipt.supported:
        raise UnsupportedMeasurement(receipt)
    worker_obj = getattr(worker, "worker", worker)
    try:
        distributed = importlib.import_module("vllm.distributed")
        pp_group, tp_group = distributed.get_pp_group(), distributed.get_tp_group()
        rank = int(getattr(worker_obj, "rank")) if rank is None else rank
        tp_rank = int(getattr(tp_group, "rank_in_group")) if tp_rank is None else tp_rank
        pp_rank = int(getattr(pp_group, "rank_in_group")) if pp_rank is None else pp_rank
    except Exception as exc:
        raise RuntimeError("physical V1 TP/PP rank receipt unavailable") from exc
    collector = NativeStageCollector(model_id=model_id, tp=tp, pp=pp, rank=rank,
                                     tp_rank=tp_rank, pp_rank=pp_rank,
                                     event_factory=event_factory)
    collector.install(worker_obj)
    return collector


def reduce_native_stage_samples(samples: Iterable[NativeStageSample], *, tp: int, pp: int,
                                role: str) -> dict[str, Any]:
    """Reduce rank samples without hiding missing physical stages or TP ranks."""
    rows = [sample for sample in samples if sample.role == role]
    if not rows:
        raise ValueError("native stage sample set is empty")
    identities = {(row.pp_rank, row.tp_rank) for row in rows}
    expected = {(p, t) for p in range(pp) for t in range(tp)}
    if identities != expected:
        raise ValueError(f"native stage coverage incomplete: missing={sorted(expected - identities)}")
    by_identity: dict[tuple[int, int], list[NativeStageSample]] = {}
    for row in rows:
        by_identity.setdefault((row.pp_rank, row.tp_rank), []).append(row)
    if len({len(values) for values in by_identity.values()}) != 1:
        raise ValueError("native TP ranks have unequal sample counts")
    values = []
    for identity in sorted(by_identity):
        durations = sorted(row.gpu_elapsed_ms for row in by_identity[identity])
        values.append({"pp_rank": identity[0], "tp_rank": identity[1],
                       "samples": len(durations), "minimum_gpu_ms": durations[0],
                       "median_gpu_ms": durations[len(durations) // 2]})
    return {"role": role, "tp": tp, "pp": pp, "ranks": values,
            "measurement_scope": rows[0].measurement_scope,
            "hardware_qualified": all(row.hardware_qualified for row in rows)}


@dataclass(frozen=True)
class EcoServeForwardSample:
    model_id: str
    tp: int
    pp: int
    length: int
    samples_ms: tuple[float, ...]
    minimum_ms: float
    measurement_scope: str = "cuda_event_full_forward"
    hardware_qualified: bool = True

    def __post_init__(self) -> None:
        if self.length < 1 or len(self.samples_ms) < 5:
            raise ValueError("EcoServe requires at least five forward samples per length")
        if any(not math.isfinite(x) or x <= 0 for x in self.samples_ms) or self.minimum_ms != min(self.samples_ms):
            raise ValueError("EcoServe samples must contain positive finite CUDA times and exact minimum")


class EcoServeForwardMeter:
    """Measure author-style prefill CSV values from actual CUDA forwards."""

    def __init__(self, *, model_id: str, tp: int, pp: int = 1,
                 event_factory: Callable[[], Any] | None = None):
        self.model_id, self.tp, self.pp = model_id, tp, pp
        self._event_factory = event_factory

    def measure(self, length: int, forward: Callable[[], Any], *, warmup: int = 2,
                samples: int = 5) -> EcoServeForwardSample:
        if type(length) is not int or length < 1 or not callable(forward):
            raise ValueError("positive length and callable forward required")
        if type(samples) is not int or samples < 5 or type(warmup) is not int or warmup < 0:
            raise ValueError("EcoServe requires warmup>=0 and at least five measured samples")
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; EcoServe forward timing cannot be collected")
            sync = torch.cuda.synchronize
        except Exception as exc:
            receipt = CapabilityReceipt("ecoserve_gpu_forward", False, "cuda_unavailable",
                                        details={"error": repr(exc), "length": length})
            raise UnsupportedMeasurement(receipt) from exc
        event_factory = self._event_factory or (lambda: torch.cuda.Event(enable_timing=True))
        for _ in range(warmup):
            forward()
        sync()
        values: list[float] = []
        for _ in range(samples):
            start, end = event_factory(), event_factory()
            start.record()
            forward()
            end.record()
            end.synchronize()
            value = float(start.elapsed_time(end))
            if not math.isfinite(value) or value <= 0:
                raise RuntimeError("CUDA forward returned a non-positive elapsed time")
            values.append(value)
        result = EcoServeForwardSample(self.model_id, self.tp, self.pp, length,
                                       tuple(values), min(values))
        return result

    @staticmethod
    def write_csv(samples: Iterable[EcoServeForwardSample], path: str | Path) -> Path:
        rows = list(samples)
        by_length = {row.length: row for row in rows}
        if 16 not in by_length or 4096 not in by_length:
            raise ValueError("EcoServe author profile requires length anchors 16 and 4096")
        if len(by_length) != len(rows):
            raise ValueError("duplicate EcoServe length measurement")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("Length", "Prefill Time"))
            writer.writeheader()
            for length in sorted(by_length):
                writer.writerow({"Length": length, "Prefill Time": format(by_length[length].minimum_ms, ".9g")})
        return path


@dataclass(frozen=True)
class DynamoShapeMeasurement:
    model_id: str
    tp: int
    frequency_mhz: int
    input_tokens: int
    context_tokens: int
    batch: int
    prefill_s: float
    iteration_s: float
    prefill_power_w: float
    decode_power_w: float
    samples: int
    source_sha256: str
    measurement_scope: str = "native_v1_mixed_shape"
    hardware_qualified: bool = True

    def __post_init__(self) -> None:
        ints = (self.tp, self.frequency_mhz, self.input_tokens, self.context_tokens, self.batch, self.samples)
        if any(type(value) is not int or value < 1 for value in ints):
            raise ValueError("Dynamo shape identity and sample count must be positive integers")
        if self.context_tokens < self.input_tokens or not SHA256.fullmatch(self.source_sha256):
            raise ValueError("Dynamo context and source SHA256 are invalid")
        values = (self.prefill_s, self.iteration_s, self.prefill_power_w, self.decode_power_w)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Dynamo shape timing and power must be positive finite values")

    def point(self) -> dict[str, Any]:
        return {"role": "mixed", "model_id": self.model_id, "tp": self.tp,
                "frequency_mhz": self.frequency_mhz, "input_tokens": self.input_tokens,
                "context_tokens": self.context_tokens, "batch": self.batch,
                "prefill_s": self.prefill_s, "iteration_s": self.iteration_s,
                "prefill_power_w": self.prefill_power_w, "decode_power_w": self.decode_power_w,
                "power_w": max(self.prefill_power_w, self.decode_power_w), "samples": self.samples,
                "source_sha256": self.source_sha256,
                "measurement_scope": self.measurement_scope,
                "hardware_qualified": self.hardware_qualified}


class DynamoShapeProfileBuilder:
    """Build a rectangular profile independently from the load history."""

    def __init__(self, *, model_id: str, tps: Sequence[int], frequencies_mhz: Sequence[int],
                 input_tokens: Sequence[int], context_tokens: Sequence[int], batches: Sequence[int],
                 engine_revision: str = TARGET_ENGINE):
        axes = (tps, frequencies_mhz, input_tokens, context_tokens, batches)
        if any(not values for values in axes) or any(len(set(values)) != len(values) for values in axes):
            raise ValueError("Dynamo rectangular profile axes must be nonempty and distinct")
        self.model_id, self.engine_revision = model_id, engine_revision
        self.expected = {(int(tp), int(freq), int(inp), int(ctx), int(batch))
                         for tp in tps for freq in frequencies_mhz for inp in input_tokens
                         for ctx in context_tokens for batch in batches}
        self.rows: dict[tuple[int, int, int, int, int], DynamoShapeMeasurement] = {}

    def record(self, row: DynamoShapeMeasurement) -> None:
        if row.model_id != self.model_id:
            raise ValueError("Dynamo shape model identity mismatch")
        key = (row.tp, row.frequency_mhz, row.input_tokens, row.context_tokens, row.batch)
        if key not in self.expected:
            raise ValueError("Dynamo shape is outside the declared rectangle")
        if key in self.rows:
            raise ValueError("duplicate Dynamo shape measurement")
        self.rows[key] = row

    def missing(self) -> tuple[tuple[int, int, int, int, int], ...]:
        return tuple(sorted(self.expected - self.rows.keys()))

    def document(self, *, require_complete: bool = True) -> dict[str, Any]:
        missing = self.missing()
        if require_complete and missing:
            raise ValueError(f"Dynamo rectangular profile is incomplete: {len(missing)} shapes missing")
        return {"schema": 2, "measurement": "hardware", "model_id": self.model_id,
                "engine_revision": self.engine_revision, "coordinate_system": "input_context_batch",
                "rectangular": not missing, "missing_shapes": [list(x) for x in missing],
                "hardware_qualified": bool(self.rows) and all(row.hardware_qualified for row in self.rows.values()),
                "points": [self.rows[key].point() for key in sorted(self.rows)]}

    def write(self, path: str | Path, *, require_complete: bool = True) -> Path:
        path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.document(require_complete=require_complete), indent=2, sort_keys=True) + "\n")
        return path


@dataclass(frozen=True)
class DynamoLoadMeasurement:
    at_s: float
    input_tokens: int
    output_tokens: int
    shape: str
    count: int = 1
    split: str = "calibration"

    def __post_init__(self) -> None:
        if not math.isfinite(self.at_s) or self.input_tokens < 1 or self.output_tokens < 1 or self.count < 1:
            raise ValueError("Dynamo load observations must be positive finite records")
        if self.split not in ("calibration", "evaluation"):
            raise ValueError("Dynamo load split must be calibration or evaluation")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class DynamoLoadProfileBuilder:
    """Separate timestamped arrival history; never merges it into shape timings."""

    def __init__(self, *, model_id: str, source_sha256: str, engine_revision: str = TARGET_ENGINE):
        if not SHA256.fullmatch(source_sha256):
            raise ValueError("Dynamo load source SHA256 required")
        self.model_id, self.source_sha256, self.engine_revision = model_id, source_sha256, engine_revision
        self.rows: list[DynamoLoadMeasurement] = []

    def record(self, row: DynamoLoadMeasurement) -> None:
        self.rows.append(row)

    def document(self) -> dict[str, Any]:
        return {"schema": 1, "measurement": "arrival_history", "model_id": self.model_id,
                "engine_revision": self.engine_revision, "source_sha256": self.source_sha256,
                "records": [row.as_dict() for row in sorted(self.rows, key=lambda row: row.at_s)]}

    def write(self, path: str | Path) -> Path:
        path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.document(), indent=2, sort_keys=True) + "\n")
        return path


class DynamoEndToEndMeasurement:
    """Independent endpoint receipt for Mixed/Dynamo replay and calibration.

    The callable must execute the real serving request and return its observed
    ``first_token_s`` and ``last_token_s`` values (or a mapping with those
    fields).  This adapter does not infer output times from max_tokens and does
    not convert endpoint time into a stage profile.
    """

    def __init__(self, *, model_id: str, system: str, tp: int, engine_revision: str = TARGET_ENGINE):
        if system not in ("mixed", "dynamollm"):
            raise ValueError("endpoint adapter supports only mixed and dynamollm")
        self.model_id, self.system, self.tp, self.engine_revision = model_id, system, tp, engine_revision
        self.records: list[dict[str, Any]] = []

    def measure(self, request_id: str, input_tokens: int, invoke: Callable[[], Any]) -> dict[str, Any]:
        if not request_id or input_tokens < 1 or not callable(invoke):
            raise ValueError("request id, input tokens and callable invoke are required")
        started = time.time()
        result = invoke()
        finished = time.time()
        if isinstance(result, Mapping):
            first = result.get("first_token_s")
            last = result.get("last_token_s")
            output_tokens = result.get("output_tokens")
        else:
            raise ValueError("invoke must return a mapping with native token timestamps")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (first, last)):
            raise ValueError("native first/last token timestamps are required")
        if first < started - 1 or last < first or finished < started:
            raise ValueError("native endpoint timestamps are inconsistent")
        row = {"request_id": request_id, "model_id": self.model_id, "system": self.system,
               "tp": self.tp, "input_tokens": input_tokens, "output_tokens": output_tokens,
               "first_token_s": first, "last_token_s": last,
               "observed_wall_start_s": started, "observed_wall_finish_s": finished,
               "measurement_scope": "native_endpoint_token_timestamps",
               "hardware_qualified": False}
        self.records.append(row)
        return row

    def write(self, path: str | Path) -> Path:
        path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for row in self.records:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return path


class MixedEndToEndMeasurement(DynamoEndToEndMeasurement):
    """The same native endpoint receipt contract for fixed-TP Mixed."""

    def __init__(self, *, model_id: str, tp: int, engine_revision: str = TARGET_ENGINE):
        super().__init__(model_id=model_id, system="mixed", tp=tp,
                         engine_revision=engine_revision)


# Neutral name for runners that share the endpoint measurement harness while
# keeping the independent system identity in every receipt.
EndpointMeasurementAdapter = DynamoEndToEndMeasurement
