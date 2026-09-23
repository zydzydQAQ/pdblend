"""Independent fixed-TP Mixed GPU functionality on the shared serving substrate.

This development driver consumes no performance profile or planner. It checks
real least-load routing, SSE completions, request-counter release and cleanup;
it does not qualify SLO, total-system energy, cancellation or KV correctness.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

from pdblend_baselines.mixed_policy import MixedLeastLoadPolicy, MixedReplica

from ..engine.client import EngineClient
from ..engine.launcher import Fleet, make_specs
from ..model_registry import ModelRegistry
from ..seed_config import SINGLE_SEED, seed_metadata
from .client import Request
from .gates import random_prompt
from .metering import Gpus
from .smoke_trace import build_smoke_trace as shared_smoke_trace

INPUT_LENGTHS = (128, 512, 2048)
OUTPUT_TOKENS = 16
RATE_RPS = .2
WARMUP_SEED = 9701


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _append(path: Path, value: dict) -> None:
    with path.open("a") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")


def build_smoke_trace(duration_s: float = 100, seed: int = SINGLE_SEED) -> list[Request]:
    """Common deterministic workload, usable unchanged by the PDBlend smoke."""
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("positive finite smoke duration required")
    if seed != SINGLE_SEED:
        raise ValueError("active smoke campaign requires seed 701")
    return shared_smoke_trace(duration_s=duration_s, seed=seed, rate_rps=RATE_RPS)


def trace_digest(trace: list[Request]) -> str:
    return hashlib.sha256(json.dumps([asdict(row) for row in trace], sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def valid_completion(row: dict, *, max_tokens: int = OUTPUT_TOKENS) -> bool:
    times = row.get("token_times_s", [])
    first, last = row.get("first_token_s"), row.get("finished_s")
    return (not row.get("error") and row.get("completion_tokens") == max_tokens
            and bool(row.get("text")) and bool(times)
            and isinstance(first, (int, float)) and isinstance(last, (int, float))
            and math.isfinite(first) and math.isfinite(last)
            and row.get("submitted_s", float("inf")) <= first <= last
            and all(isinstance(t, (int, float)) and math.isfinite(t) and first <= t <= last for t in times)
            and all(a <= b for a, b in zip(times, times[1:])))


async def replay(clients: dict, replicas: list[MixedReplica], trace: list[Request], out: Path,
                 *, duration_s: float, seed: int = SINGLE_SEED) -> dict:
    """Route every live request through the independent policy; always release counts."""
    policy = MixedLeastLoadPolicy(replicas[0].tp)
    start = asyncio.get_running_loop().time()
    started_s = time.time()
    peak_active = 0

    async def submit(request: Request) -> dict:
        nonlocal peak_active
        await asyncio.sleep(max(0, start + request.arrival_s - asyncio.get_running_loop().time()))
        request_id = f"mixed-smoke-{seed}-{request.idx}"
        route = policy.route(request_id, replicas)
        row = dict(request_id=request_id, idx=request.idx, arrival_s=request.arrival_s,
                   input_tokens=request.input_tokens, max_tokens=request.max_tokens, sampling_seed=seed)
        if route is None:
            row.update(error="no accepting Mixed replica", correct=False)
            _append(out / "outcomes.jsonl", row)
            return row
        peak_active = max(peak_active, sum(replica.active_requests for replica in replicas))
        _append(out / "routes.jsonl", dict(asdict(route), at_s=time.time(), idx=request.idx,
                                          active_after_route={r.instance_id: r.active_requests for r in replicas}))
        previous_text = ""

        def on_token(completion, timestamp):
            nonlocal previous_text
            delta = completion.text[len(previous_text):]
            previous_text = completion.text
            _append(out / "sse-events.jsonl", dict(request_id=request_id, instance_id=route.instance_id,
                                                  at_s=timestamp, text_delta=delta))

        try:
            completion = await clients[route.instance_id].complete(
                request.prompt, request.max_tokens, request_id, on_token=on_token, seed=seed)
            row.update(asdict(completion))
            row["correct"] = valid_completion(row, max_tokens=request.max_tokens)
        except Exception as exc:
            row.update(error=f"{type(exc).__name__}: {exc}", correct=False)
        finally:
            policy.complete(route.instance_id, replicas)
            _append(out / "request-events.jsonl", dict(kind="released", request_id=request_id,
                    instance_id=route.instance_id, at_s=time.time(),
                    active_after_release={r.instance_id: r.active_requests for r in replicas}))
        _append(out / "outcomes.jsonl", row)
        return row

    outcomes = await asyncio.gather(*(submit(request) for request in trace))
    await asyncio.sleep(max(0, start + duration_s - asyncio.get_running_loop().time()))
    counts = {replica.instance_id: replica.active_requests for replica in replicas}
    return dict(started_s=started_s, finished_s=time.time(), offered=len(trace), completed=len(outcomes),
                correct=sum(row.get("correct") is True for row in outcomes),
                route_count=len(policy.routes), route_instance_ids=sorted({r.instance_id for r in policy.routes}),
                active_requests=counts, counts_reclaimed=not any(counts.values()), peak_active_requests=peak_active,
                passed=bool(trace) and len(policy.routes) == len(trace)
                    and all(row.get("correct") is True for row in outcomes) and not any(counts.values()))


def gpu_manifest(meter: Gpus, gpus: list[int]) -> list[dict]:
    nv = meter.backend._nvml
    result = []
    for gpu in gpus:
        handle = meter.backend._handle(gpu)
        uuid = nv.nvmlDeviceGetUUID(handle)
        name = nv.nvmlDeviceGetName(handle)
        result.append(dict(local_gpu=gpu, uuid=uuid.decode() if isinstance(uuid, bytes) else uuid,
                           name=name.decode() if isinstance(name, bytes) else name,
                           power_limit_w=meter.backend.power_limit_w(gpu)))
    if len({row["uuid"] for row in result}) != len(gpus):
        raise ValueError("GPU identity mapping contains duplicate UUIDs")
    return result


def smoke_manifest(model, specs, hardware, trace, *, duration_s, seed) -> dict:
    policy_path = Path(sys.modules[MixedLeastLoadPolicy.__module__].__file__)
    versions = {}
    for package in ("vllm", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(schema=1, system="mixed", policy="fixed_tp_least_load", evidence_class="development",
                formal_eligible=False, energy_comparable=False, hardware_qualified=False,
                profile_required=False, predictor_required=False, model_id=model.model_id,
                model_hash=model.model_hash, tokenizer_hash=model.tokenizer_hash,
                config_sha256=model.manifest_sha256, verification_receipt=model.verification_receipt,
                tp=specs[0].tp, pp=1, specs=[asdict(spec) for spec in specs], hardware=hardware,
                source_sha256=os.environ.get("PDBLEND_SOURCE_SHA256"), image_digest=os.environ.get("PDBLEND_IMAGE_ID"),
                driver_sha256=_sha(Path(__file__)), independent_policy_sha256=_sha(policy_path), versions=versions,
                cuda=os.environ.get("CUDA_VERSION"), **seed_metadata((seed,)), duration_s=duration_s,
                warmup_seed=WARMUP_SEED,
                rate_rps=RATE_RPS, trace_sha256=trace_digest(trace), requests=len(trace),
                input_lengths=list(INPUT_LENGTHS), output_tokens=OUTPUT_TOKENS,
                measurement_scope="leased_gpu_group_service_window_and_request_tail",
                output_correctness_scope="SSE_completion_and_output_token_count",
                missing_gates=["baseline_capacity_calibration", "SLO_matrix", "whole_fleet_energy",
                               "native_cancellation", "KV_correctness", "full_model_output_correctness"])


def run(model: str, gpus: list[int], tp: int, out: Path, *, base_port: int = 8100,
        duration_s: float = 100, seed: int = SINGLE_SEED) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("completion.json", "smoke-manifest.json", "routes.jsonl")):
        raise FileExistsError("refusing to overwrite an existing Mixed smoke attempt")
    result = dict(schema=1, system="mixed", status="failed", complete=False, formal_eligible=False,
                  energy_comparable=False, hardware_qualified=False, evidence_class="development", **seed_metadata((seed,)))
    fleet = meter = sampler = None
    cleanup_errors = []
    try:
        if tp < 1 or len(gpus) < 2 * tp or len(gpus) % tp or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
            raise ValueError("Mixed smoke requires at least two disjoint fixed-TP replicas")
        if base_port < 1024 or base_port + max(gpus) > 65535:
            raise ValueError("invalid instance port range")
        trace = build_smoke_trace(duration_s, seed)
        if not trace:
            raise ValueError("smoke duration generated no requests")
        registry = ModelRegistry(os.environ.get("PDBLEND_MODELS_DIR", "/models"),
                                 verification_receipt=os.environ.get("PDBLEND_MODEL_VERIFICATION_RECEIPT"))
        model_spec = registry.get(Path(model).name)
        model_spec.validate_config()
        model_spec.validate_topology(tp, 1, available_gpus=len(gpus))
        specs = make_specs(model_spec.model_path, gpus, tp=tp, pp=1, base_port=base_port,
                           kv_connector=None, max_num_seqs=32, gpu_memory_utilization=.85)
        meter = Gpus(gpus)
        hardware = gpu_manifest(meter, gpus)
        manifest = smoke_manifest(model_spec, specs, hardware, trace, duration_s=duration_s, seed=seed)
        _write(out / "smoke-manifest.json", manifest)
        for request in trace:
            _append(out / "requests.jsonl", dict(asdict(request), sampling_seed=seed))
        result.update(model_id=model_spec.model_id, tp=tp, pp=1, gpu_uuids=[r["uuid"] for r in hardware],
                      trace_sha256=manifest["trace_sha256"])
        meter.reset_all()
        for gpu in gpus:
            meter.set_clock(gpu, 2520)
        fleet = Fleet(specs, out / "logs")
        result["startup_s"] = fleet.start_all()
        sampler = meter.sampler(interval_s=.1)

        async def serve():
            async with AsyncExitStack() as stack:
                clients = {spec.instance_id: await stack.enter_async_context(
                    EngineClient(spec.instance_id, spec.base_url, timeout_s=120)) for spec in specs}
                warm = []
                for spec in specs:
                    completion = await clients[spec.instance_id].complete(
                        random_prompt(128, WARMUP_SEED + 128), OUTPUT_TOKENS,
                        f"mixed-warm-{spec.instance_id}", seed=WARMUP_SEED)
                    row = dict(asdict(completion), sampling_seed=WARMUP_SEED)
                    warm.append(row)
                    _append(out / "warmup.jsonl", row)
                if not all(valid_completion(row) for row in warm):
                    raise RuntimeError("warmup did not produce valid SSE completions on every replica")
                replicas = [MixedReplica(spec.instance_id, tp, max_num_seqs=32) for spec in specs]
                sampler.start()
                try:
                    return await replay(clients, replicas, trace, out, duration_s=duration_s, seed=seed)
                finally:
                    sampler.stop()

        result["measurement"] = asyncio.run(serve())
        result["complete"] = result["measurement"]["passed"]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if sampler is not None:
            try:
                sampler.stop()
                _write(out / "power.json", dict(samples=sampler.samples, frequency_samples=sampler.frequency_samples,
                        power_source=sampler.power_source, power_metadata=sampler.power_metadata, error=sampler.error,
                        gpu_uuids=result.get("gpu_uuids", []), energy_comparable=False))
                result["service_energy_j"] = sampler.total_energy_j()
                result["sampler_error"] = sampler.error
                if sampler.error or len(sampler.samples) < 2 or not sampler.frequency_samples:
                    result["complete"] = False
            except Exception as exc:
                cleanup_errors.append(f"sampler finalization: {exc}")
                result["complete"] = False
        if fleet is not None:
            # Each stop targets only a child created by this Fleet.
            for instance in fleet.instances.values():
                try:
                    instance.stop()
                    if instance.alive():
                        cleanup_errors.append(f"child remains alive: {instance.spec.instance_id}")
                except Exception as exc:
                    cleanup_errors.append(f"stop {instance.spec.instance_id}: {exc}")
            try:
                _write(out / "fleet-events.json", {"events": fleet.events()})
            except Exception as exc:
                cleanup_errors.append(f"fleet event finalization: {exc}")
        if meter is not None:
            for gpu in gpus:
                try:
                    meter.unpark(gpu)
                    meter.reset_clock(gpu)
                except Exception as exc:
                    cleanup_errors.append(f"reset GPU {gpu}: {exc}")
        result.update(cleanup_errors=cleanup_errors, finished_s=time.time())
        if cleanup_errors:
            result["complete"] = False
        result["status"] = "passed" if result["complete"] else "failed"
        owned_artifacts = ("smoke-manifest.json", "requests.jsonl", "routes.jsonl", "outcomes.jsonl",
                           "sse-events.jsonl", "request-events.jsonl", "warmup.jsonl", "power.json", "fleet-events.json")
        result["artifact_sha256"] = {name: _sha(out / name) for name in owned_artifacts if (out / name).is_file()}
        _write(out / "completion.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", required=True, help="comma-separated local GPU ordinals from the lease")
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--base-port", type=int, default=8100)
    parser.add_argument("--duration", type=float, default=100)
    parser.add_argument("--seed", type=int, default=SINGLE_SEED)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.model, [int(g) for g in args.gpus.split(",")], args.tp, args.out,
                 base_port=args.base_port, duration_s=args.duration, seed=args.seed)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
