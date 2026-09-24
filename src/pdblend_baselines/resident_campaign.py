"""Run the independent DistServe and EcoServe request paths on one resident pair.

The CLI owns one pair lifecycle; the inner runner can also consume externally
resident services. System policies and receipts remain independent. No model
is unloaded between their service windows.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import aiohttp

from pdblend_runtime.probe import NativeSpec, generate
from pdblend.engine.launcher import Fleet
from pdblend.bench.metering import Gpus
from pdblend_runtime.cleanup import cleanup_owned
from pdblend.results.power_archive import write_power_archive
from .distserve.run_native import execute as distserve_execute
from .ecoserve.run_native import (execute as ecoserve_execute, validate_state,
                                 expected_identity, validate_profile, load_trace)


@contextmanager
def model_load_lock():
    """Serialize weight loading only; resident service windows stay parallel."""
    path = Path(os.environ.get('PDBLEND_RESIDENT_LOAD_LOCK', '/tmp/pdblend-resident-model-load.lock'))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield str(path)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_specs(model: str, gpus: Iterable[int], tp: int, base_port: int):
    gpus = tuple(int(x) for x in gpus)
    legal = {'Qwen2.5-7B-Instruct': (1,2,4), 'Qwen2.5-14B-Instruct': (1,2,4),
             'Qwen2.5-32B-Instruct': (2,4)}
    if (tp not in legal.get(Path(model).name, ()) or len(gpus) != 2 * tp
            or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus)):
        raise ValueError("resident campaign requires two disjoint TP groups")
    return [NativeSpec("resident-" + role, tuple(gpus[i * tp:(i + 1) * tp]),
                       base_port + 16 * i, model, tp=tp, max_num_seqs=32,
                       extra_args=("--enforce-eager",))
            for i, role in enumerate(("P", "D"))]


async def _json(session, url, path, payload=None):
    async with session.request('GET' if payload is None else 'POST', url.rstrip('/')+path, json=payload) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f'native {path} failed ({response.status}): {body[:500]}')
        return json.loads(body)


async def verify_endpoints(specs):
    expected = expected_identity(dict(model_id=Path(specs[0].model).name))
    physical = os.environ.get('PDBLEND_GPU_UUIDS', '').split(',')
    receipts = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        for spec in specs:
            cap = await _json(session, spec.base_url, '/baseline/capability')
            uuids = [physical[gpu] for gpu in spec.gpus]
            if (any(cap.get(key) != value for key, value in expected.items())
                    or cap.get('supported') is not True or cap.get('tp') != spec.tp or cap.get('pp') != 1
                    or len(set(uuids)) != spec.tp or not all(value.startswith('GPU-') for value in uuids)
                    or sorted(cap.get('gpu_uuids', [])) != sorted(uuids)):
                raise RuntimeError('resident endpoint model/source/image/topology/physical UUID differs')
            validate_state(cap['state'], spec.tp, drained=True)
            receipts[spec.instance_id] = cap
    return receipts


async def drain_endpoints(specs: Iterable[NativeSpec]) -> list[dict]:
    """Require acknowledged, fresh all-rank and scheduler inventories."""
    receipts = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=35)) as session:
        for spec in specs:
            drained = await _json(session, spec.base_url, '/baseline/drain', dict(timeout_s=30))
            if drained.get('acknowledged') is not True or drained.get('drained') is not True:
                raise RuntimeError('native drain response lacks acknowledged/drained proof')
            validate_state(drained, spec.tp, drained=True)
            state = await _json(session, spec.base_url, '/baseline/state')
            validate_state(state, spec.tp, drained=True)
            if state['generation'] != drained['generation']:
                raise RuntimeError('native generation changed across the drain boundary')
            receipts.append(dict(instance_id=spec.instance_id, drain=drained, state=state,
                                 received_s=time.time(), atomic_snapshot=False))
    return receipts


async def warmup_endpoints(specs: Iterable[NativeSpec], label: str) -> list[dict]:
    """Run a separate seed-9701 request on each resident member."""
    rows = []
    rng = random.Random(9701)
    prompt = [rng.randint(1000, 60000) for _ in range(128)]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        for spec in specs:
            before = await _json(session, spec.base_url, '/baseline/state')
            validate_state(before, spec.tp, drained=True)
            desired = dict(role='mixed', mode='temporal', accepting=True, admit_prefill=True,
                           admit_decode=True, generation=before['generation']+1)
            control = await _json(session, spec.base_url, '/baseline/control', desired)
            if control.get('acknowledged') is not True or control.get('generation') != desired['generation']:
                raise RuntimeError('resident warmup reopen lacks generation ACK')
            ready = await _json(session, spec.base_url, '/baseline/state')
            validate_state(ready, spec.tp, drained=True)
            if any(ready.get(k) != v for k, v in desired.items()):
                raise RuntimeError('resident warmup reopen differs from requested state')
            result = await generate(session, spec.base_url, dict(
                request_id=f'resident-warmup-{label}-{spec.instance_id}', prompt=prompt,
                max_tokens=16, seed=9701, temperature=0, ignore_eos=True))
            if len(result.get('token_ids', [])) != 16:
                raise RuntimeError(f'{label} warmup did not produce 16 tokens')
            rows.append(dict(instance_id=spec.instance_id, label=label, seed=9701, prompt=prompt,
                             control=control, generation=ready['generation'], **result))
    return rows


def eco_config(model: str, specs: list[NativeSpec], csv: Path) -> dict:
    model_path = str(model)
    if not csv.is_file() or not csv.read_bytes():
        raise ValueError("Eco author profile must be a nonempty existing file")
    return {
        "model_id": Path(model_path).name,
        "model_path": model_path,
        "instances": [{"id": s.instance_id, "gpus": list(s.gpus), "tp": s.tp, "pp": 1} for s in specs],
        "eco_prefill_csv": str(csv),
        "eco_profile_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
        "slo_ttft_s": 5.0, "slo_tpot_s": 0.15,
        "eco_initial_instances": 2,
        "eco_macro_lower": 2, "eco_macro_upper": 3,
        "eco_scale_period_s": 5.0,
        "eco_state_poll_s": 0.05, "eco_drain_timeout_s": 120.0,
        "request_timeout_s": 180.0, "eco_active_frequency_mhz": 2520,
    }


def preflight(*, model, tp, gpus, base_port, trace, eco_profile, duration):
    """Read-only launch validation; never create a GPU completion receipt."""
    specs = build_specs(model, gpus, tp, base_port)
    config = eco_config(model, specs, eco_profile)
    expected = expected_identity(config)
    trace_value, rows = load_trace(trace, duration)
    for key in ('model_id', 'tokenizer_hash'):
        if key in trace_value and trace_value[key] != expected[key]:
            raise ValueError('resident trace model identity differs: '+key)
    model_path = specs[0].model_path
    if not (model_path/'config.json').is_file() or not (model_path/'pdblend-model-manifest.json').is_file():
        raise ValueError('verified local model configuration/manifest is absent')
    profile = validate_profile(config, expected, tp)
    return dict(scope='preflight_only', ready=True, hardware_actions_started=False,
                complete=False, formal_eligible=False, energy_comparable=False,
                model_id=expected['model_id'], tp=tp, pp=1, seed=701, duration_s=duration,
                requests=len(rows), trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                profile=profile, identity=expected,
                model_config_sha256=hashlib.sha256((model_path/'config.json').read_bytes()).hexdigest(),
                model_manifest_sha256=hashlib.sha256((model_path/'pdblend-model-manifest.json').read_bytes()).hexdigest())


async def execute(*, model: str, tp: int, gpus: Iterable[int], base_port: int,
                  trace: Path, eco_profile: Path, out: Path, duration: float = 100.0,
                  dist_execute_fn=distserve_execute, eco_execute_fn=ecoserve_execute,
                  own_services: bool = False):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive")
    gpus = tuple(int(gpu) for gpu in gpus)
    specs = build_specs(model, gpus, tp, base_port)
    trace_sha = hashlib.sha256(trace.read_bytes()).hexdigest()
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("refusing to overwrite resident campaign evidence")
    out.mkdir(parents=True, exist_ok=True)
    dist_out, eco_out = out / "distserve", out / "ecoserve"
    urls = {s.instance_id: s.base_url for s in specs}
    receipt = {
        "schema": "resident-campaign-v1", "model": model, "tp": tp,
        "seed": 701, "duration_s": duration, "trace_sha256": trace_sha,
        "reuse_same_native_pair": True, "engine_loads": 0, "engine_load_cycles": 0,
        "owns_services": own_services, "scope": "functional", "complete_reproduction": False,
        "source_sha256": os.environ.get("PDBLEND_SOURCE_SHA256"),
        "image_digest": os.environ.get("PDBLEND_IMAGE_ID"),
        "resident_specs": [{"id": s.instance_id, "gpus": list(s.gpus),
                             "http": s.base_url, "kv": s.zmq_address} for s in specs],
        "formal_eligible": False, "energy_comparable": False,
        "status": "failed", "complete": False, "cleanup_errors": [],
    }
    fleet = meter = sampler = None
    try:
        if own_services:
            meter = Gpus(gpus, power_mode="instant"); sampler = meter.sampler(interval_s=.1)
            fleet = Fleet(specs, out / "logs")
            sampler.start()
            receipt["startup"] = {}
            receipt["engine_load_cycles"] = 1
            receipt['load_lock_wait_started_s'] = time.time()
            with model_load_lock() as lock_path:
                receipt['load_lock_path'] = lock_path
                receipt['load_lock_acquired_s'] = time.time()
                for spec in specs:
                    instance = fleet[spec.instance_id]
                    instance.start()
                    receipt["engine_loads"] += 1
                    receipt["startup"][spec.instance_id] = instance.wait_ready(timeout_s=600)
            receipt['load_lock_released_s'] = time.time()
        receipt["capabilities"] = await verify_endpoints(specs)
        receipt["warmup_distserve"] = await warmup_endpoints(specs, "distserve")
        dist_args = SimpleNamespace(
            trace=trace, out=dist_out, duration=duration, tp=tp, pp=1,
            prefill_url=specs[0].base_url, decode_url=specs[1].base_url,
            prefill_address=specs[0].zmq_address, decode_address=specs[1].zmq_address,
            max_batch_size=8, request_timeout=180.0)
        dist = await dist_execute_fn(dist_args)
        receipt["distserve"] = dist
        if (dist.get("status") != "passed" or dist.get("complete") is not True
                or dist.get("cleanup_errors", [])):
            raise RuntimeError("DistServe resident receipt did not pass")
        receipt["drain_states"] = await drain_endpoints(specs)
        receipt["warmup_ecoserve"] = await warmup_endpoints(specs, "ecoserve")
        eco = await eco_execute_fn(eco_config(model, specs, eco_profile), urls,
                                   trace, eco_out, duration)
        receipt["ecoserve"] = eco
        if (eco.get("status") != "passed" or eco.get("complete") is not True
                or eco.get("cleanup_errors", [])):
            raise RuntimeError("EcoServe resident receipt did not pass")
        receipt["final_drain_states"] = await drain_endpoints(specs)
        receipt["system_receipt_sha256"] = {name:hashlib.sha256((out/name/"completion.json").read_bytes()).hexdigest()
                                            for name in ("distserve", "ecoserve")}
        receipt.update(status="passed", complete=True)
    except BaseException as exc:
        receipt.update(status="failed", complete=False, error=repr(exc))
    finally:
        if own_services:
            try:
                if meter is not None and sampler is not None:
                    receipt["cleanup_errors"] = cleanup_owned(fleet or SimpleNamespace(instances={}), meter, sampler)
                elif fleet is not None:
                    fleet.stop_all()
            except BaseException as exc:
                receipt["cleanup_errors"].append(dict(component='owned_cleanup', error=repr(exc)))
            if sampler is not None:
                try:
                    power = dict(samples=sampler.samples, frequency_samples=sampler.frequency_samples,
                                 utilization_samples=getattr(sampler, 'utilization_samples', []),
                                 power_metadata=getattr(sampler, 'power_metadata', []),
                                 power_source=getattr(sampler, 'power_source', {}),
                                 gpu_ids=list(gpus), gpu_uuids=os.environ.get('PDBLEND_GPU_UUIDS'),
                                 error=sampler.error, formal_eligible=False, energy_comparable=False)
                    write_power_archive(out/'power.json',power)
                    receipt['power_artifact_sha256'] = hashlib.sha256((out/'power.json').read_bytes()).hexdigest()
                    receipt['energy_j'] = sampler.total_energy_j()
                    receipt['energy_samples'] = len(sampler.samples)
                    if sampler.error or len(sampler.samples) < 2 or not sampler.frequency_samples:
                        raise RuntimeError('resident GPU power/frequency evidence incomplete: '+str(sampler.error))
                except BaseException as exc:
                    receipt['cleanup_errors'].append(dict(component='power_archive', error=repr(exc)))
            if receipt['cleanup_errors']:
                receipt.update(status='failed', complete=False)
    (out / "completion.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True); p.add_argument("--tp", type=int, required=True)
    p.add_argument("--gpus", required=True, help="comma-separated local GPU indices")
    p.add_argument("--base-port", type=int, required=True); p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--eco-profile", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--duration", type=float, default=100.0)
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args(argv)
    if a.preflight_only:
        result = preflight(model=a.model, tp=a.tp, gpus=a.gpus.split(','), base_port=a.base_port,
                           trace=a.trace, eco_profile=a.eco_profile, duration=a.duration)
        a.out.mkdir(parents=True, exist_ok=True)
        destination = a.out/'preflight.json'
        if destination.exists():
            raise FileExistsError('refusing to overwrite resident preflight evidence')
        destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(result))
        return 0
    result = asyncio.run(execute(model=a.model, tp=a.tp, gpus=a.gpus.split(","),
                                 base_port=a.base_port, trace=a.trace, eco_profile=a.eco_profile,
                                 out=a.out, duration=a.duration, own_services=True))
    print(json.dumps({k: result.get(k) for k in ("status", "complete", "error")}, sort_keys=True))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
