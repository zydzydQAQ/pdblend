"""PDBlend TP wiring and explicit P/D mechanism smoke, seed 701 only.

A forced mechanism layout verifies the real request path, not policy quality or
KV/output equivalence. Native KV/golden/rollback tests remain separate gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

from ..control.planner import SLO
from ..control.topology import ResidentPool, Topology
from ..model_registry import ModelRegistry
from ..profile.model import PerfModel
from .client import Request
from .run import run_point
from .tp_runtime import mechanism_pd_plan


def build_trace(*, duration_s: float = 60, seed: int = 701, include_long: bool = True) -> list[Request]:
    if seed != 701 or duration_s <= 0:
        raise ValueError('TP smoke requires seed 701 and a positive duration')
    lengths = (512, 2048, 7168) if include_long else (512, 1024, 2048)
    count = max(6, int(duration_s * .2))
    rng = random.Random(seed)
    return [Request(i, i * duration_s / count,
                    [rng.randrange(100, 5000) for _ in range(lengths[i % len(lengths)])],
                    16, 'tp-mechanism-seed701') for i in range(count)]


def run_smoke(*, model: str, gpus: list[int], tp: int, profile: Path, out: Path,
              mode: str = 'mechanism_pd', profiles: dict[tuple[int, int], Path] | None = None,
              resident_pools: tuple[ResidentPool, ...] = (), duration_s: float = 60,
              base_port: int = 8100, proxy_port: int | None = None,
              include_long: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    trace = build_trace(duration_s=duration_s, include_long=include_long)
    payload = [asdict(request) for request in trace]
    trace_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (out / 'trace.json').write_text(json.dumps(dict(seed=701, trace_sha256=trace_hash, requests=payload), indent=2) + '\n')
    completion = dict(status='failed', complete=False, seed=701, mode=mode, formal_eligible=False,
                      energy_comparable=False, policy_decision=mode != 'mechanism_pd',
                      scope=dict(pd_path=False, dynamic_tp=False, complete_kv=False, output_golden=False), errors=[])
    try:
        if not gpus or len(gpus) > 8 or len(set(gpus)) != len(gpus) or tp not in (1, 2, 4):
            raise ValueError('TP smoke requires unique leased GPUs and TP1/2/4')
        spec = ModelRegistry().get(model)
        spec.validate_topology(tp, available_gpus=len(gpus))
        if len(gpus) % tp:
            raise ValueError('leased GPU count is not divisible by TP')
        profile_model = PerfModel.load(profile)
        if (profile_model.system != 'pdblend' or (profile_model.tp, profile_model.pp) != (tp, 1)
                or Path(profile_model.model).name != spec.model_id
                or profile_model.profile_key.get('model_id') != spec.model_id):
            raise ValueError('mechanism/TP smoke requires its own model and TP profile identity')
        forced = mechanism_pd_plan(len(gpus) // tp) if mode == 'mechanism_pd' else None
        result = run_point(
            model, gpus, tp, 'pdblend', profile, trace, SLO(15, .20), out,
            proxy_port=proxy_port or base_port + 80, base_port=base_port, sampling_seed=701,
            fixed_plan=forced, tp_mode=None if forced else mode, topology_profiles=profiles,
            resident_pools=resident_pools,
            trace_meta=dict(seed=701, trace_sha256=trace_hash, purpose=mode,
                            input_lengths=sorted({request.input_tokens for request in trace})))
        completion['summary'] = result
        if result.get('status') == 'unsupported_engine':
            completion.update(status='unsupported_engine', errors=[result['reason']])
        else:
            errors = []
            path = out / 'outcomes.jsonl'
            outcomes = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if len(outcomes) != len(trace):
                errors.append('outcome count differs from the trace')
            expected = {request.idx: request for request in trace}
            if len({row.get('idx') for row in outcomes}) != len(outcomes):
                errors.append('duplicate outcome ids')
            for row in outcomes:
                request = expected.get(row.get('idx'))
                if (request is None or row.get('error') or row.get('completion_tokens') != request.max_tokens
                        or row.get('sampling_seed') != 701 or row.get('first_token_s') is None
                        or row.get('finished_s') is None):
                    errors.append(f'invalid request outcome: {row.get("idx")}')
                if forced and request and request.input_tokens >= forced.tau and row.get('path') != 'PD':
                    errors.append(f'long request did not exercise P/D: {row.get("idx")}')
            for name in ('power.jsonl', 'routes.jsonl'):
                if not (out / name).is_file() or not (out / name).stat().st_size:
                    errors.append(f'missing {name}')
            meter = out / 'metering.json'
            if not meter.is_file() or json.loads(meter.read_text()).get('error'):
                errors.append('power metering missing or failed')
            if result.get('quarantined_instances'):
                errors.append('resident request ownership is quarantined')
            completion['scope']['pd_path'] = any(row.get('path') == 'PD' for row in outcomes)
            if forced and not completion['scope']['pd_path']:
                errors.append('P/D path was never exercised')
            completion.update(status='passed' if not errors else 'failed', complete=not errors, errors=errors,
                              correctness_status='pending_native_kv_and_golden')
    except Exception as exc:
        completion['errors'].append(f'{type(exc).__name__}: {exc}')
    (out / 'completion.json').write_text(json.dumps(completion, indent=2, default=str) + '\n')
    return completion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--tp', required=True, type=int)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--mode', default='mechanism_pd', choices=(
        'mechanism_pd', 'fixed_tp', 'offline_tp', 'resident_hetero_tp', 'slow_reshard_tp'))
    parser.add_argument('--topology-profile', action='append', default=[], help='TP=/path/to/profile.json')
    parser.add_argument('--resident-layout', type=Path, help='JSON list of pool_id,tp,replicas and optional gpus')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=60)
    parser.add_argument('--base-port', type=int, default=8100)
    parser.add_argument('--proxy-port', type=int)
    parser.add_argument('--without-long', action='store_true', help='Use 512/1024/2048; explicitly omits long-input coverage')
    args = parser.parse_args(argv)
    profiles = {(int(value.split('=', 1)[0]), 1): Path(value.split('=', 1)[1]) for value in args.topology_profile}
    pools = tuple(ResidentPool(row['pool_id'], Topology(int(row['tp']), gpus=tuple(row.get('gpus', ()))),
                               int(row['replicas'])) for row in json.loads(args.resident_layout.read_text())) if args.resident_layout else ()
    result = run_smoke(model=args.model, gpus=[int(value) for value in args.gpus.split(',')], tp=args.tp,
                       profile=args.profile, out=args.out, mode=args.mode, profiles=profiles or None,
                       resident_pools=pools, duration_s=args.duration, base_port=args.base_port,
                       proxy_port=args.proxy_port, include_long=not args.without_long)
    print(json.dumps(dict(status=result['status'], out=str(args.out), errors=result['errors']), indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
