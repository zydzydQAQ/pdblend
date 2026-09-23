"""Frozen resident TP functional qualification; CPU receipts never impersonate GPU output."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from .client import Request
from .run import _make_controller, offline_forecast, run_point
from .tp_runtime import _covers, prepare_tp_runtime
from ..control.planner import SLO
from ..control.policies import get_policy
from ..control.topology import ResidentPool, Topology
from ..proxy.router import ResidentRouter, Router


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_trace(path):
    data = json.loads(Path(path).read_text())
    rows = data['requests']
    if data.get('seed') != 701 or data.get('trace_sha256') != canonical_sha(rows):
        raise ValueError('trace seed or canonical request checksum differs')
    trace = [Request(**row) for row in rows]
    if (not trace or len({r.idx for r in trace}) != len(trace)
            or any(r.arrival_s < 0 or not 0 < r.max_tokens <= 512
                   or not r.prompt or r.input_tokens + r.max_tokens > 8192 for r in trace)
            or any(a.arrival_s > b.arrival_s for a, b in zip(trace, trace[1:]))):
        raise ValueError('invalid qualification trace')
    return trace, data['trace_sha256']


def verify_inputs(args, profiles):
    expected = json.loads(args.input_manifest.read_text())
    paths = {'trace': args.trace, 'resident_layout': args.resident_layout,
             'source_manifest': Path(os.environ['PDBLEND_SOURCE_MANIFEST']),
             'model_verification': Path(os.environ['PDBLEND_MODEL_VERIFICATION_RECEIPT'])}
    paths.update({f'tp{tp}_profile': path for (tp, pp), path in profiles.items()})
    actual = {key: sha(path) for key, path in paths.items()}
    if actual != expected['exact_inputs_sha256']:
        raise ValueError('frozen input checksum differs')
    manifest = json.loads(paths['source_manifest'].read_text())
    source_root = Path(__file__).resolve().parents[2]
    if (manifest['source_sha256'] != canonical_sha(manifest['files'])
            or manifest['source_sha256'] != expected['source_sha256']
            or os.environ['PDBLEND_SOURCE_SHA256'] != expected['source_sha256']
            or os.environ['PDBLEND_IMAGE_ID'] != expected['image_digest']):
        raise ValueError('source or image identity differs')
    for name, digest in manifest['files'].items():
        if sha(source_root / name) != digest:
            raise ValueError(f'frozen source checksum differs: {name}')
    return dict(source_sha256=manifest['source_sha256'], image_digest=expected['image_digest'],
                exact_inputs_sha256=actual, source_files_verified=len(manifest['files']))


def cpu_prepare(args, profiles, pools, trace):
    policy, slo = get_policy('pdblend'), SLO(15, .20)
    topology_pair = sorted(tp for tp, pp in profiles)
    if len(topology_pair) != 2 or len(set(topology_pair)) != 2:
        raise ValueError('qualification requires two independently profiled TP values')
    runtime = prepare_tp_runtime(
        model_name=args.model, gpus=args.gpus, fixed_tp=topology_pair[0], mode='resident_hetero_tp',
        profiles=profiles, policy=policy, forecast=offline_forecast(trace), slo=slo,
        requests=trace, resident_pools=pools, base_port=args.base_port)
    if len(args.gpus) != sum(topology_pair) or sorted(s.tp for s in runtime.specs) != topology_pair:
        raise ValueError('qualification requires one native engine per TP value and its exact GPU budget')
    controllers = {}
    for pool_id, model in runtime.pool_models.items():
        specs = [s for s in runtime.specs if s.pool_id == pool_id]
        fleet = SimpleNamespace(instances={s.instance_id: SimpleNamespace(spec=s) for s in specs})
        router = Router(list(fleet.instances))
        covered = [request for request in trace if _covers(model, (request,))]
        ctl = _make_controller(fleet, router, None, model, policy, slo, covered,
                               args.out / 'pools' / pool_id, 10.0)
        controllers[pool_id] = dict(initial_plan=None if ctl.initial_plan is None else asdict(ctl.initial_plan),
                                    covered_requests=len(covered), profile_sha256=sha(runtime.profile_paths[pool_id]))
    # A CPU admission envelope, explicitly separate from hardware route receipts.
    # No routing preference, synthetic outcomes, meters or GPU completion is created.
    largest_burst = max((list(group) for _, group in itertools.groupby(trace, key=lambda r: r.arrival_s)), key=len)
    envelope = []
    pool_ids = list(runtime.pool_models)
    for frequencies in itertools.product(*(runtime.pool_models[k].freqs for k in pool_ids)):
        frequency = dict(zip(pool_ids, frequencies))
        children = {}
        for pool_id in pool_ids:
            specs = [s for s in runtime.specs if s.pool_id == pool_id]
            metadata = {s.instance_id: dict(tp=s.tp, pp=s.pp, pool_id=s.pool_id, generation=s.generation,
                                           profile_key=s.profile_key, model_id=args.model) for s in specs}
            children[pool_id] = Router(list(metadata), instance_metadata=metadata)
        router = ResidentRouter(children, runtime.pool_models)
        router.frequency_provider = lambda iid: frequency[router._owners[iid]]
        counts = Counter()
        for request in largest_burst:
            row = router.dispatch(f'r{request.idx}', request.input_tokens, request.max_tokens)
            if row is None:
                raise ValueError('CPU admission envelope rejects burst request')
            counts[row.tp] += 1
        if set(counts) != set(topology_pair):
            raise ValueError('CPU admission envelope does not exercise both TP values')
        envelope.append(dict(frequency_mhz=frequency, admitted_by_tp=dict(counts)))
    return runtime, dict(status='cpu_preflight_passed', hardware_executed=False,
                         formal_eligible=False, energy_comparable=False,
                         resident_specs=[asdict(s) for s in runtime.specs],
                         controllers=controllers, admission_envelope=envelope,
                         admission_envelope_is_gpu_evidence=False)


def audit_gpu_result(out, trace, runtime, summary):
    """Reject successful single-pool runs and inconsistent per-request ownership."""
    errors = []
    read = lambda name: [json.loads(line) for line in (out / name).read_text().splitlines() if line.strip()]
    outcomes, routes, ownership = (read(name) for name in ('outcomes.jsonl', 'routes.jsonl', 'resident-routes.jsonl'))
    expected = {r.idx: r for r in trace}
    route_by_id = {row['request_id']: row for row in routes}
    own_by_id = {row['request_id']: row for row in ownership}
    request_ids = {f'r{idx}' for idx in expected}
    if (len(outcomes) != len(trace) or {row.get('idx') for row in outcomes} != set(expected)
            or len(routes) != len(trace) or set(route_by_id) != request_ids
            or len(ownership) != len(trace) or set(own_by_id) != request_ids):
        errors.append('request/outcome/route ownership sets differ')
    specs = {s.instance_id: s for s in runtime.specs}
    counts = Counter()
    for row in outcomes:
        request = expected.get(row.get('idx'))
        route, owner = route_by_id.get(f'r{row.get("idx")}'), own_by_id.get(f'r{row.get("idx")}')
        spec = specs.get(row.get('decode'))
        if (request is None or row.get('error') or row.get('completion_tokens') != request.max_tokens
                or row.get('sampling_seed') != 701 or row.get('first_token_s') is None
                or row.get('finished_s') is None or not route or not owner or spec is None):
            errors.append(f'invalid output/route: {row.get("idx")}')
            continue
        identity = dict(tp=spec.tp, pp=spec.pp, pool_id=spec.pool_id, generation=spec.generation)
        if (any(route.get(k) != v or owner.get(k) != v for k, v in identity.items())
                or route.get('profile_key') != spec.profile_key
                or owner.get('profile_keys') != [spec.profile_key]
                or route.get('prefill_instance') != spec.instance_id
                or owner.get('prefill_instance') != spec.instance_id
                or route.get('decode_instance') != spec.instance_id
                or owner.get('decode_instance') != spec.instance_id
                or row.get('prefill') != spec.instance_id
                or row.get('path') != 'M' or route.get('path') != 'M' or owner.get('path') != 'M'
                or owner.get('model_id') != Path(spec.model).name
                or owner.get('input_tokens') != request.input_tokens
                or owner.get('max_tokens') != request.max_tokens):
            errors.append(f'profile/topology/generation ownership differs: {request.idx}')
        counts[spec.tp] += 1
    required_tps = {s.tp for s in runtime.specs}
    if set(counts) != required_tps or any(counts[tp] < 2 for tp in required_tps):
        errors.append('both TP values require at least two completed real requests')
    meter = json.loads((out / 'metering.json').read_text())
    if meter.get('error') or meter.get('power_samples', 0) < 2 or len(read('power.jsonl')) < 2:
        errors.append('power metering missing or failed')
    if summary.get('quarantined_instances') or set(summary.get('startup_s', {})) != set(specs):
        errors.append('missing native startup or quarantined ownership')
    return errors, dict(completed_by_tp=dict(counts), routes=len(routes), outcomes=len(outcomes),
                       resident_ownership=len(ownership))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--higher-tp', type=int, default=2)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--topology-profile', action='append', required=True)
    parser.add_argument('--resident-layout', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--base-port', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    result = dict(status='failed', complete=False, hardware_executed=False, errors=[],
                  formal_eligible=False, energy_comparable=False,
                  scope=dict(resident_dual_tp=False, dynamic_tp=False, complete_kv=False, output_golden=False))
    receipt = args.out / ('preflight.json' if args.preflight_only else 'completion.json')
    try:
        args.gpus = [int(value) for value in args.gpus.split(',')]
        profiles = {(int(value.split('=', 1)[0]), 1): Path(value.split('=', 1)[1]) for value in args.topology_profile}
        if (len(args.topology_profile) != 2 or set(profiles) != {(args.tp, 1), (args.higher_tp, 1)}
                or args.tp >= args.higher_tp or args.profile != profiles[args.tp, 1]):
            raise ValueError('exactly two ordered independent TP profile arguments required')
        identity = verify_inputs(args, profiles)
        trace, trace_hash = read_trace(args.trace)
        pools = tuple(ResidentPool(row['pool_id'], Topology(int(row['tp']), gpus=tuple(row.get('gpus', ()))),
                                   int(row['replicas'])) for row in json.loads(args.resident_layout.read_text()))
        runtime, preflight = cpu_prepare(args, profiles, pools, trace)
        result.update(identity, trace_sha256=trace_hash, trace_file_sha256=sha(args.trace), requests=len(trace))
        if args.preflight_only:
            result.update(preflight)
        else:
            (args.out / 'trace.json').write_bytes(args.trace.read_bytes())
            summary = run_point(args.model, args.gpus, args.tp, 'pdblend', args.profile, trace, SLO(15, .20), args.out,
                                proxy_port=args.base_port + 80, base_port=args.base_port, sampling_seed=701,
                                tp_mode='resident_hetero_tp', topology_profiles=profiles, resident_pools=pools,
                                trace_meta=dict(seed=701, trace_sha256=trace_hash, trace_file_sha256=sha(args.trace),
                                                purpose='resident_dual_tp_functional', **identity))
            result['hardware_executed'] = True
            errors, evidence = audit_gpu_result(args.out, trace, runtime, summary)
            result.update(status='passed' if not errors else 'failed', complete=not errors,
                          errors=errors, evidence=evidence)
            result['scope']['resident_dual_tp'] = not errors
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    receipt.write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps(dict(status=result['status'], receipt=str(receipt), errors=result['errors']), indent=2))
    return 0 if result['status'] in ('passed', 'cpu_preflight_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
