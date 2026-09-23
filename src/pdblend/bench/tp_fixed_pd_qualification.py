"""Fixed/offline TP and symmetric P/D windows on one native fleet lifecycle."""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from .metering import Gpus
from .run import _make_controller, _point, offline_forecast
from .tp_qualification import read_trace, sha, verify_inputs
from .tp_runtime import mechanism_pd_plan, prepare_tp_runtime
from ..control.planner import SLO
from ..control.policies import get_policy
from ..engine.launcher import Fleet
from ..proxy.router import Router


MODES = ('fixed_tp', 'offline_tp', 'mechanism_pd')


def prepare(args, profiles, trace):
    if len(args.gpus) != 2 * args.tp or set(profiles) != {(args.tp, 1)}:
        raise ValueError('one independent TP profile and exactly two symmetric native groups required')
    policy, slo = get_policy('pdblend'), SLO(15, .20)
    runtimes, controllers = {}, {}
    for mode in MODES:
        runtime = prepare_tp_runtime(model_name=args.model, gpus=args.gpus, fixed_tp=args.tp,
                                     mode='fixed_tp' if mode == 'mechanism_pd' else mode,
                                     profiles=profiles, policy=policy, forecast=offline_forecast(trace),
                                     slo=slo, requests=trace, base_port=args.base_port)
        if runtime.selected_plan.tp != args.tp or len(runtime.specs) != 2:
            raise ValueError('fixed/offline selection changed the qualified native fleet')
        runtimes[mode] = runtime
        fleet = SimpleNamespace(instances={s.instance_id: SimpleNamespace(spec=s) for s in runtime.specs})
        ctl = _make_controller(fleet, Router(list(fleet.instances)), None, runtime.model, policy, slo,
                               trace, args.out / mode, 10.0,
                               fixed_plan=mechanism_pd_plan(2) if mode == 'mechanism_pd' else None,
                               initial_plan=None if mode == 'mechanism_pd' else runtime.selected_plan)
        controllers[mode] = dict(initial_plan=asdict(ctl.initial_plan), freeze=ctl.freeze,
                                 policy_decision=mode != 'mechanism_pd',
                                 policy_m_floor=runtime.metadata['policy_m_floor'])
    if any(runtime.specs != runtimes['fixed_tp'].specs for runtime in runtimes.values()):
        raise ValueError('windows do not share exactly the same native specs')
    return runtimes, dict(status='cpu_preflight_passed', complete=False, hardware_executed=False,
                          source_profile_sha256=sha(args.profile), controllers=controllers,
                          native_specs=[asdict(s) for s in runtimes['fixed_tp'].specs],
                          shared_load=True, eligible_offline_topologies=[dict(tp=args.tp, pp=1)],
                          offline_scope='selection among topology profiles legal for this four-GPU symmetric-pair budget')


def audit_window(path, trace, runtime, summary, mode):
    read = lambda name: [json.loads(line) for line in (path / name).read_text().splitlines() if line.strip()]
    outcomes, routes = read('outcomes.jsonl'), read('routes.jsonl')
    expected = {request.idx: request for request in trace}
    by_id = {row['request_id']: row for row in routes}
    specs = {s.instance_id: s for s in runtime.specs}
    errors, served = [], set()
    if (len(outcomes) != len(trace) or {row.get('idx') for row in outcomes} != set(expected)
            or len(routes) != len(trace) or set(by_id) != {f'r{idx}' for idx in expected}):
        errors.append('request/outcome/route sets differ')
    for row in outcomes:
        request, route = expected.get(row.get('idx')), by_id.get(f'r{row.get("idx")}')
        if (not request or not route or row.get('error') or row.get('completion_tokens') != request.max_tokens
                or row.get('sampling_seed') != 701 or row.get('first_token_s') is None
                or row.get('finished_s') is None):
            errors.append(f'invalid completed output: {row.get("idx")}')
            continue
        p, d = specs.get(row.get('prefill')), specs.get(row.get('decode'))
        required_path = 'PD' if mode == 'mechanism_pd' else 'M'
        if (p is None or d is None or route.get('path') != required_path or row.get('path') != required_path
                or route.get('prefill_instance') != row.get('prefill')
                or route.get('decode_instance') != row.get('decode')):
            errors.append(f'native route differs: {request.idx}')
            continue
        if ((p.tp, p.pp, p.pool_id, p.generation, p.profile_key) !=
                (d.tp, d.pp, d.pool_id, d.generation, d.profile_key)
                or any(route.get(k) != getattr(d, k) for k in ('tp', 'pp', 'pool_id', 'generation', 'profile_key'))
                or (mode == 'mechanism_pd' and (p.instance_id == d.instance_id or set(p.gpus) & set(d.gpus)))):
            errors.append(f'symmetric TP/profile/generation differs: {request.idx}')
        served.update((p.instance_id, d.instance_id))
    if served != set(specs):
        errors.append('both native instances must serve this window')
    meter = json.loads((path / 'metering.json').read_text())
    if meter.get('error') or meter.get('power_samples', 0) < 2 or len(read('power.jsonl')) < 2:
        errors.append('native power metering missing or failed')
    if summary.get('quarantined_instances'):
        errors.append('request ownership quarantined')
    return errors, dict(completed_requests=len(outcomes), real_route_count=len(routes), native_instances=sorted(served),
                        path='PD' if mode == 'mechanism_pd' else 'M',
                        policy_decision=mode != 'mechanism_pd', formal_eligible=False)


def run_shared(args, runtimes, trace, identity):
    specs = runtimes['fixed_tp'].specs
    meter = Gpus(args.gpus)
    meter.reset_all()
    windows, errors = {}, []
    fleet = Fleet(specs, args.out / 'logs')
    try:
        with fleet:
            startup = {}
            for iid, instance in fleet.instances.items():
                instance.start()
                startup[iid] = instance.wait_ready()
            pids = {iid: instance.process.pid for iid, instance in fleet.instances.items()}
            for mode in MODES:
                path = args.out / mode
                path.mkdir(parents=True)
                runtime = runtimes[mode]
                (path / 'trace.json').write_bytes(args.trace.read_bytes())
                (path / 'tp-runtime.json').write_text(json.dumps(runtime.metadata, indent=2) + '\n')
                summary = asyncio.run(_point(fleet, meter, runtime.model, get_policy('pdblend'), SLO(15, .20),
                                             trace, [], path, args.base_port + 80, 10.0, 300.0,
                                             fixed_plan=mechanism_pd_plan(2) if mode == 'mechanism_pd' else None,
                                             sampling_seed=701,
                                             initial_plan=None if mode == 'mechanism_pd' else runtime.selected_plan))
                summary.update(model=args.model, tp=args.tp, tp_mode=mode, startup_s=startup,
                               fleet_events=fleet.events(), source_profile_sha256=sha(args.profile),
                               source=identity, trace_file_sha256=sha(args.trace), formal_eligible=False,
                               energy_comparable=False, shared_lifecycle_pids=pids,
                               policy_decision=mode != 'mechanism_pd')
                (path / 'summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
                problems, evidence = audit_window(path, trace, runtime, summary, mode)
                if any(instance.process.pid != pids[iid] or not instance.alive()
                       for iid, instance in fleet.instances.items()):
                    problems.append('native process identity changed between shared windows')
                receipt = dict(status='failed' if problems else 'passed', complete=not problems,
                               hardware_executed=True, errors=problems, evidence=evidence, **identity)
                (path / 'completion.json').write_text(json.dumps(receipt, indent=2) + '\n')
                windows[mode] = receipt
                errors += [f'{mode}: {message}' for message in problems]
                if problems:
                    break
    finally:
        meter.reset_all()
        (args.out / 'fleet-lifecycle.json').write_text(json.dumps(dict(events=fleet.events(),
                                    expected_native_start_count=2, shared_load=True), indent=2) + '\n')
    if set(windows) != set(MODES):
        errors.append('not all fixed/offline/P-D windows completed')
    return windows, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--tp', type=int, required=True)
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
    args.gpus = [int(value) for value in args.gpus.split(',')]
    args.out.mkdir(parents=True, exist_ok=True)
    result = dict(status='failed', complete=False, hardware_executed=False, formal_eligible=False,
                  energy_comparable=False, errors=[], scope=dict(dynamic_tp=False, complete_kv=False, output_golden=False))
    try:
        profiles = {(int(value.split('=', 1)[0]), 1): Path(value.split('=', 1)[1]) for value in args.topology_profile}
        if len(args.topology_profile) != 1 or profiles.get((args.tp, 1)) != args.profile:
            raise ValueError('one explicit independent topology profile required')
        identity = verify_inputs(args, profiles)
        trace, trace_hash = read_trace(args.trace)
        runtimes, preflight = prepare(args, profiles, trace)
        result.update(identity, trace_sha256=trace_hash, trace_file_sha256=sha(args.trace), requests_per_window=len(trace))
        if args.preflight_only:
            result.update(preflight)
        else:
            windows, errors = run_shared(args, runtimes, trace, identity)
            result.update(windows=windows, errors=errors, status='failed' if errors else 'passed',
                          complete=not errors, hardware_executed=True)
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    receipt = args.out / ('preflight.json' if args.preflight_only else 'completion.json')
    receipt.write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps(dict(status=result['status'], receipt=str(receipt), errors=result['errors']), indent=2))
    return 0 if result['status'] in ('passed', 'cpu_preflight_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
