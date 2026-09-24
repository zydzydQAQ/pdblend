"""One exclusive lease: recover the 32B anchor, freeze, then measure Mixed.

The parent campaign and all existing traces remain immutable. Only a clean,
fully measured exhaustion of the predeclared SLO candidates permits fallback.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from . import longbench_anchor_recovery as recovery
from .client import load_split, poisson_trace
from .comparison_campaign import group_points, point_order, PROTOCOL, SCALES
from .resident_session import ResidentGroupSession, digest, file_sha, write_new

SCHEMA = 'longbench-mixed-combined/v1'
OVERLAYS = ('pdblend/bench/longbench_anchor_recovery.py', 'pdblend/bench/longbench_mixed_combo.py')
FLOOR_ERROR = 'RuntimeError: LongBench recovery floor reached without independent tuning confirmation'
BLOCKER = 'missing_same_model_independent_tuning_anchor'
binding, bound = recovery.binding, recovery.bound


def validate_source(parent_ref, execution_ref):
    """All parent implementation bytes, except these two wrappers, are fixed."""
    parent, execution = bound(parent_ref), bound(execution_ref)
    for ref, manifest in ((parent_ref, parent), (execution_ref, execution)):
        if digest(manifest['files']) != manifest['source_sha256']:
            raise ValueError('source manifest content address differs')
        root = Path(ref['path']).parent
        for name, sha in manifest['files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts or file_sha(root/name) != sha:
                raise ValueError('source file checksum differs: '+name)
    expected = set(parent['files']) | set(OVERLAYS)
    if set(execution['files']) != expected:
        raise ValueError('combined source contains an undeclared addition/removal')
    protected = {k:v for k,v in parent['files'].items() if k not in OVERLAYS}
    if any(execution['files'][k] != v for k,v in protected.items()):
        raise ValueError('frozen parent implementation changed')
    return dict(passed=True, protected_files=len(protected), protected_files_sha256=digest(protected),
                parent=parent_ref, execution=execution_ref, allowed_overlays=list(OVERLAYS))


def validate_baseline_source(baseline_ref, execution_ref):
    baseline, execution = bound(baseline_ref), bound(execution_ref)
    def protected(name):
        return (name.startswith(('pdblend_runtime/', 'pdblend/engine/', 'pdblend/measure/', 'pdblend_baselines/mixed/'))
                or name in ('pdblend/bench/client.py', 'pdblend/bench/metering.py',
                    'pdblend/bench/comparison_metrics.py', 'pdblend/bench/comparison_metering.py',
                    'pdblend/bench/native_mixed.py', 'pdblend_baselines/mixed_policy.py'))
    expected = {k:v for k,v in baseline['files'].items() if protected(k)}
    actual = {k:v for k,v in execution['files'].items() if protected(k)}
    if not expected or actual != expected:
        raise ValueError('frozen baseline engine/measurement/Mixed policy bytes changed')
    return dict(passed=True, baseline=baseline_ref, protected_files=len(expected), protected_files_sha256=digest(expected))


def parent_group(campaign):
    if (len(campaign['points']) != 180 or len({p['name'] for p in campaign['points']}) != 180
            or campaign['seed'] != 701 or campaign['duration_s'] != 150
            or tuple(campaign['scales']) != SCALES or campaign['measurement_protocol_version'] != PROTOCOL):
        raise ValueError('parent must be the complete fixed 180-point comparison')
    groups = [g for g in group_points(campaign['points']) if g['model_id'] == recovery.MODEL
              and all(p['system'] == 'mixed' for p in g['points'])]
    if (len(groups) != 1 or len(groups[0]['points']) != 8
            or {p['dataset'] for p in groups[0]['points']} != set(recovery.INHERITED)):
        raise ValueError('parent must retain exactly the original eight prepared 32B Mixed points')
    group = groups[0]
    for point in group['points']:
        trace = bound(point['trace'])
        if (trace['model_id'] != recovery.MODEL or trace['dataset'] != point['dataset']
                or trace['rate_rps'] != point['rate_rps'] or trace['seed'] != 701
                or trace['duration_s'] != 150 or trace['slo'] != point['slo']):
            raise ValueError('inherited evaluation trace differs')
    if any(p.get('trace') or BLOCKER not in p['blockers'] for p in campaign['points']
           if p['model_id'] == recovery.MODEL and p['dataset'] == 'longbench'):
        raise ValueError('parent LongBench rows are not the blocked missing anchor')
    return group


def recovery_args(args, plan):
    value = deepcopy(args)
    value.plan = Path(plan['recovery_plan']['path'])
    value.plan_sha256 = plan['recovery_plan']['sha256']
    value.out = args.out/'anchor'
    return value


def preflight(args):
    plan = json.loads(args.plan.read_text())
    if file_sha(args.plan) != args.plan_sha256 or plan.get('schema') != SCHEMA:
        raise ValueError('combined plan checksum/schema differs')
    parent = bound(plan['parent_campaign']); group = parent_group(parent)
    if digest(group) != plan['parent_group_sha256']:
        raise ValueError('parent group identity differs')
    protection = validate_source(plan['parent_source_manifest'], plan['source_manifest'])
    baseline_protection = validate_baseline_source(plan['baseline_source_manifest'], plan['source_manifest'])
    audit = recovery.preflight(recovery_args(args, plan))
    if (audit['source_manifest'] != plan['source_manifest']
            or audit['model_hash'] != group['engine_identity']['model_hash']
            or audit['tokenizer_hash'] != group['engine_identity']['tokenizer_hash']):
        raise ValueError('recovery and comparison execution identities differ')
    return dict(schema=SCHEMA, status='cpu_preflight_passed', hardware_executed=False,
        plan=binding(args.plan), parent_campaign=plan['parent_campaign'], source_protection=protection,
        baseline_source_protection=baseline_protection,
        inherited_points=[dict(name=p['name'], sha256=digest(p), trace=p['trace']) for p in group['points']],
        conditional_points=dict(confirmed_anchor=12, clean_slo_exhaustion=8), recovery=audit)


def validate_window(summary, root, split, rate):
    relative = Path(summary['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('recovery window escaped its output root')
    measured = bound(dict(path=str(root/relative), sha256=summary['sha256']))
    m = measured['metrics']; passed = m['passed']
    seed, duration = (9701, 60.) if split == 'calibration' else (9702, 120.)
    if (type(passed) is not bool or measured.get('system') != 'mixed'
            or measured.get('dataset') != 'longbench' or measured.get('split') != split
            or measured.get('seed') != seed or measured.get('duration_s') != duration
            or measured.get('rate_rps') != rate or summary.get('metrics') != m
            or summary.get('split') != split or summary.get('rate_rps') != rate
            or summary.get('seed') != seed or summary.get('duration_s') != duration
            or m.get('offered', 0) <= 0 or m.get('correct') != m['offered']
            or m.get('success_rate') != 1. or measured.get('counts_reclaimed') is not True
            or (m.get('slo_ttft_s'), m.get('slo_tpot_s')) != (15., .2)
            or len(measured.get('drain', [])) != 4
            or any(r.get('drain', {}).get('drained') is not True for r in measured['drain'])
            or passed != (m.get('joint_slo_rate', 0) >= .9 and m.get('ttft_p99_s', float('inf')) <= 15.
                          and m.get('tpot_p99_s', float('inf')) <= .2)):
        raise ValueError('recovery window is not a complete measured SLO-only result')
    requests = root/relative.parent/'requests.json'
    if file_sha(requests) != measured['trace_sha256']:
        raise ValueError('recovery window request checksum differs')
    workload = json.loads(requests.read_text())
    if workload.get('seed') != seed or workload.get('duration_s') != duration or len(workload.get('requests', [])) != m['offered']:
        raise ValueError('recovery request count or independent split seed differs')
    return passed


def classify_recovery(result, audit, root):
    if (result.get('hardware_executed') is not True or result.get('cleanup_errors')
            or result.get('inherited_anchors') != audit['inherited_anchors']
            or any(result.get('anchors', {}).get(k) != v for k,v in audit['inherited_anchors'].items())):
        raise ValueError('recovery hardware/cleanup or inherited anchors are invalid')
    rows = result.get('candidates', []); rates = audit['candidate_rates_rps']
    if not rows or len(rows) > len(rates):
        raise ValueError('recovery candidate inventory differs')
    winner = None
    for index, row in enumerate(rows):
        if row.get('index') != index or row.get('rate_rps') != rates[index] or 'error' in row:
            raise ValueError('recovery candidate failed or differs from predeclared schedule')
        calibrated = validate_window(row['calibration'], root, 'calibration', rates[index])
        confirmed = validate_window(row['tuning'], root, 'tuning', rates[index]) if calibrated else False
        expected = 'confirmed' if confirmed else 'tuning_failed' if calibrated else 'calibration_failed'
        if row.get('status') != expected or (not calibrated and 'tuning' in row):
            raise ValueError('recovery candidate state differs')
        if confirmed:
            if index != len(rows)-1:
                raise ValueError('recovery continued after confirmation')
            winner = row
    if winner:
        anchor = result.get('anchors', {}).get('longbench', {})
        if (result.get('status') != 'passed' or result.get('complete') is not True or result.get('error')
                or anchor.get('base_rate_rps') != winner['rate_rps']
                or anchor.get('confirmation_sha256') != winner['tuning']['sha256']
                or anchor.get('confirmation_path') != '/output/anchor/'+winner['tuning']['path']):
            raise ValueError('confirmed anchor final receipt differs')
        return 'confirmed'
    if (len(rows) != len(rates) or result.get('status') != 'failed' or result.get('complete') is not False
            or result.get('error') != FLOOR_ERROR or 'longbench' in result.get('anchors', {})):
        raise ValueError('recovery did not end in clean predeclared SLO exhaustion')
    return 'slo_exhausted'


async def verify_idle(expected):
    """Read-only physical check after the recovery's original owned cleanup."""
    from .metering import Gpus
    gpus = Gpus(range(8), power_mode='instant'); nvml = gpus.backend._nvml
    actual = [nvml.nvmlDeviceGetUUID(gpus.backend._handle(i)) for i in range(8)]
    actual = [u.decode() if isinstance(u, bytes) else u for u in actual]
    if actual != expected or os.environ.get('PDBLEND_GPU_UUIDS', '').split(',') != expected:
        raise RuntimeError('physical lease changed between anchor and comparison')
    deadline = time.monotonic()+30
    while True:
        processes = {str(i):[p.pid for p in nvml.nvmlDeviceGetComputeRunningProcesses(gpus.backend._handle(i))]
                     for i in range(8)}
        if not any(processes.values()):
            return dict(passed=True, gpu_uuids=actual, processes=processes, checked_s=time.time())
        if time.monotonic() >= deadline:
            raise RuntimeError('recovery left physical compute processes: '+repr(processes))
        await asyncio.sleep(.25)


def freeze_overlay(plan, result, audit, outcome, corpus, out):
    # Classification is repeated at the boundary where evaluation becomes readable.
    if classify_recovery(result, audit, out/'anchor') != outcome:
        raise ValueError('recovery outcome changed before freeze')
    parent = bound(plan['parent_campaign']); original = parent_group(parent)
    campaign = deepcopy(parent); traces = []
    if outcome == 'confirmed':
        anchor = result['anchors']['longbench']
        corpus_manifest = json.loads((corpus/'manifest.json').read_text())
        if (file_sha(corpus/'manifest.json') != audit['corpus_manifest_sha256']
                or file_sha(corpus/'longbench.json') != audit['corpus_sha256']['longbench']):
            raise ValueError('LongBench corpus changed after independent tuning')
        # Read evaluation only after the qualified independent tuning receipt.
        records = load_split(corpus, 'longbench', 'evaluation')
        for scale in SCALES:
            rate = anchor['base_rate_rps']*scale
            requests = [asdict(r) for r in poisson_trace(records, rate, 150., 701, 'longbench')]
            if not requests:
                raise ValueError('fixed evaluation trace is empty; do not alter seed/rate/window')
            trace = dict(schema='five-system-evaluation-trace-v1', model_id=recovery.MODEL,
                dataset='longbench', duration_s=150., rate_rps=rate, seed=701, single_seed=True,
                seed_policy='single_seed_701', selection_split='evaluation', slo=dict(ttft_s=15., tpot_s=.2),
                requests=requests, measurement_protocol_version=PROTOCOL,
                corpus_sha256=audit['corpus_sha256']['longbench'],
                corpus_manifest_sha256=audit['corpus_manifest_sha256'],
                corpus_tokenizer_sha256=corpus_manifest['tokenizer_sha256'],
                output_workload='same model-tokenized capped reference length for all systems',
                anchor=dict(binding(out/'anchor/completion.json'), base_rate_rps=anchor['base_rate_rps'],
                    confirmation_path=str((out/'anchor'/result['candidates'][-1]['tuning']['path']).resolve()),
                    confirmation_sha256=anchor['confirmation_sha256'], cleanup_verified=True,
                    dataset_confirmation_passed=True, scope=anchor['scope'], capacity_exact=False))
            path = out/'traces'/f'32b-longbench-x{scale:g}-seed701.json'; write_new(path, trace)
            ref = dict(binding(path), requests=len(requests)); traces.append(ref)
            for point in campaign['points']:
                if point['model_id'] == recovery.MODEL and point['dataset'] == 'longbench' and point['scale'] == scale:
                    point.update(trace=ref, rate_rps=rate,
                        blockers=[b for b in point['blockers'] if b != BLOCKER])
                    point['status'] = 'blocked' if point['blockers'] else 'prepared'
    for old in original['points']:
        current = next(p for p in campaign['points'] if p['name'] == old['name'])
        if current != old or file_sha(old['trace']['path']) != old['trace']['sha256']:
            raise ValueError('original eight points/traces changed')
    campaign.update(campaign_id=out.name, parent_campaign=plan['parent_campaign'],
        combined_plan=binding(Path(plan['_path'])), anchor_outcome=outcome,
        execution_source_manifest=plan['source_manifest'])
    campaign['points'].sort(key=point_order); campaign['traces'] += traces
    campaign['groups'] = group_points(campaign['points'])
    campaign['summary'].update(trace_sets=len(campaign['traces']),
        prepared_points=sum(not p['blockers'] for p in campaign['points']), resident_sessions=len(campaign['groups']))
    group = next(g for g in campaign['groups'] if g['model_id'] == recovery.MODEL
                 and all(p['system'] == 'mixed' for p in g['points']))
    if len(group['points']) != (12 if outcome == 'confirmed' else 8):
        raise ValueError('combined group point count differs')
    write_new(out/'campaign.json', campaign); write_new(out/'group.json', group)
    write_new(out/'freeze.json', dict(schema=SCHEMA, frozen_s=time.time(),
        parent_campaign=plan['parent_campaign'], campaign=binding(out/'campaign.json'), group=binding(out/'group.json'),
        anchor=binding(out/'anchor/completion.json'), anchor_outcome=outcome,
        inherited_points=[dict(name=p['name'], sha256=digest(p), trace=p['trace']) for p in original['points']],
        longbench_blocker=BLOCKER if outcome == 'slo_exhausted' else None,
        evaluation_has_started=False, point_count=len(group['points'])))
    return group


async def run(args):
    audit = preflight(args); args.out.mkdir(parents=True, exist_ok=False)
    write_new(args.out/'preflight.json', audit)
    plan = bound(audit['plan']); plan['_path'] = str(args.plan.resolve())
    report = dict(schema=SCHEMA, status='failed', complete=False, started_s=time.time(),
                  parent_campaign=plan['parent_campaign'], phases=[], service_energy_includes_extra_load=False)
    try:
        started = time.time(); recovered = await recovery.run(recovery_args(args, plan))
        report['phases'].append(dict(phase='anchor_recovery', started_s=started, finished_s=time.time(),
            completion=binding(args.out/'anchor/completion.json'),
            engine_loads=recovered.get('engine_loads', 0), engine_load_s=recovered.get('engine_load_s'),
            engine_load_cycles=int(recovered.get('engine_loads', 0)>0), allocated_to_service=False))
        outcome = classify_recovery(recovered, audit['recovery'], args.out/'anchor')
        expected = parent_group(bound(plan['parent_campaign']))['engine_identity']['fleet_gpu_uuids']
        report['anchor_cleanup_boundary'] = await verify_idle(expected)
        group = freeze_overlay(plan, recovered, audit['recovery'], outcome, args.corpus, args.out)
        report.update(anchor_outcome=outcome, longbench_blocker=BLOCKER if outcome == 'slo_exhausted' else None,
                      campaign=binding(args.out/'campaign.json'), freeze=binding(args.out/'freeze.json'))
        from .comparison_runtime import NativeResidentAdapter
        session_out = args.out/'session'
        started = time.time()
        session = ResidentGroupSession(group, NativeResidentAdapter(session_out, base_port=args.base_port), session_out)
        measured = await session.run()
        report['phases'].append(dict(phase='mixed_comparison', started_s=started, finished_s=time.time(),
            completion=binding(session_out/'completion.json'), **{k:measured.get('startup', {}).get(k)
                for k in ('engine_loads', 'engine_load_s', 'engine_load_cycles')}, allocated_to_service=False))
        if not measured['complete']:
            raise RuntimeError('resident comparison failed: '+str(measured.get('error', measured.get('cleanup_errors'))))
        report.update(status='passed', complete=True, measured_points=len(measured['windows']))
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        report.update(finished_s=time.time(), total_engine_loads=sum(p.get('engine_loads') or 0 for p in report['phases']),
            total_engine_load_cycles=sum(p.get('engine_load_cycles') or 0 for p in report['phases']))
        write_new(args.out/'completion.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--model', required=True); parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--gpus', type=lambda v:[int(x) for x in v.split(',')], required=True)
    parser.add_argument('--corpus', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base-port', type=int, required=True); parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.preflight_only:
        result = preflight(args); write_new(args.out/'preflight.json', result)
    else:
        result = asyncio.run(run(args))
    print(json.dumps({k:result.get(k) for k in ('status', 'complete', 'error', 'anchor_outcome', 'measured_points')}))
    return 0 if args.preflight_only or result['complete'] else 2


if __name__ == '__main__': raise SystemExit(main())
