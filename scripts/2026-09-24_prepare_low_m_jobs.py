#!/usr/bin/env python3
"""Freeze the independent low-M experiment and its first two exclusive trials.

Only the highest-frequency nominal trials are queued initially. Further search
requires inspecting their real SLO, physical deployment and energy evidence.
"""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from pdblend.bench.comparison_campaign import binding, load_bound
from pdblend.bench.comparison_jobs import resident_job
from pdblend.bench.low_m_tuning import prepare
from pdblend.bench.pdblend_runtime_options import DEFAULTS, CONTROL_OPTIONS
from pdblend.bench.resident_session import digest, engine_signature, write_new

ROOT = Path(__file__).resolve().parents[1]


def prepare_jobs(parent_path, out, *, after_terminal=()):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    parent = load_bound(binding(parent_path))
    execution = load_bound(parent['execution_inputs'])
    template = next(p for p in parent['points'] if p['name'] == '7b-pdblend-sharegpt-x0.25-seed701')
    spec = importlib.util.spec_from_file_location('low_m_freezer', ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    freezer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(freezer)
    source, revision = freezer.freeze_source(ROOT/'src', out/'sources')
    source_ref = binding(source/'manifest.json')
    files = load_bound(source_ref)['files']
    identity = deepcopy(template['engine_identity'])
    runtime = digest({k:v for k,v in files.items() if k.startswith(('pdblend_runtime/','pdblend/engine/'))})
    if identity['runtime_source_sha256'] != runtime:
        raise ValueError('tuning must preserve the original inference engine')
    identity['measurement_source_sha256'] = digest({k:v for k,v in files.items()
        if k.startswith('pdblend/measure/') or k in ('pdblend/bench/comparison_metrics.py',
            'pdblend/bench/comparison_metering.py','pdblend/bench/client.py')})
    recovery = dict({key:DEFAULTS[key] for key in CONTROL_OPTIONS},
        safety_max_freq=2100, startup_safety=True, deadline_safety=True,
        shield_mode='budget_aware', preserve_overload_capacity=True, safety_recovery=True,
        slo_routing=True, capacity_floor_reserve_canonical=True)
    write_new(out/'recovery.json', recovery)
    manifest = prepare(output=out/'tuning', corpus=ROOT/'datasets/prepared/2026-09-22-7b-v1',
        profile=template['inputs']['profiles'][0]['path'], recovery=recovery, source_root=source)
    group = dict(scope='independent_low_m_tuning/v2', tuning_manifest=binding(out/'tuning/manifest.json'),
        model_id=template['model_id'], engine_identity=identity, engine_signature=engine_signature(identity),
        source_manifest=source_ref, gpu_count=8, exclusive=True, reserve_host=True)
    group['session_id'] = 'low-m-initial-'+digest(group)[:20]
    group_path = out/'groups'/(group['session_id']+'.json')
    write_new(group_path, group)
    job = resident_job(group, group_path, root=ROOT, source=source, image=execution['image_digest'],
        verification=execution['model_verification']['path'], campaign=out/'campaign.json', priority=800)
    argv = job['payload']['argv']
    argv[argv.index('pdblend.bench.comparison_runtime')] = 'pdblend.bench.low_m_tuning_runtime'
    argv[argv.index('--out')] = '--output'
    first = ['r2-m2-f2100-s8801-nominal', 'r4-m3-f2100-s8801-nominal']
    if not set(first) <= {trial['id'] for trial in manifest['trials']}:
        raise ValueError('initial trial IDs missing from immutable manifest')
    argv.extend(['--trial-ids', *first])
    job['payload'].update(system='pdblend', model_id=template['model_id'], after_terminal=list(after_terminal),
        independent_tuning=True, selection_split='tuning', trial_ids=first, timeout_s=7200)
    write_new(out/'jobs.json', [job])
    write_new(out/'campaign.json', dict(scope=group['scope'], parent_campaign=binding(parent_path),
        group=binding(group_path), source_manifest=source_ref, tuning_manifest=group['tuning_manifest'],
        first_trials=first, subsequent_trials_require_observed_decision=True, formal_eligible=False))
    write_new(out/'preparation.json', dict(source_revision=revision, job_id=job['job_id'],
        manifest_trials=len(manifest['trials']), initially_scheduled_trials=first,
        hardware_executed=False, enqueued=False))
    return job


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--after-terminal', nargs='*', default=[])
    args = parser.parse_args()
    job = prepare_jobs(args.parent, args.out, after_terminal=args.after_terminal)
    print(json.dumps(dict(job_id=job['job_id'], trials=job['payload']['trial_ids'])))
