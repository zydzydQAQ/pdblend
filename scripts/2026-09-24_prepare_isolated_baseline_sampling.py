#!/usr/bin/env python3
"""Prepare process-isolated sampling for unstarted native comparison jobs."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

from pdblend.bench.comparison_campaign import binding, group_points, load_bound
from pdblend.bench.comparison_jobs import resident_job
from pdblend.bench.resident_session import digest, write_new

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--job', action='append', required=True)
    parser.add_argument('--overlay', action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    base_ref = binding(args.base)
    base = load_bound(base_ref)
    queue = json.loads((ROOT/'results/2026-09-22/three-model/queue.json').read_text())
    selected, old_jobs = {}, []
    for job_id in args.job:
        job = queue['jobs'][job_id]
        if job['status'] != 'queued' or job['attempts'] != 0 or job.get('lease_id'):
            raise ValueError('sampling replacement requires an unstarted queued job')
        argv = job['payload']['argv']
        group = load_bound(binding(argv[argv.index('--group')+1]))
        if {p['system'] for p in group['points']} not in ({'distserve'}, {'dynamollm'}):
            raise ValueError('only original native DistServe/Dynamo wrappers may be selected')
        for point in group['points']:
            if point != next(p for p in base['points'] if p['name'] == point['name']):
                raise ValueError('queued point differs from parent campaign')
            if point['name'] in selected:
                raise ValueError('duplicate unstarted point')
            selected[point['name']] = job_id
        old_jobs.append(job)
    refs = {p['source_manifest']['path'] for p in base['points'] if p['name'] in selected}
    if len(refs) != 1:
        raise ValueError('selected jobs do not share one frozen source base')
    parent_ref = binding(next(iter(refs)))
    parent_source = Path(parent_ref['path']).parent
    parent_files = load_bound(parent_ref)['files']
    spec = importlib.util.spec_from_file_location('sampling_source_freezer', ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    freezer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(freezer)
    freezer.verify_snapshot(parent_source, parent_files)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    staged = out/'staged-source'
    for name in parent_files:
        target = staged/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent_source/name, target)
    for name in args.overlay:
        target = staged/name
        if not target.resolve().is_relative_to(staged.resolve()) or not name.startswith('pdblend/bench/'):
            raise ValueError('sampling wrapper overlay escapes its scope')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/'src'/name, target)
    source, revision = freezer.freeze_source(staged, out/'sources')
    shutil.rmtree(staged)
    source_ref = binding(source/'manifest.json')
    files = load_bound(source_ref)['files']
    if any(files.get(k) != v for k, v in parent_files.items() if k not in args.overlay):
        raise ValueError('non-sampling implementation changed')
    protected = [k for k in parent_files if k.startswith(('pdblend_baselines/', 'pdblend_runtime/',
                 'pdblend/engine/', 'pdblend/measure/')) or k in (
                 'pdblend/bench/client.py', 'pdblend/bench/comparison_metrics.py',
                 'pdblend/bench/comparison_metering.py', 'pdblend/bench/isolated_comparison_meter.py')]
    if any(files.get(k) != parent_files[k] for k in protected):
        raise ValueError('native algorithm, sampler or numerical measurement method changed')
    points = deepcopy(base['points'])
    for point in points:
        if point['name'] not in selected:
            continue
        point.update(metering_execution='isolated_process', source_manifest=source_ref, revision=revision)
        point['engine_identity']['metering_execution'] = 'isolated_process'
        point['inputs']['source_manifest'] = source_ref
    selected_groups = group_points([p for p in points if p['name'] in selected])
    execution = dict(load_bound(base['execution_inputs']), source=str(source), source_sha256=revision)
    write_new(out/'execution-inputs.json', execution)
    write_new(out/'source-extension.json', dict(base_manifest=parent_ref,
        overlays={k:binding(ROOT/'src'/k) for k in args.overlay}, source_manifest=source_ref,
        baseline_algorithms_unchanged=True, original_sampler_and_integration_unchanged=True))
    campaign = dict(base, campaign_id=out.name, parent_campaign=base_ref, points=points,
        groups=group_points(points), execution_inputs=binding(out/'execution-inputs.json'),
        execution_source_manifest=source_ref,
        execution_campaigns=sorted(set(base.get('execution_campaigns', [])+[str(args.base.resolve())])),
        summary=dict(points=len(points), isolated_sampling_points=len(selected),
                     existing_or_running_points_changed=0, baseline_results_retried=0))
    write_new(out/'campaign.json', campaign)
    jobs = []
    for group in selected_groups:
        owners = {selected[p['name']] for p in group['points']}
        if len(owners) != 1:
            raise ValueError('replacement combined distinct original jobs')
        old = next(j for j in old_jobs if j['job_id'] in owners)
        group_path = out/'groups'/(group['session_id']+'.json')
        write_new(group_path, group)
        job = resident_job(group, group_path, root=ROOT, source=source,
            image=execution['image_digest'], verification=execution['model_verification']['path'],
            campaign=out/'campaign.json', priority=old['priority'])
        external = old['payload'].get('external_readonly_inputs', [])
        at = job['payload']['argv'].index(execution['image_digest'])
        job['payload']['argv'][at:at] = [v for row in external for v in ('-v', f"{row['path']}:{row['path']}:ro")]
        job['payload'].update(system=old['payload']['system'], model_id=group['model_id'],
            observation_scope=old['payload']['observation_scope'], result_policy='all_recorded_windows/v1',
            after_terminal=old['payload'].get('after_terminal', []), external_readonly_inputs=external,
            metering_execution='isolated_process', supersedes_job_id=old['job_id'],
            original_job_payload_sha256=digest(old['payload']), source_review=binding(out/'source-extension.json'))
        jobs.append(job)
    write_new(out/'jobs.json', jobs)
    print(json.dumps(dict(points=len(selected), jobs=len(jobs), source_sha256=revision, out=str(out))))


if __name__ == '__main__':
    main()
