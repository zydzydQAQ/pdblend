#!/usr/bin/env python3
"""Append one immutable first-gap guard trial to the existing paired campaign.

The candidate is the exact v4 source plus router.py, isolating the uncovered
two-token handoff guard from concurrently developed capacity-floor changes.
"""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

from pdblend.bench.comparison_campaign import binding, load_bound
from pdblend.bench.comparison_jobs import resident_job
from pdblend.bench.resident_session import digest, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    parent = load_bound(binding(args.parent))
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('new campaign directory required')
    template = next(p for p in parent['points'] if p['recovery_experiment']['arm'] == 'handoff')
    source_base = Path(template['source_manifest']['path']).parent
    spec = importlib.util.spec_from_file_location('first_gap_freezer',
        root/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    freezer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(freezer)
    old_manifest = load_bound(template['source_manifest'])
    freezer.verify_snapshot(source_base, old_manifest['files'])
    with tempfile.TemporaryDirectory(prefix='pd-first-gap-') as tmp:
        staging = Path(tmp)/'source'
        shutil.copytree(source_base, staging)
        shutil.copyfile(root/'src/pdblend/online/router.py', staging/'pdblend/online/router.py')
        source, revision = freezer.freeze_source(staging, out/'sources')
    manifest = load_bound(binding(source/'manifest.json'))
    changed = [name for name, sha in old_manifest['files'].items() if manifest['files'].get(name) != sha]
    if changed != ['pdblend/online/router.py'] or set(manifest['files']) != set(old_manifest['files']):
        raise ValueError('first-gap trial must isolate exactly one router change')
    point = deepcopy(template)
    point.update(name=template['name'].replace('-handoff-', '-first-gap-guard-'), revision=revision,
                 source_manifest=binding(source/'manifest.json'))
    point['inputs']['source_manifest'] = point['source_manifest']
    point['recovery_experiment'].update(arm='first_gap_guard',
        paired_source_point_sha256=digest(template),
        selection_reason='CPU semantic counterexample: legacy transfer is not measured first-gap coverage',
        thresholds_tuned_from_evaluation=False)
    original_group = next(g for g in parent['groups'] if template in g['points'])
    group = dict(original_group, session_id='first-gap-'+digest(point)[:20], points=[point])
    group_path = out/'groups'/(group['session_id']+'.json')
    write_new(group_path, group)
    jobs = json.loads((args.parent.parent/'scheduled-jobs.json').read_text())
    original_job = next(j for j in jobs if j['payload']['session_id'] == original_group['session_id'])
    execution = load_bound(load_bound(parent['parent_campaign'])['execution_inputs'])
    job = resident_job(group, group_path, root=root, source=source,
        image=execution['image_digest'], verification=execution['model_verification']['path'],
        campaign=out/'campaign.json', priority=1850)
    job['payload'].update(system='pdblend', model_id=point['model_id'],
        after_terminal=[original_job['job_id']], depends_on=[],
        observation_scope=point['observation_scope'], result_policy=point['result_policy'], formal_eligible=False)
    campaign = dict(parent, campaign_id=out.name, guard_parent=binding(args.parent),
        points=parent['points']+[point], groups=parent['groups']+[group],
        candidate_source=point['source_manifest'],
        summary=dict(points=len(parent['points'])+1, jobs=len(jobs)+1, service_seconds=(len(parent['points'])+1)*150),
        single_change=dict(parent_source=template['source_manifest'], files=changed,
                           evaluation_used_for_parameter_tuning=False))
    write_new(out/'campaign.json', campaign)
    write_new(out/'jobs.json', jobs+[job])
    write_new(out/'new-jobs.json', [job])
    write_new(out/'guard-preflight-campaign.json', dict(campaign, points=[point], groups=[group]))
    print(json.dumps(dict(job_id=job['job_id'], after_terminal=job['payload']['after_terminal'],
                         source_sha256=revision, changed_files=changed)))


if __name__ == '__main__':
    main()
