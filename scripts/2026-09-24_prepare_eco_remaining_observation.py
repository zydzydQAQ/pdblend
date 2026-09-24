#!/usr/bin/env python3
"""Schedule only unexecuted EcoServe windows under the recorded-result policy."""
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
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--previous', type=Path, action='append', required=True)
    parser.add_argument('--after-terminal', action='append', default=[])
    args = parser.parse_args()
    base_ref = binding(args.base)
    base = load_bound(base_ref)
    module = importlib.util.spec_from_file_location('eco_selection', ROOT/'scripts/2026-09-24_prepare_ecoserve_recovery.py')
    selector = importlib.util.module_from_spec(module)
    module.loader.exec_module(selector)
    selection = selector.unattempted_selection(base, args.previous,
        ROOT/'results/2026-09-22/three-model/queue.json')
    if not selection['points']:
        raise ValueError('no remaining unexecuted EcoServe windows')
    execution = load_bound(base['execution_inputs'])
    selected = [p for p in base['points'] if p['name'] in selection['points']]
    parent_sources = {p['source_manifest']['path'] for p in selected}
    if len(parent_sources) != 1:
        raise ValueError('remaining points require distinct source packages')
    parent_source_ref = binding(next(iter(parent_sources)))
    parent_source = Path(parent_source_ref['path']).parent
    parent_files = load_bound(parent_source_ref)['files']
    spec = importlib.util.spec_from_file_location('source_freezer', ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
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
    overlay = 'pdblend/bench/resident_session.py'
    shutil.copyfile(ROOT/'src'/overlay, staged/overlay)
    source, revision = freezer.freeze_source(staged, out/'sources')
    shutil.rmtree(staged)
    source_ref = binding(source/'manifest.json')
    current_files = load_bound(source_ref)['files']
    if any(current_files.get(name) != sha for name, sha in parent_files.items() if name != overlay):
        raise ValueError('source changed outside the result-policy coordinator')
    write_new(out/'source-extension.json', dict(base_manifest=parent_source_ref,
        overlays={overlay: binding(ROOT/'src'/overlay)}))
    execution = dict(execution, source=str(source), source_sha256=revision)
    points = deepcopy(base['points'])
    changed = []
    for point in points:
        if point['name'] not in selection['points']:
            continue
        old_files = load_bound(point['source_manifest'])['files']
        protected = [name for name in old_files if name.startswith((
            'pdblend_baselines/ecoserve/', 'pdblend_runtime/', 'pdblend/engine/', 'pdblend/measure/'))
            or name in ('pdblend/bench/client.py', 'pdblend/bench/comparison_metrics.py',
                        'pdblend/bench/comparison_metering.py')]
        if any(old_files[name] != current_files.get(name) for name in protected):
            raise ValueError('EcoServe core or common execution/measurement implementation changed')
        point.update(result_policy='all_recorded_windows/v1', revision=execution['source_sha256'],
                     source_manifest=source_ref, status='prepared', blockers=[])
        point['inputs']['source_manifest'] = source_ref
        changed.append(point['name'])
    groups = group_points([point for point in points if point['name'] in changed])
    write_new(out/'unattempted-selection.json', selection)
    write_new(out/'execution-inputs.json', execution)
    campaign = dict(base, campaign_id=out.name, parent_campaign=base_ref,
        execution_inputs=binding(out/'execution-inputs.json'), points=points, groups=groups,
        execution_campaigns=sorted(set(base.get('execution_campaigns', [])+[str(args.base.resolve())])),
        unattempted_selection=binding(out/'unattempted-selection.json'),
        summary=dict(points=len(points), new_ecoserve_points=len(changed),
                     executed_points_retried=0, baseline_core_changed=False))
    write_new(out/'campaign.json', campaign)
    jobs = []
    for group in groups:
        path = out/'groups'/(group['session_id']+'.json')
        write_new(path, group)
        job = resident_job(group, path, root=ROOT, source=source, image=execution['image_digest'],
            verification=execution['model_verification']['path'], campaign=out/'campaign.json',
            priority=1800-len(jobs))
        job['payload'].update(system='ecoserve', model_id=group['model_id'],
            result_policy='all_recorded_windows/v1', after_terminal=args.after_terminal,
            unattempted_selection=binding(out/'unattempted-selection.json'))
        jobs.append(job)
    write_new(out/'jobs.json', jobs)
    print(json.dumps(dict(out=str(out), jobs=len(jobs), unexecuted_points=changed)))


if __name__ == '__main__':
    main()
