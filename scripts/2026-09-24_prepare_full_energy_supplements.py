#!/usr/bin/env python3
"""Append complete-cycle energy reruns while retaining frozen baseline cores."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from pdblend.bench.comparison_campaign import binding, group_points, load_bound
from pdblend.bench.comparison_jobs import resident_job
from pdblend.bench.resident_session import digest, write_new

ROOT = Path(__file__).resolve().parents[1]


def prepare(parent_path, gaps_path, out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    parent_ref, gap_ref = binding(parent_path), binding(gaps_path)
    parent, gaps = load_bound(parent_ref), load_bound(gap_ref)
    execution = load_bound(parent['execution_inputs'])
    if gaps.get('schema') != 'pdblend-baseline-energy-gap-inventory/v2':
        raise ValueError('full service-and-tail gap inventory is required')
    spec = importlib.util.spec_from_file_location('supplement_freezer', ROOT/'scripts/2026-09-24_prepare_saturation_round.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sources, points = {}, []
    for gap in gaps['gaps']:
        original = load_bound(gap['point'])
        size, system = original['model_id'].split('-')[1].lower(), original['system']
        parent_source = parent['baseline_execution_sources'][size][system]
        key = parent_source['sha256']
        if key not in sources:
            sources[key] = module.frozen_baseline_coordinator(parent_source, out)
        source_ref = sources[key]
        files = load_bound(source_ref)['files']
        old_files = load_bound(original['source_manifest'])['files']
        protected = [k for k in old_files if k.startswith((
            'pdblend_baselines/', 'pdblend_runtime/', 'pdblend/engine/', 'pdblend/measure/'))
            or k in ('pdblend/bench/client.py', 'pdblend/bench/comparison_metrics.py',
                     'pdblend/bench/comparison_metering.py')]
        if any(files.get(k) != old_files[k] for k in protected):
            raise ValueError('baseline core or energy implementation changed: '+original['name'])
        point = deepcopy(original)
        point.update(name=original['name']+'-full-energy-supplement', source_manifest=source_ref,
            revision=load_bound(source_ref)['source_sha256'], metering_execution='isolated_process',
            run_id=out.name, status='prepared', blockers=[], result_policy='all_recorded_windows/v1',
            energy_supplement_of=gap['receipt'], original_point_name=original['name'],
            comparison_selection='first_predeclared_complete_service_and_tail_attempt', formal_eligible=False)
        point['inputs']['source_manifest'] = source_ref
        point['engine_identity']['metering_execution'] = 'isolated_process'
        points.append(point)
    groups, jobs = group_points(points), []
    for group in groups:
        group['session_id'] = 'full-energy-'+digest(group)[:20]
        path = out/'groups'/(group['session_id']+'.json')
        write_new(path, group)
        member = group['points'][0]
        source = Path(member['source_manifest']['path']).parent
        job = resident_job(group, path, root=ROOT, source=source, image=execution['image_digest'],
            verification=execution['model_verification']['path'], campaign=out/'campaign.json', priority=500)
        job['payload'].update(system=member['system'], model_id=member['model_id'],
            depends_on=[], after_terminal=[jobs[-1]['job_id']] if jobs else [],
            formal_eligible=False, result_policy='all_recorded_windows/v1')
        jobs.append(job)
    campaign = dict(parent, campaign_id=out.name, run_id=out.name, parent_campaign=parent_ref,
                    points=points, groups=groups, baseline_energy_gaps=gap_ref,
                    summary=dict(points=len(points), jobs=len(jobs), missing_tail_only=sum(
                        g['missing_components'] == ['energy_tail_j'] for g in gaps['gaps'])))
    write_new(out/'campaign.json', campaign)
    write_new(out/'jobs.json', jobs)
    write_new(out/'preparation.json', dict(campaign=binding(out/'campaign.json'), jobs=binding(out/'jobs.json'),
        baseline_core_unchanged=True, energy_measurement_unchanged=True,
        historical_receipts_unchanged=True, hardware_executed=False, enqueued=False))
    return campaign


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--gaps', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.parent, args.gaps, args.out)['summary']))
