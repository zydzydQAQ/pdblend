#!/usr/bin/env python3
"""Prepare four combined-controller observations, reusing recorded controls.

Only new-jobs.json is intended for enqueueing. No GPU, Docker or queue writes
occur here. The parent campaign's eleven windows remain available to the report.
"""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from pdblend.bench.comparison_campaign import binding, load_bound
from pdblend.bench.comparison_jobs import resident_job
from pdblend.bench.comparison_pdblend_observation import validate_observation_inputs
from pdblend.bench.comparison_runtime import pdblend_window_resources
from pdblend.bench.pdblend_runtime_options import ARTIFACTS, DEFAULTS
from pdblend.bench.recovery_campaign import method_hashes, report
from pdblend.bench.resident_session import digest, engine_signature, write_new


CASES = ('7b-pdblend-longbench-x1-seed701', '7b-pdblend-sharegpt-x0.25-seed701',
         '14b-pdblend-longbench-x1-seed701', '32b-pdblend-alpaca-x1-seed701')


def combined_options():
    """Freeze current defaults explicitly; optional artifact mechanisms stay off."""
    values = dict(DEFAULTS, **{key: None for key in ARTIFACTS})
    if not (values['shield_mode'] == 'budget_aware' and values['slo_routing']
            and values['preserve_overload_capacity'] and values['safety_recovery']
            and values['experiment_mode'] == 'adaptive' and not values['joint_resident']
            and not values['capacity_floor_reserve_canonical']):
        raise ValueError('combined trial requires the reviewed default recovery/routing/Shield configuration')
    return values


def base_controls(parent, observed):
    controls, receipts = [], {}
    for case in CASES:
        points = [p for p in parent['points'] if p['recovery_experiment']['case'] == case
                  and p['recovery_experiment']['arm'] == 'control'
                  and p['recovery_experiment']['repeat'] == 0]
        rows = [r for r in observed['windows'] if r['case'] == case and r['arm'] == 'control'
                and r['repeat'] == 0]
        if len(points) != 1 or len(rows) != 1:
            raise ValueError('each case needs exactly one existing control point and recorded window: '+case)
        point, row = points[0], rows[0]
        if not (row['artifact_valid'] and row['canonical_metrics_valid'] and row['energy_metrics_valid']
                and row['cleanup_passed'] and row['point_sha256'] == digest(point)):
            raise ValueError('control lacks intact canonical/energy/cleanup evidence: '+case)
        # Reuse descriptive observations even when a clock/profile gate failed;
        # retain the original audit rather than manufacture measurement eligibility.
        load_bound(row['receipt'])
        receipts[case] = dict(receipt=row['receipt'], point_sha256=digest(point),
            measurement_qualified=row['measurement_valid'], original_qualification_retained=True)
        controls.append(point)
    return controls, receipts


def prepare(parent_path, out, *, root, queue_path, profile_jobs_path):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('new immutable combined campaign directory required')
    parent_ref = binding(parent_path)
    parent = load_bound(parent_ref)
    if parent.get('repeats') != 1 or parent.get('schema') != 'pdblend-recovery-campaign/v1':
        raise ValueError('combined extension needs the single-repeat diagnostic parent')
    parent_jobs_ref = binding(Path(parent_path).parent/'jobs.json')
    parent_jobs = load_bound(parent_jobs_ref)
    profile_jobs_ref = binding(profile_jobs_path)
    profile_jobs = load_bound(profile_jobs_ref)
    dependencies = [job['job_id'] for job in profile_jobs]
    if (len(dependencies) != len(set(dependencies)) or len(dependencies) != 3
            or {job['payload']['model_id'] for job in profile_jobs}
               != {'Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct', 'Qwen2.5-32B-Instruct'}):
        raise ValueError('bind exactly the three current model profile jobs')
    controls, receipts = base_controls(parent, report(parent_path, queue_path))
    execution = load_bound(load_bound(parent['parent_campaign'])['execution_inputs'])
    reference_source = parent['single_change']['parent_source']  # v4, before first-gap guard
    reference_manifest = load_bound(reference_source)
    spec = importlib.util.spec_from_file_location('combined_source_freezer',
        root/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    freezer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(freezer)
    freezer.verify_snapshot(Path(reference_source['path']).parent, reference_manifest['files'])
    # Validate source compatibility before publishing any campaign files.
    current_files = freezer.source_files(root/'src')
    if method_hashes(current_files) != method_hashes(reference_manifest['files']):
        raise ValueError('common engine or numerical metering differs from v4')
    options = combined_options()
    source, revision = freezer.freeze_source(root/'src', out/'sources')
    source_ref = binding(source/'manifest.json')
    frozen_manifest = load_bound(source_ref)
    if frozen_manifest['files'] != current_files:
        raise ValueError('workspace source changed during combined preparation')
    points, groups, jobs = [], [], []
    for control in controls:
        point = deepcopy(control)
        case = control['recovery_experiment']['case']
        point.update(name=case+'-combined-r0', revision=revision, source_manifest=source_ref,
                     status='prepared', formal_eligible=False, profile_qualified=False)
        point['inputs']['source_manifest'] = source_ref
        config = load_bound(control['inputs']['system_config'])
        config['pdblend_runtime'] = deepcopy(options)
        config_path = out/'configs'/(point['name']+'.json')
        write_new(config_path, config)
        point['inputs']['system_config'] = binding(config_path)
        point['recovery_experiment'].update(arm='combined',
            reused_control_receipt=receipts[case]['receipt'], paired_control_point_sha256=digest(control),
            combined_parent=parent_ref, thresholds_tuned_from_evaluation=False,
            selection_reason='verify interactions of reviewed recovery, SLO routing and budget Shield defaults',
            all_optimizations_enabled=False, qualification='diagnostic_only')
        validate_observation_inputs(point, point['inputs'])
        specs = [SimpleNamespace(tp=r['tp'], pp=r['pp'], generation=0)
                 for r in point['engine_identity']['instances']]
        pdblend_window_resources(point, specs)
        for key in ('trace', 'system_config', 'offline_choice', 'planning_trace', 'source_manifest'):
            load_bound(point['inputs'][key])
        points.append(point)
    for size in ('7b', '14b', '32b'):
        selected = [point for point in points if point['name'].startswith(size+'-')]
        identity = selected[0]['engine_identity']
        if any(point['engine_identity'] != identity for point in selected):
            raise ValueError('combined model windows must share identical engine identity')
        if identity['image_digest'] != execution['image_digest']:
            raise ValueError('combined image differs from common execution inputs')
        group = dict(session_id='combined-'+digest(selected)[:20], model_id=selected[0]['model_id'],
            engine_identity=identity, engine_signature=engine_signature(identity), points=selected,
            gpu_count=8, exclusive=True, reserve_host=True)
        path = out/'groups'/(group['session_id']+'.json')
        write_new(path, group)
        job = resident_job(group, path, root=root, source=source, image=execution['image_digest'],
            verification=execution['model_verification']['path'], campaign=out/'campaign.json', priority=600)
        job['payload'].update(system='pdblend', model_id=group['model_id'],
            after_terminal=dependencies+([jobs[-1]['job_id']] if jobs else []), depends_on=[],
            observation_scope='pdblend_profile_unqualified_evaluation/v1',
            result_policy='all_recorded_windows/v1', formal_eligible=False)
        groups.append(group)
        jobs.append(job)
    campaign = dict(parent)
    campaign.pop('single_change', None)
    campaign.update(campaign_id=out.name, combined_parent=parent_ref, candidate_source=source_ref,
        base_control_receipts=receipts, points=parent['points']+points, groups=parent['groups']+groups,
        summary=dict(points=len(parent['points'])+4, jobs=len(parent_jobs)+3,
                     service_seconds=(len(parent['points'])+4)*150,
                     new_points=4, new_jobs=3, new_service_seconds=600, reused_controls=4),
        combined_protocol=dict(options=options, reference_source=reference_source,
            common_runtime_and_meter_unchanged=True, evaluation_used_for_parameter_tuning=False,
            full_workspace_source_frozen=True, all_optimizations_enabled=False,
            controls_reexecuted=False, historical_controls_reused=True,
            same_session_pairing=False, temporal_confound_not_excluded=True,
            profile_job_dependencies=profile_jobs_ref, parent_jobs=parent_jobs_ref))
    write_new(out/'campaign.json', campaign)
    write_new(out/'jobs.json', parent_jobs+jobs)
    write_new(out/'new-jobs.json', jobs)
    write_new(out/'new-preflight-campaign.json', dict(campaign, points=points, groups=groups))
    write_new(out/'preparation.json', dict(schema='pdblend-combined-recovery-preparation/v1',
        campaign=binding(out/'campaign.json'), new_jobs=binding(out/'new-jobs.json'),
        new_preflight=binding(out/'new-preflight-campaign.json'), source=source_ref,
        preparer=binding(Path(__file__)), base_control_receipts=receipts,
        common_runtime_and_meter_unchanged=True, cpu_input_validation_passed=True,
        container_preflight_passed=False, container_preflight_pending=True,
        hardware_executed=False, enqueued=False, formal_eligible=False, statistical_claim=False))
    return campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--profile-jobs', type=Path, required=True)
    args = parser.parse_args()
    value = prepare(args.parent, args.out, root=Path(__file__).resolve().parents[1],
                    queue_path=args.queue, profile_jobs_path=args.profile_jobs)
    print(json.dumps(dict(campaign=str(args.out.resolve()/'campaign.json'), summary=value['summary'],
        source=value['candidate_source'], hardware_executed=False, enqueued=False)))


if __name__ == '__main__':
    main()
