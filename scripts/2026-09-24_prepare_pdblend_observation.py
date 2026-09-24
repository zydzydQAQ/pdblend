#!/usr/bin/env python3
"""Prepare real original-policy windows using unchanged historical PD profiles.

This prepares observational measurements, not profile qualification, and never
enqueues work or edits a baseline point. The scheduling owner publishes jobs.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import json
import math
from pathlib import Path
import shutil

from pdblend.bench.comparison_campaign import binding, group_points, load_bound
from pdblend.bench.comparison_jobs import resident_job
from pdblend.bench.independent_dispatch import request_rows
from pdblend.bench.resident_session import digest, write_new
from pdblend.bench.run import offline_forecast
from pdblend.control.policies import get_policy
from pdblend.planner.pool import Plan, PlannerConfig, PoolPlanner, SLO
from pdblend.profile.query.versions import load_profile

ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'pdblend_profile_unqualified_evaluation/v1'
OVERLAYS = ('pdblend/bench/comparison_runtime.py',
            'pdblend/bench/independent_dispatch.py',
            'pdblend/bench/comparison_pdblend_observation.py',
            'pdblend/bench/resident_session.py',
            'pdblend/bench/run.py',
            'pdblend/bench/comparison_pdblend_acceptance.py',
            'pdblend/bench/comparison_pdblend_lifecycle.py',
            'pdblend/bench/comparison_native_acceptance.py')


def make_tuning(row, out):
    original = load_bound(row['planning_trace'])
    confirmation = load_bound(row['confirmation'])
    if (confirmation.get('split') != 'tuning'
            or confirmation.get('dataset') != row['dataset']
            or confirmation.get('trace_sha256') != row['planning_trace']['sha256']):
        raise ValueError('independent tuning ownership differs')
    requests = request_rows(original)
    if any(r.source != row['dataset'] for r in requests):
        raise ValueError('planning trace dataset differs')
    rate = offline_forecast(requests).rate_rps
    factor = rate / row['target_rate_rps']
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('invalid independent tuning time scale')
    trace = dict(original, model_id=row['model_id'], dataset=row['dataset'],
        selection_split='tuning', source_trace=row['planning_trace'],
        source_confirmation=row['confirmation'], time_scale=factor,
        prescribed_rate_rps=row['target_rate_rps'], evaluation_used_for_selection=False,
        duration_s=original['duration_s']*factor,
        requests=[dict(r, arrival_s=r['arrival_s']*factor) for r in original['requests']])
    actual = offline_forecast(request_rows(trace)).rate_rps
    if not math.isclose(actual, row['target_rate_rps'], rel_tol=1e-12):
        raise ValueError('bootstrap tuning rate differs from the offline forecast')
    write_new(out, trace)
    return binding(out), trace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--after-terminal', action='append', default=[])
    parser.add_argument('--result-policy', choices=['all_recorded_windows/v1'])
    args = parser.parse_args(argv)
    base_ref, replay_ref = binding(args.base), binding(args.replay)
    base, replay = load_bound(base_ref), load_bound(replay_ref)
    rows = {(r['model_id'], r['dataset'], r['scale']): r for r in replay['rows']}
    if (len(replay['rows']) != 36 or len(rows) != 36
            or any(not r.get('plan_returned') or not r.get('finite_estimates') for r in rows.values())):
        raise ValueError('need the complete three-model original-policy CPU replay')
    execution_ref = base['execution_inputs']
    execution = load_bound(execution_ref)
    source_base = Path(execution['source'])
    frozen_ref = binding(source_base/'manifest.json')
    frozen = load_bound(frozen_ref)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    module = importlib.util.spec_from_file_location('source_freezer', ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    freezer = importlib.util.module_from_spec(module); module.loader.exec_module(freezer)
    freezer.verify_snapshot(source_base, frozen['files'])
    staged = out/'staged-source'; staged.mkdir()
    for name in frozen['files']:
        target = staged/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_base/name, target)
    for name in OVERLAYS:
        target = staged/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/'src'/name, target)
    write_new(out/'source-extension.json', dict(base_manifest=frozen_ref,
        overlays={name:binding(ROOT/'src'/name) for name in OVERLAYS}))
    source, revision = freezer.freeze_source(staged, out/'sources')
    shutil.rmtree(staged)
    source_ref = binding(source/'manifest.json'); files = load_bound(source_ref)['files']
    runtime = {k:v for k,v in files.items() if k.startswith(('pdblend_runtime/', 'pdblend/engine/'))}
    measurement = {k:v for k,v in files.items() if k.startswith('pdblend/measure/') or k in (
        'pdblend/bench/comparison_metrics.py', 'pdblend/bench/comparison_metering.py', 'pdblend/bench/client.py')}
    protected = [k for k in frozen['files'] if k not in OVERLAYS]
    if any(files.get(k) != frozen['files'][k] for k in protected):
        raise ValueError('unexpected source change outside the observational adapter')
    points, checks = deepcopy(base['points']), []
    for point in points:
        if point['system'] != 'pdblend':
            continue
        if point.get('evidence_valid') or point.get('baseline_frozen'):
            raise ValueError('cannot replace an executed point identity')
        row = rows[(point['model_id'], point['dataset'], point['scale'])]
        if row['target_rate_rps'] != point['rate_rps']:
            raise ValueError('tuning rate and declared evaluation rate differ')
        size = point['model_id'].split('-')[1].lower()
        provenance = replay['profiles'][size]
        profile = row['profile']; profile_data = load_bound(profile)
        if provenance['profile'] != profile or profile_data.get('bounded_coverage'):
            raise ValueError('use the unchanged original training profile only')
        owner = load_bound(provenance['owner_manifest'])
        if (owner.get('system') != 'pdblend'
                or any(owner.get(k) != provenance[k] for k in
                       ('model_id','tp','pp','model_hash','tokenizer_hash'))
                or owner['model_id'] != point['model_id'] or owner['tp'] != row['tp']
                or Path(owner['training_profile']).resolve() != Path(profile['path']).resolve()
                or owner['training_profile_sha256'] != profile['sha256']
                or Path(owner['training_raw']).resolve() != Path(provenance['training_raw']['path']).resolve()
                or owner['training_raw_sha256'] != provenance['training_raw']['sha256']):
            raise ValueError('original training profile ownership binding differs')
        mixed = next(p for p in base['points'] if p['system']=='mixed'
            and p['model_id']==point['model_id'] and p.get('engine_identity'))
        identity = deepcopy(mixed['engine_identity'])
        if (identity['runtime_source_sha256'] != digest(runtime)
                or identity['measurement_source_sha256'] != digest(measurement)):
            raise ValueError('public runtime or metering differs from the baseline')
        if (provenance['model_hash'] != identity['model_hash']
                or provenance['tokenizer_hash'] != identity['tokenizer_hash']):
            raise ValueError('training profile model/tokenizer differs from evaluation')
        for index, instance in enumerate(identity['instances']):
            if instance['tp'] != row['tp'] or instance['pp'] != 1:
                raise ValueError('original profile TP does not match the resident inventory')
            instance['instance_id'] = 'pd'+str(index)
            instance['launch_options']['kv_connector'] = 'P2pNcclConnector'
        tuning_ref, tuning = make_tuning(row, out/'tuning'/(point['name']+'.json'))
        loaded = load_profile(profile['path'], system='pdblend', model_id=point['model_id'],
                              tp=row['tp'], pp=1, usage='development')
        policy = get_policy('pdblend')
        cfg = policy.planner_config(PlannerConfig(slots=8//row['tp'],
            slo=SLO(point['slo']['ttft_s'], point['slo']['tpot_s']),
            freqs=loaded.model.freqs, max_num_seqs=32))
        plan = PoolPlanner(loaded.model, cfg).plan(offline_forecast(request_rows(tuning)))
        plan = replace(plan, tp=row['tp'], pp=1,
            profile_key=json.dumps(loaded.profile_key, sort_keys=True, separators=(',', ':')))
        # Replaying scaled tuning must preserve the reviewed actual plan.
        if plan.key() != Plan(**row['plan']).key():
            raise ValueError('tuning-only plan replay changed deployment')
        choice = dict(system='pdblend', model_id=point['model_id'], selection_split='tuning',
            evaluation_used_for_selection=False, profile_sha256=profile['sha256'],
            planning_trace_sha256=tuning_ref['sha256'], plan=asdict(plan),
            offline_tp_scope='available_original_own_training_topology_only',
            candidate_tps=[row['tp']], topology_optimality_established=False,
            original_profile_provenance=provenance, planner_replay=replay_ref)
        choice_path = out/'choices'/(point['name']+'.json'); write_new(choice_path, choice)
        config = dict(system='pdblend', model_id=point['model_id'], profile=profile,
            observation_scope=SCOPE, profile_usage='development', policy='pdblend',
            profile_qualified=False, profile_interpolation_or_extrapolation='original_affine_model_unchanged')
        config_path = out/'configs'/(point['name']+'.json'); write_new(config_path, config)
        inputs = dict(trace=point['trace'], profiles=[profile], system_config=binding(config_path),
            offline_choice=binding(choice_path), planning_trace=tuning_ref, qualifications=[],
            source_manifest=source_ref)
        point.update(inputs=inputs, engine_identity=identity, source_manifest=source_ref,
            revision=revision, observation_scope=SCOPE, qualification_mode=SCOPE,
            blockers=[], status='prepared', formal_eligible=False,
            profile_qualified=False, profile_usage='development')
        point['topology'] = dict(point['topology'], selected=dict(tp=row['tp'], pp=1),
            candidate_tps=[row['tp']], topology_optimality_established=False)
        if args.result_policy:
            point['result_policy'] = args.result_policy
        from pdblend.bench.comparison_pdblend_observation import validate_observation_inputs
        validate_observation_inputs(point, inputs)
        checks.append(dict(point=point['name'], profile=profile, tuning=tuning_ref,
            finite_estimates=all(math.isfinite(getattr(plan,k)) for k in ('power_w','ttft_s','tpot_s')),
            original_policy=asdict(policy), formal_eligible=False, profile_qualified=False))
    if len(checks) != 36 or not all(r['finite_estimates'] for r in checks):
        raise ValueError('incomplete executable original-policy inventory')
    parents = {p['name']:p for p in base['points']}
    if any(p != parents[p['name']] for p in points if p['system'] != 'pdblend'):
        raise ValueError('baseline point changed')
    groups = group_points(points)
    pd_groups = [g for g in groups if {p['system'] for p in g['points']} == {'pdblend'}]
    if (len(pd_groups) != 3 or len({g['model_id'] for g in pd_groups}) != 3
            or any(len(g['points']) != 12 for g in pd_groups)):
        raise ValueError('expected twelve windows in each model resident group')
    write_new(out/'execution-inputs.json', dict(execution, source=str(source), source_sha256=revision))
    campaign = dict(base, campaign_id=out.name, parent_campaign=base_ref,
        execution_inputs=binding(out/'execution-inputs.json'), execution_source_manifest=source_ref,
        parent_execution_inputs=execution_ref, parent_execution_source_manifest=base.get('execution_source_manifest'),
        execution_campaigns=sorted(set(base.get('execution_campaigns', [])+[str(args.base.resolve())])),
        points=points, groups=groups, observational_pd_revision=revision,
        summary=dict(points=len(points), new_pd_observations=36, pure_service_s=36*150,
                     baseline_points_unchanged=144, profile_qualified=False))
    write_new(out/'campaign.json', campaign); write_new(out/'input-preflight.json', checks)
    jobs=[]
    for group in groups:
        if {p['system'] for p in group['points']} != {'pdblend'}:
            continue
        group_path=out/'groups'/(group['session_id']+'.json'); write_new(group_path, group)
        job=resident_job(group,group_path,root=ROOT,source=source,image=execution['image_digest'],
            verification=execution['model_verification']['path'],campaign=out/'campaign.json',priority=3000-len(jobs))
        job['payload'].update(after_terminal=list(args.after_terminal), system='pdblend',
            model_id=group['model_id'], observation_scope=SCOPE, formal_eligible=False,
            source_review=binding(out/'input-preflight.json'))
        jobs.append(job)
    if len(jobs)!=3:
        raise ValueError('expected one resident PD job per model')
    write_new(out/'jobs.json',jobs)
    print(json.dumps(dict(out=str(out),jobs=len(jobs),points=36,source_sha256=revision,formal_eligible=False)))


if __name__=='__main__':
    main()
