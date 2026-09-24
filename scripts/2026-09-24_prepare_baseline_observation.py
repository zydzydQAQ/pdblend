#!/usr/bin/env python3
"""Prepare only unmeasured DistServe/Dynamo points; never operate the GPU queue."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

from pdblend.bench.comparison_campaign import binding, group_points, load_bound
from pdblend.bench.comparison_jobs import resident_job
from pdblend.bench.comparison_baseline_observation import SCOPE, RESULT_POLICY, validate_observation_inputs
from pdblend.bench.comparison_dynamo_runtime import ENTRYPOINT, WORKER_EXTENSION, dynamo_launch_options
from pdblend.bench.resident_session import digest, write_new

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = {'distserve', 'dynamollm'}
OVERLAYS = ('pdblend/bench/comparison_runtime.py', 'pdblend/bench/comparison_baseline_observation.py')


def measured_names(inventory):
    """Every real window is preserved, irrespective of prior acceptance flags."""
    return {row['point'] for row in inventory['windows']}


def extend_source(base, out):
    execution = load_bound(base['execution_inputs']); source_base = Path(execution['source'])
    frozen_ref = binding(source_base/'manifest.json'); frozen = load_bound(frozen_ref)
    spec = importlib.util.spec_from_file_location('observation_source_freezer', ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    freezer = importlib.util.module_from_spec(spec); spec.loader.exec_module(freezer)
    freezer.verify_snapshot(source_base, frozen['files'])
    staged = out/'staged-source'; staged.mkdir()
    for name in frozen['files']:
        target = staged/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_base/name, target)
    for name in OVERLAYS:
        target = staged/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/'src'/name, target)
    source, revision = freezer.freeze_source(staged, out/'sources'); shutil.rmtree(staged)
    manifest = load_bound(binding(source/'manifest.json'))
    changed = sorted(k for k in set(frozen['files']) | set(manifest['files'])
                     if frozen['files'].get(k) != manifest['files'].get(k))
    if set(changed)-set(OVERLAYS):
        raise ValueError('source changed outside the two explicit observation wrappers')
    write_new(out/'source-extension.json', dict(base_manifest=frozen_ref,
        builder=binding(__file__), overlays={k:binding(ROOT/'src'/k) for k in OVERLAYS},
        changed_files=changed, source_manifest=binding(source/'manifest.json'),
        baseline_core_unchanged=True, public_engine_and_measurement_unchanged=True))
    return execution, source, revision, manifest


def configure_point(point, mixed, row, out, source_ref, revision):
    identity = deepcopy(mixed['engine_identity']); system = point['system']
    if system == 'distserve':
        choice = load_bound(row['distserve_plan'])
        choice.update(trace=point['trace'], rate_rps=point['rate_rps'], slo=point['slo'])
        for i, instance in enumerate(identity['instances']):
            instance['instance_id'] = f'dist-{i//2}-' + ('P' if i % 2 == 0 else 'D')
            instance['launch_options']['kv_connector'] = 'P2pNcclConnector'
        config = dict(system=system, model_id=point['model_id'], observation_scope=SCOPE,
            formal_eligible=False, max_batch_size=32, request_timeout_s=240.)
        choice_path = out/'choices'/(point['name']+'.json'); write_new(choice_path, choice)
        inputs = dict(offline_choice=binding(choice_path), profiles=choice['profiles'])
        topology = dict(point['topology'], mode='predeclared_fixed_native_pairs',
            selected=choice['selected'], scope=choice['scope'], topology_optimality_established=False,
            offline_topology_search_performed=False)
    else:
        config = load_bound(row['dynamo_config'])
        config.update(observation_scope=SCOPE, formal_eligible=False, trace=point['trace']['path'],
                      slo_ttft_s=point['slo']['ttft_s'], slo_tpot_s=point['slo']['tpot_s'])
        identity.update(entrypoint=ENTRYPOINT, worker_extension=WORKER_EXTENSION,
            instances=[dict(instance_id=r.get('instance_id', r.get('id')), tp=r['tp'], pp=r.get('pp',1),
                gpu_uuids=[identity['fleet_gpu_uuids'][g] for g in r['gpus']],
                launch_options=dynamo_launch_options(config)) for r in config['instances']])
        inputs = dict(profiles=[row['profile']], predictor_manifest=row['predictor_manifest'],
            history=row['history'], history_summary=row['history_summary'],
            goldens=row['goldens'], transition_cost=row['transition_cost'])
        topology = dict(point['topology'], require_qualified_control_space=False,
                        qualification_waiver_scope=SCOPE, original_periods_preserved=True)
    config_path = out/'configs'/(point['name']+'.json'); write_new(config_path, config)
    inputs.update(trace=point['trace'], system_config=binding(config_path), source_manifest=source_ref,
                  qualifications=[], original_input_preparation=row)
    point.update(inputs=inputs, engine_identity=identity, source_manifest=source_ref, revision=revision,
        observation_scope=SCOPE, qualification_mode=SCOPE, result_policy=RESULT_POLICY,
        blockers=[], status='prepared', formal_eligible=False, profile_qualified=False,
        topology=topology, baseline_frozen=False)
    checked = validate_observation_inputs(point, identity)
    return dict(point=point['name'], system=system, model_id=point['model_id'],
        immutable_input_check_passed=True, formal_eligible=False, profile_qualified=False,
        profile_missing_gates=checked['profile_missing_gates'], trace=point['trace'])


def external_history_mounts(config):
    """Bind the old expected SHA without rereading a 692 MB author CSV here."""
    history = json.loads(Path(config['dynamo_weekly_history']).read_text())
    path = Path(history['source_path']).resolve()
    if path.is_relative_to(ROOT):
        return []
    if not path.is_file():
        raise ValueError('original author history file is unavailable at its bound absolute path')
    return [dict(path=str(path), expected_sha256=history['source_sha256'], bytes=path.stat().st_size,
                 mode='ro', rehashed_during_preparation=False,
                 execution_verifier='pdblend_baselines.dynamollm.validation.verified_history')]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--readiness', type=Path, required=True)
    parser.add_argument('--recorded-inventory', type=Path, required=True)
    parser.add_argument('--non-matrix-review', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--after-terminal', action='append', default=[])
    parser.add_argument('--priority', type=int, default=1900)
    args = parser.parse_args(argv)
    if not 0 <= args.priority < 2000:
        raise ValueError('baseline jobs must remain below the PD-first priority range')
    refs = {k:binding(getattr(args,k)) for k in ('base','readiness','recorded_inventory','non_matrix_review')}
    load_bound(refs['non_matrix_review'])
    base, readiness, inventory = (load_bound(refs[k]) for k in ('base','readiness','recorded_inventory'))
    if len(base['points']) != 180 or len({p['name'] for p in base['points']}) != 180:
        raise ValueError('complete immutable parent matrix required')
    measured = measured_names(inventory)
    for row in inventory['windows']:
        # Small receipt binding only; do not sweep raw request/power archives.
        load_bound(row['receipt'])
    rows = {r['model_id']:r for r in readiness['rows']}
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    execution, source, revision, manifest = extend_source(base, out)
    source_ref = binding(source/'manifest.json')
    runtime = {k:v for k,v in manifest['files'].items() if k.startswith(('pdblend_runtime/', 'pdblend/engine/'))}
    measurement = {k:v for k,v in manifest['files'].items() if k.startswith('pdblend/measure/') or k in (
        'pdblend/bench/comparison_metrics.py', 'pdblend/bench/comparison_metering.py', 'pdblend/bench/client.py')}
    points, checks = deepcopy(base['points']), []
    selected_names = {p['name'] for p in points if p['system'] in SYSTEMS and p['name'] not in measured}
    for point in points:
        if point['name'] not in selected_names:
            continue
        if point.get('evidence_valid') or point.get('baseline_frozen'):
            raise ValueError('attempt to replace an already observed/frozen point')
        mixed = next(p for p in base['points'] if p['system']=='mixed' and p['model_id']==point['model_id'])
        identity = mixed['engine_identity']
        if identity['runtime_source_sha256'] != digest(runtime) or identity['measurement_source_sha256'] != digest(measurement):
            raise ValueError('frozen public engine or common measurement identity differs')
        checks.append(configure_point(point, mixed, rows[point['model_id']], out, source_ref, revision))
    parents = {p['name']:p for p in base['points']}
    if any(p != parents[p['name']] for p in points if p['name'] not in selected_names):
        raise ValueError('an already measured or other-system point changed')
    groups = group_points(points)
    new_groups = [dict(g, points=[p for p in g['points'] if p['name'] in selected_names]) for g in groups
                  if any(p['name'] in selected_names for p in g['points'])]
    if any(not g['points'] or len(g['points']) > 12 or len({p['system'] for p in g['points']}) != 1 for g in new_groups):
        raise ValueError('unmeasured points do not form independent per-model baseline groups')
    write_new(out/'execution-inputs.json', dict(execution, source=str(source), source_sha256=revision))
    campaign = dict(base, campaign_id=out.name, parent_campaign=refs['base'], points=points, groups=groups,
        execution_inputs=binding(out/'execution-inputs.json'), execution_source_manifest=source_ref,
        parent_execution_inputs=base['execution_inputs'], parent_execution_source_manifest=base['execution_source_manifest'],
        execution_campaigns=sorted(set(base.get('execution_campaigns', [])+[str(args.base.resolve())])),
        observational_baseline_revision=revision,
        summary=dict(points=180, new_baseline_observations=len(checks), new_groups=len(new_groups),
            preserved_points=180-len(checks), preserved_pd_points=36, observed_points_not_rerun=sorted(measured),
            pure_service_s=len(checks)*150, profile_qualified=False))
    write_new(out/'campaign.json', campaign)
    write_new(out/'input-preflight.json', dict(status='passed', hardware_executed=False, formal_eligible=False,
        inputs=refs, source_manifest=source_ref, checked_points=checks, selected_names=sorted(selected_names),
        observed_point_exclusion=sorted(measured), preserved_point_sha256={p['name']:digest(p) for p in points
            if p['name'] not in selected_names}, source_extension=binding(out/'source-extension.json')))
    jobs=[]
    for group in new_groups:
        group_path=out/'groups'/(group['session_id']+'.json'); write_new(group_path,group)
        system=group['points'][0]['system']
        job=resident_job(group,group_path,root=ROOT,source=source,image=execution['image_digest'],
            verification=execution['model_verification']['path'],campaign=out/'campaign.json',priority=args.priority-len(jobs))
        mounts=[]
        if system=='dynamollm':
            mounts=external_history_mounts(load_bound(group['points'][0]['inputs']['system_config']))
            at=job['payload']['argv'].index(execution['image_digest'])
            job['payload']['argv'][at:at]=[v for row in mounts for v in ('-v',f"{row['path']}:{row['path']}:ro")]
        job['payload'].update(after_terminal=list(args.after_terminal),system=system,model_id=group['model_id'],
            observation_scope=SCOPE,result_policy=RESULT_POLICY,formal_eligible=False,
            external_readonly_inputs=mounts,source_review=binding(out/'input-preflight.json'))
        jobs.append(job)
    write_new(out/'jobs.json',jobs)
    write_new(out/'preparation.json',dict(schema='baseline-observation-preparation/v1',builder=binding(__file__),
        inputs=refs,source_manifest=source_ref,campaign=binding(out/'campaign.json'),jobs=binding(out/'jobs.json'),
        prepared_only=True,queue_modified=False,hardware_executed=False,writer_started=False,
        baseline_algorithms_unchanged=True,groups=len(jobs),points=len(checks),source_sha256=revision))
    print(json.dumps(dict(out=str(out),jobs=len(jobs),points=len(checks),source_sha256=revision,hardware_executed=False)))


if __name__=='__main__':main()
