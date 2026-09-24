#!/usr/bin/env python3
"""Prepare the original 36 PDblend points against an explicit frozen source.

This writes immutable inputs and exclusive serial jobs; it never enqueues or
executes GPU work. CPU validation imports the same source the jobs will use.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SIZES = ('7b', '14b', '32b')
DATASETS = ('alpaca', 'sharegpt', 'longbench')
SCALES = (.25, .5, .75, 1.)
CASES = tuple(f'{size}-pdblend-{dataset}-x{scale:g}-seed701'
              for size in SIZES for dataset in DATASETS for scale in SCALES)
FLOOR_CASES = {'7b-pdblend-sharegpt-x0.25-seed701': 2.,
               '7b-pdblend-sharegpt-x0.5-seed701': 4.}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify_source(path):
    """Verify the complete frozen inventory before importing any runtime code."""
    path = Path(path).absolute()
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError('source manifest and source directory must not be symlinks')
    path = path.resolve()
    data = json.loads(path.read_text())
    files = data.get('files', {})
    if (not files or data.get('source_sha256') != _digest(files)
            or path.parent.name != data['source_sha256']):
        raise ValueError('explicit source must be a content-addressed complete frozen inventory')
    actual = {}
    for item in sorted(path.parent.rglob('*')):
        if item == path:
            continue
        if item.is_symlink():
            raise ValueError('frozen source contains a symlink')
        if item.is_file():
            actual[str(item.relative_to(path.parent))] = hashlib.sha256(item.read_bytes()).hexdigest()
    if files != actual:
        raise ValueError('frozen source bytes or inventory differ')
    if 'pdblend/bench/comparison_runtime.py' not in files:
        raise ValueError('frozen source lacks the resident comparison runtime')
    return path, data


def activate_source(path):
    source = path.parent
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] in ('pdblend', 'pdblend_runtime', 'pdblend_baselines'):
            location = getattr(module, '__file__', None)
            if location and not Path(location).resolve().is_relative_to(source):
                raise ValueError('preparer already imported another algorithm; run this CLI in a fresh process')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))


def select_cases(parent, requested=None):
    names = list(CASES if requested is None else requested)
    if not names or len(names) != len(set(names)) or set(names)-set(CASES):
        raise ValueError('cases must be a nonempty unique subset of the original 36 standard points')
    by_name = {}
    for point in parent['points']:
        if point['name'] in names:
            if point['name'] in by_name:
                raise ValueError('parent contains duplicate standard points')
            by_name[point['name']] = point
    if set(by_name) != set(names):
        raise ValueError('parent lacks requested original standard points: '+str(sorted(set(names)-set(by_name))))
    for name, point in by_name.items():
        size, _, dataset, scale, _ = name.split('-')
        if (point.get('system') != 'pdblend' or point.get('model_id') != 'Qwen2.5-'+size.upper()+'-Instruct'
                or point.get('dataset') != dataset or point.get('scale') != float(scale[1:])
                or point.get('seed') != 701 or point.get('duration_s') != 150):
            raise ValueError('original standard point identity differs: '+name)
    return [by_name[name] for name in CASES if name in by_name]


def load_helper():
    path = ROOT/'scripts/2026-09-24_prepare_energy_repair.py'
    spec = importlib.util.spec_from_file_location('standard_energy_repair_helper', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qualified_capacity(path, source, ceiling):
    from pdblend.bench.comparison_campaign import binding, load_bound
    from pdblend.bench.capacity_floor_v2 import validate_manifest
    from pdblend.bench.low_m_tuning import source_inventory
    from pdblend.planner.capacity import load_capacity_floors
    from pdblend.profile.query.versions import load_profile
    artifact_ref = binding(Path(path))
    artifact = load_bound(artifact_ref)
    if artifact.get('kind') != 'pdblend_capacity_floor_v2':
        raise ValueError('standard low-M deployment requires direct v2 raw-trial capacity evidence')
    load_bound(artifact['tuning_manifest'])
    manifest = validate_manifest(artifact['tuning_manifest']['path'])
    if manifest['identity']['model_id'] != 'Qwen2.5-7B-Instruct' or (manifest['identity']['tp'], manifest['identity']['pp']) != (1, 1):
        raise ValueError('this standard deployment only accepts 7B TP1 ShareGPT capacity evidence')
    model = load_profile(manifest['profile']['path'], system='pdblend', model_id=manifest['identity']['model_id'],
                         tp=1, pp=1, usage='development').model
    floors = load_capacity_floors(Path(path), model=model)  # Replays every raw trial and the complete summary.
    if (not floors or any(floor.version != 2 or not floor.qualified for floor in floors)
            or _digest(source_inventory(source)) != manifest['context']['algorithm_source_sha256']
            or manifest['recovery_policy']['safety_max_freq'] != ceiling):
        raise ValueError('capacity evidence differs from the exact frozen algorithm or recovery ceiling')
    return SimpleNamespace(artifact=artifact_ref, manifest=manifest, floors=floors)


def capacity_binding(point, capacity, options):
    """Bind only a measured nominal domain; runtime still checks every shape."""
    from pdblend.bench.pdblend_runtime_options import capacity_workload_context
    nominal = FLOOR_CASES.get(point['name'])
    if capacity is None or nominal is None:
        return None
    if (point['model_id'] != 'Qwen2.5-7B-Instruct' or point['dataset'] != 'sharegpt'
            or point['rate_rps'] != nominal or point['slo'] != {'ttft_s': 5., 'tpot_s': .15}
            or capacity.manifest['profile'] not in point['inputs']['profiles']):
        raise ValueError('capacity point model, nominal rate, original SLO or exact profile differs')
    matches = [floor for floor in capacity.floors if floor.context.get('nominal_rate_rps') == nominal]
    if not matches:
        return None
    family = capacity.manifest['workload_identity']
    workload = dict(family, nominal_rate_rps=nominal)
    options.update(capacity_floor_path=capacity.artifact, capacity_floor_reserve_canonical=True,
                   capacity_workload_binding=workload)
    context = capacity_workload_context(capacity.artifact['path'], workload, options)
    if not context or any(floor.context != context for floor in matches):
        raise ValueError('capacity runtime source/workload/recovery/nominal context differs')
    return dict(workload=workload, context=context, nominal_rate_rps=nominal,
                matched_frequencies_mhz=sorted({floor.frequency_mhz for floor in matches}),
                unknown_shape_action='restore_canonical_M4', evaluation_used_for_selection=False)


def prepare(parent_path, out, *, source_manifest, capacity_floor=None, cases=None, ceiling=2100,
            after_terminal=()):
    source_path, frozen = verify_source(source_manifest)
    activate_source(source_path)
    helper = load_helper()
    from pdblend.bench.comparison_campaign import binding, load_bound
    from pdblend.bench.comparison_jobs import resident_job
    from pdblend.bench.pdblend_observation_plan import decode_plan, encode_plan
    from pdblend.bench.pdblend_runtime_options import DEFAULTS, ARTIFACTS
    from pdblend.bench.resident_session import digest, engine_signature, write_new
    parent_ref = binding(Path(parent_path)); parent = load_bound(parent_ref)
    originals = select_cases(parent, cases)
    source_ref = binding(source_path); revision = frozen['source_sha256']
    runtime_sha, measurement_sha = helper.method_hashes(frozen['files'])
    execution = load_bound(parent['execution_inputs'])
    capacity = qualified_capacity(capacity_floor, source_path.parent, ceiling) if capacity_floor else None
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('execution package output already exists')
    points, pending, capacity_points = [], [], []
    for original in originals:
        if original['engine_identity']['runtime_source_sha256'] != runtime_sha:
            raise ValueError('standard controller repair must preserve the original inference engine')
        options = dict(DEFAULTS, **{key:None for key in ARTIFACTS}, safety_max_freq=ceiling,
                       startup_safety=True, deadline_safety=True)
        bound_capacity = capacity_binding(original, capacity, options)
        choice = helper.startup_choice(original, ceiling)
        plan = decode_plan(choice, observation=True)
        if plan.detail.get('startup_prediction') == 'unavailable':
            raise ValueError('startup counts lack profile coverage at the requested ceiling: '+original['name'])
        if bound_capacity:
            if len(original['engine_identity']['instances']) != 8 or (plan.tp, plan.pp) != (1, 1):
                raise ValueError('qualified low-M startup requires the original eight TP1 instances')
            plan = replace(plan, counts={'M':4, 'L1':4}, f_P=ceiling, f_D=ceiling, f_M=ceiling, tau=0)
            # Recompute the final canonical layout, retaining honest unavailable estimates.
            from pdblend.planner.pool import PlannerConfig, PoolPlanner, SLO
            loaded = helper.load_profile(capacity.manifest['profile']['path'], system='pdblend',
                model_id=original['model_id'], tp=1, pp=1, usage='development')
            cfg = helper.get_policy('pdblend').planner_config(PlannerConfig(8, SLO(**original['slo']),
                freqs=tuple(f for f in loaded.model.freqs if f <= ceiling), max_num_seqs=32))
            cfg.preserve_overload_capacity = True
            forecast = helper.offline_forecast(helper.request_rows(load_bound(original['inputs']['planning_trace'])))
            plan = PoolPlanner(loaded.model, cfg).refresh_estimate(plan, forecast)
            if plan.detail.get('prediction_unavailable'):
                raise ValueError('canonical low-M startup lacks measured profile coverage')
            choice.update(encode_plan(plan), startup_policy='capacity_v2_canonical_M4_L1_startup')
        choice['runtime_options'] = options
        point = deepcopy(original)
        point.update(name=original['name']+'-repaired-standard-'+out.name, run_id=out.name,
            revision=revision, source_manifest=source_ref, status='prepared', blockers=[], formal_eligible=False)
        point['engine_identity']['measurement_source_sha256'] = measurement_sha
        point['repair_experiment'] = dict(case=original['name'], arm='candidate', repeat=0, source_manifest=source_ref,
            same_evaluation_trace=True, clock_qualification_pending=True, capacity_domain=bound_capacity)
        config = load_bound(original['inputs']['system_config']); config['pdblend_runtime'] = options
        choice_path = out/'choices'/(point['name']+'.json')
        config_path = out/'configs'/(point['name']+'.json')
        pending.append((point, choice_path, choice, config_path, config, options))
        if bound_capacity:
            capacity_points.append(original['name'])
    out.mkdir(parents=True, exist_ok=False)
    for point, choice_path, choice, config_path, config, options in pending:
        write_new(choice_path, choice); write_new(config_path, config)
        point['inputs'].update(source_manifest=source_ref, offline_choice=binding(choice_path),
                               system_config=binding(config_path))
        point['optimization_version'] = dict(source_manifest=source_ref, profile=point['inputs']['profiles'][0],
            requested=options, qualification='development_only', clock_qualification_pending=True)
        helper.validate_observation_inputs(point, point['inputs'])
        specs = [SimpleNamespace(tp=row['tp'], pp=row['pp'], generation=0)
                 for row in point['engine_identity']['instances']]
        helper.pdblend_window_resources(point, specs)
        points.append(point)
    groups, jobs = [], []
    for size in SIZES:
        model_id = 'Qwen2.5-'+size.upper()+'-Instruct'
        members = [point for point in points if point['model_id'] == model_id]
        if not members:
            continue
        identity = members[0]['engine_identity']
        if any(point['engine_identity'] != identity for point in members):
            raise ValueError('standard model group contains conflicting engine identities')
        group = dict(session_id='repaired-standard-'+size+'-'+digest(members)[:16], model_id=model_id,
            engine_identity=identity, engine_signature=engine_signature(identity), points=members,
            gpu_count=8, exclusive=True, reserve_host=True)
        path = out/'groups'/(group['session_id']+'.json'); write_new(path, group)
        job = resident_job(group, path, root=ROOT, source=source_path.parent, image=execution['image_digest'],
            verification=execution['model_verification']['path'], campaign=out/'campaign.json', priority=700)
        job['payload'].update(system='pdblend', model_id=model_id,
            after_terminal=[jobs[-1]['job_id']] if jobs else list(after_terminal),
            result_policy='all_recorded_windows/v1',
            observation_scope='pdblend_profile_unqualified_evaluation/v1', formal_eligible=False)
        groups.append(group); jobs.append(job)
    campaign = dict(parent, campaign_id=out.name, run_id=out.name, parent_campaign=parent_ref,
        points=points, groups=groups, candidate_source=source_ref, execution_source_manifest=source_ref,
        summary=dict(points=len(points), jobs=len(jobs), service_seconds=150*len(points)),
        repair_protocol=dict(cases=[row['name'] for row in originals], standard_cases=list(CASES),
            capacity_floor=None if capacity is None else capacity.artifact, capacity_points=capacity_points,
            frozen_source_reused=True, canonical_startup_for_capacity_v2=True,
            other_initial_counts_preserved=True, ceiling_mhz=ceiling,
            service_and_tail_total_energy=True, frequency_qualification_required=True,
            unknown_capacity_shapes_restore_canonical=True, formal_eligible=False))
    for key in ('extension_policy_refs', 'extension_manifest_roots'):
        campaign.pop(key, None)
    write_new(out/'campaign.json', campaign); write_new(out/'jobs.json', jobs)
    write_new(out/'preparation.json', dict(campaign=binding(out/'campaign.json'), source=source_ref,
        jobs=binding(out/'jobs.json'), preparer=binding(Path(__file__)),
        helper=binding(ROOT/'scripts/2026-09-24_prepare_energy_repair.py'),
        hardware_executed=False, cpu_preflight_passed=True, container_preflight_pending=True, enqueued=False))
    return campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--capacity-floor', type=Path)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--ceiling', type=int, default=2100)
    parser.add_argument('--after-terminal', nargs='*', default=[])
    args = parser.parse_args()
    result = prepare(args.parent, args.out, source_manifest=args.source_manifest,
        capacity_floor=args.capacity_floor, cases=args.cases, ceiling=args.ceiling,
        after_terminal=args.after_terminal)
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
