#!/usr/bin/env python3
"""Prepare a matrix from an explicit reviewed source; never freeze or enqueue.

Imports only the selected frozen algorithm in a fresh process. --preview-only
writes a CPU review, with no runnable campaign, policy, jobs or queue changes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
POLICY = 'all_recorded_windows/v1'
SCOPE = 'pdblend_profile_unqualified_evaluation/v1'
DATASETS = ('alpaca', 'sharegpt', 'longbench')
SIZES = ('7b', '14b', '32b')
DIAGNOSIS_CASES = ('7b-pdblend-sharegpt-x0.25-seed701',
    '7b-pdblend-sharegpt-x0.5-seed701', '7b-pdblend-alpaca-x0.5-seed701',
    '7b-pdblend-alpaca-x1-seed701', '7b-pdblend-longbench-x1-seed701')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load_bound(ref):
    if not isinstance(ref, dict) or binding(ref['path'])['sha256'] != ref['sha256']:
        raise ValueError('artifact binding differs')
    return json.loads(Path(ref['path']).read_text())


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')


def verify_source(path):
    path = Path(path).absolute()
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError('frozen source must not be a symlink')
    ref = binding(path); data = load_bound(ref); files = data.get('files', {})
    if not files or data.get('source_sha256') != digest(files) or path.parent.name != data['source_sha256']:
        raise ValueError('candidate source must be a complete content-addressed inventory')
    actual = {}
    for item in path.parent.rglob('*'):
        if item == path: continue
        if item.is_symlink(): raise ValueError('frozen source contains a symlink')
        if item.is_file(): actual[str(item.relative_to(path.parent))] = binding(item)['sha256']
    if actual != files:
        raise ValueError('candidate frozen source bytes or inventory differ')
    return ref, data


def activate_source(ref):
    source = Path(ref['path']).parent
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] in ('pdblend', 'pdblend_runtime', 'pdblend_baselines'):
            location = getattr(module, '__file__', None)
            if location and not Path(location).resolve().is_relative_to(source):
                raise ValueError('another algorithm is already imported; use a fresh process')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))


def method_hashes(files):
    return (digest({k:v for k,v in files.items() if k.startswith(('pdblend_runtime/', 'pdblend/engine/'))}),
        digest({k:v for k,v in files.items() if k.startswith('pdblend/measure/') or k in (
            'pdblend/bench/client.py', 'pdblend/bench/comparison_metrics.py', 'pdblend/bench/comparison_metering.py')}))


def select_standard_points(base):
    points = base['points']
    if len(points) != 180 or len({p['name'] for p in points}) != 180:
        raise ValueError('base must retain exactly the original 180 logical points')
    pd = [p for p in points if p['system'] == 'pdblend']
    expected = {f'{size}-pdblend-{dataset}-x{scale:g}-seed701'
                for size in SIZES for dataset in DATASETS for scale in (.25, .5, .75, 1.)}
    if {p['name'] for p in pd} != expected or len(pd) != 36:
        raise ValueError('expected exactly the original thirty-six PD points')
    for p in pd:
        size = p['model_id'].split('-')[1].lower()
        if (p['name'] != f'{size}-pdblend-{p["dataset"]}-x{p["scale"]:g}-seed701'
                or p['seed'] != 701 or p['duration_s'] != 150):
            raise ValueError('standard workload identity differs')
    return pd


def complete_options(path, defaults, artifacts):
    ref = binding(path); options = load_bound(ref)
    if not isinstance(options, dict) or set(options) != set(defaults) | set(artifacts):
        raise ValueError('freeze all runtime options explicitly, including null artifact bindings')
    return ref, options


def point_options(default, overrides, name, defaults, artifacts):
    """Evidence-specific floors can vary; global controller tuning cannot."""
    selected = deepcopy(overrides.get(name, default))
    required = set(defaults) | set(artifacts)
    if not required <= set(selected) or set(selected)-required-{'capacity_workload_binding'}:
        raise ValueError('each point override must contain the complete runtime options')
    local = set(artifacts) | {'capacity_floor_reserve_canonical', 'capacity_workload_binding'}
    if any(selected[k] != default[k] for k in required-local):
        raise ValueError('point evidence binding cannot change the global controller policy')
    if selected.get('capacity_floor_path') is not None and not selected.get('capacity_workload_binding'):
        raise ValueError('point capacity evidence requires explicit model/corpus/nominal-rate binding')
    return selected


def validate_measurement_pair(old_ref, new_ref, old_hash, new_hash, reviews):
    if old_hash == new_hash:
        return []
    for review in reviews:
        sources = review['sources']
        for left, right in (sources, sources[::-1]):
            if (left['source_manifest'] == old_ref and right['source_manifest'] == new_ref
                    and left['measurement_source_sha256'] == old_hash
                    and right['measurement_source_sha256'] == new_hash):
                return [review['manifest_binding']]
    raise ValueError('changed measurement method lacks an exact-source compatibility review')


def policy_declaration(evidence_ref):
    evidence = load_bound(evidence_ref)
    cases = evidence.get('evidence_declaration', {}).get('direct_candidate_comparison_cases')
    if set(cases or []) != set(DIAGNOSIS_CASES):
        raise ValueError('policy development evidence must name the five observed diagnostic cases')
    return dict(policy_design_used_prior_evaluation=True, evaluation_used_for_parameter_tuning=True,
        evaluation_used_for_parameter_tuning_scope='shared_policy_and_ceiling_after_prior_evaluation',
        evaluation_used_for_selection=False,
        evaluation_used_for_selection_scope='initial_plan_from_bound_independent_planning_trace',
        final_observation_after_freeze=True, not_blind_holdout=True,
        direct_candidate_comparison_cases=list(cases), evidence=evidence_ref,
        ceiling_selection_basis='engineered_policy_setting_with_evaluation_feedback',
        no_optimal_frequency_claim=True)


def baseline_identity(point):
    """Ignore only attempt naming/source wrapper and isolated meter execution."""
    engine = deepcopy(point['engine_identity']); engine.pop('metering_execution', None)
    inputs = point.get('inputs', {})
    return dict(**{k:point[k] for k in ('model_id', 'system', 'dataset', 'scale', 'seed', 'duration_s', 'slo')},
        trace=point['trace'], engine_identity=engine,
        profiles=inputs.get('profiles'), offline_choice=inputs.get('offline_choice'),
        system_config=inputs.get('system_config'))


def complete_energy_receipt(ref, expected_point):
    """Job termination is never a substitute for this result predicate."""
    receipt = load_bound(ref)
    point_path = Path(ref['path']).parent/'point.json'
    point_ref = binding(point_path)
    if receipt.get('artifacts', {}).get('point.json') != point_ref['sha256']:
        raise ValueError('completion receipt lacks its hash-bound point')
    point = load_bound(point_ref)
    if receipt.get('point_sha256') != digest(point) or baseline_identity(point) != baseline_identity(expected_point):
        raise ValueError('completion receipt is for a different original baseline identity')
    metrics = receipt.get('result', {}).get('metrics', {})
    finite = lambda x: type(x) in (int, float) and math.isfinite(x) and x >= 0
    return (receipt.get('recorded_window_complete') is True and receipt.get('cleanup_passed') is True
        and receipt.get('measurement_evidence_valid') is True
        and finite(metrics.get('energy_service_j')) and finite(metrics.get('energy_tail_j')))


def baseline_ownership(gaps_ref, external_campaign_refs=(), state_ref=None, completion_refs=()):
    """Prepared-only external jobs never remove an authorized gap from pending."""
    inventory = load_bound(gaps_ref)
    if len(inventory['gaps']) != 20:
        raise ValueError('the authorized original service-energy scope remains exactly twenty')
    originals = [(gap, load_bound(gap['point'])) for gap in inventory['gaps']]
    identities = [digest(baseline_identity(p)) for _,p in originals]
    if len(set(identities)) != 20: raise ValueError('duplicate original baseline gap identity')
    external = {}
    for ref in external_campaign_refs:
        campaign = load_bound(ref)
        for point in campaign['points']:
            if point['system'] == 'pdblend': continue
            external.setdefault(digest(baseline_identity(point)), []).append((ref, point))
    state = load_bound(state_ref) if state_ref else dict(jobs=[])
    if state_ref and state.get('schema') != 'baseline-owner-state/v1':
        raise ValueError('explicit external job-state snapshot schema required')
    states = {(entry['campaign']['sha256'], name):entry
              for entry in state.get('jobs', []) for name in entry['point_names']}
    receipts = [(ref, load_bound(ref)) for ref in completion_refs]
    ledger, pending, waiting, completed, newly_frozen = [], [], [], [], []
    for (gap, point), identity in zip(originals, identities):
        row = dict(point_id=point['name'], identity_sha256=identity, original_point=gap['point'],
                   original_receipt=gap['receipt'], status='pending', external_matches=[])
        valid_receipts=[]
        for ref, receipt in receipts:
            if receipt.get('point') not in [point['name']] + [p['name'] for _,p in external.get(identity, [])]:
                continue
            if complete_energy_receipt(ref, point):
                valid_receipts.append(ref)
        if len(valid_receipts)>1:
            raise ValueError('multiple complete gap receipts require explicit first-frozen selection')
        if valid_receipts:
            ref=valid_receipts[0];receipt=load_bound(ref)
            actual_ref=binding(Path(ref['path']).parent/'point.json');actual=load_bound(actual_ref)
            row.update(status='completed',completion_receipt=ref)
            snapshot=deepcopy(gap)
            snapshot.update(point_id=actual['name'],original_point_id=point['name'],point=actual_ref,point_sha256=receipt['point_sha256'],
                receipt=ref,engine_identity=actual['engine_identity'],source_manifest=actual['source_manifest'],
                revision=actual['revision'],inputs=actual.get('inputs'),trace=actual['trace'],
                result=receipt['result'],energy_service_j=receipt['result']['metrics']['energy_service_j'],
                energy_tail_j=receipt['result']['metrics']['energy_tail_j'],
                frozen_completion_of=gap['receipt'],selection_rule='first_explicit_complete_gap_receipt')
            newly_frozen.append(snapshot)
        for campaign_ref, other in external.get(identity, []):
            owner = states.get((campaign_ref['sha256'], other['name']))
            match = dict(campaign=campaign_ref, point_name=other['name'],
                         status=owner['status'] if owner else 'prepared_only')
            if owner:
                if owner['campaign'] != campaign_ref: raise ValueError('owner snapshot campaign binding differs')
                match['job_id'] = owner['job_id']
            row['external_matches'].append(match)
            if row['status'] == 'pending' and owner and owner['status'] in ('queued', 'running'):
                row.update(status='external_inflight', external_job=owner['job_id'], state_evidence=state_ref)
        ledger.append(row)
        if row['status'] == 'completed': completed.append(row)
        elif row['status'] == 'external_inflight': waiting.append(row)
        else: pending.append(gap)
    return dict(schema='baseline-gap-ownership/v1', authorized_gaps=gaps_ref,
        authorized_count=20, rows=ledger, completed_count=len(completed), pending_count=len(pending),
        external_inflight_count=len(waiting), all_complete=len(completed)==20,
        pending_gap_rows=pending, external_campaigns=list(external_campaign_refs),
        newly_frozen_baseline_rows=newly_frozen,
        state_snapshot=state_ref, completion_receipts=list(completion_refs),
        recheck_before_submission=True, terminal_without_valid_receipt_returns_to_pending=True,
        terminal_job_is_not_completion=True)


def publication_history(base, campaign_refs=()):
    """Keep full historical point variants without adding any executable group."""
    campaigns = [(None, base)] + [(ref, load_bound(ref)) for ref in campaign_refs]
    variants, policies, manifests, roots, executions = {}, {}, {}, set(), set()
    for ref, campaign in campaigns:
        if not isinstance(campaign.get('points'), list):
            raise ValueError('publication campaign must bind its point specifications')
        for point in campaign['points']:
            if not isinstance(point, dict) or not point.get('name') or not point.get('revision'):
                raise ValueError('historical point requires its own name and revision')
            variants.setdefault(digest(point), deepcopy(point))
        for key, target in (('extension_policy_refs', policies), ('historical_extension_manifests', manifests)):
            for item in campaign.get(key, []):
                load_bound(item); target[(item['path'], item['sha256'])] = item
        roots.update(campaign.get('extension_manifest_roots', []))
        executions.update(campaign.get('execution_campaigns', []))
        if ref: executions.add(ref['path'])
    return dict(schema='matrix-publication-history/v1', campaigns=list(campaign_refs),
        point_variants=list(variants.values()), point_sha256s=list(variants),
        extension_policy_refs=list(policies.values()), historical_extension_manifests=list(manifests.values()),
        extension_manifest_roots=sorted(roots), execution_campaigns=sorted(executions),
        scheduled_groups_inherited=False, boundary_state_reused=False)


def merge_publication_points(active_points, publication):
    # A repeated logical name can have multiple sources/revisions. Full point SHA
    # also preserves changed same-revision specs needed by receipt identity.
    unique = {digest(point): point for point in active_points}
    for point in publication['point_variants']:
        unique.setdefault(digest(point), deepcopy(point))
    return list(unique.values())


def extension_history(base, manifest_refs):
    policies = list(base.get('extension_policy_refs', []))
    for ref in policies: load_bound(ref)
    manifests = list(base.get('historical_extension_manifests', [])) + list(manifest_refs)
    unique = {}
    for ref in manifests:
        value = load_bound(ref)
        if value.get('mode') == 'pointer': ref=value['manifest']; value=load_bound(ref)
        if value.get('mode') != 'manifest' or value.get('policy') not in policies:
            raise ValueError('historical extension manifest is not bound to a retained old policy')
        unique[(ref['path'], ref['sha256'])] = ref
    return dict(policy_refs=policies, manifest_refs=list(unique.values()),
        manifest_roots=list(base.get('extension_manifest_roots', [])),
        carry_forward_boundaries=False, previous_manifest_for_new_policy=None)


def offline_plan(point, profile, options, startup, options_path):
    from pdblend.bench.independent_dispatch import request_rows
    from pdblend.bench.run import offline_forecast
    from pdblend.control.policies import get_policy
    from pdblend.planner.pool import PlannerConfig, PoolPlanner, SLO
    from pdblend.profile.query.versions import load_profile
    from pdblend.bench.pdblend_runtime_options import comparison_options, comparison_capacity_floors
    tp = point['topology']['selected']['tp']
    loaded = load_profile(profile['path'], system='pdblend', model_id=point['model_id'], tp=tp, pp=1,
                          usage='development')
    policy = get_policy('pdblend')
    cfg = policy.planner_config(PlannerConfig(slots=len(point['engine_identity']['instances']),
        slo=SLO(**point['slo']), freqs=loaded.model.freqs,
        max_num_seqs=point['engine_identity']['instances'][0]['launch_options']['max_num_seqs']))
    startup.apply_frequency_ceiling(cfg, loaded.model, options['safety_max_freq'])
    cfg.preserve_overload_capacity = options['preserve_overload_capacity']
    cfg.pressure_controls = policy.dynamic_m_floor
    cfg.capacity_floor_reserve_canonical = options['capacity_floor_reserve_canonical']
    checked = comparison_options(dict(pdblend_runtime=options),options_path,point=point)
    cfg.capacity_floor_context = checked.get('capacity_floor_context', {})
    floors = []
    if checked['values']['capacity_floor_path'] is not None:
        floors, _ = comparison_capacity_floors(checked['values']['capacity_floor_path'],model=loaded.model)
        cfg.capacity_floors = floors
    if checked['values']['transition_catalog_path'] is not None:
        from pdblend.planner.capacity import select_artifact
        from pdblend.planner.transitions import TransitionCatalog
        cfg.transition_estimator = TransitionCatalog.load(
            select_artifact(checked['values']['transition_catalog_path'],loaded.model),
            model=loaded.model,qualified_only=options['transition_qualified_only'])
    tuning = load_bound(point['inputs']['planning_trace'])
    if tuning.get('selection_split') not in ('calibration', 'tuning'):
        raise ValueError('initial planning input is not an independent split')
    rows = request_rows(tuning); forecast = offline_forecast(rows)
    planner = PoolPlanner(loaded.model, cfg); plan = planner.plan(forecast)
    if policy.dynamic_m_floor:
        pressure = planner.mixed_pressure(forecast, plan.counts.get('M', 0), plan.f_M)
        cfg.pd_pressure_active = (forecast.input_p95 >= cfg.pd_min_input_tokens
                                 and pressure['pressure'] >= policy.pd_pressure_enter)
        plan = planner.plan(forecast)
    plan = replace(plan, tp=tp, pp=1,
        profile_key=json.dumps(loaded.profile_key, sort_keys=True, separators=(',', ':')))
    if any(floor.version == 2 for floor in floors):
        # An independent measured low-M domain permits later adaptive selection;
        # the validated recovery protocol still begins with canonical M reserve.
        slots=len(point['engine_identity']['instances']);ceiling=options['safety_max_freq'] or max(loaded.model.freqs)
        plan=replace(plan,counts={'M':min(4,slots),'L1':max(0,slots-4)},
                     f_P=ceiling,f_D=ceiling,f_M=ceiling,tau=0)
        plan=planner.refresh_estimate(plan,forecast)
    if any(math.isnan(getattr(plan,k)) or getattr(plan,k)<0 for k in ('power_w','ttft_s','tpot_s')):
        raise ValueError('invalid initial estimate')
    contract = startup.startup_contract(point, loaded.model, plan, options, rows,
        profile=profile, planning_trace=point['inputs']['planning_trace'])
    contract['artifact_bindings_sha256']=digest({k:options.get(k) for k in (
        'capacity_floor_path','transition_catalog_path','incremental_energy_path')})
    contract['capacity_workload_binding_sha256']=digest(options.get('capacity_workload_binding'))
    if any(floor.version == 2 for floor in floors):
        actual=contract['expected_first_plan'];counts={k:v for k,v in actual['counts'].items() if v}
        expected={k:v for k,v in {'M':min(4,slots),'L1':max(0,slots-4)}.items() if v}
        contract['capacity_v2_effective_startup_matches_canonical']=(counts==expected and actual['tau']==0
            and all(actual['f_'+role]==ceiling for role in ('P','D','M')))
    return loaded, plan, contract


# Legacy helper API retained for existing baseline-only preparation scripts.
# The explicit-candidate prepare() below never calls either helper.
def freezer():
    spec = importlib.util.spec_from_file_location('round_freezer',
        ROOT / 'scripts/2026-09-22_enqueue_parallel_profiles.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_baseline_coordinator(parent_ref, out):
    """Only update lease continuation; baseline algorithms/meters stay byte exact."""
    parent = load_bound(parent_ref)
    source = Path(parent_ref['path']).parent
    overlay = 'pdblend/bench/resident_session.py'
    staged = out / 'baseline-staging' / parent['source_sha256']
    for name in parent['files']:
        target = staged / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    shutil.copyfile(ROOT / 'src' / overlay, staged / overlay)
    frozen, revision = freezer().freeze_source(staged, out / 'baseline-sources')
    shutil.rmtree(staged)
    ref = binding(frozen / 'manifest.json')
    current = load_bound(ref)['files']
    if any(current.get(name) != sha for name, sha in parent['files'].items() if name != overlay):
        raise ValueError('baseline changed outside its lease continuation coordinator')
    write_new(out / 'baseline-source-reviews' / (revision + '-' + parent['source_sha256'][:12] + '.json'), dict(
        parent=parent_ref, source_manifest=ref, overlay=binding(ROOT / 'src' / overlay),
        baseline_core_unchanged=True, measurement_implementation_unchanged=True))
    return ref



def prepare(base_path, profiles_path, gaps_path, out, *, candidate_source, runtime_options_path,
            policy_evidence_path, compatibility_paths=(), external_campaign_paths=(),
            external_state_path=None, completion_receipt_paths=(), history_manifest_paths=(),
            root_path=None, preview_only=False, startup_helper_path=None, point_options_path=None,
            startup_contract_mode='external_audit', publication_campaign_paths=()):
    root = Path(root_path).resolve() if root_path else ROOT
    source_ref, frozen = verify_source(candidate_source); activate_source(source_ref)
    from pdblend.bench.pdblend_runtime_options import DEFAULTS, ARTIFACTS, comparison_options
    from pdblend.bench.measurement_compatibility import load_compatibility
    from pdblend.bench.pdblend_observation_plan import encode_plan
    if startup_contract_mode not in ('external_audit','inprocess'):
        raise ValueError('unknown startup contract consumption mode')
    if startup_helper_path:
        if not preview_only and startup_contract_mode!='external_audit':
            raise ValueError('inprocess mode requires the helper and consumers inside the frozen source')
        helper_ref = binding(startup_helper_path)
        spec = importlib.util.spec_from_file_location('pdblend.bench.comparison_startup', helper_ref['path'])
        startup = importlib.util.module_from_spec(spec); spec.loader.exec_module(startup)
    else:
        from pdblend.bench import comparison_startup as startup
        helper_ref = binding(startup.__file__)
    base_ref, profiles_ref, gaps_ref = map(binding, (base_path, profiles_path, gaps_path))
    base, profiles = map(load_bound, (base_ref, profiles_ref)); originals = select_standard_points(base)
    options_ref, options = complete_options(runtime_options_path, DEFAULTS, ARTIFACTS)
    comparison_options(dict(pdblend_runtime=options), options_ref['path'])
    if any(options.get(k) is not None for k in ARTIFACTS):
        raise ValueError('domain artifacts belong in explicit per-point options, never a global default')
    point_options_ref=binding(point_options_path) if point_options_path else None
    overrides=load_bound(point_options_ref) if point_options_ref else {}
    if not isinstance(overrides,dict) or set(overrides)-{p['name'] for p in originals}:
        raise ValueError('point options name an unrecognized standard point')
    declaration = policy_declaration(binding(policy_evidence_path))
    reviews = [load_compatibility(p) for p in compatibility_paths]
    runtime_hash, measurement_hash = method_hashes(frozen['files'])
    ownership = baseline_ownership(gaps_ref, tuple(map(binding,external_campaign_paths)),
        binding(external_state_path) if external_state_path else None, tuple(map(binding,completion_receipt_paths)))
    publication = publication_history(base, tuple(map(binding,publication_campaign_paths)))
    history = extension_history(publication, tuple(map(binding,history_manifest_paths)))
    out = Path(out).resolve()
    if out.exists(): raise ValueError('preparation output already exists')
    points, checks, choices, configs = deepcopy(base['points']), [], {}, {}
    for point in points:
        if point['system'] != 'pdblend': continue
        size = point['model_id'].split('-')[1].lower(); profile=profiles['models'][size]['profile']; load_bound(profile)
        if point['engine_identity']['runtime_source_sha256'] != runtime_hash:
            raise ValueError('candidate changed original inference-engine identity')
        pair = validate_measurement_pair(point['source_manifest'],source_ref,
            point['engine_identity']['measurement_source_sha256'],measurement_hash,reviews)
        selected_options=point_options(options,overrides,point['name'],DEFAULTS,ARTIFACTS)
        selected_options_ref=point_options_ref if point['name'] in overrides else options_ref
        loaded, plan, contract = offline_plan(point, profile, selected_options, startup,selected_options_ref['path'])
        if not preview_only and contract.get('capacity_v2_effective_startup_matches_canonical') is False:
            raise ValueError('actual pressure-replanned startup overrides the canonical capacity-v2 contract: '+point['name'])
        choice = dict(system='pdblend',model_id=point['model_id'],selection_split='tuning',
            **declaration,profile_sha256=profile['sha256'], planning_trace_sha256=point['inputs']['planning_trace']['sha256'],
            **encode_plan(plan),offline_tp_scope='available_own_training_topology_only', candidate_tps=[plan.tp],
            topology_optimality_established=False,parent_choice=point['inputs']['offline_choice'],
            profile_composition=profiles_ref,runtime_options=selected_options,runtime_options_source=selected_options_ref,
            startup_contract=contract,deployment_selection_policy='replanned_independent_tuning_then_effective_startup',
            startup_contract_mode=startup_contract_mode,startup_helper=helper_ref,
            direct_candidate_comparison=point['name'] in DIAGNOSIS_CASES)
        config = dict(system='pdblend',model_id=point['model_id'],profile=profile,observation_scope=SCOPE,
            profile_usage='development',policy='pdblend',profile_qualified=False,pdblend_runtime=selected_options,
            startup_contract_mode=startup_contract_mode,startup_helper=helper_ref,
            profile_composition=profiles_ref,policy_development=declaration)
        choices[point['name']]=choice; configs[point['name']]=config
        point.update(revision=frozen['source_sha256'],source_manifest=source_ref,run_id=out.name,status='prepared',
            blockers=[],result_policy=POLICY,observation_scope=SCOPE,qualification_mode=SCOPE,formal_eligible=False,
            profile_qualified=False,profile_usage='development',profile_composition=profiles_ref,profile_sha256=profile['sha256'],
            comparison_scope='original_matrix_full_group',measurement_compatibility=pair,
            optimization_version=dict(source_manifest=source_ref,profile=profile,requested=selected_options,
                independent_tuning_only=False,initial_plan_independent_tuning_only=True,policy_development=declaration))
        point['engine_identity']['measurement_source_sha256']=measurement_hash
        checks.append(dict(point=point['name'],profile=profile,planning_trace=point['inputs']['planning_trace'],
            **encode_plan(plan),startup_contract=contract,runtime_options=selected_options,
            capacity_domain=dict(artifact=selected_options['capacity_floor_path'],
                workload=selected_options.get('capacity_workload_binding'),
                fallback='canonical_reserve_on_absent_or_unmatched_domain')))
    review=dict(schema='explicit-candidate-matrix-preview/v1',candidate_source=source_ref,runtime_options=options_ref,
        point_options=point_options_ref,
        startup_contract_mode=startup_contract_mode,
        compatibility=[r['manifest_binding'] for r in reviews],policy_development=declaration,
        source_startup_helper=helper_ref,startup_helper_external=bool(startup_helper_path),plans=checks,
        baseline_ownership=ownership,extension_history=history,
        publication_history=dict(publication,point_variants_count=len(publication['point_variants'])),
        hardware_executed=False,enqueued=False)
    if preview_only:
        write_new(out/'preview.json',review); return review
    from pdblend.bench.comparison_campaign import group_points
    from pdblend.bench.comparison_pdblend_observation import validate_observation_inputs
    from pdblend.bench.comparison_jobs import resident_job
    from pdblend.bench.single_observation_slo_boundary import build_policy
    for point in points:
        if point['system'] != 'pdblend': continue
        name=point['name']; choice_path=out/'choices'/(name+'.json'); config_path=out/'configs'/(name+'.json')
        write_new(choice_path,choices[name]);write_new(config_path,configs[name])
        point['inputs'].update(profiles=[configs[name]['profile']],source_manifest=source_ref,
            offline_choice=binding(choice_path),system_config=binding(config_path))
        validate_observation_inputs(point,point['inputs'])
    parents={p['name']:p for p in base['points']}
    if any(p!=parents[p['name']] for p in points if p['system']!='pdblend'):
        raise ValueError('frozen baseline point changed')
    groups=group_points([p for p in points if p['system']=='pdblend'])
    for group in groups:
        size=group['model_id'].split('-')[1].lower()
        if len(group['points'])!=12: raise ValueError('expected one twelve-point group per model')
        group['session_id']='saturation-'+digest(dict(run=out.name,engine=group['engine_signature'],model=group['model_id']))[:20]
        corpus=root/'datasets/prepared'/('2026-09-22-'+size+'-v1')
        rates={d:next(p['rate_rps']/p['scale'] for p in group['points'] if p['dataset']==d) for d in DATASETS}
        group['extension_policy']=build_policy(group,out/'policies'/size,run_id=out.name,
            corpus_refs={d:binding(corpus/(d+'.json')) for d in DATASETS},base_rates=rates,
            max_steps_per_dataset=12,soft_deadline_s=14400)
    execution=dict(load_bound(base['execution_inputs']),source=str(Path(source_ref['path']).parent),source_sha256=frozen['source_sha256'])
    write_new(out/'execution-inputs.json',execution)
    ledger_path=out/'baseline-ownership.json';write_new(ledger_path,ownership)
    pending_inventory=dict(load_bound(gaps_ref),gaps=ownership['pending_gap_rows'],authorized_inventory=gaps_ref,
        scope_projection='pending_after_receipts_and_temporary_external_live_jobs',ownership=binding(ledger_path))
    pending_inventory['frozen_baselines']=pending_inventory['frozen_baselines']+ownership['newly_frozen_baseline_rows']
    pending_inventory.update(complete_energy_points=len(pending_inventory['frozen_baselines']),
        missing_energy_points=len(pending_inventory['gaps']),authorized_missing_energy_points=20,
        external_inflight_energy_points=ownership['external_inflight_count'])
    write_new(out/'pending-baseline-gaps.json',pending_inventory)
    write_new(out/'extension-history.json',history)
    write_new(out/'publication-history.json',publication)
    published_points=merge_publication_points(points,publication)
    campaign=dict(base,campaign_id=out.name,run_id=out.name,parent_campaign=base_ref,points=published_points,groups=groups,
        active_standard_point_sha256s={p['name']:digest(p) for p in points},
        candidate_source=source_ref,execution_source_manifest=source_ref,execution_inputs=binding(out/'execution-inputs.json'),
        profiles=profiles_ref,authorized_baseline_energy_gaps=gaps_ref,
        baseline_energy_gaps=binding(out/'pending-baseline-gaps.json'),baseline_ownership=binding(ledger_path),
        baseline_comparison_policy=dict(base.get('baseline_comparison_policy',{}),
            frozen_baselines=binding(out/'pending-baseline-gaps.json')),
        measurement_compatibility=[r['manifest_binding'] for r in reviews],policy_development=declaration,
        startup_contract_mode=startup_contract_mode,startup_helper=helper_ref,
        extension_policy_refs=history['policy_refs']+[g['extension_policy'] for g in groups],
        active_extension_policy_refs=[g['extension_policy'] for g in groups],
        historical_extension_manifests=history['manifest_refs'],extension_manifest_roots=history['manifest_roots'],
        extension_history=binding(out/'extension-history.json'),old_boundary_state_reused=False,
        publication_campaigns=publication['campaigns'],publication_history=binding(out/'publication-history.json'),
        baseline_completion_contract=dict(authorized_gaps=gaps_ref,require_count=20,
            predicate='complete_energy_receipt',terminal_jobs_sufficient=False,recheck_external_state_before_submit=True),
        execution_campaigns=sorted(set(publication['execution_campaigns']+[str(Path(base_path).resolve())]
                                      +[str(Path(p).resolve()) for p in external_campaign_paths])),
        summary=dict(original_points=180,new_pd_points=36,published_point_variants=len(published_points),
            historical_point_variants=len(publication['point_variants']),authorized_baseline_energy_supplements=20,
            pending_baseline_energy_supplements=ownership['pending_count'],external_inflight_baseline_supplements=ownership['external_inflight_count'],
            completed_baseline_supplements=ownership['completed_count'],adaptive_slo_boundary_sequences=9,repeats_per_point=1))
    write_new(out/'campaign.json',campaign);write_new(out/'input-preflight.json',checks)
    jobs=[]
    for group in groups:
        path=out/'groups'/(group['session_id']+'.json');write_new(path,group)
        job=resident_job(group,path,root=root,source=Path(source_ref['path']).parent,image=execution['image_digest'],
            verification=execution['model_verification']['path'],campaign=out/'campaign.json',priority=3000-len(jobs))
        job['payload'].update(system='pdblend',model_id=group['model_id'],run_id=out.name,
            observation_scope=SCOPE,result_policy=POLICY,formal_eligible=False,source_review=binding(out/'input-preflight.json'))
        jobs.append(job)
    write_new(out/'jobs.json',jobs)
    write_new(out/'preparation.json',dict(schema='explicit-candidate-matrix/v1',builder=binding(__file__),
        candidate_source=source_ref,options=options_ref,policy_evidence=declaration['evidence'],
        baseline_ownership=binding(ledger_path),old_baseline_algorithms_unchanged=True,old_baseline_values_unchanged=True,
        actual_image_preflight_pending=True,enqueued=False,driver_completion_integration_required=True))
    return campaign


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base','profiles','gaps','out','candidate-source','runtime-options','policy-evidence'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('compatibility','external-campaign','completion-receipt','history-manifest','publication-campaign'):
        parser.add_argument('--'+name,type=Path,action='append',default=[])
    parser.add_argument('--external-state',type=Path);parser.add_argument('--root',type=Path)
    parser.add_argument('--point-options',type=Path)
    parser.add_argument('--preview-only',action='store_true');parser.add_argument('--startup-helper',type=Path)
    parser.add_argument('--startup-contract-mode',choices=('external_audit','inprocess'),default='external_audit')
    a=parser.parse_args()
    value=prepare(a.base,a.profiles,a.gaps,a.out,candidate_source=a.candidate_source,
        runtime_options_path=a.runtime_options,policy_evidence_path=a.policy_evidence,
        compatibility_paths=a.compatibility,external_campaign_paths=a.external_campaign,
        external_state_path=a.external_state,completion_receipt_paths=a.completion_receipt,
        history_manifest_paths=a.history_manifest,root_path=a.root,preview_only=a.preview_only,
        startup_helper_path=a.startup_helper,point_options_path=a.point_options,startup_contract_mode=a.startup_contract_mode,
        publication_campaign_paths=a.publication_campaign)
    print(json.dumps(dict(out=str(a.out),preview_only=a.preview_only,points=len(value.get('plans',value.get('points',[]))))))


if __name__=='__main__': main()
