"""One-observation SLO boundary exploration inside a pre-authorized lease.

This deliberately does not weaken ``slo_capacity``'s repeated tuning protocol.
Evaluation observations choose load only; controller/profile parameters remain
fixed. Each generated point and decision is immutable before it can execute.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import time

from .resident_session import digest, engine_signature, file_sha, write_new

SCHEMA = 'single_observation_slo_boundary/v1'
DATASETS = ('alpaca', 'sharegpt', 'longbench')


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def read_bound(ref):
    if not isinstance(ref, dict) or file_sha(ref['path']) != ref['sha256']:
        raise ValueError('boundary artifact binding differs')
    return json.loads(Path(ref['path']).read_text())


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_policy(policy, group=None):
    if (policy.get('schema') != SCHEMA or policy.get('mode') != 'policy'
            or policy.get('duration_s') != 150 or policy.get('seed') != 701
            or policy.get('growth_factor') != 1.25
            or type(policy.get('max_steps_per_dataset')) is not int
            or policy['max_steps_per_dataset'] < 1
            or not positive(policy.get('soft_deadline_s'))
            or not policy.get('run_id') or not policy.get('revision')):
        raise ValueError('invalid single-observation boundary policy')
    if not policy.get('datasets') or set(policy['datasets']) - set(DATASETS):
        raise ValueError('boundary policy dataset inventory differs')
    for dataset, values in policy['datasets'].items():
        point = read_bound(values['template_point'])
        if (point['system'] != 'pdblend' or point['dataset'] != dataset
                or point['model_id'] != policy['model_id'] or point['revision'] != policy['revision']
                or point['seed'] != 701 or point['duration_s'] != 150
                or engine_signature(point['engine_identity']) != policy['engine_signature']
                or not positive(values['base_rate_rps'])
                or not math.isclose(point['rate_rps']/point['scale'], values['base_rate_rps'])):
            raise ValueError('boundary template differs from authorized execution')
        trace = read_bound(point['trace'])
        if trace.get('corpus_sha256') != values['corpus']['sha256']:
            raise ValueError('extension corpus differs from the original evaluation corpus')
        prior = read_bound(point['inputs']['planning_trace'])
        if prior.get('selection_split') not in ('calibration', 'tuning'):
            raise ValueError('boundary choices require independent planning input')
    if group is not None:
        if (group['model_id'] != policy['model_id']
                or group['engine_signature'] != policy['engine_signature']
                or any(p['system'] != 'pdblend' or p['revision'] != policy['revision']
                       for p in group['points'])):
            raise ValueError('extension policy does not authorize this resident group')
    return policy


def build_policy(group, out, *, run_id, corpus_refs, base_rates,
                 max_steps_per_dataset=12, soft_deadline_s=14400, previous_manifest=None):
    """Freeze authorization before publishing the immutable group/job."""
    out = Path(out)
    datasets = {}
    for dataset in DATASETS:
        candidates = [p for p in group['points'] if p['dataset'] == dataset and p['scale'] == 1.]
        if not candidates:
            continue
        if len(candidates) != 1:
            raise ValueError('one scale-one template per dataset is required')
        path = out/(dataset+'-template.json')
        write_new(path, candidates[0])
        datasets[dataset] = dict(template_point=binding(path), corpus=corpus_refs[dataset],
                                 base_rate_rps=base_rates[dataset])
    policy = dict(schema=SCHEMA, mode='policy', run_id=run_id, model_id=group['model_id'],
        revision=group['points'][0]['revision'], engine_signature=group['engine_signature'],
        duration_s=150, seed=701, growth_factor=1.25, max_steps_per_dataset=max_steps_per_dataset,
        soft_deadline_s=soft_deadline_s, datasets=datasets,
        statistical_scope='single_observation_no_significance_or_exact_capacity',
        selection_scope='evaluation_selects_next_load_only_parameters_frozen')
    if previous_manifest is not None:
        raise ValueError('resume must reuse the original policy and pass the previous session to --previous')
    validate_policy(policy, group)
    path = out/'policy.json'
    write_new(path, policy)
    return binding(path)


def observation_verdict(point, receipt):
    """Energy/frequency diagnostics never turn into a failing SLO bound."""
    metrics = receipt.get('result', {}).get('metrics', {})
    if receipt.get('cleanup_passed') is not True or receipt.get('error'):
        return dict(verdict='incomplete', reason='execution_or_drain_failure')
    start = metrics.get('service_start_s', metrics.get('service_started_s'))
    end = metrics.get('service_end_s', metrics.get('service_finished_s'))
    if (metrics.get('duration_s') != 150 or not positive(start) or not positive(end)
            or not math.isclose(end-start, 150., abs_tol=1e-5, rel_tol=0)):
        return dict(verdict='incomplete', reason='missing_complete_service_timing')
    keys = ('offered_requests', 'successful_requests', 'failed_requests', 'joint_slo_requests',
            'ttft_samples', 'tpot_samples')
    if any(type(metrics.get(k)) is not int or metrics[k] < 0 for k in keys):
        return dict(verdict='incomplete', reason='missing_request_counts')
    offered, success, failed, good = (metrics[k] for k in keys[:4])
    if (not offered or success+failed != offered or good > success
            or metrics.get('invalid_timing_requests', 0) or metrics.get('unresolved_requests', 0)
            or metrics['ttft_samples'] != success or metrics['tpot_samples'] != success):
        return dict(verdict='incomplete', reason='incomplete_or_inconsistent_request_timing')
    if success and any(type(metrics.get(k+'_p99_s')) not in (int, float)
            or not math.isfinite(metrics[k+'_p99_s']) or metrics[k+'_p99_s'] < 0 for k in ('ttft', 'tpot')):
        return dict(verdict='incomplete', reason='missing_latency_percentiles')
    passed = success == offered and good/offered >= .9 and all(
        metrics[k+'_p99_s'] <= point['slo'][k+'_s'] for k in ('ttft', 'tpot'))
    return dict(verdict='pass' if passed else 'fail', reason='measured_slo',
                low_sample=success < 100)


def validate_generated_point(policy, point, *, load=read_bound):
    values = policy['datasets'][point['dataset']]
    template = load(values['template_point'])
    for key in ('model_id','system','revision','seed','duration_s','slo','engine_identity',
                'source_manifest','topology','observation_scope','qualification_mode','result_policy'):
        if point.get(key) != template.get(key):
            raise ValueError('generated point changed frozen configuration: '+key)
    for key in ('system_config','profiles','source_manifest','qualifications'):
        if point['inputs'].get(key) != template['inputs'].get(key):
            raise ValueError('generated point changed frozen inputs: '+key)
    if (not positive(point['scale']) or not math.isclose(point['rate_rps'],
            values['base_rate_rps']*point['scale'], rel_tol=1e-12)
            or point['inputs']['trace'] != point['trace']):
        raise ValueError('generated point rate/trace differs from its authorization')
    return point


def boundary_state(observations, *, factor=1.25):
    """An observed bracket, never a statistical or global monotonic claim."""
    rates = {}
    for row in observations:
        scale = row['scale']
        if not positive(scale) or row['verdict'] not in ('pass', 'fail', 'incomplete'):
            raise ValueError('invalid boundary observation')
        if scale in rates and rates[scale] != row['verdict']:
            return dict(status='conflicting_observations', next_rate_scale=None,
                        passed_lower=None, failed_upper=None, saturation_observed=False, nonmonotonic=True)
        rates[scale] = row['verdict']
    passed = sorted(r for r, v in rates.items() if v == 'pass')
    failed = sorted(r for r, v in rates.items() if v == 'fail')
    lower = max(passed, default=None)
    upper = min((r for r in failed if lower is None or r > lower), default=None)
    nonmonotonic = lower is not None and any(r < lower for r in failed)
    base = dict(passed_lower=lower, failed_upper=upper, nonmonotonic=nonmonotonic,
                saturation_observed=upper is not None, next_rate_scale=None,
                statistical_scope='single_observation_no_significance_or_exact_capacity')
    if 'incomplete' in rates.values():
        return dict(base, status='incomplete_observation')
    if lower is not None and upper is not None and upper/lower <= factor+1e-12:
        return dict(base, status='bracketed', relative_width=upper/lower-1)
    if lower is None:
        next_rate = upper/factor if upper is not None else 1.
        status = 'searching_lower' if upper is not None else 'unmeasured'
    else:
        next_rate, status = lower*factor, 'expanding' if upper is None else 'narrowing'
    if not positive(next_rate) or next_rate in rates:
        return dict(base, status='numeric_search_limit')
    return dict(base, status=status, next_rate_scale=next_rate)


def make_extension_point(policy_ref, dataset, scale, out):
    """Fresh full evaluation trace; independent tuning time scaling and planning."""
    from .client import poisson_trace
    from .independent_dispatch import request_rows
    from .run import offline_forecast
    from pdblend.control.policies import get_policy
    from pdblend.planner.pool import PlannerConfig, PoolPlanner, SLO
    from pdblend.profile.query.versions import load_profile
    policy = validate_policy(read_bound(policy_ref))
    values = policy['datasets'][dataset]
    point = deepcopy(read_bound(values['template_point']))
    out = Path(out)
    rate = values['base_rate_rps']*scale
    if not positive(rate):
        raise ValueError('invalid generated rate')
    corpus = read_bound(values['corpus'])
    records = [r for r in corpus['evaluation'] if r['output_tokens'] >= 2]
    trace = deepcopy(read_bound(point['trace']))
    requests = [asdict(r) for r in poisson_trace(records, rate, 150., 701, dataset)]
    trace.update(rate_rps=rate, duration_s=150., seed=701, selection_split='evaluation',
                 requests=requests, boundary_policy=policy_ref)
    write_new(out/'trace.json', trace)
    prior = read_bound(point['inputs']['planning_trace'])
    factor = offline_forecast(request_rows(prior)).rate_rps/rate
    tuning = dict(prior, source_trace=point['inputs']['planning_trace'], time_scale=factor,
        prescribed_rate_rps=rate, duration_s=prior['duration_s']*factor,
        evaluation_used_for_selection=False,
        requests=[dict(r, arrival_s=r['arrival_s']*factor) for r in prior['requests']])
    write_new(out/'planning-trace.json', tuning)
    config = read_bound(point['inputs']['system_config'])
    previous = read_bound(point['inputs']['offline_choice'])
    selected = config['profile']; read_bound(selected)
    tp, pp = previous['plan']['tp'], previous['plan']['pp']
    loaded = load_profile(selected['path'], system='pdblend', model_id=point['model_id'], tp=tp, pp=pp,
                          usage='development')
    batch = {r['launch_options']['max_num_seqs'] for r in point['engine_identity']['instances']}
    if len(batch) != 1:
        raise ValueError('heterogeneous batch limits cannot share a boundary lease')
    policy_impl = get_policy('pdblend')
    cfg = policy_impl.planner_config(PlannerConfig(slots=len(point['engine_identity']['instances']),
        slo=SLO(**point['slo']), freqs=loaded.model.freqs, max_num_seqs=batch.pop()))
    from .pdblend_runtime_options import comparison_options, comparison_capacity_floors
    checked_options = comparison_options(config, point['inputs']['system_config']['path'], point=point)
    options = checked_options['values']
    cfg.capacity_floor_context = checked_options.get('capacity_floor_context', {})
    cfg.preserve_overload_capacity = options['preserve_overload_capacity']
    cfg.pressure_controls = policy_impl.dynamic_m_floor
    cfg.capacity_floor_reserve_canonical = options['capacity_floor_reserve_canonical']
    if options['capacity_floor_path'] is not None:
        cfg.capacity_floors, _ = comparison_capacity_floors(options['capacity_floor_path'], model=loaded.model)
    if options['transition_catalog_path'] is not None:
        from pdblend.planner.capacity import select_artifact
        from pdblend.planner.transitions import TransitionCatalog
        cfg.transition_estimator = TransitionCatalog.load(select_artifact(options['transition_catalog_path'], loaded.model),
            model=loaded.model, qualified_only=options['transition_qualified_only'])
    forecast = offline_forecast(request_rows(tuning))
    planner = PoolPlanner(loaded.model, cfg)
    plan = planner.plan(forecast)
    if policy_impl.dynamic_m_floor:
        pressure = planner.mixed_pressure(forecast, plan.counts.get('M', 0), plan.f_M)
        cfg.pd_pressure_active = (forecast.input_p95 >= cfg.pd_min_input_tokens
                                 and pressure['pressure'] >= policy_impl.pd_pressure_enter)
        plan = planner.plan(forecast)
    plan = replace(plan, tp=tp, pp=pp, profile_key=json.dumps(loaded.profile_key, sort_keys=True, separators=(',', ':')))
    from .pdblend_observation_plan import encode_plan
    choice = dict(previous, **encode_plan(plan), planning_trace_sha256=binding(out/'planning-trace.json')['sha256'],
        profile_sha256=selected['sha256'], selection_split='tuning', evaluation_used_for_selection=False,
        boundary_policy=policy_ref)
    write_new(out/'choice.json', choice)
    size = point['model_id'].split('-')[1].lower()
    point.update(name=f'{size}-pdblend-{dataset}-x{scale:.12g}-seed701', scale=scale, rate_rps=rate,
        trace=binding(out/'trace.json'), status='prepared', blockers=[],
        boundary_policy=policy_ref, run_id=policy['run_id'], experiment_phase='slo_boundary_extension')
    point['inputs'].update(trace=point['trace'], offline_choice=binding(out/'choice.json'),
                           planning_trace=binding(out/'planning-trace.json'))
    write_new(out/'point.json', point)
    return point, binding(out/'point.json')


def read_extension_manifest(ref_or_path, *, load_bound=None):
    """Validate small immutable manifests; raw window audits belong to exporter."""
    load = load_bound or read_bound
    ref = ref_or_path if isinstance(ref_or_path, dict) else binding(ref_or_path)
    manifest = load(ref)
    if manifest.get('schema') == SCHEMA and manifest.get('mode') == 'pointer':
        ref = manifest['manifest']; manifest = load(ref)
    if manifest.get('schema') != SCHEMA or manifest.get('mode') != 'manifest':
        raise ValueError('unsupported boundary extension manifest')
    policy = load(manifest['policy'])
    if policy.get('schema') != SCHEMA or policy.get('mode') != 'policy':
        raise ValueError('extension manifest lacks a frozen policy')
    for key in ('run_id', 'model_id', 'revision', 'engine_signature'):
        if manifest.get(key) != policy.get(key):
            raise ValueError('extension manifest policy identity differs: '+key)
    points, receipts, seen = [], [], set()
    observations = manifest.get('observations', [])
    observed_rates = set()
    for row in observations:
        point = load(row['point']); receipt = load(row['receipt'])
        key = (row['dataset'], row['scale'])
        if (key in observed_rates or point['system'] != 'pdblend'
                or point['model_id'] != policy['model_id'] or point['revision'] != policy['revision']
                or engine_signature(point['engine_identity']) != policy['engine_signature']
                or point['dataset'] != row['dataset'] or point['scale'] != row['scale']
                or point['dataset'] not in policy['datasets']
                or receipt.get('point_sha256') != digest(point)
                or receipt.get('point') != point['name']):
            raise ValueError('boundary observation identity differs')
        verdict = observation_verdict(point, receipt)
        if any(row.get(k) != value for k, value in verdict.items()):
            raise ValueError('boundary observation verdict differs from measured receipt')
        observed_rates.add(key); receipts.append(row['receipt'])
    expected_states = states_with_endpoints(observations, policy['datasets'])
    if manifest.get('boundaries') != expected_states:
        raise ValueError('boundary endpoints differ from measured observations')
    previous = None
    for ref_decision in manifest.get('decisions', []):
        decision = load(ref_decision)
        if (decision.get('policy') != manifest['policy'] or decision.get('previous_decision') != previous):
            raise ValueError('extension decision chain differs')
        authorized = load(decision['point'])
        state = boundary_state(decision.get('observations', []))
        if (state != decision.get('state_before') or decision.get('scale') != state.get('next_rate_scale')
                or authorized['dataset'] != decision['dataset'] or authorized['scale'] != decision['scale']):
            raise ValueError('extension decision does not replay the frozen load rule')
        for observed in decision.get('observations', []):
            if observed not in observations:
                raise ValueError('decision evidence is absent from its observation ledger')
        previous = ref_decision
    decision_refs = {r['sha256'] for r in manifest.get('decisions', [])}
    for entry in manifest.get('points', []):
        point = load(entry['point'])
        validate_generated_point(policy, point, load=load)
        if (point['name'] in seen or point['system'] != 'pdblend'
                or point['model_id'] != policy['model_id'] or point['revision'] != policy['revision']
                or point.get('boundary_policy') != manifest['policy']
                or point['seed'] != 701 or point['duration_s'] != 150
                or engine_signature(point['engine_identity']) != policy['engine_signature']
                or entry['decision']['sha256'] not in decision_refs):
            raise ValueError('extension point is outside its immutable policy')
        decision = load(entry['decision'])
        if decision.get('point') != entry['point']:
            raise ValueError('decision does not authorize generated point')
        seen.add(point['name']); points.append(point)
        if entry.get('receipt'):
            receipt = load(entry['receipt'])
            if receipt.get('point_sha256') != digest(point):
                raise ValueError('extension receipt point differs')
            if entry['receipt'] not in receipts:
                receipts.append(entry['receipt'])
    return dict(manifest=manifest, manifest_ref=ref, points=points, receipts=receipts, policy=policy)


def states_with_endpoints(observations, datasets):
    states = {}
    for dataset in datasets:
        rows = [r for r in observations if r['dataset'] == dataset]
        state = boundary_state(rows)
        for label in ('passed_lower', 'failed_upper'):
            endpoint = next((r for r in rows if r['scale'] == state.get(label)), None)
            state[label+'_point'] = endpoint['point'] if endpoint else None
            state[label+'_receipt'] = endpoint['receipt'] if endpoint else None
        states[dataset] = state
    return states


class BoundaryPointSource:
    def __init__(self, group, out, *, previous=(), point_factory=None, now=time.time):
        self.ref, self.out, self.now = group['extension_policy'], Path(out), now
        self.policy = validate_policy(read_bound(self.ref), group)
        self.factory = point_factory or make_extension_point
        self.started = now(); self.entries = []; self.decisions = []; self.observations = []
        self.steps = {d:0 for d in self.policy['datasets']}; self.counter = 0
        self.last_manifest = None; self.stop_reason = None
        candidates = []
        if self.policy.get('previous_manifest'):
            candidates.append(self.policy['previous_manifest'])
        for root in previous:
            pointer = Path(root)/'extensions/latest.json'
            if pointer.exists():
                candidates.append(binding(pointer))
        for candidate in candidates:
            loaded = read_extension_manifest(candidate)
            if loaded['manifest']['policy'] != self.ref:
                raise ValueError('resume requires the identical extension policy binding')
            # A resumed manifest contains its entire predecessor ledger.
            manifest = loaded['manifest']
            if len(manifest.get('decisions', [])) >= len(self.decisions):
                self.entries = deepcopy(manifest.get('points', []))
                self.decisions = deepcopy(manifest.get('decisions', []))
                self.observations = deepcopy(manifest.get('observations', []))
                self.last_manifest = loaded['manifest_ref']
        self.publish()

    def observe(self, point, receipt_ref):
        receipt = read_bound(receipt_ref)
        if receipt.get('point_sha256') != digest(point):
            raise ValueError('boundary observation receipt differs from executed point')
        if any(r['receipt'] == receipt_ref for r in self.observations):
            return
        if any(r['dataset'] == point['dataset'] and r['scale'] == point['scale'] for r in self.observations):
            raise ValueError('boundary cannot select between repeated observations')
        point_path = Path(receipt_ref['path']).parent/'point.json'
        point_ref = binding(point_path)
        if read_bound(point_ref) != point:
            raise ValueError('boundary receipt point artifact differs')
        self.observations.append(dict(dataset=point['dataset'], scale=point['scale'],
            point=point_ref, receipt=receipt_ref, **observation_verdict(point, receipt)))
        for entry in self.entries:
            if read_bound(entry['point'])['name'] == point['name']:
                entry['receipt'] = receipt_ref
        self.publish()

    def states(self):
        return states_with_endpoints(self.observations, self.policy['datasets'])

    def publish(self):
        manifest = dict(schema=SCHEMA, mode='manifest', policy=self.ref,
            **{k:self.policy[k] for k in ('run_id','model_id','revision','engine_signature')},
            points=self.entries, decisions=self.decisions, observations=self.observations,
            boundaries=self.states(), stop_reason=self.stop_reason,
            continuation_required=self.stop_reason == 'lease_budget_reached',
            previous_manifest=self.last_manifest)
        path = self.out/f'manifest-{self.counter:06d}.json'; self.counter += 1
        write_new(path, manifest); self.last_manifest = binding(path)
        self.out.mkdir(parents=True, exist_ok=True)
        temp = self.out/'latest.tmp'
        temp.write_text(json.dumps(dict(schema=SCHEMA, mode='pointer', manifest=self.last_manifest), sort_keys=True)+'\n')
        os.replace(temp, self.out/'latest.json')
        return self.last_manifest

    def next_point(self):
        for entry in self.entries:
            if not entry.get('receipt'):
                return read_bound(entry['point'])
        if self.now()-self.started >= self.policy['soft_deadline_s']:
            self.stop_reason = 'lease_budget_reached'; self.publish(); return None
        states = self.states()
        datasets = list(self.policy['datasets'])
        offset = len(self.decisions) % len(datasets)
        for dataset in datasets[offset:]+datasets[:offset]:
            state = states[dataset]
            scale = state.get('next_rate_scale')
            if scale is None or self.steps[dataset] >= self.policy['max_steps_per_dataset']:
                continue
            point, ref = self.factory(self.ref, dataset, scale,
                self.out/'generated'/f'{len(self.decisions):06d}-{dataset}')
            validate_generated_point(self.policy, point)
            if engine_signature(point['engine_identity']) != self.policy['engine_signature']:
                raise ValueError('extension factory changed engine compatibility')
            decision = dict(schema=SCHEMA, mode='decision', policy=self.ref,
                previous_decision=self.decisions[-1] if self.decisions else None,
                dataset=dataset, scale=scale, state_before=boundary_state(
                    [r for r in self.observations if r['dataset'] == dataset]),
                observations=[r for r in self.observations if r['dataset'] == dataset], point=ref)
            path = self.out/'decisions'/f'{len(self.decisions):06d}.json'
            write_new(path, decision); decision_ref = binding(path)
            self.decisions.append(decision_ref)
            self.entries.append(dict(point=ref, decision=decision_ref, receipt=None))
            self.steps[dataset] += 1; self.publish()
            return point
        self.stop_reason = ('boundaries_complete' if all(v['status'] == 'bracketed' for v in states.values())
                            else 'lease_budget_reached' if any(v.get('next_rate_scale') is not None for v in states.values())
                            else 'incomplete_boundary_evidence')
        self.publish()
        return None
