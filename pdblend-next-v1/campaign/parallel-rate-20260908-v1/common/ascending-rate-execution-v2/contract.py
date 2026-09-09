"""Pure ascending schedule, exact observation identity and paired-workload contract."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import math
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[1]
SYSTEMS = ('pdblend', 'mixed', 'distserve', 'dynamollm', 'ecoserve')
HOSTS = {'7b': 'C', '14b': 'A', '32b': 'B'}
SLOS = {'alpaca': (1., .1), 'sharegpt': (5., .15), 'longbench': (15., .2)}
INPUTS = {
    'plan_manifest': {'path': str(ROOT/'planning/ascending-rate-redesign-20260909-v1/manifest.json'), 'sha256': '7dfa8c80291606fe076967cab310ae4392be8caf18db399ccfd0505d81777e20'},
    'review_manifest': {'path': str(ROOT/'reports/ascending-rate-review-002/manifest.json'), 'sha256': 'c2bc23f21dae0abf940c09295da22f537f3347c18509af86c08d57ef6e5fa4e7'},
    'off95': {'path': str(ROOT/'common/idle-domain-off-existing95-reuse-declaration-v7.json'), 'sha256': '51a42d5ac1a27c6b39657f124c6b8319ee6d14389c00c3b60d269e976150d302'},
}


def need(value, message):
    if not value:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n').encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def checked(reference):
    need(isinstance(reference, dict) and set(reference) >= {'path', 'sha256'}, 'explicit saved reference required')
    need(sha(reference['path']) == reference['sha256'], 'saved reference changed: '+reference['path'])
    return read(reference['path'])


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def number(value):
    need(not isinstance(value, bool), 'rate cannot be boolean')
    value = Decimal(str(value))
    need(value.is_finite() and value > 0, 'positive finite rate required')
    return format(value.normalize(), 'f')


def position_id(node, model, dataset, rate):
    return f'ascending-v1-{node}-{model}-{dataset}-r{number(rate)}-s701'


def cell_id(node, model, dataset, rate, system, repeat):
    return f'{position_id(node, model, dataset, rate)}-w100-{system}-slo1-repeat{repeat}'


def identity(row):
    return (row['model'], row['dataset'], number(row['rate_rps']), row['seed'],
            row['trace_sha256'], row['content_pairing_sha256'],
            row['slo_ttft_s'], row['slo_tpot_s'],
            row.get('measurement_host', row.get('node')))


def load_inputs():
    plan_m = checked(INPUTS['plan_manifest'])
    review_m = checked(INPUTS['review_manifest'])
    plan_path = Path(INPUTS['plan_manifest']['path']).parent/'plan.json'
    review_path = Path(INPUTS['review_manifest']['path']).parent/'results.json'
    plan = checked(dict(path=str(plan_path), sha256=plan_m['files']['plan.json']))
    review = checked(dict(path=str(review_path), sha256=review_m['files']['results.json']))
    snapshot = checked(review['source_snapshot'])
    a_path = ROOT/'A/final-p9-code-001/work-declaration.json'
    a = read(a_path)
    sources = dict(review_m['sources'])
    for mr, m in ((INPUTS['plan_manifest'], plan_m), (INPUTS['review_manifest'], review_m)):
        sources[mr['path']] = mr['sha256']
        sources.update(m.get('sources', {}))
        for name, digest in m['files'].items():
            sources[str(Path(mr['path']).parent/name)] = digest
    sources.update({r['path']: r['sha256'] for r in INPUTS.values()})
    sources[str(a_path)] = sha(a_path)
    return plan, review, snapshot, a, sources


def saved_task(observation):
    return dict(action='reuse', observation_id='saved:'+observation['checkpoint']['sha256'],
                cell_id=observation['cell_id'], repeat=observation.get('repeat', 1),
                checkpoint=observation['checkpoint'], actual_source_preserved=True)


def new_task(row):
    return dict(action='execute', observation_id='new:'+row['cell_id'],
                cell_id=row['cell_id'], repeat=row['repeat'], row=row)


def row_for(workload, node, system, repeat, sequence):
    row = copy.deepcopy(workload)
    row.update(cell_id=cell_id(node, row['model'], row['dataset'], row['rate_rps'], system, repeat),
               system=system, repeat=repeat, node=node, measurement_host=node,
               phase='main', part='main', slo_scale=1., allowed_slo_scales=[1.],
               slo_ttft_s=row['slo']['ttft_s'], slo_tpot_s=row['slo']['tpot_s'],
               slo_attainment_target=.9, sequence=sequence, execution_status='not_run',
               original_actual_source_relabelled=False, execution_binding_required=True,
               policy_binding_required=True, controller_config=None, strategy=None,
               reuse_main_cell_id=None, actual_fixed_window_verified=False,
               request_hard_timeout_s=120., drain_after_arrival_window_s=120.,
               formal_eligible=False, fixed_slo_only=True)
    return row


def derive(plan, review, snapshot, a_declaration, new_workloads):
    """Derive all rows and pairings from pinned inventories, never scan by ID."""
    reused = copy.deepcopy(review['reuse_candidate_observations'])
    need(len(reused) == 479 and len({p['checkpoint']['path'] for p in reused}) == 479, 'exact479 distinct source observations')
    pdb = [p for p in reused if p['system'] == 'pdblend']
    bases = [p for p in reused if p['system'] != 'pdblend']
    need(len(pdb) == 95 and len(bases) == 384, 'exact95 PDB and384 baseline sources')
    by_saved_id = {(p['cell_id'], p['measurement_host'], p['system']): p for p in reused}
    need(len(by_saved_id) == 479, 'ambiguous actual saved observation identity')
    new_map = {(w['model'], w['dataset'], number(w['rate_rps'])): w for w in new_workloads}
    need(len(new_map) == 8, 'exact8 new trace coordinates')
    a_sources = {(r['dataset'], number(r['source_row']['rate_rps']), r['repeat']): r
                 for r in a_declaration['cells'] if r['dataset'] in ('sharegpt', 'longbench')}
    cells, workloads, positions = [], [], []
    for old_position in plan['rate_positions']:
        model, dataset, node, rate = (old_position[k] for k in ('model', 'dataset', 'node', 'rate_rps'))
        key = model, dataset, number(rate)
        existing = [p for p in pdb if (p['model'], p['dataset'], number(p['rate_rps'])) == key]
        renewing = model == '14b' and dataset in ('sharegpt', 'longbench')
        if old_position['new_rate_coordinate']:
            workload = copy.deepcopy(new_map[key])
        elif renewing:
            source = a_sources[(dataset, number(rate), 1)]
            workload = copy.deepcopy(source['source_row'])
            workload['parent_source_row'] = dict(declaration=ref(ROOT/'A/final-p9-code-001/work-declaration.json'),
                                                cell_id=source['cell_id'])
        else:
            need(existing, 'missing declared retained PDB point')
            workload = copy.deepcopy(checked(existing[0]['checkpoint'])['row'])
            workload['parent_source_row'] = dict(checkpoint=existing[0]['checkpoint'])
        workload.update(model=model, dataset=dataset, rate_rps=float(rate), rate_rps_decimal=number(rate),
                        seed=701, arrival_seed=701, sampling_seed=20260907,
                        arrival_window_s=100., trace_duration_s=100., node=node,
                        workload_id=f'{model}-{dataset}-r{number(rate)}-s701-w100')
        trace_ref = dict(path=workload['trace_path'], sha256=workload['trace_sha256'])
        trace = checked(trace_ref)
        need((trace['model'], trace['dataset'], number(trace['rate']), trace['seed'], trace['sampling_seed']) ==
             (model, dataset, number(rate), 701, 20260907), 'trace model/rate/seed differs')
        need(trace['arrival_window_s'] == trace['duration_s'] == 100 and trace['request_hard_timeout_s'] == 120,
             'trace window/deadline differs')
        need(trace['slo'] == workload['slo'] and trace['n_requests'] == workload['n_requests'] == len(trace['requests']),
             'trace SLO/request count differs')
        workload.update(trace=trace_ref['path'], trace_reference=trace_ref,
                        expected_generated_tokens=sum(r['output_len'] for r in trace['requests']),
                        source_indices_sha256=hashlib.sha256(encode(trace['source_pool_indices'])).hexdigest())
        need(trace['content_pairing_sha256'] == workload['content_pairing_sha256'], 'workload content mismatch')
        workloads.append(workload)
        tasks = {s: [] for s in SYSTEMS}
        for s, count in old_position['new_executions_by_system'].items():
            for repeat in range(1, count+1):
                row = row_for(workload, node, s, repeat, len(cells)+1)
                cells.append(row)
                tasks[s].append(new_task(row))
        tasks['pdblend'] += [saved_task(p) for p in sorted(existing, key=lambda p: p['repeat'])]
        position = dict(position_id=position_id(node, model, dataset, rate), node=node, model=model,
                        dataset=dataset, order=old_position['order'], rate_rps=float(rate),
                        rate_rps_decimal=number(rate), trace=trace_ref,
                        workload=workload, newly_added_coordinate=old_position['new_rate_coordinate'],
                        required_pdb_repeats=old_position['pdblend_repeats'], systems=tasks,
                        new_executions_before_cap_pruning=old_position['new_executions_before_cap_pruning'],
                        conditional_on_no_lower_rate_cap=True)
        if renewing and not old_position['new_rate_coordinate']:
            for s in SYSTEMS[1:4]:
                candidates = [p for p in bases if p['system'] == s and p['measurement_host'] == node
                              and (p['model'], p['dataset'], number(p['rate_rps'])) == key
                              and p['review_use'] == 'restore_original_A_three_baseline60_in_new_declaration']
                need(len(candidates) == 1 and candidates[0]['repeat'] == 1, 'exact A original baselineR1')
                need(identity(candidates[0]) == identity(dict(workload, measurement_host=node)), 'A old baseline exact trace/SLO mismatch')
                tasks[s].append(saved_task(candidates[0]))
        positions.append(position)
    by_position = {p['position_id']: p for p in positions}
    pairing = []
    for pair in snapshot['pairs']:
        p = by_saved_id[(pair['cell_id'], pair['measurement_host'], 'pdblend')]
        b = by_saved_id[(pair['baseline_cell_id'], pair['baseline_measurement_host'], pair['baseline_system'])]
        need(identity(p) == identity(b), 'inherited pairing identity changed')
        pos = by_position[position_id(p['measurement_host'], p['model'], p['dataset'], p['rate_rps'])]
        btask = saved_task(b)
        if btask not in pos['systems'][b['system']]:
            pos['systems'][b['system']].append(btask)
        pairing.append(dict(position_id=pos['position_id'], pdb=saved_task(p), baseline=btask,
                            baseline_system=b['system'], baseline_repeat_reused=p['repeat'] != b['repeat'],
                            pair_origin='exact_prior_pair', trace=pos['trace'],
                            prior_pair_reference=dict(snapshot=review['source_snapshot'], pdb_cell_id=p['cell_id'],
                                                      baseline_cell_id=b['cell_id'], baseline_system=b['system'])))
    for pos in positions:
        for pt in pos['systems']['pdblend']:
            if pt['action'] != 'execute':
                continue
            for s in SYSTEMS[1:]:
                available = pos['systems'][s]
                match = [b for b in available if b['repeat'] == pt['repeat']]
                if not match:
                    need(not pos['newly_added_coordinate'] and len(available) == 1 and available[0]['repeat'] == 1,
                         'only declared existing-grid baselineR1 may serve PDBR2')
                    match = available
                need(len(match) == 1, 'pair must have exactly one baseline observation')
                pairing.append(dict(position_id=pos['position_id'], pdb=pt, baseline=match[0],
                    baseline_system=s, baseline_repeat_reused=pt['repeat'] != match[0]['repeat'],
                    pair_origin='new_exact_repeat' if pos['newly_added_coordinate'] else 'A_existing_grid_explicit_R1_reuse',
                    trace=pos['trace']))
    groups = []
    for group in plan['groups']:
        relevant = [p for p in positions if p['model'] == group['model'] and p['dataset'] == group['dataset']]
        groups.append(dict(group_id=f"{group['node']}-{group['model']}-{group['dataset']}",
                           **group, position_ids=[p['position_id'] for p in relevant],
                           pdb_source_contract=dict(target_manifest=ref(ROOT/f"hosts/{group['model']}-capacity-p12/manifest.json"),
                             idle_domain_reacquire_v1=False, actual_reused_sources_preserved=True,
                             existing95_equivalence=INPUTS['off95'],
                             numerical_profile_and_config_qualification_required=True),
                           actual_execution_binding_required=True))
    return dict(groups=groups, positions=positions, workloads=workloads, cells=cells,
                reused_observations=reused, pairings=pairing,
                historical_A_observations=copy.deepcopy(review['preserved_historical_A_observations']),
                stopped_P12_history=copy.deepcopy(review['preserved_stopped_P12_observations']))


def validate_structure(d):
    need(d['schema'] == 'ascending-rate-execution-declaration-v1', 'wrong execution schema')
    need(d['legacy_scope_inherited'] is False and d['old104_scope_resumed'] is False
         and d['fixed_slo_only'] is True and d['baseline_saturation_search_enabled'] is False, 'oldscope or SLO scale forbidden')
    need(d['counts'] == dict(rate_positions=102, pending_positions=28, new_rate_coordinates=8,
         pending_logical_system_cells=80, initial_runs_upper_bound=124,
         initial_runs_by_node={'A':64,'B':10,'C':50}, reused_PDB=95,
         unique_reused_paired_baselines=324, restored_A_baselines=60,
         exact_potential_pairings=540), 'declared count metadata differs')
    need(d['campaign_deadline_s'] is None and d['arrival_seed'] == 701
         and d['arrival_window_s'] == 100 and d['request_hard_timeout_s'] == 120,
         'top-level workload/deadline changed')
    pos, cells, reused = d['positions'], d['cells'], d['reused_observations']
    need(len(pos) == 102 and len(cells) == 124 and len(reused) == 479 and len(d['pairings']) == 540, 'scope count mismatch')
    need(len({p['position_id'] for p in pos}) == 102 and len({r['cell_id'] for r in cells}) == 124, 'duplicate position/cell')
    need(sum(bool(p['new_executions_before_cap_pruning']) for p in pos) == 28, 'exact28 pending positions')
    need(sum(p['newly_added_coordinate'] for p in pos) == 8, 'exact8 new coordinates')
    need({n: sum(r['node'] == n for r in cells) for n in 'ABC'} == {'A': 64, 'B': 10, 'C': 50}, 'node run counts differ')
    need(len({(r['model'], r['dataset'], r['rate_rps'], r['system']) for r in cells}) == 80, 'exact80 logical system cells')
    saved = {'saved:'+p['checkpoint']['sha256']: p for p in reused}
    new = {'new:'+p['cell_id']: p for p in cells}
    need(len(saved) == 479 and not set(saved).intersection(new), 'source observation identity duplicate')
    for group in d['groups']:
        ps = [p for p in pos if p['model'] == group['model'] and p['dataset'] == group['dataset']]
        rates = [Decimal(p['rate_rps_decimal']) for p in ps]
        need(rates == sorted(set(rates)), 'group rate order not strictly ascending')
        need(group['node'] == HOSTS[group['model']], 'cross-host model allocation changed')
        need(group['pdb_source_contract']['idle_domain_reacquire_v1'] is False, 'OFF95 cannot authorize opt-in source reuse')
    for p in pos:
        need(p['node'] == HOSTS[p['model']] and p['position_id'] == position_id(p['node'], p['model'], p['dataset'], p['rate_rps']), 'position identity changed')
        need(sorted(t['repeat'] for t in p['systems']['pdblend']) == p['required_pdb_repeats'], 'PDB repeat requirement changed')
        for system, tasks in p['systems'].items():
            need(system in SYSTEMS, 'unknown comparison system')
            for t in tasks:
                source = saved.get(t['observation_id']) if t['action'] == 'reuse' else new.get(t['observation_id'])
                need(source is not None and source['system'] == system and source['cell_id'] == t['cell_id'], 'task not bound to exact observation')
                need(t == (saved_task(source) if t['action'] == 'reuse' else new_task(source)), 'task body or saved reference altered')
                need(identity(source) == identity(dict(p['workload'], measurement_host=p['node'])), 'task host/trace/SLO/seed mismatch')
        if p['newly_added_coordinate']:
            need(all([t['repeat'] for t in p['systems'][s]] == [1, 2] for s in SYSTEMS), 'new rates need five systems x two actual repeats')
    for row in cells:
        need(row['seed'] == row['arrival_seed'] == 701 and row['sampling_seed'] == 20260907, 'seed altered')
        need(row['arrival_window_s'] == row['trace_duration_s'] == 100 and row['request_hard_timeout_s'] == 120, 'window/deadline changed')
        need(row['slo_scale'] == 1 and row['allowed_slo_scales'] == [1.] and
             (row['slo_ttft_s'], row['slo_tpot_s']) == SLOS[row['dataset']], 'SLO changed')
    pairkeys = set()
    for pair in d['pairings']:
        p, b = pair['pdb'], pair['baseline']
        ps = saved.get(p['observation_id'], new.get(p['observation_id']))
        bs = saved.get(b['observation_id'], new.get(b['observation_id']))
        need(ps is not None and bs is not None and identity(ps) == identity(bs), 'pair identity differs')
        need(p == (saved_task(ps) if p['action'] == 'reuse' else new_task(ps)) and
             b == (saved_task(bs) if b['action'] == 'reuse' else new_task(bs)), 'pair task body or saved reference changed')
        need(bs['system'] == pair['baseline_system'] and ps['system'] == 'pdblend', 'pair system differs')
        need(pair['baseline_repeat_reused'] == (p['repeat'] != b['repeat']), 'repeat reuse hidden')
        key = p['observation_id'], pair['baseline_system']
        need(key not in pairkeys, 'duplicate paired result')
        pairkeys.add(key)
    need(len(d['stopped_P12_history']) == 8 and len(d['historical_A_observations']) == 40, 'history dropped')
    need(not {p['checkpoint']['path'] for p in d['stopped_P12_history']}.intersection(p['checkpoint']['path'] for p in reused), 'P12 history relabelled into reuse')
    return True


def load_declaration(declaration_ref):
    d = checked(declaration_ref)
    validate_structure(d)
    return d


def lookup(declaration_ref, model, dataset, rate, system, repeat):
    """Return one standard NEW execution row. Reused observations have no new row."""
    d = load_declaration(declaration_ref)
    rows = [r for r in d['cells'] if (r['model'], r['dataset'], number(r['rate_rps']), r['system'], r['repeat']) ==
            (model, dataset, number(rate), system, repeat)]
    need(len(rows) == 1, 'undeclared, reused, or ambiguous execution request')
    checked(rows[0]['trace_reference'])
    return copy.deepcopy(rows[0])


def resolve_group(declaration_ref, model, dataset, actual_host=None):
    d = load_declaration(declaration_ref)
    groups = [g for g in d['groups'] if g['model'] == model and g['dataset'] == dataset]
    need(len(groups) == 1, 'unknown group')
    group = copy.deepcopy(groups[0])
    need(actual_host is None or actual_host == group['node'], 'wrong physical host')
    group['positions'] = [copy.deepcopy(p) for p in d['positions'] if p['position_id'] in group['position_ids']]
    group['pairings'] = [copy.deepcopy(p) for p in d['pairings'] if p['position_id'] in group['position_ids']]
    group['reused_observations'] = [copy.deepcopy(p) for p in d['reused_observations'] if p['model'] == model and p['dataset'] == dataset]
    return group


def evaluate_rate(position, observations):
    """Decision arithmetic only: caller must independently validate saved raw CPs.

    Each input is the audited summary plus cell_id, repeat and optional failure_class.
    This routine never grants measurement, hardware, or capacity qualification.
    """
    required = {t['cell_id']: t['repeat'] for t in position['systems']['pdblend']}
    seen = {}
    for obs in observations:
        need(obs['cell_id'] in required and obs['cell_id'] not in seen, 'unknown/duplicate PDB observation')
        need(obs.get('repeat') == required[obs['cell_id']], 'observation repeat differs')
        expected_identity = dict(model=position['model'], dataset=position['dataset'],
                                 node=position['node'], measurement_host=position['node'],
                                 rate_rps=position['rate_rps'], seed=701,
                                 trace_sha256=position['trace']['sha256'])
        for field, expected in expected_identity.items():
            need(field not in obs or obs[field] == expected, 'observation position identity differs: '+field)
        q = obs.get('slo_attainment')
        need(q is None or (type(q) in (int, float) and math.isfinite(q) and 0 <= q <= 1), 'invalid SLO fraction')
        seen[obs['cell_id']] = obs
    missing = [cid for cid in required if cid not in seen]
    faults, losses, capacities, passes = [], [], [], []
    for cid, obs in seen.items():
        if obs.get('measurement_valid') is not True:
            faults.append(cid)
        elif obs.get('work_complete') is True:
            need(obs.get('slo_attainment') is not None, 'complete work requires SLO')
            (losses if obs['slo_attainment'] < .9 else passes).append(cid)
        elif obs.get('failure_class') == 'independently_diagnosed_capacity_deadline' and obs.get('diagnosis_reference'):
            capacities.append(cid)
        else:
            faults.append(cid)
    if faults:
        status = 'stop_for_engineering_diagnosis'
    elif missing:
        status = 'complete_current_rate_repeats'
    elif losses:
        status = 'cap_complete_work_SLO_below_90'
    elif capacities:
        status = 'cap_diagnosed_service_failure'
    else:
        status = 'advance'
    return dict(status=status, missing_cell_ids=missing, engineering_fault_cell_ids=faults,
                loss_cell_ids=losses, diagnosed_capacity_cell_ids=capacities,
                threshold_straddles=bool(losses and passes),
                increase_rate_allowed=status == 'advance',
                decision_only_not_raw_or_hardware_qualification=True)


def select_group(group, observations):
    """Return current-rate tasks or cap-pruned baseline stage; never launch anything."""
    observed = {r['cell_id']: r for r in observations}
    need(len(observed) == len(observations), 'duplicate observations')
    allowed = {t['cell_id'] for p in group['positions'] for t in p['systems']['pdblend'] if t['action'] == 'execute'}
    need(set(observed) <= allowed, 'observations outside this new PDB group')
    reused = {r['cell_id']: r for r in group['reused_observations'] if r['system'] == 'pdblend'}
    need(not set(reused).intersection(observed), 'new observation cannot replace a pinned reused CP')
    for p in group['positions']:
        available = []
        for task in p['systems']['pdblend']:
            value = reused.get(task['cell_id'], observed.get(task['cell_id']))
            if value is not None:
                available.append(value)
        decision = evaluate_rate(p, available)
        if decision['status'] == 'advance':
            continue
        if decision['status'].startswith('cap_'):
            cap = Decimal(p['rate_rps_decimal'])
            reached = [x for x in group['positions'] if Decimal(x['rate_rps_decimal']) <= cap]
            higher = [x for x in group['positions'] if Decimal(x['rate_rps_decimal']) > cap]
            return dict(phase='baselines', cap_position_id=p['position_id'], cap_rate_rps=p['rate_rps'],
                        decision=decision, eligible_position_ids=[x['position_id'] for x in reached],
                        baseline_tasks=[t for x in reached for s in SYSTEMS[1:] for t in x['systems'][s]],
                        pruned_new_cell_ids=[t['cell_id'] for x in higher for tasks in x['systems'].values() for t in tasks if t['action'] == 'execute'],
                        history_preserved=True)
        return dict(phase='diagnosis' if decision['engineering_fault_cell_ids'] else 'pdblend',
                    position_id=p['position_id'], rate_rps=p['rate_rps'], decision=decision,
                    next_tasks=[t for t in p['systems']['pdblend'] if t['cell_id'] in decision['missing_cell_ids']],
                    higher_rate_dispatch_forbidden=True)
    next_rate = Decimal(group['positions'][-1]['rate_rps_decimal']) + Decimal(str(group['near_boundary_increment_rps']))
    return dict(phase='extension_declaration_required', next_rate_rps_decimal=number(next_rate),
                proposed_maximum_new_runs=10, existing_declaration_does_not_authorize_undeclared_cells=True)
