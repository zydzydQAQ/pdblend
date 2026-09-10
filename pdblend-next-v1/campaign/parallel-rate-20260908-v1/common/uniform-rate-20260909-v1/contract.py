"""Exact arithmetic and immutable evidence for the authorized uniform-rate run.

Scheduling does not qualify a GPU runtime. Observation inputs must first pass
the node's raw workload, clock, power, source and cleanup audit.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[1]
SYSTEMS = ('pdblend', 'mixed', 'distserve', 'dynamollm', 'ecoserve')
MODELS = ('7b', '14b', '32b')
DATASETS = ('alpaca', 'sharegpt', 'longbench')
HOSTS = {'7b': 'C', '14b': 'Anew20260909', '32b': 'B'}
SLOS = {'alpaca': (1., .1), 'sharegpt': (5., .15), 'longbench': (15., .2)}
STEPS = {m: {d: (dict(zip(MODELS, ('3', '1.5', '.5')))[m] if d == 'alpaca' else '.25')
             for d in DATASETS} for m in MODELS}
INITIAL_LIMITS = {'7b': {'alpaca': '21', 'sharegpt': '3.75', 'longbench': '3'},
                  '14b': {'alpaca': '12', 'sharegpt': '2', 'longbench': '1.25'},
                  '32b': {'alpaca': '5', 'sharegpt': '1.5', 'longbench': '.5'}}
SCHEMA = 'uniform-rate-execution-declaration-v1'


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                       allow_nan=False) + '\n').encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def checked(reference):
    need(isinstance(reference, dict) and set(reference) >= {'path', 'sha256'}, 'saved reference required')
    need(sha(reference['path']) == reference['sha256'], 'saved reference changed: ' + reference['path'])
    return read(reference['path'])


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def number(value):
    need(not isinstance(value, bool), 'boolean is not a rate')
    try:
        value = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('invalid decimal rate') from exc
    need(value.is_finite() and value > 0, 'rate must be positive and finite')
    return format(value.normalize(), 'f')


def step(model, dataset):
    need(model in MODELS and dataset in DATASETS, 'unknown model/dataset')
    return Decimal(STEPS[model][dataset])


def on_grid(model, dataset, rate):
    rate = Decimal(number(rate))
    return rate % step(model, dataset) == 0


def grid(model, dataset, limit):
    limit = Decimal(number(limit)); increment = step(model, dataset)
    need(limit % increment == 0, 'limit must be on the declared grid')
    return [number(increment * i) for i in range(1, int(limit / increment) + 1)]


def position_id(node, model, dataset, rate):
    return f'uniform-v1-{node}-{model}-{dataset}-r{number(rate)}-s701'


def cell_id(node, model, dataset, rate, system, repeat):
    return f'{position_id(node, model, dataset, rate)}-w100-{system}-slo1-repeat{repeat}'


def identity(row):
    return (row['model'], row['dataset'], number(row['rate_rps']), row['seed'],
            row['trace_sha256'], row['content_pairing_sha256'], row['slo_ttft_s'], row['slo_tpot_s'],
            row.get('measurement_host', row.get('node')))


def saved_task(observation):
    return dict(action='reuse', observation_id='saved:' + observation['checkpoint']['sha256'],
                cell_id=observation['cell_id'], repeat=observation.get('repeat', 1),
                checkpoint=observation['checkpoint'], actual_source_preserved=True)


def new_task(row):
    return dict(action='execute', observation_id='new:' + row['cell_id'],
                cell_id=row['cell_id'], repeat=row['repeat'], row=row)


def row_for(workload, node, system, repeat, sequence):
    need(system in SYSTEMS and repeat in (1, 2), 'unknown system/repeat')
    row = copy.deepcopy(workload)
    row.update(cell_id=cell_id(node, row['model'], row['dataset'], row['rate_rps'], system, repeat),
        node=node, measurement_host=node, system=system, repeat=repeat, sequence=sequence,
        phase='main', part='main', slo_scale=1., allowed_slo_scales=[1.],
        slo_ttft_s=SLOS[row['dataset']][0], slo_tpot_s=SLOS[row['dataset']][1],
        slo_attainment_target=.9, execution_status='not_run', execution_binding_required=True,
        policy_binding_required=True, controller_config=None, strategy=None, reuse_main_cell_id=None,
        original_actual_source_relabelled=False, actual_fixed_window_verified=False,
        request_hard_timeout_s=120., drain_after_arrival_window_s=120.,
        formal_eligible=False, fixed_slo_only=True)
    return row


def strict_slo_pass(dataset, ttft_s, tpot_s, *, work_complete):
    """The original joint TTFT/TPOT rule; equality is a miss, TTLT is absent."""
    need(dataset in DATASETS, 'unknown SLO dataset')
    ttft, tpot = SLOS[dataset]
    return bool(work_complete is True and type(ttft_s) in (int, float) and type(tpot_s) in (int, float)
                and math.isfinite(ttft_s) and math.isfinite(tpot_s)
                and 0 <= ttft_s < ttft and 0 <= tpot_s < tpot)


def validate_structure(d):
    need(d['schema'] == SCHEMA and d['fixed_slo_only'] is True, 'wrong uniform declaration')
    need(d['arrival_seed'] == 701 and d['arrival_window_s'] == 100 and
         d['request_hard_timeout_s'] == 120 and d['sampling_seed'] == 20260907, 'workload contract changed')
    need(d['baseline_saturation_search_enabled'] is False and d['slo_uses_ttlt'] is False,
         'baseline saturation or TTLT gate added')
    positions = d['positions']; cells = d['cells']
    need(len({p['position_id'] for p in positions}) == len(positions), 'duplicate positions')
    need(len({r['cell_id'] for r in cells}) == len(cells), 'duplicate new cells')
    new = {r['cell_id']: r for r in cells}
    saved = {r['cell_id']: r for r in d['reused_observations']}
    need(len(saved) == len(d['reused_observations']), 'duplicate saved observation')
    for g in d['groups']:
        need(g['node'] == HOSTS[g['model']], 'wrong physical host')
        ps = [p for p in positions if p['position_id'] in g['position_ids']]
        need([p['rate_rps_decimal'] for p in ps] == grid(g['model'], g['dataset'], ps[-1]['rate_rps_decimal']),
             'grid must start at one step and contain every arithmetic increment')
        need(g['rate_step_rps_decimal'] == number(step(g['model'], g['dataset'])), 'step changed')
    for p in positions:
        need(p['node'] == HOSTS[p['model']] and on_grid(p['model'], p['dataset'], p['rate_rps_decimal']), 'off-grid or cross-host point')
        need(p['position_id'] == position_id(p['node'], p['model'], p['dataset'], p['rate_rps_decimal']), 'position ID differs')
        need(sorted(t['repeat'] for t in p['systems']['pdblend']) == p['required_pdb_repeats'], 'PDB repeats changed')
        need(set(p['systems']) == set(SYSTEMS), 'five comparison systems required')
        for system, tasks in p['systems'].items():
            need(tasks and len({t['repeat'] for t in tasks}) == len(tasks), 'missing/duplicate system repeats')
            for task in tasks:
                source = saved.get(task['cell_id']) if task['action'] == 'reuse' else new.get(task['cell_id'])
                need(source is not None and source['system'] == system, 'unbound task')
                need(task == (saved_task(source) if task['action'] == 'reuse' else new_task(source)), 'task altered')
                need(identity(source) == identity(p['workload']), 'host/trace/SLO pairing differs')
                if p['node'] == 'Anew20260909':
                    need(task['action'] == 'execute', 'new A cannot reuse old A observations')
        if p['newly_added_coordinate']:
            need(all([t['repeat'] for t in p['systems'][s]] == [1, 2] for s in SYSTEMS), 'new point needs two actual repeats per system')
    for r in cells:
        need(r['seed'] == r['arrival_seed'] == 701 and r['arrival_window_s'] == r['trace_duration_s'] == 100,
             'row seed/window changed')
        need(r['slo_scale'] == 1 and r['allowed_slo_scales'] == [1.] and
             (r['slo_ttft_s'], r['slo_tpot_s']) == SLOS[r['dataset']], 'row fixed SLO changed')
    return True


def load_declaration(declaration_ref):
    d = checked(declaration_ref)
    validate_structure(d)
    return d


def lookup(declaration_ref, model, dataset, rate, system, repeat):
    d = load_declaration(declaration_ref)
    rows = [r for r in d['cells'] if (r['model'], r['dataset'], number(r['rate_rps']), r['system'], r['repeat']) ==
            (model, dataset, number(rate), system, repeat)]
    need(len(rows) == 1, 'undeclared or reused execution row')
    checked(rows[0]['trace_reference'])
    return copy.deepcopy(rows[0])


def resolve_group(declaration_ref, model, dataset, actual_host=None):
    d = load_declaration(declaration_ref)
    found = [g for g in d['groups'] if g['model'] == model and g['dataset'] == dataset]
    need(len(found) == 1, 'unknown group')
    g = copy.deepcopy(found[0])
    need(actual_host is None or actual_host == g['node'], 'wrong physical host')
    g['positions'] = [copy.deepcopy(p) for p in d['positions'] if p['position_id'] in g['position_ids']]
    g['pairings'] = [copy.deepcopy(p) for p in d['pairings'] if p['position_id'] in g['position_ids']]
    g['reused_observations'] = [copy.deepcopy(r) for r in d['reused_observations'] if r['model'] == model and r['dataset'] == dataset]
    g['declaration'] = declaration_ref
    return g


def evaluate_rate(position, observations):
    required = {t['cell_id']: t['repeat'] for t in position['systems']['pdblend']}
    seen = {}; faults = []; losses = []; passes = []
    for obs in observations:
        cid = obs['cell_id']
        need(cid in required and cid not in seen and obs.get('repeat') == required[cid], 'unknown/duplicate PDB repeat')
        for key in ('model', 'dataset', 'seed', 'trace_sha256', 'content_pairing_sha256', 'slo_ttft_s', 'slo_tpot_s', 'measurement_host'):
            need(key not in obs or obs[key] == position['workload'][key], 'observation identity differs: ' + key)
        if 'rate_rps' in obs:
            need(number(obs['rate_rps']) == position['rate_rps_decimal'], 'observation rate differs')
        q = obs.get('slo_attainment')
        need(q is None or type(q) in (int, float) and math.isfinite(q) and 0 <= q <= 1, 'invalid attainment')
        seen[cid] = obs
        if obs.get('measurement_valid') is not True or obs.get('work_complete') is not True:
            faults.append(cid)
        else:
            need(q is not None, 'complete observation missing SLO')
            (losses if q < .90 else passes).append(cid)
    missing = [cid for cid in required if cid not in seen]
    status = ('stop_for_engineering_diagnosis' if faults else 'complete_current_rate_repeats' if missing
              else 'cap_complete_work_SLO_below_90' if losses else 'advance')
    return dict(status=status, missing_cell_ids=missing, engineering_fault_cell_ids=faults,
        loss_cell_ids=losses, diagnosed_capacity_cell_ids=[], threshold_straddles=bool(losses and passes),
        cap_observed=bool(losses), increase_rate_allowed=status == 'advance',
        decision_only_not_raw_or_hardware_qualification=True)


def select_group(group, observations):
    observed = {r['cell_id']: r for r in observations}
    need(len(observed) == len(observations), 'duplicate observations')
    tasks = {t['cell_id']: t for p in group['positions'] for ts in p['systems'].values() for t in ts}
    need(set(observed) <= set(tasks), 'observations outside declared group')
    reused = {r['cell_id']: r for r in group['reused_observations']}
    need(not set(reused).intersection(observed), 'cannot replace pinned historical observation')
    available = dict(reused, **observed)
    for p in group['positions']:
        decision = evaluate_rate(p, [available[t['cell_id']] for t in p['systems']['pdblend'] if t['cell_id'] in available])
        if decision['status'] == 'advance':
            continue
        if decision['status'].startswith('cap_'):
            reached = [x for x in group['positions'] if Decimal(x['rate_rps_decimal']) <= Decimal(p['rate_rps_decimal'])]
            baselines = [t for x in reached for s in SYSTEMS[1:] for t in x['systems'][s]]
            faults = []
            for task in baselines:
                obs = observed.get(task['cell_id'])
                if obs is None:
                    continue
                accepted = obs.get('measurement_valid') is True and (obs.get('work_complete') is True or
                    obs.get('failure_class') == 'independently_diagnosed_capacity_deadline' and obs.get('diagnosis_reference'))
                if not accepted:
                    faults.append(task['cell_id'])
            if faults:
                return dict(phase='diagnosis', cap_rate_rps=p['rate_rps'],
                    engineering_fault_cell_ids=faults, higher_rate_dispatch_forbidden=True,
                    baseline_tasks=[], history_preserved=True)
            pending = [t for t in baselines if t['cell_id'] not in available]
            return dict(phase='baselines' if pending else 'complete', cap_position_id=p['position_id'],
                cap_rate_rps=p['rate_rps'], cap_rate_rps_decimal=p['rate_rps_decimal'], decision=decision,
                eligible_position_ids=[x['position_id'] for x in reached], baseline_tasks=pending,
                all_reached_baseline_tasks=baselines, history_preserved=True, higher_rate_dispatch_forbidden=True)
        return dict(phase='diagnosis' if decision['engineering_fault_cell_ids'] else 'pdblend',
            position_id=p['position_id'], rate_rps=p['rate_rps'], decision=decision,
            next_tasks=[t for t in p['systems']['pdblend'] if t['cell_id'] in decision['missing_cell_ids']],
            higher_rate_dispatch_forbidden=True)
    return dict(phase='extension_declaration_required',
        next_rate_rps_decimal=number(Decimal(group['positions'][-1]['rate_rps_decimal']) + step(group['model'], group['dataset'])),
        proposed_maximum_new_runs=10, existing_declaration_does_not_authorize_undeclared_cells=True)


def apply_audited_reuse(group, observations):
    """Dynamically bind completed B/C work from explicitly pinned audit reports.

The report has an observations/points array containing each exact checkpoint.
No filesystem-name scan and no running checkpoint can qualify for reuse.
"""
    g = copy.deepcopy(group)
    need(g['node'] in ('B', 'C'), 'new A has no historical reuse authorization')
    for obs in observations:
        need(obs['measurement_valid'] is True and obs['measurement_host'] == g['node'], 'unqualified or cross-host reuse')
        if obs.get('work_complete') is not True:
            need(obs['system'] != 'pdblend' and
                 obs.get('failure_class') == 'independently_diagnosed_capacity_deadline' and
                 obs.get('diagnosis_reference'), 'incomplete reuse requires independently diagnosed baseline deadline')
            checked(obs['diagnosis_reference'])
        checked(obs['checkpoint'])
        report = checked(obs['audit_reference'])
        need(report.get('slo_threshold_comparison') == 'strict_lt', 'dynamic audit must recompute strict TTFT/TPOT thresholds')
        matching = [x for x in report.get('observations', report.get('points', [])) if x.get('checkpoint') == obs['checkpoint']]
        need(len(matching) == 1 and all(obs.get(k) == matching[0].get(k) for k in
             ('cell_id', 'repeat', 'system', 'model', 'dataset', 'rate_rps', 'seed', 'trace_sha256',
              'content_pairing_sha256', 'slo_ttft_s', 'slo_tpot_s', 'measurement_host',
              'measurement_valid', 'work_complete', 'slo_attainment')), 'audit does not establish this observation')
        ps = [p for p in g['positions'] if identity(obs) == identity(p['workload'])]
        need(len(ps) == 1, 'reuse is off-grid or has another workload/SLO/host')
        p = ps[0]; system = obs['system']
        found = [t for t in p['systems'][system] if t['repeat'] == obs['repeat']]
        need(len(found) == 1, 'reuse repeat is not declared')
        previous = found[0]
        if previous['action'] == 'reuse':
            need(previous['checkpoint'] == obs['checkpoint'], 'cannot select a replacement historical outcome')
            continue
        replacement = saved_task(obs)
        p['systems'][system] = [replacement if t == previous else t for t in p['systems'][system]]
        for pair in g['pairings']:
            for key in ('pdb', 'baseline'):
                if pair[key]['cell_id'] == previous['cell_id']:
                    pair[key] = replacement
        g['reused_observations'].append(copy.deepcopy(obs))
    return g
