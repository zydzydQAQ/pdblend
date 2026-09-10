"""Immutable, same-host fixed-SLO decisions for the authorized one-run campaign."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path

HERE = Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1/common/uniform-rate-20260909-v2')
ROOT = HERE.parents[1]
REPO = ROOT.parents[1]
SYSTEMS = ('pdblend', 'mixed', 'distserve', 'dynamollm', 'ecoserve')
MODELS = ('7b', '14b', '32b')
DATASETS = ('alpaca', 'sharegpt', 'longbench')
SLOS = {'alpaca': (1., .1), 'sharegpt': (5., .15), 'longbench': (15., .2)}
HOSTS = {(m, d): ('C' if m == '7b' else 'B' if m == '32b' or d == 'sharegpt' else 'Anew20260909')
         for m in MODELS for d in DATASETS}
STEPS = {m: {d: (dict(zip(MODELS, ('3', '1.5', '.5')))[m] if d == 'alpaca'
                    else '.05' if (m, d) == ('32b', 'longbench') else '.25') for d in DATASETS} for m in MODELS}
INITIAL_LIMITS = {'7b': {'alpaca': '18', 'sharegpt': '3.25', 'longbench': '2.5'},
                  '14b': {'alpaca': '12', 'sharegpt': '1.75', 'longbench': '1.25'},
                  '32b': {'alpaca': '4.5', 'sharegpt': '1.5', 'longbench': '.3'}}
SCHEMA = 'uniform-rate-execution-declaration-v2'
SUPPLEMENTS = {('7b', 'alpaca', '12', 'ecoserve'), ('7b', 'alpaca', '15', 'ecoserve'),
               ('32b', 'alpaca', '4.5', 'distserve'), ('32b', 'sharegpt', '1.25', 'dynamollm'),
               ('32b', 'sharegpt', '1.5', 'distserve'), ('32b', 'sharegpt', '1.5', 'dynamollm')}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def checked(reference):
    need(isinstance(reference, dict) and set(reference) >= {'path', 'sha256'}, 'immutable reference required')
    need(sha(reference['path']) == reference['sha256'], 'immutable input changed: ' + reference['path'])
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


def host(model, dataset):
    need((model, dataset) in HOSTS, 'unknown model/dataset')
    return HOSTS[(model, dataset)]


def step(model, dataset):
    host(model, dataset)
    return Decimal(STEPS[model][dataset])


def on_grid(model, dataset, rate):
    return Decimal(number(rate)) % step(model, dataset) == 0


def grid(model, dataset, limit):
    limit, increment = Decimal(number(limit)), step(model, dataset)
    need(limit % increment == 0, 'limit must be on the exact grid')
    return [number(increment * i) for i in range(1, int(limit / increment) + 1)]


def position_id(node, model, dataset, rate):
    return f'uniform-v2-{node}-{model}-{dataset}-r{number(rate)}-s701'


def cell_id(node, model, dataset, rate, system, repeat, purpose='normal'):
    base = f'{position_id(node, model, dataset, rate)}-w100-{system}-slo1-repeat{repeat}'
    return base if purpose == 'normal' else base + '-metric-supplement1'


def identity(row):
    return (row['model'], row['dataset'], number(row['rate_rps']), row['seed'], row['trace_sha256'],
            row['content_pairing_sha256'], row['slo_ttft_s'], row['slo_tpot_s'], row['measurement_host'])


def saved_task(observation):
    return dict(action='reuse', observation_id='saved:' + observation['checkpoint']['sha256'],
                cell_id=observation['cell_id'], repeat=observation.get('repeat', 1),
                checkpoint=observation['checkpoint'], actual_source_preserved=True)


def new_task(row):
    return dict(action='execute', observation_id='new:' + row['cell_id'],
                cell_id=row['cell_id'], repeat=row['repeat'], row=row)


def row_for(workload, node, system, repeat, sequence, purpose='normal'):
    need(system in SYSTEMS and repeat in (1, 2), 'unknown system/repeat')
    need(node == host(workload['model'], workload['dataset']), 'wrong group host')
    row = copy.deepcopy(workload)
    row.update(cell_id=cell_id(node, row['model'], row['dataset'], row['rate_rps'], system, repeat, purpose),
        node=node, measurement_host=node, system=system, repeat=repeat, sequence=sequence,
        phase='main', part='main', slo_scale=1., allowed_slo_scales=[1.],
        slo_ttft_s=SLOS[row['dataset']][0], slo_tpot_s=SLOS[row['dataset']][1],
        slo_attainment_target=.9, execution_status='not_run', execution_binding_required=True,
        policy_binding_required=True, controller_config=None, strategy=None, reuse_main_cell_id=None,
        original_actual_source_relabelled=False, actual_fixed_window_verified=False,
        request_hard_timeout_s=120., drain_after_arrival_window_s=120., formal_eligible=False,
        fixed_slo_only=True, measurement_purpose=purpose,
        execution_reason=('missing_exact_token_metric' if purpose == 'metric_supplement'
                          else 'pdb_first_below_90_confirmation' if system == 'pdblend' and repeat == 2
                          else 'missing_normal_measurement'),
        conditional=(system == 'pdblend' and repeat == 2 and purpose == 'normal'),
        changes_historical_repeat_count=False if purpose == 'metric_supplement' else None)
    return row


def strict_slo_pass(dataset, ttft_s, tpot_s, *, work_complete):
    ttft, tpot = SLOS[dataset]
    return bool(work_complete is True and type(ttft_s) in (int, float) and type(tpot_s) in (int, float)
                and math.isfinite(ttft_s) and math.isfinite(tpot_s)
                and 0 <= ttft_s < ttft and 0 <= tpot_s < tpot)


def all_tasks(position):
    return [t for tasks in position['systems'].values() for t in tasks] + position.get('metric_supplements', [])


def validate_structure(d):
    need(d['schema'] == SCHEMA and d['fixed_slo_only'] is True and d['completion_scope'] == 'five_systems', 'wrong declaration')
    need((d['arrival_seed'], d['sampling_seed'], d['arrival_window_s'], d['request_hard_timeout_s']) ==
         (701, 20260907, 100, 120), 'workload contract changed')
    need(d['slo_uses_ttlt'] is False and d['normal_repeats'] == 1, 'SLO/repetition contract changed')
    cells = {r['cell_id']: r for r in d['cells']}
    saved = {r['cell_id']: r for r in d['reused_observations']}
    need(len(cells) == len(d['cells']) and len(saved) == len(d['reused_observations']), 'duplicate cells')
    need(len({p['position_id'] for p in d['positions']}) == len(d['positions']), 'duplicate positions')
    for g in d['groups']:
        ps = [p for p in d['positions'] if p['position_id'] in g['position_ids']]
        need(g['node'] == host(g['model'], g['dataset']), 'wrong physical host')
        need([p['rate_rps_decimal'] for p in ps] == grid(g['model'], g['dataset'], ps[-1]['rate_rps_decimal']), 'nonuniform grid')
    for p in d['positions']:
        need(p['node'] == host(p['model'], p['dataset']) and on_grid(p['model'], p['dataset'], p['rate_rps_decimal']), 'off-grid/cross-host')
        need(set(p['systems']) == set(SYSTEMS), 'five systems required')
        for system, tasks in p['systems'].items():
            need(tasks and len({t['repeat'] for t in tasks}) == len(tasks), 'missing/duplicate repeat')
            need(tasks[0]['repeat'] == 1, 'a second repeat cannot replace the first')
            need(all(t['repeat'] in (1, 2) for t in tasks), 'normal third repeat forbidden')
            for task in tasks:
                source = saved.get(task['cell_id']) if task['action'] == 'reuse' else cells.get(task['cell_id'])
                need(source is not None and source['system'] == system and identity(source) == identity(p['workload']), 'unpaired task')
                need(task == (saved_task(source) if task['action'] == 'reuse' else new_task(source)), 'task changed')
        for task in p.get('metric_supplements', []):
            row = cells.get(task['cell_id'])
            need(row and row['measurement_purpose'] == 'metric_supplement' and row['system'] != 'pdblend', 'invalid metric supplement')
            need(identity(row) == identity(p['workload']), 'supplement pairing differs')
    return True


def load_declaration(reference):
    declaration = checked(reference)
    validate_structure(declaration)
    return declaration


def lookup_cell(reference, cid):
    rows = [r for r in load_declaration(reference)['cells'] if r['cell_id'] == cid]
    need(len(rows) == 1, 'undeclared or reused execution row')
    checked(rows[0]['trace_reference'])
    return copy.deepcopy(rows[0])


def lookup(reference, model, dataset, rate, system, repeat, purpose='normal'):
    return lookup_cell(reference, cell_id(host(model, dataset), model, dataset, rate, system, repeat, purpose))


def resolve_group(reference, model, dataset, actual_host=None):
    d = load_declaration(reference)
    groups = [g for g in d['groups'] if (g['model'], g['dataset']) == (model, dataset)]
    need(len(groups) == 1, 'unknown group')
    g = copy.deepcopy(groups[0])
    need(actual_host is None or actual_host == g['node'], 'wrong physical host')
    g.update(positions=[copy.deepcopy(p) for p in d['positions'] if p['position_id'] in g['position_ids']],
        pairings=[copy.deepcopy(p) for p in d['pairings'] if p['position_id'] in g['position_ids']],
        reused_observations=[copy.deepcopy(r) for r in d['reused_observations'] if (r['model'], r['dataset']) == (model, dataset)],
        declaration=reference)
    return g


def evaluate_rate(position, observations):
    tasks = position['systems']['pdblend']
    required = {t['cell_id']: t['repeat'] for t in tasks}
    seen, faults, losses, passes = {}, [], [], []
    for obs in sorted(observations, key=lambda r: r['repeat']):
        cid = obs['cell_id']
        need(cid in required and cid not in seen and obs['repeat'] == required[cid], 'unknown/duplicate PDB repeat')
        need(identity(obs) == identity(position['workload']), 'observation identity differs')
        q = obs.get('slo_attainment')
        need(q is None or type(q) in (float, int) and math.isfinite(q) and 0 <= q <= 1, 'invalid attainment')
        seen[cid] = obs
        if obs.get('measurement_valid') is not True or obs.get('work_complete') is not True:
            faults.append(cid)
        else:
            need(q is not None and obs.get('strict_slo_recomputed') is True, 'strict SLO audit required')
            (losses if q < .90 else passes).append(cid)
    mandatory = [t for t in tasks if not t.get('row', {}).get('conditional', False) or losses]
    missing = [t['cell_id'] for t in mandatory if t['cell_id'] not in seen]
    status = ('stop_for_engineering_diagnosis' if faults else 'complete_current_rate_repeats' if missing
              else 'cap_complete_work_SLO_below_90' if losses else 'advance')
    return dict(status=status, missing_cell_ids=missing, engineering_fault_cell_ids=faults,
        loss_cell_ids=losses, cap_trigger_cell_id=losses[0] if losses else None,
        threshold_straddles=bool(losses and passes), cap_observed=bool(losses), increase_rate_allowed=status == 'advance')


def acceptable_baseline(obs):
    original = obs.get('measurement_valid') is True and (obs.get('work_complete') is True or
        obs.get('historical_partial_work_audited') is True or
        obs.get('failure_class') == 'independently_diagnosed_capacity_deadline' and obs.get('diagnosis_reference'))
    if original:
        return original
    if not (obs.get('measurement_valid') is True and obs.get('work_complete') is False
            and obs.get('system') != 'pdblend'
            and obs.get('failure_class') == 'independently_diagnosed_capacity_rejection'
            and obs.get('diagnosis_reference')):
        return False
    diagnosis = checked(obs['diagnosis_reference'])
    classification = obs.get('baseline_service_failure', {})
    return (diagnosis.get('passed') is True and diagnosis.get('independently_recomputed') is True
        and diagnosis.get('checkpoint') == obs.get('checkpoint')
        and diagnosis.get('classification') == classification
        and (classification.get('classification') == 'baseline_explicit_controller_admission_queue_full'
            or classification.get('classification') == 'baseline_explicit_native_admission_queue_full'
            and (obs.get('measurement_host'), obs.get('model'), obs.get('system')) == ('C', '7b', 'ecoserve')
            and classification.get('native_rejections_are_not_timeouts') is True
            and classification.get('native_rejections', 0) > 0
            and classification.get('native_queue_evidence', {}).get('passed') is True
            and classification.get('native_queue_evidence', {}).get('checkpoint') == obs.get('checkpoint'))
        and classification.get('independently_diagnosed') is True
        and diagnosis.get('no_unknown_errors') is True and diagnosis.get('no_PDB_complete_boundary_claim') is True)


def select_group(group, observations):
    observed = {r['cell_id']: r for r in observations}
    need(len(observed) == len(observations), 'duplicate observations')
    tasks = {t['cell_id']: t for p in group['positions'] for t in all_tasks(p)}
    reused = {r['cell_id']: r for r in group['reused_observations']}
    need(set(observed) <= set(tasks) and not set(observed).intersection(reused), 'outside group or replacing history')
    available = dict(reused, **observed)
    for cid, observation in observed.items():
        task = tasks[cid]
        need(task['action'] == 'execute' and identity(observation) == identity(task['row']), 'observation task identity differs')
        need(observation['system'] == task['row']['system'] and observation['repeat'] == task['repeat'], 'observation system/repeat differs')
    for p in group['positions']:
        decision = evaluate_rate(p, [available[t['cell_id']] for t in p['systems']['pdblend'] if t['cell_id'] in available])
        if decision['status'] == 'advance':
            continue
        if decision['status'].startswith('cap_'):
            reached = [x for x in group['positions'] if Decimal(x['rate_rps_decimal']) <= Decimal(p['rate_rps_decimal'])]
            baseline = [t for x in reached for s in SYSTEMS[1:] for t in x['systems'][s]]
            supplements = [t for x in reached for t in x.get('metric_supplements', [])]
            faults = [t['cell_id'] for t in baseline + supplements if t['cell_id'] in available and
                      not acceptable_baseline(available[t['cell_id']])]
            faults += [t['cell_id'] for t in supplements if t['cell_id'] in available and
                       available[t['cell_id']].get('token_throughput_is_exact') is not True]
            if faults:
                return dict(phase='diagnosis', engineering_fault_cell_ids=faults, baseline_tasks=[], higher_rate_dispatch_forbidden=True)
            # A supplement is unnecessary if an independently audited normal observation already closes its exact metric.
            def supplement_needed(task):
                row = task['row']
                return not any(identity(o) == identity(row) and o['system'] == row['system'] and
                               o.get('token_throughput_is_exact') is True for o in available.values())
            pending = [t for t in baseline if t['cell_id'] not in available]
            pending += [t for t in supplements if t['cell_id'] not in available and supplement_needed(t)]
            metric_gaps = []
            for x in reached:
                for system in SYSTEMS[1:]:
                    matching = [o for o in available.values() if identity(o) == identity(x['workload']) and o['system'] == system]
                    if matching and not any(o.get('token_throughput_is_exact') is True for o in matching):
                        planned = any(t['row']['system'] == system for t in x.get('metric_supplements', []))
                        if not planned:
                            metric_gaps.append(dict(position_id=x['position_id'], system=system))
            if metric_gaps:
                return dict(phase='metric_supplement_declaration_required', metric_gaps=metric_gaps,
                    pdb_boundary_complete=True, five_system_complete=False,
                    baseline_tasks=pending, higher_rate_dispatch_forbidden=True)
            return dict(phase='baselines' if pending else 'complete', cap_position_id=p['position_id'],
                cap_rate_rps=p['rate_rps'], cap_rate_rps_decimal=p['rate_rps_decimal'], decision=decision,
                eligible_position_ids=[x['position_id'] for x in reached], baseline_tasks=pending,
                all_reached_baseline_tasks=baseline, pdb_boundary_complete=True, five_system_complete=not pending,
                history_preserved=True, higher_rate_dispatch_forbidden=True)
        return dict(phase='diagnosis' if decision['engineering_fault_cell_ids'] else 'pdblend',
            position_id=p['position_id'], rate_rps=p['rate_rps'], decision=decision,
            next_tasks=[t for t in p['systems']['pdblend'] if t['cell_id'] in decision['missing_cell_ids']],
            higher_rate_dispatch_forbidden=True)
    return dict(phase='extension_declaration_required', next_rate_rps_decimal=number(
        Decimal(group['positions'][-1]['rate_rps_decimal']) + step(group['model'], group['dataset'])),
        proposed_maximum_new_runs=6, existing_declaration_does_not_authorize_undeclared_cells=True)


def apply_audited_reuse(group, observations):
    """Only explicitly pinned, same-host audit entries may replace pending work."""
    result = copy.deepcopy(group)
    for obs in observations:
        need(obs['measurement_host'] == result['node'] and obs['measurement_valid'] is True, 'cross-host/unqualified reuse')
        checked(obs['checkpoint'])
        audit = checked(obs['audit_reference'])
        need(audit.get('slo_threshold_comparison') == 'strict_lt', 'strict audit required')
        matching = [r for r in audit['observations'] if r['checkpoint'] == obs['checkpoint']]
        need(len(matching) == 1 and all(matching[0].get(k) == obs.get(k) for k in
            ('cell_id', 'repeat', 'system', 'slo_attainment', 'measurement_valid', 'work_complete')), 'audit mismatch')
        position = [p for p in result['positions'] if identity(p['workload']) == identity(obs)]
        need(len(position) == 1, 'different host/trace/SLO')
        p = position[0]
        existing = [t for t in p['systems'][obs['system']] if t['repeat'] == obs['repeat']]
        need(len(existing) == 1, 'undeclared repeat')
        old = existing[0]
        if old['action'] == 'reuse':
            need(old['checkpoint'] == obs['checkpoint'], 'cannot replace historical outcome')
            continue
        p['systems'][obs['system']] = [saved_task(obs) if t == old else t for t in p['systems'][obs['system']]]
        result['reused_observations'].append(copy.deepcopy(obs))
    return result
