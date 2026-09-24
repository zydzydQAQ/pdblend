import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import pytest

MODULE = Path(os.environ.get('PDBLEND_EXPLICIT_REJECTION_MODULE',
                            Path(__file__).resolve().parents[2] / 'scripts/explicit_rejection_boundary.py'))
spec = importlib.util.spec_from_file_location('explicit_rejection_boundary_test', MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture(tmp_path):
    def put(path, value):
        path = tmp_path / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
        return mod.binding(path)

    template = dict(system='pdblend', model_id='model', run_id='run', revision='revision',
                    dataset='sharegpt', seed=701, duration_s=150, slo=dict(ttft_s=5, tpot_s=.15),
                    engine_identity={'frozen': 'all_launch_options'}, inputs={}, source_manifest={'sha256': 'source'})
    policy = dict(schema=mod.BOUNDARY_SCHEMA, mode='policy', run_id='run', model_id='model',
                  revision='revision', engine_signature=mod.digest(template['engine_identity']),
                  seed=701, duration_s=150, growth_factor=1.25,
                  datasets={'sharegpt': dict(base_rate_rps=4, template_point=put('template.json', template))})
    policy_ref = put('policy.json', policy)

    def window(label, scale, rejected):
        point = dict(template, name=label, scale=scale, rate_rps=4*scale, boundary_policy=policy_ref,
                     trace={'path': 'never-open-trace', 'sha256': label},
                     inputs={'trace': {'path': 'never-open-trace', 'sha256': label}})
        point_ref = put(label+'/point.json', point)
        requests = [dict(idx=i, successful=True, timeout=False, token_events_complete=True,
                         timing_complete=True, error=None, ttft_s=.2+i*.1, tpot_s=.05,
                         joint_slo=True, completion_tokens=10) for i in range(3)]
        if rejected:
            requests += [dict(idx=3, successful=False, timeout=False, token_events_complete=True,
                              timing_complete=False, error='503: {"error":"no instance accepting requests"}',
                              ttft_s=None, tpot_s=None, completion_tokens=0, native_reported_completion_tokens=None,
                              joint_slo=False, window_good=False, pending_at_window_end=False,
                              completed_in_window=False, terminal_after_window=False)]
        metrics = dict(service_start_s=1000, service_end_s=1150, duration_s=150, tail_end_s=1151,
                       request_elapsed_s=151, offered_requests=len(requests), successful_requests=3,
                       failed_requests=int(rejected), joint_slo_requests=3, invalid_timing_requests=int(rejected),
                       ttft_samples=3, tpot_samples=3, timeout_requests=0, unresolved_requests=0,
                       token_timing_complete=True, success_rate=3/len(requests), joint_slo_rate=3/len(requests),
                       slo_pass=not rejected)
        for name in ('ttft', 'tpot'):
            values = sorted(r[name+'_s'] for r in requests if r['successful'])
            for p in (50, 90, 95, 99):
                metrics[f'{name}_p{p}_s'] = values[math.ceil(p/100*len(values))-1]
        result = {'metrics': metrics}
        artifacts = {'point.json': point_ref['sha256']}
        for name, value in [('result.json', result), ('drain.json', {'passed': True, 'tail_end_s': 1151}),
                            ('run/comparison-requests.json', requests)]:
            artifacts[name] = put(label+'/'+name, value)['sha256']
        receipt = dict(point=label, point_sha256=mod.digest(point), engine_signature=policy['engine_signature'],
                       cleanup_passed=True, recorded_window_complete=True, result=result, artifacts=artifacts)
        receipt_ref = put(label+'/receipt.json', receipt)
        return dict(point=point_ref, receipt=receipt_ref, scale=scale)

    lower, upper = window('lower', 1.5625, False), window('upper', 1.953125, True)
    observations = [dict(dataset='sharegpt', **lower, verdict='pass', reason='measured_slo'),
                    dict(dataset='sharegpt', **upper, verdict='incomplete', reason='incomplete_or_inconsistent_request_timing')]
    decisions = [put('decision-'+label+'.json', dict(policy=policy_ref, point=e['point'], dataset='sharegpt', scale=e['scale']))
                 for label,e in [('lower',lower),('upper',upper)]]
    state = dict(status='incomplete_observation', failed_upper=None, next_rate_scale=None,
                 passed_lower=lower['scale'], passed_lower_point=lower['point'], passed_lower_receipt=lower['receipt'])
    head = dict(schema=mod.BOUNDARY_SCHEMA, mode='manifest', policy=policy_ref, observations=observations,
                decisions=decisions, points=[dict(point=e['point'], receipt=e['receipt'], decision=d)
                                             for e,d in zip([lower,upper], decisions)],
                boundaries={'sharegpt':state}, **{k:policy[k] for k in ['run_id','model_id','revision','engine_signature']})
    head_ref = put('head.json', head)
    sidecar_ref = mod.build_sidecar(head_ref, 'sharegpt', tmp_path/'sidecar.json')
    return dict(sidecar=sidecar_ref, head=head_ref, lower=lower, upper=upper, policy=policy_ref)


def test_complete_explicit_rejections_form_observed_bound(tmp_path):
    f = fixture(tmp_path)
    validated = mod.validate_explicit_rejection_sidecar(f['sidecar'])
    assert validated['lower']['scale'] == 1.5625
    assert validated['upper']['scale'] == 1.953125
    assert validated['summary']['upper']['counts']['invalid_timing_requests'] == 1
    assert validated['summary']['upper']['failed_request_indices'] == [3]
    assert validated['summary']['upper']['slo_pass'] is False
    assert validated['original_boundary']['failed_upper'] is None
    assert validated['canonical_metrics_modified'] is False
    assert mod.build_sidecar(f['head'], 'sharegpt', tmp_path/'sidecar.json') == f['sidecar']


@pytest.mark.parametrize('mutation', [
    'unknown_error', 'timeout', 'unresolved', 'successful_timing_missing', 'duplicate_idx',
    'request_missing', 'unknown_terminal', 'mixed_failure', 'wrong_percentile', 'wrong_slo',
    'incomplete_drain', 'execution_error', 'wrong_engine', 'wrong_policy', 'wrong_head',
    'wrong_summary', 'wrong_seed_bool', 'wrong_upper_scale', 'different_trace',
])
def test_rejects_incomplete_or_misbound_evidence(tmp_path, mutation):
    f = fixture(tmp_path)
    def load(ref):
        value = copy.deepcopy(mod.read_bound(ref))
        path = ref['path']
        if path.endswith('/upper/run/comparison-requests.json'):
            if mutation == 'unknown_error': value[-1]['error'] = '500: engine crash'
            if mutation == 'timeout': value[-1]['timeout'] = True
            if mutation == 'successful_timing_missing': value[0]['ttft_s'] = None
            if mutation == 'duplicate_idx': value[1]['idx'] = 0
            if mutation == 'request_missing': value.pop()
            if mutation == 'unknown_terminal': value[-1]['pending_at_window_end'] = True
            if mutation == 'mixed_failure': value[0]['successful'] = False
        if path.endswith('/upper/receipt.json'):
            if mutation == 'unresolved': value['result']['metrics']['unresolved_requests'] = 1
            if mutation == 'wrong_percentile': value['result']['metrics']['ttft_p99_s'] = 1.2
            if mutation == 'wrong_slo': value['result']['metrics']['slo_pass'] = True
            if mutation == 'execution_error': value['error'] = 'TimeoutError'
            if mutation == 'wrong_engine': value['engine_signature'] = 'wrong'
        if path.endswith('/upper/drain.json') and mutation == 'incomplete_drain': value['passed'] = False
        if path.endswith('/upper/point.json'):
            if mutation == 'wrong_policy': value['boundary_policy'] = f['head']
            if mutation == 'wrong_seed_bool': value['seed'] = True
            if mutation == 'different_trace': value['trace']['sha256'] = 'different'
        if ref == f['head']:
            if mutation == 'wrong_head': value['model_id'] = 'another'
            if mutation == 'wrong_upper_scale': value['observations'][-1]['scale'] = 9
        if ref == f['sidecar'] and mutation == 'wrong_summary': value['summary']['upper']['counts']['failed_requests'] = 0
        return value
    with pytest.raises(ValueError):
        mod.validate_explicit_rejection_sidecar(f['sidecar'], load_bound=load)


def test_rejects_real_artifact_tampering(tmp_path):
    f = fixture(tmp_path)
    path = Path(f['upper']['receipt']['path'])
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match='SHA mismatch'):
        mod.validate_explicit_rejection_sidecar(f['sidecar'])


def test_execution_obstruction_cannot_be_promoted(tmp_path):
    f = fixture(tmp_path)
    def load(ref):
        value = copy.deepcopy(mod.read_bound(ref))
        if ref == f['head']: value['observations'][-1]['reason'] = 'execution_or_drain_failure'
        return value
    with pytest.raises(ValueError, match='execution obstruction'):
        mod.validate_explicit_rejection_sidecar(f['sidecar'], load_bound=load)


def test_small_read_limit_and_pointer_rejected(tmp_path):
    p = tmp_path/'large.json'; p.write_bytes(b' '*(mod.MAX_SMALL_BYTES+1))
    with pytest.raises(ValueError, match='small-read limit'): mod.binding(p)
    f = fixture(tmp_path)
    def load(ref):
        value = copy.deepcopy(mod.read_bound(ref))
        if ref == f['head']: value['mode'] = 'pointer'
        return value
    with pytest.raises(ValueError, match='immutable manifest'):
        mod.validate_explicit_rejection_sidecar(f['sidecar'], load_bound=load)


def csv_rows(f, tmp_path):
    rows = []
    for side in ('lower', 'upper'):
        e = f[side]; p = mod.read_bound(e['point'])
        rows.append(dict(system='pdblend', point_id=p['name'], point_sha256=mod.digest(p),
                         receipt_sha256=e['receipt']['sha256'], receipt_path=e['receipt']['path'],
                         model_id=p['model_id'], dataset=p['dataset'], revision=p['revision'], run_id=p['run_id'],
                         trace_sha256=p['trace']['sha256'], energy_service_j='100.000000',
                         invalid_timing_requests='1' if side == 'upper' else '0',
                         boundary_failed_upper='', boundary_status='incomplete_observation',
                         comparison_complete_five_systems='False', available_candidate_energy_rank=''))
    selection = tmp_path/'selection.json'
    selection.write_text(json.dumps({'classification_sidecars':[f['sidecar']]}))
    for system in ('mixed', 'distserve', 'ecoserve', 'dynamollm'):
        for side in ('lower', 'upper'):
            p = mod.read_bound(f[side]['point'])
            rows.append(dict(system=system, point_id=system+'-'+side, receipt_sha256='',
                             model_id=p['model_id'], dataset=p['dataset'], trace_sha256=p['trace']['sha256'],
                             pd_boundary_point=json.dumps(f[side]['point']), boundary_selection=json.dumps(mod.binding(selection)),
                             comparison_complete_five_systems='False', available_candidate_energy_rank='',
                             energy_service_j='', invalid_timing_requests=''))
    rows.append(dict(system='pdblend', point_id='unrelated', receipt_sha256='other', energy_service_j='1.234567'))
    return rows


def test_annotation_keeps_all_old_cells_and_is_idempotent(tmp_path):
    f = fixture(tmp_path); rows = csv_rows(f, tmp_path); before = copy.deepcopy(rows)
    annotated = mod.annotate_classified_boundary_rows(rows, [f['sidecar']])
    assert rows == before
    assert [{k:r[k] for k in original} for r, original in zip(annotated, rows)] == rows
    assert sum(bool(r['classified_boundary_role']) for r in annotated) == 10
    assert annotated[0]['classified_boundary_role'] == 'passed_lower'
    assert annotated[1]['classified_boundary_role'] == 'failed_upper'
    assert annotated[1]['boundary_failed_upper'] == ''
    assert annotated[1]['invalid_timing_requests'] == '1'
    assert annotated[1]['available_candidate_energy_rank'] == ''
    assert annotated[-1]['classified_boundary_status'] == ''
    assert mod.annotate_classified_boundary_rows(annotated, [f['sidecar']]) == annotated


@pytest.mark.parametrize('mutation', ['wrong_trace', 'missing_pd', 'wrong_pd_digest', 'unauthorized_selection', 'conflicting_annotation'])
def test_annotation_rejects_unbound_group(tmp_path, mutation):
    f = fixture(tmp_path); rows = csv_rows(f, tmp_path)
    if mutation == 'wrong_trace': rows[2]['trace_sha256'] = 'unrelated'
    if mutation == 'missing_pd': rows.pop(1)
    if mutation == 'wrong_pd_digest': rows[1]['point_sha256'] = 'wrong'
    if mutation == 'unauthorized_selection':
        p=tmp_path/'empty-selection.json'; p.write_text('{}'); rows[2]['boundary_selection']=json.dumps(mod.binding(p))
    if mutation == 'conflicting_annotation': rows[0]['classified_boundary_failed_upper'] = 'different'
    with pytest.raises(ValueError): mod.annotate_classified_boundary_rows(rows, [f['sidecar']])


def test_csv_staging_only_and_no_formal_write(tmp_path):
    with pytest.raises(ValueError, match='sole writer'):
        mod.annotate_classified_boundary_csv(tmp_path/'compare.csv', [])
