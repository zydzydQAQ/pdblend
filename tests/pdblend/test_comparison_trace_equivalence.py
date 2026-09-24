import importlib.util
import json
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
from pdblend.bench import comparison_trace_equivalence as eq


def setup(tmp_path):
    def put(name, data):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, sort_keys=True, indent=2)+'\n')
        return eq.binding(path)
    policy = put('new-policy.json', dict(run_id='new-run', model_id='m', revision='new-pd'))
    common = dict(model_id='m', dataset='sharegpt', rate_rps=2., duration_s=150., seed=701,
                  slo=dict(ttft_s=5., tpot_s=.15), measurement_protocol_version='native-comparison-150s/v1')
    trace = dict(common, selection_split='evaluation', corpus_sha256='corpus',
                 requests=[dict(idx=0, arrival_s=.25, prompt=[1, 2, 3], max_tokens=4),
                           dict(idx=1, arrival_s=1.5, prompt=[4, 5], max_tokens=8)])
    old_trace = put('old-trace.json', dict(trace, boundary_policy=dict(path='old-policy', sha256='old')))
    new_trace = put('new-trace.json', dict(trace, boundary_policy=policy))
    engine = dict(image_digest='image', model_hash='weights', tokenizer_hash='tokenizer',
                  fleet_gpu_uuids=['g'+str(i) for i in range(8)], runtime_source_sha256='runtime',
                  measurement_source_sha256='meter')
    old = dict(common, name='mixed-case', system='mixed', scale=1.5625, revision='old-mixed',
               engine_identity=engine, trace=old_trace, source_manifest=dict(path='source', sha256='source'))
    old_ref = put('old/point.json', old)
    metrics = dict(duration_s=150., service_start_s=100., service_end_s=250., energy_service_j=100.,
                   energy_tail_j=5., offered_requests=2, successful_requests=2, failed_requests=0,
                   joint_slo_requests=2, ttft_p99_s=1., tpot_p99_s=.01, ttft_samples=2, tpot_samples=2)
    result = dict(metrics=metrics)
    result_ref = put('old/result.json', result)
    receipt = dict(point=old['name'], point_sha256=eq.digest(old), recorded_window_complete=True,
                   cleanup_passed=True, measurement_evidence_valid=True, result=result,
                   artifacts={'point.json':old_ref['sha256'], 'result.json':result_ref['sha256']})
    receipt_ref = put('old/receipt.json', receipt)
    new = dict(old, name='pd-case', system='pdblend', revision='new-pd', trace=new_trace,
               boundary_policy=policy, run_id='new-run', experiment_phase='slo_boundary_extension')
    new_ref = put('new/point.json', new)
    registry = eq.make_registry([dict(point=old_ref, receipt=receipt_ref)])
    registry_ref = put('registry.json', registry)
    records = eq.read_registry(registry_ref)
    campaign = dict(run_id='new-run', active_extension_policy_refs=[policy])
    return dict(put=put, old=old, new=new, old_ref=old_ref, new_ref=new_ref,
                receipt=receipt, receipt_ref=receipt_ref, registry=registry,
                registry_ref=registry_ref, records=records, campaign=campaign, trace=trace)


def test_exact_provenance_only_equivalence_and_raw_identity_preserved(tmp_path):
    f = setup(tmp_path); proof = eq.build_equivalence(f['records'][0], f['new_ref'])
    ref = f['put']('proof.json', proof)
    assert eq.read_equivalence(ref, f['campaign'], f['records']) == proof
    rows = [dict(trace_sha256=t['sha256'], energy_service_j=42) for t in proof['traces']]
    before = deepcopy(rows); eq.annotate(rows, [(proof, ref)])
    assert rows[0]['analysis_trace_identity_sha256'] == rows[1]['analysis_trace_identity_sha256']
    for a, b in zip(before, rows):
        assert all(b[k] == v for k, v in a.items())
    assert proof['full_request_sequence_compared'] and proof['request_count'] == 2


@pytest.mark.parametrize('kind', ['arrival', 'prompt', 'tokens', 'order', 'extra_request_field',
                                  'unknown_metadata', 'slo', 'corpus', 'seed'])
def test_reject_every_event_or_workload_change(tmp_path, kind):
    f = setup(tmp_path); trace = eq.load_bound(f['new']['trace'])
    if kind == 'arrival': trace['requests'][0]['arrival_s'] += 1e-12
    elif kind == 'prompt': trace['requests'][0]['prompt'][0] += 1
    elif kind == 'tokens': trace['requests'][0]['max_tokens'] += 1
    elif kind == 'order': trace['requests'].reverse()
    elif kind == 'extra_request_field': trace['requests'][0]['provenance'] = 'not_ignored'
    elif kind == 'unknown_metadata': trace['new_metadata'] = 'not_whitelisted'
    elif kind == 'slo': trace['slo']['ttft_s'] += 1
    elif kind == 'corpus': trace['corpus_sha256'] = 'different'
    elif kind == 'seed': trace['seed'] = 702
    new = dict(f['new'], trace=f['put']('changed-trace.json', trace))
    with pytest.raises(ValueError):
        eq.build_equivalence(f['records'][0], f['put']('changed-point.json', new))


@pytest.mark.parametrize('kind', ['policy_not_authorized', 'run_id', 'proof_digest', 'receipt_sha'])
def test_reject_tampered_or_unauthorized_proof(tmp_path, kind):
    f = setup(tmp_path); proof = eq.build_equivalence(f['records'][0], f['new_ref']); campaign=deepcopy(f['campaign'])
    if kind == 'policy_not_authorized': campaign['active_extension_policy_refs'] = []
    elif kind == 'run_id': campaign['run_id'] = 'other'
    elif kind == 'proof_digest': proof['analysis_trace_identity_sha256'] = 'forged'
    elif kind == 'receipt_sha': proof['historical_receipt']['sha256'] = 'forged'
    with pytest.raises(ValueError):
        eq.read_equivalence(f['put']('proof.json', proof), campaign, f['records'])


def test_registry_freezes_slo_failure_without_energy_selection(tmp_path):
    f=setup(tmp_path)
    first_result=deepcopy(f['receipt']['result']); first_result['metrics'].update(successful_requests=0,
        failed_requests=2, joint_slo_requests=0, energy_service_j=999.)
    result_ref=f['put']('old/result.json', first_result)
    first=deepcopy(f['receipt']);first.update(result=first_result)
    first['artifacts']['result.json']=result_ref['sha256']
    first_ref=f['put']('old/receipt.json',first)
    later=deepcopy(first);later['result']['metrics'].update(service_start_s=300.,energy_service_j=1.,
        successful_requests=2,failed_requests=0,joint_slo_requests=2)
    later_point=f['put']('later/point.json',f['old']);later_result=f['put']('later/result.json',later['result'])
    later['artifacts']={'point.json':later_point['sha256'],'result.json':later_result['sha256']}
    later_ref=f['put']('later/receipt.json',later)
    result=eq.make_registry([dict(point=later_point,receipt=later_ref),dict(point=f['old_ref'],receipt=first_ref)])
    assert result['entries'][0]['receipt']==first_ref


def test_registry_rejects_missing_energy_and_mutated_result(tmp_path):
    f=setup(tmp_path)
    result=deepcopy(f['receipt']['result']);result['metrics']['energy_service_j']=None
    result_ref=f['put']('old/result.json',result);receipt=deepcopy(f['receipt']);receipt['result']=result
    receipt['artifacts']['result.json']=result_ref['sha256'];ref=f['put']('old/receipt.json',receipt)
    with pytest.raises(ValueError, match='complete service'):
        eq.checked_record(f['old_ref'],ref)
    f['put']('old/result.json',{'mutated':True})
    with pytest.raises(ValueError, match='exact result'):
        eq.checked_record(f['old_ref'],ref)


def test_preparer_creates_proof_only_for_existing_matching_endpoint(tmp_path):
    f=setup(tmp_path)
    registry,refs=eq.prepare_reuse(f['campaign'],[f['new_ref']],f['registry_ref'],tmp_path/'proofs')
    assert len(refs)==1
    proof=eq.read_equivalence(refs[0],f['campaign'],registry)
    assert eq.permits([proof],f['old']['trace'],f['new']['trace'])
    assert not eq.permits([proof],f['old']['trace'],dict(f['new']['trace'],sha256='other'))


def test_five_system_analysis_uses_proof_and_preserves_old_receipt(tmp_path):
    f=setup(tmp_path); proof=eq.build_equivalence(f['records'][0],f['new_ref']); ref=f['put']('proof.json',proof)
    import pdblend.bench.comparison_recorded as original
    spec=importlib.util.spec_from_file_location('pdblend.bench.proposed_recorded',ROOT/'src/pdblend/bench/comparison_recorded.py')
    proposed=importlib.util.module_from_spec(spec);spec.loader.exec_module(proposed)
    rows=[];records={}
    for system in ('mixed','distserve','ecoserve','dynamollm','pdblend'):
        metrics=deepcopy(f['receipt']['result']['metrics'])
        if system=='pdblend':metrics['energy_service_j']=80.
        row=dict(point_id=system,system=system,revision=system+'-rev',model_id='m',dataset='sharegpt',
            rate_scale=1.5625,seed=701,trace_sha256=proof['traces'][0 if system=='mixed' else 1]['sha256'],
            measurement_protocol_version='native-comparison-150s/v1',model_hash='weights',tokenizer_hash='tokenizer',
            gpu_uuids=['g'+str(i) for i in range(8)],image_digest='image',slo_ttft_s=5.,slo_tpot_s=.15,
            status='measured',receipt_path='/'+system,receipt_sha256=system,**metrics)
        rows.append(row);records[row['receipt_path']]=(dict(metrics=metrics),dict(cleanup_passed=True))
    before=deepcopy(rows)
    original.analyze(deepcopy(rows), records, frozen_baselines={})
    eq.annotate(rows,[(proof,ref)]); proposed.analyze(rows, records, frozen_baselines={})
    assert rows[-1]['comparison_complete_five_systems'] is True
    assert rows[-1]['energy_rank']==1
    assert rows[-1]['pdblend_saving_vs_best_feasible_baseline']==pytest.approx(.2)
    assert len(rows)==5 and len({r['receipt_sha256'] for r in rows})==5
    assert all(r['trace_sha256']==b['trace_sha256'] and all(r[k]==b[k] for k in f['receipt']['result']['metrics'])
               for r,b in zip(rows,before))


def test_preparer_requires_explicit_proof_and_preserves_hardware_check(tmp_path, monkeypatch):
    f=setup(tmp_path);proof=eq.build_equivalence(f['records'][0],f['new_ref'])
    monkeypatch.setitem(sys.modules,'pdblend.bench.comparison_trace_equivalence',eq)
    spec=importlib.util.spec_from_file_location('proposed_baselines',ROOT/'scripts/2026-09-24_prepare_boundary_baselines.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old=dict(f['old'],rate_scale=f['old']['scale'])
    assert module.same_workload(old,f['new']) is False
    assert module.same_workload(old,f['new'],trace_proofs=[proof]) is True
    changed=deepcopy(f['new']);changed['engine_identity']['model_hash']='other'
    assert module.same_workload(old,changed,trace_proofs=[proof]) is False


def test_trace_proof_does_not_waive_meter_compatibility(tmp_path):
    f=setup(tmp_path);target=deepcopy(f['new']);target['engine_identity']['measurement_source_sha256']='new-meter'
    target_ref=f['put']('other-meter-point.json',target)
    proof=eq.build_equivalence(f['records'][0],target_ref)
    with pytest.raises(ValueError,match='measurement compatibility'):
        eq.read_equivalence(f['put']('proof.json',proof),f['campaign'],f['records'])


def test_history_inventory_cannot_silently_omit_prior_frozen_receipt(tmp_path):
    f=setup(tmp_path);history=f['put']('history.json',dict(completed_receipts=[f['receipt_ref']]))
    with pytest.raises(ValueError,match='all predeclared'):
        eq.make_registry([],history_inventory=history)


def test_raw_trace_identity_stays_distinct_without_authorized_annotation(tmp_path):
    f=setup(tmp_path)
    spec=importlib.util.spec_from_file_location('pdblend.bench.proposed_identity',ROOT/'src/pdblend/bench/comparison_recorded.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    row=dict(model_id='m',dataset='sharegpt',rate_scale=1.5625,seed=701,duration_s=150.,
        trace_sha256=f['old']['trace']['sha256'],measurement_protocol_version='protocol',
        model_hash='weights',tokenizer_hash='tokenizer',gpu_uuids=['g0'],image_digest='image',
        slo_ttft_s=5.,slo_tpot_s=.15)
    assert module._identity(row)!=module._identity(dict(row,trace_sha256=f['new']['trace']['sha256']))
    assert module._identity(dict(row, analysis_trace_identity_sha256='same')) != module._identity(dict(row,trace_sha256='same'))


def test_completion_reuse_requires_the_same_proof_and_first_frozen_receipt(tmp_path):
    f=setup(tmp_path);proof=eq.build_equivalence(f['records'][0],f['new_ref']);ref=f['put']('proof.json',proof)
    campaign=dict(f['campaign'],baseline_energy_gaps=f['put']('inventory.json',dict(frozen_baselines=[])))
    entry=dict(receipt=f['receipt_ref'],system='mixed',dataset='sharegpt',scale=1.5625,
               trace_equivalence_refs=[ref])
    assert eq.validate_reuse(entry,f['new_ref'],campaign,f['records'],[(proof,ref)])['receipt']==f['receipt_ref']
    with pytest.raises(ValueError,match='authorized proof'):
        eq.validate_reuse(dict(entry,trace_equivalence_refs=[]),f['new_ref'],campaign,f['records'],[(proof,ref)])
    with pytest.raises(ValueError,match='predeclared frozen'):
        eq.validate_reuse(entry,f['new_ref'],campaign,[],[(proof,ref)])
    with pytest.raises(ValueError,match='another endpoint'):
        eq.validate_reuse(dict(entry,scale=2.),f['new_ref'],campaign,f['records'],[(proof,ref)])
