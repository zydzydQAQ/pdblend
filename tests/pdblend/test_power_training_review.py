"""CPU-only tests for the separate training power proposal, not the live API."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2]/'scripts/2026-09-23_compare_7b_tp4_power.py'
spec = importlib.util.spec_from_file_location('power_training_review', SCRIPT)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def rows():
    return [dict(freq_mhz=f,batch=b,context_tokens=c,
        repeats=[dict(effective_context_tokens=c+10+i,power_w=100+b+c/100+i*.1) for i in range(3)])
        for f in (900,1200) for b in (1,4,8,16) for c in (256,1024,4096)]


def test_errors_use_observed_denominator_and_preserve_unsupported():
    report=review.errors([100,100,100],[90,110,None])
    assert report['max_error']==pytest.approx(.1)
    assert report['supported']==2 and report['unsupported']==1
    assert review.errors([100],[None])['power_training_gate'] is False


def test_training_groups_keep_all_repeats_together():
    data=rows(); held=data[4]
    for axis in ('shape','batch','nominal_context'):
        train,test=review.split_group(data,held,axis)
        assert not {review.identity(r) for r in train}&{review.identity(r) for r in test}
        assert all(r['freq_mhz']==held['freq_mhz'] for r in train+test)
        assert all(len(r['repeats'])==3 for r in train+test)
        assert held in test and held not in train
    train,test=review.split_group(data,held,'batch')
    assert len(test)==3 and all(r['batch']!=held['batch'] for r in train)


def test_table_never_claims_unmeasured_b1_to_b4_or_context_rectangle():
    nodes=[dict(batch=1,context_min=700,context_max=710,power_w=600),
           dict(batch=1,context_min=1500,context_max=1510,power_w=610),
           dict(batch=4,context_min=650,context_max=660,power_w=550),
           dict(batch=4,context_min=1400,context_max=1410,power_w=560),
           dict(batch=8,context_min=600,context_max=610,power_w=545),
           dict(batch=8,context_min=1300,context_max=1310,power_w=565)]
    assert review.table_predict(nodes,1,705)==600
    assert review.table_predict(nodes,1,699) is None
    assert review.table_predict(nodes,1,1511) is None
    assert review.table_predict(nodes,2,1000) is None
    assert review.table_predict(nodes,3,1000) is None
    assert review.table_predict(nodes,6,1000)>0
    # At c=1350, B4 is supported but B8 is not. Broad min/max would lie.
    assert review.table_predict(nodes,6,1350) is None
    assert review.table_predict(nodes,9,1000) is None
    assert review.table_predict(nodes,4,1500) is None


def test_cv_counts_boundary_missingness_separately_from_error():
    result=review.compare(rows())['bounded_table_linear_batch']
    assert result['training_resubstitution']['supported']==72
    assert result['shape']['supported']==24
    assert result['shape']['unsupported']==48
    assert result['nominal_context']['supported']==24
    assert result['batch']['unsupported']>0
    assert all(r['relative_error'] is None for r in result['batch']['rows']
               if r['status']=='outside_remaining_training_coverage')


def test_holdout_cannot_be_passed_as_training(tmp_path):
    with pytest.raises(ValueError,match='never holdout'):
        review.validate_training(dict(system='pdblend',model_id='Qwen2.5-7B-Instruct',tp=4,pp=1,
                                      holdout_independent=True),tmp_path/'raw.json')


def test_shared_plan_keeps_three_actual_windows_and_original_timing(tmp_path):
    raw=dict(model_id='Qwen2.5-7B-Instruct',model_hash='model',tokenizer_hash='tokenizer',
             kv_capacity_tokens=2441936,decode=[],prefill=[])
    tables={}
    for f in review.FREQUENCIES:
        tables[str(f)]=[]
        for b in (1,64,128,256):
            for c in (256,1024,4096):
                raw['decode'].append(dict(freq_mhz=f,batch=b,effective_context_tokens=c+300,step_seconds=.03))
                tables[str(f)].append(dict(batch=b,context_min=c+295,context_max=c+305,power_w=600+c/50))
        for c in (128,512,2048,4096,7168):
            raw['prefill'].append(dict(freq_mhz=f,input_tokens=c,seconds=c*.0001))
    candidate=tmp_path/'candidate.json';candidate.write_text('{}')
    plan=review.fresh_plan(raw,dict(tables=tables),candidate)
    assert len(plan['points'])==24 and plan['minimum_decode_window_seconds']==504
    assert plan['timing_recollection_required'] is False
    assert plan['executable_collector_available'] is False and plan['enqueue'] is False
    assert plan['cpu_scheduling_proxy']['shared_prefill_serial_work_proxy_seconds'] < plan['cpu_scheduling_proxy']['separate_prefill_serial_work_proxy_seconds']
    for point in plan['points']:
        option=point['shared_prefill_option']
        assert option['distinct_windows']==3 and option['actual_each_window_context_and_prediction_required']
        assert len(option['estimated_effective_contexts'])==3
        assert all(review.table_predict(tables[str(point['freq_mhz'])],point['batch'],c) is not None
                   for c in option['estimated_effective_contexts'])
        assert point['batch']*(option['context_tokens']+option['max_tokens'])<=.9*raw['kv_capacity_tokens']


def test_review_outputs_refuse_overwrite(tmp_path):
    path=tmp_path/'proposal.json';review.write(path,dict(a=1));review.write(path,dict(a=1))
    with pytest.raises(ValueError,match='refusing to overwrite'):
        review.write(path,dict(a=2))
    assert json.loads(path.read_text())==dict(a=1)
