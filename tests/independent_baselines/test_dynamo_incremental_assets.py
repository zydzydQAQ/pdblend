import hashlib
import json
from pathlib import Path

import pytest
from types import SimpleNamespace

from pdblend_baselines.dynamollm import history_adapter, history_provenance
from pdblend_baselines.dynamollm.prediction_cache import prompt_sha, read_prefix, row_identity
from pdblend_baselines.dynamollm.transition_costs import validate_cost
from pdblend_baselines.dynamollm.transition_evidence import integrate
from pdblend_baselines.dynamollm.validation import verified_history
from pdblend_baselines.dynamollm.profile_v1 import measurement_points


def test_history_revalidation_reaggregates_and_never_restores_old_receipt(tmp_path, monkeypatch):
    source=tmp_path/'AzureLLMInferenceTrace_code_1week.csv'
    source.write_text('TIMESTAMP,ContextTokens,GeneratedTokens\n'
                      '2024-01-01T12:00:00+00:00,128,16\n'
                      '2024-01-07T12:00:00+00:00,512,32\n')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setitem(history_adapter.PINNED_ASSETS, source.name, digest)
    monkeypatch.setattr(history_provenance, 'REFERENCES',
                        {'LICENSE':hashlib.sha256(b'fixed reference').hexdigest()})
    result=history_provenance.prepare(source,tmp_path/'new',get=lambda _:b'fixed reference')
    assert result['counts']==2 and result['raw_reaggregated']
    assert not result['chronological_holdout_available']
    receipt=Path(result['source_receipt'])
    value=json.loads(receipt.read_text())
    assert value['downloaded'] is False and value['prior_download_receipt_restored'] is False
    _,_,audit=verified_history(result['history'])
    assert audit['requests']==2
    value['downloaded']=True
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='receipt differs'):
        verified_history(result['history'])


def test_history_wrong_source_fails_before_creating_output(tmp_path):
    source=tmp_path/'AzureLLMInferenceTrace_code_1week.csv';source.write_text('wrong')
    with pytest.raises(ValueError,match='source SHA'):
        history_provenance.prepare(source,tmp_path/'new')
    assert not (tmp_path/'new').exists()


def test_transition_power_interpolates_actual_bracket_only():
    result=integrate([(0,100),(1,200),(2,100)],.5,1.5,max_gap_s=1)
    assert result['energy_j']==pytest.approx(175)
    assert result['extrapolation'] is False
    with pytest.raises(ValueError,match='bracket'):
        integrate([(0,100),(1,200)],-.1,.9,max_gap_s=1)
    with pytest.raises(ValueError,match='gap'):
        integrate([(0,100),(1,200)],.1,.9)


def test_empty_drain_cost_cannot_qualify_nonempty_workload():
    cost=dict(measurement='hardware',system='dynamollm',engine_revision='vllm-0.10.1.1',
              model_id='Qwen2.5-7B-Instruct',duration_s=20,energy_j=5000,
              workload_envelope=dict(max_input_tokens=0,max_output_tokens=0))
    with pytest.raises(ValueError,match='coverage absent'):
        validate_cost(cost,dict(input_tokens=128,output_tokens=16))


def test_prediction_cache_rejects_another_prompt_or_noncontiguous_repeat(tmp_path):
    rows=[dict(prompt=[1,2],input_tokens=2,request_shape_sha256='shape-a'),
          dict(prompt=[3,4],input_tokens=2,request_shape_sha256='shape-b')]
    record=dict(row_identity(rows[0],0),binding_sha256='binding',predicted_output=74)
    path=tmp_path/'predictions.jsonl';path.write_text(json.dumps(record)+'\n')
    assert len(read_prefix(path,rows,'binding'))==1
    path.write_text((json.dumps(record)+'\n')*2)
    with pytest.raises(ValueError,match='prefix'):
        read_prefix(path,rows,'binding')
    path.write_text(json.dumps(dict(record,prompt_tokens_sha256=prompt_sha([9,9])))+'\n')
    with pytest.raises(ValueError,match='identity'):
        read_prefix(path,rows,'binding')


def test_sparse_profile_plan_does_not_cartesian_expand_or_borrow_model(tmp_path):
    plan=dict(schema='dynamo-missing-profile-cells-v1',system='dynamollm',
        model_id='Qwen2.5-32B-Instruct',tp=2,pp=1,fit_from_holdout=False,
        points=[dict(frequency_mhz=900,input_tokens=7168,output_tokens=512,batch=1),
                dict(frequency_mhz=1200,input_tokens=16,output_tokens=74,batch=4)])
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    args=SimpleNamespace(tp=2,points_file=path)
    assert measurement_points(args,plan['model_id'])==plan['points']
    with pytest.raises(ValueError,match='identity'):
        measurement_points(args,'Qwen2.5-7B-Instruct')
    plan['points'].append(plan['points'][0]);path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='duplicate'):
        measurement_points(args,plan['model_id'])
