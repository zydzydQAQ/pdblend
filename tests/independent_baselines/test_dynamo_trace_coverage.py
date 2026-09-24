import json
from itertools import product

import pytest

from pdblend_baselines.dynamollm.deployment import save, sha
from pdblend_baselines.dynamollm.prediction_cache import row_identity
from pdblend_baselines.dynamollm.profiles import CoverageError, PaperProfiles
from pdblend_baselines.dynamollm.run_v1 import qualify
from pdblend_baselines.dynamollm.trace_coverage import audit_predictions, bind_trace, supports
from pdblend_baselines.dynamollm.validation import preflight


def fixture_assets(tmp_path):
    corpus, cache, predictor = [tmp_path/name for name in ('corpus','cache','predictor')]
    corpus.mkdir(); cache.mkdir(); predictor.mkdir()
    rows=[dict(prompt=[1,2],input_tokens=2,output_tokens=9,request_shape_sha256='shape0'),
          dict(prompt=[3,4],input_tokens=2,output_tokens=10,request_shape_sha256='shape1')]
    save(corpus/'alpaca.json',dict(evaluation=rows))
    save(corpus/'manifest.json',dict(dataset_sha256={'alpaca':sha(corpus/'alpaca.json')},
                                    tokenizer_sha256='corpus-tokenizer'))
    save(predictor/'manifest.json',dict(model='own-predictor'))
    binding=dict(model_identity={'model':'Qwen2.5-7B-Instruct'},
                 corpus_manifest_sha256=sha(corpus/'manifest.json'),predictor=str(predictor),
                 predictor_manifest_sha256=sha(predictor/'manifest.json'))
    save(cache/'binding.json',binding)
    predictions=[dict(row_identity(r,i),binding_sha256=sha(cache/'binding.json'),
        dataset='alpaca',split='evaluation',predicted_output=74,
        corpus_file_sha256=sha(corpus/'alpaca.json'),actual_output_limit=r['output_tokens'],
        output_limit_used_for_prediction=False) for i,r in enumerate(rows)]
    path=cache/'alpaca-evaluation.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in predictions))
    save(cache/'completion.json',dict(complete=True,binding_sha256=sha(cache/'binding.json'),
        artifacts={str(path.resolve()):sha(path)},counts={'alpaca-evaluation':2}))
    trace=tmp_path/'trace.json'
    save(trace,dict(schema='five-system-evaluation-trace-v1',selection_split='evaluation',
        seed=701,duration_s=300,model_id='Qwen2.5-7B-Instruct',dataset='alpaca',rate_rps=1,
        slo=dict(ttft_s=1,tpot_s=.1),corpus_sha256=sha(corpus/'alpaca.json'),
        corpus_manifest_sha256=sha(corpus/'manifest.json'),corpus_tokenizer_sha256='corpus-tokenizer',
        requests=[dict(idx=i,arrival_s=i+.1,prompt=rows[j]['prompt'],max_tokens=rows[j]['output_tokens'])
                  for i,j in enumerate((1,0,1))]))
    return trace,cache,corpus


def test_trace_cache_binding_matches_prompt_and_limit_not_reordered_index(tmp_path):
    trace,cache,corpus=fixture_assets(tmp_path)
    value=bind_trace(trace,cache=cache,corpus=corpus)
    assert [r['corpus_indices'] for r in value['predictions']] == [[1],[0],[1]]
    assert {r['predicted_output'] for r in value['predictions']} == {74}
    assert not value['prediction_cache_used_for_runtime']
    altered=json.loads(trace.read_text()); altered['requests'][0]['max_tokens']=11
    save(trace,altered)
    with pytest.raises(ValueError,match='lacks unambiguous'):
        bind_trace(trace,cache=cache,corpus=corpus)


def test_trace_binding_rejects_changed_corpus_or_predictor(tmp_path):
    trace,cache,corpus=fixture_assets(tmp_path)
    (tmp_path/'predictor/manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='predictor manifest'):
        bind_trace(trace,cache=cache,corpus=corpus)


def test_150_second_coverage_binds_actual_duration_without_relaxing_original_cycles(tmp_path):
    trace,cache,corpus=fixture_assets(tmp_path)
    value=json.loads(trace.read_text());value['duration_s']=150;save(trace,value)
    bound=bind_trace(trace,cache=cache,corpus=corpus)
    assert bound['duration_s']==150 and len(bound['predictions'])==3
    assert not bound['formal_eligible'] and not bound['predictor_fitted_here']
    missing=preflight({},mode='comparison',duration_s=150)['missing_evidence']
    assert 'duration' not in missing and 'missing_original_cycle_mechanisms' in missing
    value['requests'][-1]['arrival_s']=150;save(trace,value)
    with pytest.raises(ValueError,match='invalid shared trace'):
        bind_trace(trace,cache=cache,corpus=corpus)


def test_geometry_matches_own_complete_rectangle_rule_and_missing_batch():
    cells=set(product((16,7168),(74,512),(1,16)))
    cells.add((128,74,1)) # Extra partial grid cannot hide the complete rectangle.
    points=[dict(role='mixed',tp=1,frequency_mhz=900,input_tokens=n,context_tokens=n+o,
                 batch=b,prefill_s=.01,iteration_s=.02,power_w=100,samples=3,
                 source_sha256='a'*64) for n,o,b in cells]
    profile=PaperProfiles(points,coordinate_system='input_output_batch')
    for target in ((256,317,4),(128,74,1),(7168,512,16),(8,74,1),(128,74,17)):
        try:
            profile.query(1,900,target[0],target[0]+target[1],target[2]); numerical=True
        except CoverageError:
            numerical=False
        assert supports(cells,target) == numerical
    assert not supports({(16,74,1),(7168,512,1)},(128,317,1))


def test_b1_receipt_does_not_cover_native_batches_or_other_tp():
    bound=dict(dataset='alpaca',trace_sha256='trace',predictions=[dict(input_tokens=128,
        predicted_output=74,actual_output_limit=256,arrival_s=.1,shape='SS')])
    points=[dict(tp=1,frequency_mhz=f,input_tokens=n,output_tokens=o,batch=1)
            for f,n,o in product((900,1200,1500,1800,2100,2520),(16,7168),(74,512))]
    audit=audit_predictions(bound,points,tps=(1,2),max_batch=4)
    assert audit['topologies'][0]['fully_supported_batches']==[1]
    assert audit['topologies'][1]['fully_supported_batches']==[]
    assert not audit['formal_eligible'] and not audit['native_batch_observed']


def test_short_comparison_keeps_original_control_asset_gates():
    result=preflight({},mode='comparison',duration_s=300)
    assert 'duration' not in result['missing_evidence']
    assert set(result['missing_evidence']) >= {'missing_history','missing_shape_profile',
        'missing_transition_cost','missing_golden','missing_workload_coverage',
        'missing_original_cycle_mechanisms'}
    assert result['periods_s']=={'ScaleInst':1800.,'ScaleShard':300.,'ScaleFreq':5.}
    assert not result['ready']
    wrong=preflight({},mode='comparison',duration_s=100)
    assert 'duration' in wrong['missing_evidence']
    success=qualify([dict(event='dynamo_route',request_id='r')],
                    [dict(request_id='r',ok=True)],mode='comparison',duration_s=300)
    assert success['status']=='passed' and not success['formal_eligible']
    assert not success['short_window_dynamic_tp_benefit_claim']
