import json
import pytest
from pdblend_baselines.native_profile import audit, _sha, _select_samples, measured_points, _sample_ms, _phase_max_tokens


def test_native_profile_phase_token_budgets_are_explicit():
    assert _phase_max_tokens('prefill') == 1
    assert _phase_max_tokens('decode') == 32
    with pytest.raises(ValueError):
        _phase_max_tokens('mixed')


def sample(role='prefill',context=512,tp=2,scope='runner'):
    return {'ranks':[{'rank':rank,'samples':[dict(role=role,input_tokens=context if role=='prefill' else 1,
        context_tokens=context if role=='prefill' else context+1,batch=1,gpu_elapsed_ms=1.0+rank,
        measurement_scope=scope,request_ids=['request'])]} for rank in range(tp)]}


def artifact(system='distserve'):
    meta={k:'x' for k in ('model_hash','tokenizer_hash','engine_version','image_digest','source_revision')}
    meta.update(gpu_uuids=['GPU-1','GPU-2'],tp=2,pp=1)
    if system=='distserve':
        rows=[]
        for freq in (900,1200,1500,1800,2100,2520):
            value=sample(); sha=_sha(value)
            selected=_select_samples(value['ranks'],role='prefill',context=512,batch=1,scope='runner',tp=2)
            rows.append(dict(frequency_mhz=freq,role='prefill',context_tokens=512,batch_size=1,
                             sample=value,sample_sha256=sha,points=measured_points(selected,tp=2,raw_sha256=sha)))
    else:
        rows=[]
        for length in (16,128,512,1024,2048,4096,7168):
            values=[sample(context=length,scope='forward') for _ in range(5)]
            rows.append(dict(input_tokens=length,repetitions=values,samples_sha256=[_sha(v) for v in values],minimum_ms=2.0))
    return dict(schema='pdblend-baseline-profile-v1',system=system,metadata=meta,rows=rows,complete=True)


@pytest.mark.parametrize('system',['distserve','ecoserve'])
def test_profile_audit_checks_actual_rank_samples_and_provenance(tmp_path,system):
    path=tmp_path/'p.json'; value=artifact(system);path.write_text(json.dumps(value))
    assert audit(path)['valid'] and not audit(path)['formal_eligible']
    value['metadata'].pop('image_digest');path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='provenance'):audit(path)
    value=artifact(system)
    if system=='distserve':value['rows'][0]['sample']['ranks'][0]['samples'][0]['gpu_elapsed_ms']=99
    else:value['rows'][0]['minimum_ms']=1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='checksum|MIN'):audit(path)


def test_tp_forward_uses_slowest_rank_and_decode_does_not_contaminate_prefill():
    value=sample(scope='forward')
    assert _sample_ms(value,context=512,tp=2)==2.0
    for rank in value['ranks']:
        rank['samples'].append(dict(rank['samples'][0],role='decode',context_tokens=513,input_tokens=1,gpu_elapsed_ms=.1))
    assert _sample_ms(value,context=512,tp=2)==2.0
    value['ranks'][1]['samples'][0]['request_ids']=['other']
    with pytest.raises(ValueError,match='alignment'):_sample_ms(value,context=512,tp=2)


def test_missing_rank_and_chunk_fragment_fail_closed():
    value=sample()
    with pytest.raises(ValueError,match='rank coverage'):
        _select_samples(value['ranks'][:1],role='prefill',context=512,batch=1,scope='runner',tp=2)
    value['ranks'][0]['samples'][0]['input_tokens']=128
    with pytest.raises(ValueError,match='full batch'):
        _select_samples(value['ranks'],role='prefill',context=512,batch=1,scope='runner',tp=2)
