"""Dynamo/common-meter integration with real immutable files and CPU fakes."""
import asyncio
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from pdblend.bench import comparison_dynamo_runtime as module
from pdblend.bench.comparison_campaign import binding
from pdblend.bench.resident_session import digest, engine_signature, write_new


@pytest.fixture
def harness(tmp_path, monkeypatch):
    model = 'Qwen2.5-7B-Instruct'
    uuids = ['GPU-'+str(i) for i in range(8)]
    models = tmp_path/'models'
    profile_path = tmp_path/'profile.json'
    write_new(profile_path,dict(system='dynamollm',model_id=model))
    source = tmp_path/'source.json'
    write_new(source,dict(files={}))
    for name,value in dict(PDBLEND_GPU_UUIDS=','.join(uuids),PDBLEND_IMAGE_ID='image',
        PDBLEND_SOURCE_SHA256='source',PDBLEND_SOURCE_MANIFEST=str(source),
        PDBLEND_MODELS_DIR=str(models),PDBLEND_MODEL_VERIFICATION_RECEIPT='model-receipt').items():
        monkeypatch.setenv(name,value)
    trace = dict(seed=701,model_id=model,dataset='chat',duration_s=150,rate_rps=.25,
        selection_split='evaluation',slo=dict(ttft_s=1.,tpot_s=.1),
        requests=[dict(idx=71,arrival_s=0.,prompt=[10,11],max_tokens=2)])
    trace_path = tmp_path/'trace.json'
    write_new(trace_path,trace)
    config = dict(system='dynamollm',model_id=model,model_path=str(models/model),
        node_gpus=list(range(8)),profiles=str(profile_path),trace=str(trace_path),
        base_port=16000,instances=[dict(id='one',gpus=[0],tp=1,port=16000)])
    config_path = tmp_path/'config.json'
    write_new(config_path,config)
    qualifications=[]
    for gate in ('source_identity','profile_calibration','workload_coverage','mechanisms','energy_protocol'):
        p=tmp_path/(gate+'.json')
        write_new(p,dict(system='dynamollm',model_id=model,gate=gate,formal_eligible=True,
                         trace_sha256=binding(trace_path)['sha256']))
        qualifications.append(binding(p))
    identity=dict(model_hash='model',tokenizer_hash='tokenizer',image_digest='image',
        runtime_source_sha256='runtime',measurement_source_sha256='measure',
        entrypoint=module.ENTRYPOINT,worker_extension=module.WORKER_EXTENSION,dtype='bfloat16',
        fleet_gpu_uuids=uuids,environment={},instances=[dict(instance_id='one',tp=1,pp=1,
            gpu_uuids=uuids[:1],launch_options=module.dynamo_launch_options(config))])
    point=dict(name='first',system='dynamollm',model_id=model,dataset='chat',duration_s=150,
        rate_rps=.25,seed=701,slo=trace['slo'],trace=binding(trace_path),inputs=dict(
            trace=binding(trace_path),system_config=binding(config_path),profiles=[binding(profile_path)],
            qualifications=qualifications))
    group=dict(model_id=model,engine_identity=identity,engine_signature=engine_signature(identity),points=[point])
    calls=dict(starts=[],windows=[],boundaries=[],preflights=[],meter=[])

    def checked(value,**kwargs):
        calls['preflights'].append((deepcopy(value),kwargs))
        return dict(ready=not calls.get('native_reject'),missing_evidence={'native':'closed'}
                    if calls.get('native_reject') else {},evidence=dict(model_identity=model))
    monkeypatch.setattr(module,'preflight',checked)
    verified=SimpleNamespace(model_hash='model',tokenizer_hash='tokenizer',validate_config=lambda:None)
    monkeypatch.setattr(module,'ModelRegistry',lambda *a,**kw:SimpleNamespace(get=lambda _:verified))
    monkeypatch.setattr(module,'model_load_lock',nullcontext)

    class Meter:
        def __init__(self,gpus,gpu_uuids):
            calls['meter'].append((list(gpus),gpu_uuids))
        def start(self):return self
        def summarize(self,**kwargs):
            calls['metering_window']=kwargs
            return dict(energy_comparable=not calls.get('missing_power'),energy_service_j=100.,energy_tail_j=5.,
                energy_service_tail_j=105.,gpu_util_mean_pct=8.,util_coverage_fraction=1.,
                service=dict(utilization=dict(per_gpu={u:dict(mean_pct=i,peak_pct=i+10)
                    for i,u in enumerate(uuids)})))
        def snapshot(self):return dict(samples=[])
        def stop(self,**kwargs):calls['meter_stopped']=True
    monkeypatch.setattr(module,'ComparisonMeteringSession',Meter)

    class Session:
        @classmethod
        async def start(cls,value,out,**kwargs):
            calls['starts'].append((deepcopy(value),kwargs))
            self=cls();self.capabilities={'one':{'model_hash':'model','tokenizer_hash':'tokenizer'}}
            self.identity_sha256='resident'
            async def stream(iid,payload):yield dict(token_ids=list(range(16)),finished=True)
            self.transport=SimpleNamespace(stream=stream)
            return self
        async def boundary(self,**kwargs):
            calls['boundaries'].append(kwargs)
            if calls.get('dirty'):raise RuntimeError('native KV state dirty')
            return {'one':{'drained':True}}
        async def execute_window(self,rows,**kwargs):
            calls['windows'].append((deepcopy(rows),kwargs))
            started=time.time()-150.1
            out=Path(kwargs['output']);out.mkdir()
            outcome=dict(request_id='dynamo-701-0',scheduled_s=started,submitted_s=started+.01,
                first_token_s=started+.1,last_token_s=started+.15,finished_s=started+.17,
                completion_tokens=2,terminal=True,ok=True)
            if calls.get('slow'):
                outcome.update(first_token_s=started+2.,last_token_s=started+2.05,finished_s=started+2.07)
            write_new(out/'outcomes.json',[outcome])
            events=[dict(event='dynamo_sse',request_id='dynamo-701-0',at_s=outcome[k],
                payload=dict(token_ids=[i],received_s=outcome[k],finished=i==1))
                for i,k in enumerate(('first_token_s','last_token_s'))]
            (out/'events.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in events))
            return dict(service_started_s=started,complete=True,own_cleanup_complete=True,
                resident_reusable=True,resident_boundaries=dict(after={'one':{'drained':True}}))
        async def close(self):
            calls['closed']=calls.get('closed',0)+1
            if calls.get('close_failed'):raise RuntimeError('owned cleanup failed')
    monkeypatch.setattr(module,'ResidentSession',Session)
    async def no_sleep(_):pass
    monkeypatch.setattr(module.asyncio,'sleep',no_sleep)
    out=tmp_path/'result';out.mkdir()
    return module.DynamoResidentAdapter(out,base_port=18000),group,calls,tmp_path,config


def test_two_windows_one_owned_lifecycle_meter_and_actual_request_clock(harness):
    adapter,group,calls,tmp_path,config=harness
    second=deepcopy(group['points'][0]);second['name']='second'
    group['points'].append(second)
    async def run():
        startup=await adapter.start(group)
        results=[]
        for i,point in enumerate(group['points']):
            assert (await adapter.reset(point))['passed']
            result=await adapter.execute(point,tmp_path/f'window-{i}')
            results.append(result)
            assert (await adapter.drain(point))['passed']
        assert (await adapter.close())['passed']
        assert (await adapter.close())['passed']
        return startup,results
    startup,results=asyncio.run(run())
    assert startup['engine_loads']==1 and len(calls['starts'])==1
    assert len(calls['windows'])==2 and len(calls['meter'])==1 and calls['closed']==1
    assert calls['starts'][0][0]['base_port']==18000
    assert calls['starts'][0][0]['instances'][0]['port']==18000
    assert calls['starts'][0][0]['target_port']==18032
    assert config['base_port']==16000
    for result in results:
        assert result['evidence_valid'] and result['formal_eligible'] is False
        m=result['metrics']
        assert m['offered_requests']==1 and m['successful_requests']==1
        assert m['ttft_p99_s']==pytest.approx(.1,abs=1e-6)
        assert m['tpot_p99_s']==pytest.approx(.05,abs=1e-6)
        assert m['goodput_request_s']==1/150 and m['window_delivered_tokens']==2
        assert m['gpu7_util_mean_pct']==7 and m['gpu7_util_peak_pct']==17
    assert calls['meter_stopped']
    assert all(k['mode']=='comparison' and k['duration_s']==150 for _,k in calls['preflights'])


@pytest.mark.parametrize('kind',['native','qualification','inventory','lease','profile','entrypoint'])
def test_rejects_closed_gates_and_substitutions_before_hardware(harness,kind):
    adapter,group,calls,_,_=harness
    if kind=='native':calls['native_reject']=True
    elif kind=='qualification':group['points'][0]['inputs']['qualifications']=[]
    elif kind=='inventory':group['engine_identity']['instances'][0]['launch_options']['max_num_seqs']=32
    elif kind=='lease':group['engine_identity']['fleet_gpu_uuids'].reverse()
    elif kind=='profile':group['points'][0]['inputs']['profiles']=[]
    elif kind=='entrypoint':group['engine_identity']['entrypoint']='native'
    group['engine_signature']=engine_signature(group['engine_identity'])
    with pytest.raises(ValueError):asyncio.run(adapter.start(group))
    assert not calls['starts'] and not calls['meter']


def test_slo_miss_is_valid_evidence_and_missing_power_is_not(harness):
    adapter,group,calls,tmp_path,_=harness
    calls['slow']=True
    async def run():
        await adapter.start(group)
        first=await adapter.execute(group['points'][0],tmp_path/'slow')
        calls['missing_power']=True
        second=await adapter.execute(group['points'][0],tmp_path/'missing-power')
        await adapter.close()
        return first,second
    first,second=asyncio.run(run())
    assert first['evidence_valid'] and not first['metrics']['slo_pass']
    assert first['metrics']['goodput_request_s']==0
    assert not second['evidence_valid']


def test_changed_point_and_dirty_boundary_cannot_run_next_window(harness):
    adapter,group,calls,_,_=harness
    async def run():
        await adapter.start(group)
        changed=deepcopy(group['points'][0]);changed['rate_rps']=99
        with pytest.raises(ValueError,match='changed after'):
            await adapter.reset(changed)
        calls['dirty']=True
        with pytest.raises(RuntimeError,match='KV state dirty'):
            await adapter.reset(group['points'][0])
        await adapter.close()
    asyncio.run(run())
    assert calls['windows']==[]


def test_close_failure_is_explicit_and_meter_still_closes(harness):
    adapter,group,calls,_,_=harness
    async def run():
        await adapter.start(group)
        calls['close_failed']=True
        return await adapter.close()
    result=asyncio.run(run())
    assert not result['passed'] and not result['process_cleanup_verified']
    assert calls['meter_stopped']


def test_port_relocation_refuses_remote_and_out_of_lease_endpoints():
    base=dict(base_port=16000,instances=[dict(port=16001)])
    assert module.lease_config(base,18000)['instances'][0]['port']==18001
    for changed in (dict(base,store_port=17000),dict(base,target_port=16099),
                    dict(base,instances=[dict(port=16001,url='http://remote:16001')])):
        with pytest.raises(ValueError):module.lease_config(changed,18000)


def test_request_binding_refuses_ids_not_in_native_schedule():
    with pytest.raises(ValueError,match='outside'):
        module.bind_dynamo_request_indices({'requests':[dict(idx=99)]},[dict(request_id='other-701-0')],None)
