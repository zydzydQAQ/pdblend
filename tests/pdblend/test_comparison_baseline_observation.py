import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext

import pytest

from pdblend.bench import comparison_baseline_observation as obs
from pdblend.bench.comparison_campaign import binding
from pdblend.bench.comparison_dynamo_runtime import dynamo_launch_options, ENTRYPOINT, WORKER_EXTENSION
from pdblend.bench.resident_session import digest, write_new, engine_signature
from pdblend_baselines.dynamollm import resident, validation
from test_comparison_metering import evidence


def prepared(tmp_path, *, system='dynamollm'):
    model='Qwen2.5-7B-Instruct'; uuids=[f'GPU-{i}' for i in range(8)]
    trace=dict(seed=701,duration_s=150,model_id=model,dataset='alpaca',rate_rps=.1,
        slo=dict(ttft_s=1.,tpot_s=.1),selection_split='evaluation',
        requests=[dict(idx=0,arrival_s=0.,prompt=[10,11],max_tokens=2)])
    write_new(tmp_path/'trace.json',trace)
    write_new(tmp_path/'profile.json',dict(system=system,model=model,formal_eligible=False))
    profiles=[binding(tmp_path/'profile.json')]
    config=dict(system=system,model_id=model,model_path='/models/'+model,observation_scope=obs.SCOPE,
        formal_eligible=False,mode='comparison',dynamo_require_full_mechanisms=False,
        profiles=profiles[0]['path'],trace=str(tmp_path/'trace.json'),node_gpus=list(range(8)),
        periods_s=dict(ScaleInst=1800.,ScaleShard=300.,ScaleFreq=5.),base_port=16000,
        slo_ttft_s=1.,slo_tpot_s=.1,
        instances=[dict(id=f'dynamo-{i}',gpus=[i],tp=1,pp=1,port=16000+2*i) for i in range(8)])
    identity=dict(model_hash='model',tokenizer_hash='tokenizer',image_digest='image',
        runtime_source_sha256='runtime',measurement_source_sha256='measurement',
        dtype='bfloat16',entrypoint=ENTRYPOINT,worker_extension=WORKER_EXTENSION,
        fleet_gpu_uuids=uuids,environment={},instances=[dict(instance_id=r['id'],tp=1,pp=1,
            gpu_uuids=[uuids[i]],launch_options=dynamo_launch_options(config)) for i,r in enumerate(config['instances'])])
    inputs=dict(trace=binding(tmp_path/'trace.json'),profiles=profiles)
    if system=='distserve':
        from pdblend_runtime.probe import NativeSpec
        from dataclasses import asdict
        config.update(max_batch_size=32,request_timeout_s=240.)
        identity.update(entrypoint='pdblend_runtime.serve',worker_extension='native_v1')
        for i,row in enumerate(identity['instances']):
            row['instance_id']=f'dist-{i//2}-'+('P' if i%2==0 else 'D')
            spec=NativeSpec(row['instance_id'],(i,),17000+16*i,'/models/'+model,max_num_seqs=32)
            row['launch_options']={k:asdict(spec)[k] for k in ('max_model_len','max_num_seqs',
                'max_num_batched_tokens','gpu_memory_utilization','kv_connector')}
        choice=dict(system=system,model_id=model,status='ready_for_native_execution',selection_split='calibration',
            evaluation_used_for_selection=False,selection_used_dataset_requests=False,
            selection='predeclared_fixed_native_topology_no_offline_search',offline_topology_search_performed=False,
            formal_eligible=False,selected=dict(config=[1,1,1,1,1],tp=1,pp=1,replicas=4,total_gpu_count=8),
            profiles=profiles,trace=inputs['trace'],rate_rps=.1,slo=trace['slo'],frequency_mhz=2520,gpu_budget=8,
            identity={k:identity[k] for k in ('model_hash','tokenizer_hash','image_digest')})
        write_new(tmp_path/'choice.json',choice);inputs['offline_choice']=binding(tmp_path/'choice.json')
    write_new(tmp_path/'config.json',config);inputs['system_config']=binding(tmp_path/'config.json')
    point=dict(name='point',system=system,model_id=model,dataset='alpaca',rate_rps=.1,slo=trace['slo'],seed=701,
        duration_s=150,trace=inputs['trace'],inputs=inputs,engine_identity=identity,
        observation_scope=obs.SCOPE,qualification_mode=obs.SCOPE,result_policy=obs.RESULT_POLICY)
    return point,identity,config


@pytest.mark.parametrize('system',['distserve','dynamollm'])
def test_input_validation_keeps_profiles_original(tmp_path,system):
    point,identity,_=prepared(tmp_path,system=system);path=tmp_path/'profile.json';before=path.read_bytes()
    checked=obs.validate_observation_inputs(point,identity)
    assert not checked['formal_eligible'] and not checked['profile_qualified']
    assert checked['profile_missing_gates'] and path.read_bytes()==before


def test_legacy_own_model_path_is_normalized_without_rewriting_profile(tmp_path):
    point,identity,_=prepared(tmp_path)
    path=tmp_path/'profile.json'
    path.write_text(json.dumps(dict(system='dynamollm',model='/models/'+point['model_id'])))
    point['inputs']['profiles']=[binding(path)];before=path.read_bytes()
    assert not obs.validate_observation_inputs(point,identity)['profile_qualified']
    assert path.read_bytes()==before


@pytest.mark.parametrize('fault',['trace_bytes','other_model','period','scope','fleet','profile_system'])
def test_observation_never_weakens_source_model_trace_identity(tmp_path,fault):
    point,identity,cfg=prepared(tmp_path)
    if fault=='trace_bytes':(tmp_path/'trace.json').write_text('{}')
    elif fault=='other_model':point['model_id']='Qwen2.5-14B-Instruct'
    elif fault=='scope':point['qualification_mode']='formal'
    elif fault=='fleet':identity['instances'].pop()
    elif fault=='period':
        cfg['periods_s']['ScaleShard']=1.;(tmp_path/'config.json').write_text(json.dumps(cfg))
        point['inputs']['system_config']=binding(tmp_path/'config.json')
    else:
        (tmp_path/'profile.json').write_text(json.dumps(dict(system='pdblend',model=point['model_id'])))
        point['inputs']['profiles']=[binding(tmp_path/'profile.json')]
    with pytest.raises(ValueError):obs.validate_observation_inputs(point,identity)


def test_private_resident_preserves_original_lifecycle_and_reports_real_failed_qualification(tmp_path,monkeypatch):
    _,_,cfg=prepared(tmp_path);calls=[]
    def preflight(value,**kw):
        calls.append(kw)
        return dict(ready=kw['mode']=='functional',mode=kw['mode'],evidence={'model_identity':'real-asset'},
            missing_evidence={} if kw['mode']=='functional' else {'missing_original_cycle_mechanisms':'not qualified'})
    monkeypatch.setattr(validation,'preflight',preflight)
    private=obs.dynamo_resident_implementation()
    assert resident._config(cfg,'comparison')['dynamo_require_full_mechanisms'] is True
    assert private._config(cfg,'comparison')['dynamo_require_full_mechanisms'] is False
    original_base=private.ResidentSession.__mro__[1]
    for name in ('execute_window','_boundary','boundary','_close','close'):
        assert getattr(original_base,name).__code__.co_code==getattr(resident.ResidentSession,name).__code__.co_code
    assert private.ResidentSession.start.__func__.__code__.co_code==resident.ResidentSession.start.__func__.__code__.co_code
    checked=private.preflight(cfg,mode='comparison',duration_s=150,seed=701)
    assert checked['ready'] and checked['readiness_scope']=='functional_assets_only'
    assert checked['comparison_qualification']['ready'] is False and checked['mode']=='functional'
    assert checked['execution_mode']=='comparison' and not checked['formal_eligible']
    assert [r['mode'] for r in calls]==['functional','comparison']
    assert resident.preflight is not private.preflight


def test_real_asset_rejection_is_not_made_ready(tmp_path,monkeypatch):
    _,_,cfg=prepared(tmp_path)
    monkeypatch.setattr(validation,'preflight',lambda *a,**kw:dict(ready=False,evidence={},missing_evidence={'model':'wrong'}))
    private=obs.dynamo_resident_implementation()
    assert not private.preflight(cfg,mode='comparison',duration_s=150,seed=701)['ready']
    with pytest.raises(ValueError):private.preflight(cfg,mode='functional',duration_s=150,seed=701)


def test_dynamo_adapter_keeps_one_meter_and_one_loaded_session(tmp_path,monkeypatch):
    point,identity,cfg=prepared(tmp_path);calls=[]
    from pdblend.bench import comparison_dynamo_runtime as owner
    original_session=owner.ResidentSession
    class Session:
        @classmethod
        async def start(cls,value,out,**kw):
            calls.append(('start',value,kw));s=cls();s.identity_sha256='session'
            s.capabilities={r['instance_id']:dict(model_hash='model',tokenizer_hash='tokenizer') for r in identity['instances']}
            async def stream(iid,request):yield dict(token_ids=list(range(16)),finished=True)
            s.transport=SimpleNamespace(stream=stream);return s
        async def boundary(self,**kwargs):calls.append(('boundary',kwargs));return {'real_cpu_contract':True}
        async def close(self):calls.append(('close',))
    impl=SimpleNamespace(ResidentSession=Session,_config=lambda c,m:c,
        preflight=lambda *a,**kw:dict(ready=True,evidence={},missing_evidence={}))
    monkeypatch.setattr(obs,'dynamo_resident_implementation',lambda:impl)
    adapter=obs.make_dynamo_adapter(tmp_path/'adapter',base_port=19000)
    source=tmp_path/'source.json';write_new(source,dict(files={}))
    for k,v in dict(PDBLEND_GPU_UUIDS=','.join(identity['fleet_gpu_uuids']),PDBLEND_IMAGE_ID='image',
        PDBLEND_SOURCE_MANIFEST=str(source),PDBLEND_MODELS_DIR='/models',
        PDBLEND_MODEL_VERIFICATION_RECEIPT='already-bound-model-receipt').items():monkeypatch.setenv(k,v)
    namespace=adapter.start.__func__.__globals__
    monkeypatch.setitem(namespace,'ModelRegistry',lambda *a,**kw:SimpleNamespace(get=lambda _:SimpleNamespace(
        model_hash='model',tokenizer_hash='tokenizer',validate_config=lambda:None)))
    monkeypatch.setitem(namespace,'model_load_lock',nullcontext)
    class Meter:
        def __init__(self,gpus,uuids):calls.append(('meter',list(gpus),uuids))
        def start(self):return self
        def stop(self,**kwargs):calls.append(('meter_stop',))
        def snapshot(self):return dict(samples=[])
    monkeypatch.setitem(namespace,'ComparisonMeteringSession',Meter)
    group=dict(engine_identity=identity,engine_signature=engine_signature(identity),model_id=point['model_id'],points=[point])
    async def run():
        start=await adapter.start(group)
        for _ in range(2):
            assert (await adapter.reset(point))['passed']
            assert (await adapter.drain(point))['passed']
        first=await adapter.close();assert await adapter.close() is first
        return start
    result=asyncio.run(run())
    assert result['engine_loads']==8 and result['engine_load_cycles']==1
    assert sum(r[0]=='start' for r in calls)==sum(r[0]=='meter' for r in calls)==1
    assert sum(r[0]=='close' for r in calls)==sum(r[0]=='meter_stop' for r in calls)==1
    assert next(r for r in calls if r[0]=='start')[2]==dict(mode='comparison',duration_s=150)
    assert owner.ResidentSession is original_session


def test_dynamo_runner_reuses_original_session_and_comparison_hooks(tmp_path):
    point,identity,_=prepared(tmp_path);calls=[]
    class Session:
        async def execute_window(self,trace,**kwargs):calls.append((trace,kwargs));return dict(actual=True)
    session=Session()
    for name in ('one','two'):
        result=asyncio.run(obs.execute_native_observation(point,identity,out=tmp_path/name,session=session,base_port=19000))
        assert result==dict(actual=True)
    assert len(calls)==2
    for trace,kw in calls:
        assert kw['mode']=='comparison' and kw['duration_s']==150 and kw['config']['instances'][0]['port']==19000
        assert trace[0]['prompt']==[10,11] and kw['config']['dynamo_require_full_mechanisms'] is False


def test_distserve_uses_original_multipair_fcfs_runtime(tmp_path,monkeypatch):
    point,identity,_=prepared(tmp_path,system='distserve');seen={}
    from pdblend_baselines.distserve import deployment
    async def run(*args,**kw):seen.update(args=args,kw=kw);return {'actual':True}
    monkeypatch.setattr(deployment,'execute_on_resident',run)
    specs=['real-caller-owned-specs']
    result=asyncio.run(obs.execute_native_observation(point,identity,out=tmp_path/'run',specs=specs))
    assert result['actual'] and seen['args'][1] is specs and seen['args'][4]==150
    assert seen['args'][0]['selected']['replicas']==4 and seen['kw']=={'request_timeout':240.}


def window(tmp_path,*,failed=False,gap=False):
    point,identity,_=prepared(tmp_path);out=tmp_path/'run';out.mkdir()
    events=[dict(event='dynamo_service_window_start',at_s=100.)]
    if not failed:
        events.extend(dict(event='dynamo_sse',request_id='dynamo-701-0',at_s=t,
            payload=dict(token_ids=[token],token_index=i+1,finished=i==1))
            for i,(t,token) in enumerate([(100.1,10),(100.15,11)]))
    (out/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
    outcome=dict(request_id='dynamo-701-0',submitted_s=100.,finished_s=100.2,completion_tokens=0 if failed else 2,
                 ok=not failed,terminal=not failed)
    if failed:outcome['error']='actual request failure'
    write_new(out/'outcomes.json',[outcome])
    raw=dict(system='dynamollm',trace_sha256=point['trace']['sha256'],events_sha256=binding(out/'events.jsonl')['sha256'],
        status='failed' if failed else 'passed',service_started_s=100.,requests_done_s=250.,own_cleanup_complete=True,
        resident_reusable=not failed,cleanup_errors=[])
    times=[99.5+i*.5 for i in range(303)]
    if gap:times=[t for t in times if not 120<=t<=123]
    snapshot=evidence(times);snapshot.update(gpu_uuids=identity['fleet_gpu_uuids'],gpu_uuid_binding_verified=True)
    return point,identity,out,raw,snapshot


@pytest.mark.parametrize('failed,gap',[(False,False),(True,False),(False,True),(True,True)])
def test_all_completed_observations_retained_without_qualification(tmp_path,failed,gap):
    point,identity,out,raw,snapshot=window(tmp_path,failed=failed,gap=gap)
    result=obs.finalize_observation(point,identity,out=out,native_result=raw,snapshot=snapshot,tail_end_s=250.,
                                    startup={},reset={'passed':True},drain={'passed':True})
    assert result['measurement_evidence_valid'],result['observation_acceptance']
    assert obs.valid_observation_result(point,result)
    assert not result['evidence_valid'] and not result['formal_eligible'] and not result['profile_qualified']
    assert result['metrics']['slo_pass'] is (not failed)
    assert result['native_status']==('failed' if failed else 'passed')
    if gap:
        assert result['metrics']['energy_service_j'] is None
        assert 'metering.coverage' in result['observation_acceptance']['diagnostic_failures']
    else:assert result['metrics']['energy_service_j']==pytest.approx(120000.)


def test_failure_before_service_records_no_invented_window(tmp_path):
    point,identity,out,raw,snapshot=window(tmp_path)
    raw.pop('service_started_s');raw.update(status='failed',error='actual engine startup failure')
    (out/'events.jsonl').write_text('');raw['events_sha256']=binding(out/'events.jsonl')['sha256']
    result=obs.finalize_observation(point,identity,out=out,native_result=raw,snapshot=snapshot,tail_end_s=250.,
                                    startup={},reset={'passed':False},drain={'passed':False})
    assert not result['measurement_evidence_valid'] and result['metrics']=={}
    assert result['native_error']=='actual engine startup failure'


@pytest.mark.parametrize('fault',[None,'short','cleanup','boundary','closed'])
def test_failed_request_reuse_requires_original_successful_boundary(tmp_path,fault):
    session=SimpleNamespace(closed=False,quarantined=True)
    raw=dict(service_started_s=100.,requests_done_s=250.,own_cleanup_complete=True,
        cleanup_errors=[],resident_reusable=False,status='failed',resident_boundaries={'after':{'actual':'native drain'}})
    if fault=='short':raw['requests_done_s']=249.
    elif fault=='cleanup':raw['cleanup_errors']=['actual native failure']
    elif fault=='boundary':raw['resident_boundaries']={}
    elif fault=='closed':session.closed=True
    original=deepcopy(raw);receipt=obs.restore_observational_reuse(session,raw)
    assert raw==original and receipt['original_native_result_sha256']==digest(original)
    assert receipt['reuse_permitted'] is (fault is None)
    assert session.quarantined is (fault is not None)


@pytest.mark.parametrize('fault',[None,'cleanup','boundary','short','serialization','close_error'])
def test_result_policy_defers_qualification_close_but_never_unsafe_cleanup(tmp_path,fault):
    class Original:
        def __init__(self):
            self.closed=False;self.quarantined=False;self.close_calls=0
            self.raw=dict(service_started_s=100.,requests_done_s=250.,own_cleanup_complete=True,
                cleanup_errors=[],resident_reusable=False,status='failed',
                resident_boundaries={'after':{'actual':'native all-rank drain'}})
            if fault in ('cleanup','close_error'):self.raw['cleanup_errors']=['native failure']
            elif fault=='boundary':self.raw['resident_boundaries']={}
            elif fault=='short':self.raw['requests_done_s']=249.
        async def execute_window(self,trace,**kwargs):
            # Original result policy closes before returning; subclass must
            # defer this call until it can inspect the actual result safely.
            self.quarantined=True
            await self._close()
            assert self.close_calls==0
            if fault=='serialization':raise RuntimeError('actual serialization failure')
            return self.raw
        async def _close(self):
            self.close_calls+=1;self.closed=True
            if fault=='close_error':raise RuntimeError('actual close failure')
    session=obs._observational_session_class(Original)();original=deepcopy(session.raw)
    async def run():return await session.execute_window([],output=tmp_path)
    if fault in ('serialization','close_error'):
        with pytest.raises(RuntimeError):asyncio.run(run())
    else:assert asyncio.run(run())==original
    assert session.raw==original
    assert session.close_calls==(0 if fault is None else 1)
    assert session.closed is (fault is not None)
    assert session.quarantined is (fault is not None)
    if fault=='close_error':
        receipt=json.loads((tmp_path/'observation-resident-cleanup.json').read_text())
        assert 'actual close failure' in receipt['close_error']
