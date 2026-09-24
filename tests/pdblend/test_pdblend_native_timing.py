import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from pdblend.profile.collection.native_timing_plan import build_plan,binding,validate_plan
from pdblend.profile.collection.native_timing_audit import measured_events,audit_window,fit_component
from pdblend.profile.collection.native_timing_collect import burst,measurement_barrier,gather_owned
from test_comparison_acceptance import state


def plan_fixture(tmp_path):
    ledger=dict(schema='pdblend-offline-query-ledger/v2',evaluation_read=False,min_m_floor_overridden=False,
        ledgers=[dict(model_id='Qwen2.5-7B-Instruct',selection_split='tuning',tp=1,pp=1,min_m_instances=4,
                      frequency_scope=[1500,2520])])
    p=tmp_path/'ledger.json';p.write_text(json.dumps(ledger));b=tmp_path/'bindings.json'
    b.write_text(json.dumps(dict(outputs={'query-ledger.json':binding(p)})))
    return build_plan(binding(p),binding(b))


def test_plan_covers_boundaries_without_evaluation_or_promoting_power(tmp_path):
    p=validate_plan(plan_fixture(tmp_path));assert len(p['points'])==118
    for f in (1500,2520):
        train=[r for r in p['points'] if r['frequency_mhz']==f and r['role']=='decode' and r['purpose']=='training']
        assert {r['batch'] for r in train}=={1,2,4,8,16,24,32}
        assert {r['prompt_tokens'] for r in train}=={64,256,2048,4096,7424}
        assert all(r['repeats']==3 and r['seed']==9701 for r in train)
    assert not p['formal_eligible'] and 'not_pure_decode' in p['power_scope']


def test_modified_holdout_or_evaluation_plan_is_rejected(tmp_path):
    p=plan_fixture(tmp_path);p['points'][-1]['purpose']='training'
    with pytest.raises(ValueError):validate_plan(p)
    p=plan_fixture(tmp_path);p['evaluation_used_for_selection']=True
    with pytest.raises(ValueError):validate_plan(p)


def event(*,rank=0,batch=2,context=129,role='decode',at=104.):
    return dict(system='pdblend',measurement_scope='runner',rank=rank,tp_rank=rank,pp_rank=0,tp=1,pp=1,
        role=role,batch=batch,prompt_lengths=[128]*batch,context_lengths=[context]*batch,
        scheduled_lengths=[1]*batch,request_ids=['r'+str(i) for i in range(batch)],
        gpu_elapsed_ms=2.,at_s=at)


def raw_fixture():
    ident=dict(model_id='Qwen2.5-7B-Instruct',model_hash='model',tokenizer_hash='tokenizer',engine_revision='vllm-0.10.1.1',
               source_revision='source',image_digest='image',tp=1,pp=1,gpu_uuids=['GPU-a'])
    point=dict(role='decode',batch=2,prompt_tokens=128,output_tokens=64,frequency_mhz=1500,purpose='training')
    drained=dict(state(1,110.,5),acknowledged=True,drained=True)
    return ident,dict(schema='pdblend-native-timing-window-v1',system='pdblend',status='measured',capability=ident,
        point=point,window_id='w',start_s=103.,end_s=108.,measurement_started_s=100.,settle_started_s=100.,settle_finished_s=102.,
        measurement_start=dict(acknowledged=True),measurement_stop=dict(ranks=[dict(rank=0,acknowledged=True)]),
        cleanup_errors=[],sampler_error=None,drain=drained,
        clock_receipt=dict(acknowledged=True,success=True,requested_frequency_mhz=1500,gpus=[dict(gpu_uuid='GPU-a')]),
        frequency_samples=[[103+i*.5,[1500]] for i in range(11)],
        client_requests=[dict(request_id='r'+str(i),terminal=True,completion_tokens=64,submitted_s=103.,finished_s=108.) for i in range(2)],
        sample=dict(ranks=[dict(rank=0,samples=[event(at=104+i*.1) for i in range(8)])]))


def test_native_window_retains_only_actual_service_shapes():
    identity,raw=raw_fixture();raw['sample']['ranks'][0]['samples'] += [event(at=102.),event(at=109.),event(batch=1)]
    rows=audit_window(raw,identity=identity)
    assert len(rows)==8 and all(r['batch']==2 for r in rows)


@pytest.mark.parametrize('change',[
    lambda r:r['sample']['ranks'][0]['samples'][0].update(system='distserve'),
    lambda r:r['sample']['ranks'][0]['samples'][0].update(gpu_elapsed_ms=0),
    lambda r:r['sample']['ranks'][0]['samples'][0].update(scheduled_lengths=[2,1]),
    lambda r:r['sample']['ranks'][0]['samples'][0].update(context_lengths=[129,130]),
    lambda r:r.update(frequency_samples=[[103.,[1500]],[108.,[1500]]]),
    lambda r:r['measurement_stop'].update(ranks=[]),
    lambda r:r['client_requests'][0].update(terminal=False),
    lambda r:r['clock_receipt']['gpus'][0].update(gpu_uuid='GPU-other'),
])
def test_incomplete_or_different_native_evidence_is_rejected(change):
    identity,raw=raw_fixture();change(raw)
    with pytest.raises(ValueError):audit_window(raw,identity=identity)


def test_rank_alignment_cannot_be_replaced_by_independent_maxima():
    a=event();b=event(rank=1);a['tp']=b['tp']=2;b['request_ids'].reverse()
    with pytest.raises(ValueError):measured_events(dict(ranks=[dict(rank=0,samples=[a]),dict(rank=1,samples=[b])]),tp=2)


def test_prefill_chunks_never_become_complete_prefill_samples():
    e=event(role='prefill');assert measured_events(dict(ranks=[dict(rank=0,samples=[e])]))==[]


def component_rows():
    train=[];hold=[]
    for frequency in (1500,2520):
        for role in ('prefill','decode'):
            for target,shapes in [(train,[(1,64),(1,4096),(32,64),(32,4096)] if role=='decode' else [(1,16),(1,512),(1,4096)]),
                                  (hold,[(8,1024)] if role=='decode' else [(1,1024)])]:
                purpose='t' if target is train else 'h'
                for b,c in shapes:
                    target.append(dict(frequency_mhz=frequency,role=role,batch=b,context_tokens=c,prompt_tokens=c,
                        latency_ms=1+b+b*c/8192 if role=='decode' else 1+c/8192+(c/8192)**2,
                        window_id=f'{purpose}-{frequency}-{role}-{b}-{c}',point_sha256=f'{purpose}-{frequency}-{role}-{b}-{c}'))
    return train,hold


def test_timing_component_qualification_does_not_grant_full_profile():
    train,hold=component_rows()
    result=fit_component(train,hold,identity={'system':'pdblend'},raw_bindings=[],
        measurement_qualification={'qualified':True},limits=dict(mean_relative_error=.1,p95_relative_error=.2,max_relative_error=.25))
    assert result['component_qualified'] and not result['formal_eligible'] and not result['energy_comparable']


def test_holdout_is_never_fit_or_promoted_when_outside_actual_hull():
    train,hold=component_rows();hold[0].update(context_tokens=7168,prompt_tokens=7168)
    result=fit_component(train,hold,identity={'system':'pdblend'},raw_bindings=[],
        measurement_qualification={'qualified':True},limits=dict(mean_relative_error=.1,p95_relative_error=.2,max_relative_error=.25))
    assert not result['component_qualified']
    hold[0]['window_id']=train[0]['window_id']
    with pytest.raises(ValueError):fit_component(train,hold,identity={},raw_bindings=[],measurement_qualification={},limits={})


def test_python310_barrier_releases_every_member_and_failure_cancels_owned_peers():
    async def run():
        barrier=measurement_barrier(3);seen=[]
        async def member(i):await barrier();seen.append(i)
        await gather_owned(member(i) for i in range(3));assert sorted(seen)==[0,1,2]
        stopped=asyncio.Event()
        async def wait():
            try:await asyncio.Event().wait()
            finally:stopped.set()
        async def fail():await asyncio.sleep(0);raise ValueError('probe failed')
        with pytest.raises(ValueError):await gather_owned([wait(),fail()])
        assert stopped.is_set()
    asyncio.run(run())


def test_decode_admission_barrier_prevents_context_skew(monkeypatch):
    from pdblend.profile.collection import native_timing_collect as module
    from pdblend_runtime import probe
    controls=[];queued=set();flags=dict(admit_prefill=False,admit_decode=False)
    async def call(session,url,path,body=None):
        if path.endswith('/control'):controls.append(dict(body));flags.update(body);return {'acknowledged':True}
        if path.endswith('/state'):return {'all_queue':list(queued)}
        return {'acknowledged':True}
    async def request(session,url,payload,row):
        queued.add(payload['request_id'])
        while not flags['admit_prefill']:await asyncio.sleep(0)
        row['seen_tokens']=1
        while not flags['admit_decode']:await asyncio.sleep(0)
        row.update(terminal=True,completion_tokens=64)
    monkeypatch.setattr(probe,'call',call);monkeypatch.setattr(module,'_request',request)
    point=dict(seed=9701,prompt_tokens=128,batch=4,output_tokens=64);clients=[]
    asyncio.run(burst(None,'http://local',point,'r',clients))
    assert len(clients)==4 and controls[0]['admit_prefill'] is False
    assert controls[1]==dict(admit_prefill=True,admit_decode=False)
    assert controls[2]==dict(admit_prefill=True,admit_decode=True)
