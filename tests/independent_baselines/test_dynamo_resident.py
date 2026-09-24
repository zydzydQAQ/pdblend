"""Resident boundary contracts with real journals, no CUDA/NVML or engine launch."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from pdblend_baselines.dynamollm import resident
from pdblend_baselines.dynamollm.run_v1 import execute_on_resident, load_trace, qualify
from pdblend_baselines.dynamollm.validation import preflight


@pytest.fixture
def harness(tmp_path, monkeypatch):
    calls = dict(controllers=[], starts=[], stops=[], clocks=[], preflights=[])
    model = 'Qwen2.5-7B-Instruct'
    trace_path = tmp_path/'trace.json'
    trace_path.write_text(json.dumps(dict(seed=701, requests=[dict(arrival_s=0., prompt=[100,101], max_tokens=2)])))
    config = dict(model_id=model, model_path='/models/'+model, node_gpus=[0],
        profiles='profile.json', trace=str(trace_path), dynamo_predictor_dir='predictor',
        instances=[dict(id='one', gpus=[0], tp=1, port=18000, url='http://127.0.0.1:18000',
                        shape='SS', frequency_mhz=1500)], base_port=18000)
    monkeypatch.setenv('PDBLEND_SOURCE_SHA256','source')
    monkeypatch.setenv('PDBLEND_IMAGE_ID','image')
    monkeypatch.setenv('PDBLEND_GPU_UUIDS','GPU-one')

    def checked(value, **kwargs):
        calls['preflights'].append((deepcopy(value), kwargs))
        return dict(ready=True, evidence=dict(model_identity=dict(model=model)), missing_evidence={})
    monkeypatch.setattr(resident,'preflight',checked)
    monkeypatch.setattr(resident.PaperProfiles,'load',lambda _: SimpleNamespace(frequencies=lambda _: [900,1500,2520]))

    class Telemetry:
        def __init__(self,gpus,journal):
            self.uuids={0:'GPU-one'};self.journal=journal
        def start(self):pass
        def clock(self,gpus,frequency):
            calls['clocks'].append(frequency)
            for _ in range(2):self.journal('dynamo_power',gpu=0,power_w=50.,timestamp=time.time())
            return dict(gpus=gpus,requested_frequency_mhz=frequency)
        async def close(self):calls['telemetry_closed']=True

    class Transport:
        def __init__(self,instances,clock,journal):
            self.instances={r['id']:dict(r) for r in instances}
            self.set_clock,self.journal=clock,journal
            self.accepting=True;self.dynamo_topology=None;self.dirty=False
            calls['transport']=self
        async def start(self):pass
        async def state(self,iid):
            now=time.time()
            return dict(tp=1,pp=1,generation=0,acknowledged_generation=0,
                all_queue=['leak'] if self.dirty else [],running=[],waiting=[],kv_allocations={},
                retained_kv_requests=[],transfer_allocations={},pending_transfers=0,
                free_kv_tokens=144,total_kv_tokens=144,native_evidence_complete=True,
                transport_healthy=True,healthy=True,dynamo_weights_ready=True,native_at_s=now,
                rank_observation_started_s=now,rank_observation_finished_s=now,
                ranks=[dict(rank=0,generation=0,healthy=True,native_evidence_complete=True,
                            at_s=now,pending_transfers=0,transfer_allocations={})],
                role='mixed',mode='temporal',accepting=self.accepting,admit_prefill=True,admit_decode=True)
        async def json(self,iid,path,payload=None,method='POST'):
            if path.endswith('/capability'):
                return dict(supported=True,model_id=model,model_hash='weight',tokenizer_hash='tokenizer',
                    engine_revision='vllm-0.10.1.1',source_revision='source',image_digest='image',
                    gpu_uuids=['GPU-one'],tp=1,pp=1)
            if path.endswith('/quiesce'):self.accepting=False;return dict(accepting=False)
            if path.endswith('/resume'):self.accepting=True;return dict(accepting=True)
            if path.endswith('/drain'):
                return dict(drained=True,owner_ack=True,
                            ranks=[dict(rank=0,ok=True,drained=True,generation=0,
                                        cuda_synchronized=True,active_weight_sessions=0)])
            raise AssertionError(path)
        async def clock(self,gpus,frequency):return self.set_clock(gpus,frequency)
        async def stream(self,iid,payload):
            yield dict(token_ids=list(range(payload['max_tokens'])),finished=True)
        async def close(self):calls['transport_closed']=True

    class Lifecycle:
        def __init__(self,value,transport,journal,output):
            self.instances={};self.processes={};self.transport=transport
            calls['lifecycle']=self
        async def start(self,row):
            iid=row.get('instance_id',row.get('id'))
            self.instances[iid]=dict(row);self.processes[iid]=SimpleNamespace(pid=123,returncode=None)
            calls['starts'].append(iid)
        async def close(self):
            calls['stops'].extend(self.instances)
            self.instances.clear();self.processes.clear()

    class Controller:
        def __init__(self,value,transport,journal):
            self.config=value;self.transport=transport;self.journal=journal
            self.profiles=SimpleNamespace(fingerprint='profile')
            self.predictor=object();self.requests=[];self.closed=False
            calls['controllers'].append(self)
        async def startup(self):self.epoch=time.time()
        async def handle(self,payload,rid):
            self.requests.append(rid)
            self.journal('dynamo_route',request_id=rid)
            yield dict(token_ids=[1],finished=False)
            if calls.get('fail_request'):raise RuntimeError('injected request failure')
            if calls.get('change_layout'):
                calls['lifecycle'].processes['one'].pid=456
            if calls.get('leak_kv'):self.transport.dirty=True
            yield dict(token_ids=[2],finished=True)
        async def close(self):self.closed=True
    monkeypatch.setattr(resident,'GroupTelemetry',Telemetry)
    monkeypatch.setattr(resident,'V1Transport',Transport)
    monkeypatch.setattr(resident,'SubprocessLifecycle',Lifecycle)
    monkeypatch.setattr(resident,'DynamoController',Controller)
    return config,calls,tmp_path


def test_external_boundary_is_fresh_and_quarantines_unclean_native_state(harness):
    config,calls,tmp_path=harness
    async def run():
        session=await resident.ResidentSession.start(config,tmp_path/'external',mode='functional')
        first=await session.boundary(config=config,label='reset')
        assert first['one']['ready_state']['free_kv_tokens']==144
        calls['transport'].dirty=True
        with pytest.raises(resident.ResidentReuseError,match='not clean'):
            await session.boundary(config=config,label='drain')
        assert session.closed and session.quarantined
        with pytest.raises(resident.ResidentReuseError,match='idle healthy'):
            await session.boundary(config=config)
    asyncio.run(run())
    assert calls['stops']==['one'] and calls['controllers']==[]


def test_two_windows_retain_engines_but_reset_controller_and_epoch(harness):
    config,calls,tmp=harness
    async def scenario():
        session=await resident.ResidentSession.start(config,tmp/'session',duration_s=.005)
        trace=load_trace(config['trace'],.005)
        for index in range(2):
            result=await execute_on_resident(config,trace,session=session,output=tmp/f'w{index}',
                                            duration_s=.005,mode='functional')
            assert result['status']=='passed' and result['resident_reusable']
            assert result['offered_requests']==1 and not result['formal_eligible']
            outcomes=json.loads((tmp/f'w{index}'/'outcomes.json').read_text())
            item=outcomes[0]
            assert item['scheduled_s']<=item['first_token_s']<=item['last_token_s']<=item['finished_s']
            assert item['completion_tokens']==2 and item['terminal']
        assert calls['starts']==['one'] and calls['stops']==[]
        a,b=calls['controllers']
        assert a is not b and a.predictor is not b.predictor and a.epoch<b.epoch
        assert a.closed and b.closed and a.requests==b.requests==['dynamo-701-0']
        await session.close()
        await session.close()
        assert calls['stops']==['one']
        assert calls['telemetry_closed'] and calls['transport_closed']
        assert len(calls['preflights'])==3
    asyncio.run(scenario())


@pytest.mark.parametrize('fault',['change_layout','leak_kv','fail_request'])
def test_dirty_failed_or_changed_session_is_closed_and_cannot_be_reused(harness,fault):
    config,calls,tmp=harness
    async def scenario():
        session=await resident.ResidentSession.start(config,tmp/'session',duration_s=.005)
        calls[fault]=True
        trace=load_trace(config['trace'],.005)
        result=await session.execute_window(trace,output=tmp/'window',duration_s=.005,mode='functional')
        assert result['status']=='failed' and result['independent_launch_required']
        assert not result['resident_reusable'] and session.closed and session.quarantined
        assert calls['stops']==['one']
        with pytest.raises(resident.ResidentReuseError,match='closed, quarantined'):
            await session.execute_window(trace,output=tmp/'second',duration_s=.005,mode='functional')
    asyncio.run(scenario())


def test_incompatible_engine_or_substituted_trace_rejected_before_window(harness):
    config,calls,tmp=harness
    async def scenario():
        session=await resident.ResidentSession.start(config,tmp/'session',duration_s=.005)
        trace=load_trace(config['trace'],.005)
        with pytest.raises(resident.ResidentReuseError,match='configuration changed'):
            await session.execute_window(trace,config=dict(config,max_num_seqs=32),
                                         output=tmp/'changed',duration_s=.005,mode='functional')
        with pytest.raises(resident.ResidentReuseError,match='bound window trace'):
            await session.execute_window([dict(trace[0],max_tokens=3)],
                                         output=tmp/'substituted',duration_s=.005,mode='functional')
        assert calls['controllers']==[] and not (tmp/'changed').exists()
        await session.close()
    asyncio.run(scenario())


def test_comparison_rechecks_full_preflight_and_never_uses_caller_pass(harness,monkeypatch):
    config,calls,tmp=harness
    async def scenario():
        session=await resident.ResidentSession.start(config,tmp/'session',duration_s=.005)
        def deny(value,**kwargs):
            assert value['dynamo_require_full_mechanisms'] is True
            return dict(ready=False,evidence={},missing_evidence={'missing_original_cycle_mechanisms':'absent'})
        monkeypatch.setattr(resident,'preflight',deny)
        with pytest.raises(resident.ResidentReuseError,match='original_cycle'):
            await session.execute_window(load_trace(config['trace'],150),output=tmp/'window',
                duration_s=150,mode='comparison',receipt=dict(ready=True,evidence={}))
        assert calls['controllers']==[]
        await session.close()
    asyncio.run(scenario())


def test_window_refusal_does_not_hide_planned_requests(harness):
    config,calls,tmp=harness
    async def scenario():
        session=await resident.ResidentSession.start(config,tmp/'session',duration_s=.005)
        calls['transport'].dirty=True
        result=await session.execute_window(load_trace(config['trace'],.005),
            output=tmp/'window',duration_s=.005,mode='functional')
        outcomes=json.loads((tmp/'window'/'outcomes.json').read_text())
        assert result['offered_requests']==len(outcomes)==1
        assert outcomes[0]['submitted_s'] is None and outcomes[0]['ok'] is False
        assert outcomes[0]['error']=='window_failed_before_submission'
        assert calls['controllers']==[] and session.closed
    asyncio.run(scenario())


def test_150_second_comparison_keeps_original_cycle_gate():
    rows=[dict(event='dynamo_route',request_id='r')]
    outcomes=[dict(request_id='r',ok=True)]
    result=qualify(rows,outcomes,mode='comparison',duration_s=150)
    assert result['status']=='passed' and result['periods_s']==dict(ScaleInst=1800.,ScaleShard=300.,ScaleFreq=5.)
    assert result['formal_eligible'] is False
    incomplete=preflight(dict(model_id='Qwen2.5-7B-Instruct'),mode='comparison',duration_s=150)
    assert 'duration' not in incomplete['missing_evidence']
    assert 'missing_original_cycle_mechanisms' in incomplete['missing_evidence']
    assert not incomplete['ready']
    full=qualify(rows,outcomes,mode='full',duration_s=150)
    assert 'original_period_window_missing' in full['failures']
