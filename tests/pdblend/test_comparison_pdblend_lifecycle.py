import asyncio
from contextlib import nullcontext
from dataclasses import replace
import time
from types import SimpleNamespace

import pytest

from pdblend.bench import comparison_pdblend_lifecycle as module
from pdblend_runtime.probe import NativeSpec


def state(generation=0, **values):
    return dict(generation=generation, tp=1, pp=1, native_evidence_complete=True,
        transport_healthy=True, native_at_s=time.time(), all_queue=[], running=[], waiting=[],
        retained_kv_requests=[], pending_transfers=0, transfer_allocations={}, kv_allocations={},
        total_blocks=10, free_blocks=10, reserved_blocks=0,
        ranks=[dict(rank=0,generation=generation,native_evidence_complete=True,healthy=True,
                    pending_transfers=0,transfer_allocations={})], **values)


class Instance:
    def __init__(self, spec):
        self.spec=spec;self.process=SimpleNamespace(pid=100+spec.gpus[0]);self.state='ready'
        self.events=[dict(kind='start',t_s=1.,pid=self.process.pid,instance=spec.instance_id)]
        self.starts=1;self.waits=0
    def alive(self):return self.process is not None
    def start(self):
        if self.alive():return
        self.starts+=1;self.process=SimpleNamespace(pid=100*self.starts+self.spec.gpus[0]);self.state='starting'
        self.events.append(dict(kind='start',t_s=time.time(),pid=self.process.pid,instance=self.spec.instance_id))
    def stop(self):
        self.events.append(dict(kind='stop',t_s=time.time(),returncode=0,instance=self.spec.instance_id))
        self.process=None;self.state='off'
    def wait_ready(self,*args,**kwargs):self.waits+=1;self.state='ready';return .1


class Fleet:
    def __init__(self,specs):self.instances={s.instance_id:Instance(s) for s in specs}
    def __getitem__(self,iid):return self.instances[iid]


@pytest.fixture
def setup(monkeypatch):
    specs=[NativeSpec(f'p{i}',(i,),19000+4*i,'/models/Qwen2.5-7B-Instruct',generation=7) for i in range(2)]
    uuids=[f'GPU-{i}' for i in range(8)];calls=[];processes={i:[] for i in range(8)}
    nvml=SimpleNamespace(nvmlDeviceGetUUID=lambda g:uuids[g],
        nvmlDeviceGetComputeRunningProcesses=lambda g:[SimpleNamespace(pid=p) for p in processes[g]])
    gpus=SimpleNamespace(backend=SimpleNamespace(_nvml=nvml,_handle=lambda g:g),
        unpark=lambda g:calls.append(('unpark',g)),set_clock=lambda g,f:calls.append(('clock',g,f)))
    identity=dict(fleet_gpu_uuids=uuids,model_hash='weights',tokenizer_hash='tokenizer',image_digest='image')
    adapter=SimpleNamespace(specs=specs,fleet=Fleet(specs),gpus=gpus,identity=identity,load_count=2)
    monkeypatch.setenv('PDBLEND_GPU_UUIDS',','.join(uuids));monkeypatch.setenv('PDBLEND_SOURCE_SHA256','source')
    async def drain(specs):
        assert all(adapter.fleet[s.instance_id].alive() for s in specs)
        calls.append(('drain',[s.instance_id for s in specs]))
        return [dict(instance_id=s.instance_id,drain=dict(drained=True),state=state(s.generation)) for s in specs]
    async def caps(specs):
        assert all(adapter.fleet[s.instance_id].alive() for s in specs)
        calls.append(('verify',[s.instance_id for s in specs]));return {s.instance_id:{} for s in specs}
    monkeypatch.setattr(module,'drain_endpoints',drain);monkeypatch.setattr(module,'verify_endpoints',caps)
    monkeypatch.setattr(module,'model_load_lock',nullcontext)
    native={s.instance_id:state() for s in specs};http=[]
    def query(spec,endpoint,payload=None):
        http.append((spec.instance_id,endpoint,payload))
        if endpoint=='capability':return dict(supported=True,tp=1,pp=1,model_id='Qwen2.5-7B-Instruct',
            source_revision='source',gpu_uuids=[uuids[g] for g in spec.gpus],**{k:identity[k]
                for k in ('model_hash','tokenizer_hash','image_digest')})
        if endpoint=='control':
            native[spec.instance_id]=state(**payload)
            return dict(acknowledged=True,generation=payload['generation'])
        return dict(native[spec.instance_id],native_at_s=time.time())
    monkeypatch.setattr(module,'_json',query)
    return adapter,calls,processes,http,native


def test_finish_drains_only_live_and_preserves_off_and_parked_clocks(setup):
    adapter,calls,processes,http,native=setup;boundary=module.PDblendResidentBoundary(adapter)
    boundary.begin_window();adapter.fleet['p1'].stop()
    result=asyncio.run(boundary.finish_window(dict(final_roles={'p0':'L1','p1':'off'})))
    assert calls==[('drain',['p0'])] and not http
    assert adapter.fleet['p1'].state=='off' and adapter.fleet['p1'].starts==1
    assert result['policy_off_preserved'] and result['live_instance_ids']==['p0']
    assert result['off_instances'][0]['physical_gpus']==[dict(local_index=1,uuid='GPU-1',compute_pids=[])]
    assert result['off_instances'][0]['owned_stop']['kind']=='stop'
    assert result['window_engine_loads']==0 and result['engine_load_accounting']['engine_loads']==2


def test_next_reset_restores_missing_inventory_and_native_epoch_before_warmup(setup):
    adapter,calls,processes,http,native=setup;boundary=module.PDblendResidentBoundary(adapter)
    boundary.begin_window();adapter.fleet['p1'].stop()
    asyncio.run(boundary.finish_window(dict(final_roles={'p0':'L1','p1':'off'})))
    calls.clear();restored=asyncio.run(boundary.restore())
    assert calls[0]==('drain',['p0']) and calls[-1]==('verify',['p0','p1'])
    assert restored['restored_instances']==['p1'] and restored['engine_loads']==1
    assert restored['cumulative_engine_loads']==3 and not restored['allocated_to_previous_service']
    assert adapter.fleet['p0'].starts==1 and adapter.fleet['p1'].starts==2
    assert native['p1']['generation']==7 and native['p1']['accepting'] is False
    assert len(boundary.restart_epochs)==1
    before=len(http);adapter.fleet['p1'].wait_ready()
    assert len(http)==before, 'readiness retry must not reset active admission'


def test_policy_owned_restart_is_counted_and_preserves_its_epoch(setup):
    adapter,calls,processes,http,native=setup;boundary=module.PDblendResidentBoundary(adapter)
    boundary.begin_window();adapter.fleet['p1'].stop()
    # This is the actual controller off->wake sequence before NativeControl.resume.
    adapter.fleet['p1'].start();adapter.fleet['p1'].wait_ready()
    result=asyncio.run(boundary.finish_window(dict(final_roles={'p0':'M','p1':'M'})))
    assert result['window_engine_loads']==1 and adapter.load_count==3
    assert result['restart_epoch_receipts'][0]['state']['generation']==7
    assert result['off_instances']==[]


def test_off_gpu_residue_blocks_reuse_without_start_or_clock_changes(setup):
    adapter,calls,processes,http,native=setup;boundary=module.PDblendResidentBoundary(adapter)
    adapter.fleet['p1'].stop();processes[1]=[9876]
    with pytest.raises(RuntimeError,match='retains compute'):
        asyncio.run(boundary._off_evidence([adapter.specs[1]],timeout_s=0))
    assert not calls and not http and adapter.fleet['p1'].starts==1


@pytest.mark.parametrize('roles,action,message',[
    ({'p0':'M'},lambda a:None,'role inventory'),
    ({'p0':'M','p1':'off'},lambda a:None,'live/undeclared'),
    ({'p0':'M','p1':'M'},lambda a:a.fleet['p1'].stop(),'lost its process'),
])
def test_unexpected_lifecycle_is_not_excused_as_policy_off(setup,roles,action,message):
    adapter,*_=setup;boundary=module.PDblendResidentBoundary(adapter);action(adapter)
    with pytest.raises(RuntimeError,match=message):asyncio.run(boundary.drain(roles))


def test_changed_inventory_or_gpu_identity_is_not_restored(setup):
    adapter,calls,processes,http,native=setup;boundary=module.PDblendResidentBoundary(adapter)
    adapter.specs[0]=replace(adapter.specs[0],tp=2,gpus=(0,2))
    with pytest.raises(RuntimeError,match='topology/options'):asyncio.run(boundary.restore())
    assert not calls


def test_failed_restart_epoch_is_not_published_as_ready_for_policy(setup,monkeypatch):
    adapter,calls,processes,http,native=setup;boundary=module.PDblendResidentBoundary(adapter)
    adapter.fleet['p1'].stop();adapter.fleet['p1'].start()
    original=module._json
    def wrong(spec,endpoint,payload=None):
        row=original(spec,endpoint,payload)
        return dict(row,acknowledged=False) if endpoint=='control' else row
    monkeypatch.setattr(module,'_json',wrong)
    with pytest.raises(RuntimeError,match='restoration failed'):adapter.fleet['p1'].wait_ready()
    assert not boundary.restart_epochs
    assert boundary.refresh_load_count()['engine_loads']==3


def test_adapter_next_window_restores_then_establishes_one_fresh_epoch(setup,monkeypatch,tmp_path):
    from pdblend.bench import comparison_runtime as runtime
    original,calls,processes,http,native=setup
    adapter=runtime.NativeResidentAdapter(tmp_path,base_port=19000)
    for key,value in vars(original).items():setattr(adapter,key,value)
    adapter.pdblend_boundary=module.PDblendResidentBoundary(adapter)
    adapter.pdblend_boundary.begin_window();adapter.fleet['p1'].stop()
    asyncio.run(adapter.pdblend_boundary.finish_window(dict(final_roles={'p0':'L1','p1':'off'})))
    async def drain(specs):
        assert all(adapter.fleet[s.instance_id].alive() for s in specs)
        return [dict(instance_id=s.instance_id,state=native[s.instance_id]) for s in specs]
    async def call(session,url,path,payload=None):
        spec=next(s for s in adapter.specs if s.base_url==url)
        if payload:
            native[spec.instance_id]=state(**payload)
            return dict(acknowledged=True,generation=payload['generation'])
        return native[spec.instance_id]
    async def warmup(specs,label):
        assert all(adapter.fleet[s.instance_id].alive() for s in specs)
        for spec in specs:
            native[spec.instance_id]=state(generation=spec.generation+1,accepting=True)
        return [dict(instance_id=s.instance_id) for s in specs]
    monkeypatch.setattr(runtime,'drain_endpoints',drain);monkeypatch.setattr(runtime,'call',call)
    monkeypatch.setattr(runtime,'warmup_endpoints',warmup)
    reset=asyncio.run(adapter.reset(dict(system='pdblend',name='second-window')))
    assert reset['pdblend_inventory_reset']['restored_instances']==['p1']
    assert set(reset['generation'].values())=={10}
    assert {s.generation for s in adapter.specs}=={10}
    assert all(adapter.fleet[s.instance_id].spec==s for s in adapter.specs)
    assert all(value['accepting'] for value in reset['reopen_state'].values())
    assert adapter.load_count==3
