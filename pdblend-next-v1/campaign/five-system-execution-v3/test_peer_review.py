"""Focused CPU checks of the actual wrapper functions; no hardware or network."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

runner=load('five_system_peer_run',ROOT/'run.py')
child=load('five_system_peer_child',ROOT/'child.py')

def test_invalid_native_barrier_still_resumes_actual_instance(monkeypatch):
    calls=[]
    async def wait_idle(*a,**k):return {'generation':9}
    async def http(*a,**k):return {'drained':False}
    async def resume(session,instance,tokens):calls.append((instance['id'],tokens));return {'actual_generation':11}
    monkeypatch.setattr(runner,'wait_idle',wait_idle);monkeypatch.setattr(runner,'http',http);monkeypatch.setattr(runner,'resume',resume)
    r=asyncio.run(runner.restore(None,dict(id='a',tp=1,native_kind='v3',restore_budget_tokens=8192)))
    assert calls==[('a',8192)] and r['complete'] is False and r['resumed']['actual_generation']==11
    assert len(r['errors'])==1 and 'native proof' in r['errors'][0]

def test_legacy_rank_proof_cannot_be_relabelled_v3():
    before={'generation':5};proof=dict(drained=True,accepting=False,generation=6,
        drain_proof_type='synchronous_put_owner_barrier',transfers=[dict(buffered_tensors=0,inflight_receives=0,listener_alive=True)])
    runner.barrier(before,proof,dict(tp=1,native_kind='legacy_sync_put'))
    with pytest.raises(RuntimeError,match='rank sends'):runner.barrier(before,proof,dict(tp=1,native_kind='v3'))

def fake_measurement_environment(monkeypatch,tmp_path,*,setup_fails):
    import ecopadg.measure.power as power
    import ecopadg.serving.backend as backend
    from ecopadg.measure.backends import INSTANT_POWER_SOURCE_ID
    events=[];samplers=[]
    class Sampler:
        def __init__(self,gpus,**kw):
            assert list(gpus)==list(range(8)) and kw['sample_clocks'] is True
            self.error=None;self.samples=[];self.utilization_samples=[];self.frequency_samples=[];self.power_metadata=[]
            self.power_source=dict(mode='instant',source_id=INSTANT_POWER_SOURCE_ID,field_id=186,scope_id=0);samplers.append(self)
        def append(self,t):
            self.samples.append((t,[50.]*8));self.utilization_samples.append((t,[20.]*8));self.frequency_samples.append((t,[1500.]*8))
            self.power_metadata.append(dict(t_s=t,gpus=list(range(8)),mode=['instant']*8,source_id=[INSTANT_POWER_SOURCE_ID]*8,
                field_id=[186]*8,scope_id=[0]*8,value_type=[1]*8,return_code=[0]*8,nvml_timestamp_us=[int(t*1e6)]*8,
                nvml_latency_us=[0]*8,read_started_s=[t]*8,read_finished_s=[t]*8))
        def start(self):self.append(time.time()-.02);self.append(time.time()-.01)
        def stop(self):self.append(time.time()+.001);events.append('sampler_stopped')
    class ClockOwner:
        def __init__(self,hardware,gpus):assert tuple(gpus)==tuple(range(8))
        async def close(self):events.append('clocks_closed')
    async def identity(*a):return [{'identity_verified':True}]
    async def resume(session,instance,tokens=None):
        if setup_fails and instance['id']=='a':events.append('a_setup_failed');raise RuntimeError('injected setup failure')
        await asyncio.sleep(.01);events.append(instance['id']+'_setup_complete');return {'complete':True}
    async def restore(session,instance):
        assert 'b_setup_complete' in events
        events.append(instance['id']+'_restore');return {'complete':True}
    async def cpu_thread_call(fn,*args,**kwargs):
        # All substituted functions are CPU stand-ins; thread scheduling itself
        # is outside this lifecycle/energy test and needs no sandbox wakeup FD.
        return fn(*args,**kwargs)
    monkeypatch.setattr(power,'PowerSampler',Sampler);monkeypatch.setattr(backend,'ClockOwner',ClockOwner)
    monkeypatch.setattr(runner,'identity',identity);monkeypatch.setattr(runner,'resume',resume);monkeypatch.setattr(runner,'restore',restore)
    monkeypatch.setattr(runner.asyncio,'to_thread',cpu_thread_call)
    trace=tmp_path/'trace.json';trace.write_text('{}')
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps({'strategy':'mixed'}))
    instances=[dict(id=n,port=30000+k,native_kind='legacy_sync_put',tp=1) for k,n in enumerate(('a','b'))]
    binding=dict(configs={'alpaca':str(cfg)},instances=instances,protocol_id=runner.PROTOCOL,
        hostname=runner.socket.gethostname(),deadline_s=runner.GLOBAL_DEADLINE,system='mixed',
        files={str(cfg):runner.sha(cfg),str(trace):runner.sha(trace)})
    row=dict(dataset='alpaca',cell_id='cpu-only',system='mixed',trace=str(trace),trace_sha256=runner.sha(trace),n_requests=10,
        slo_scale=1.,slo_ttft_s=1.,slo_tpot_s=.1)
    class Process:
        pid=123;returncode=0
        async def wait(self):return 0
    async def create(*args,**kw):
        assert not setup_fails
        job=runner.read(args[-1]);out=Path(job['out']);out.mkdir(parents=True)
        summary=dict(trace_sha256=row['trace_sha256'],measurement_window_protocol=runner.PROTOCOL,fixed_window_valid=True,
            measurement_valid=True,post_measurement_cleanup={'cleanup_complete':True},measurement_end_s=time.time(),
            work_complete=False,n_expected=10,completed=7,good_requests=0,energy_j=1234.)
        runner.write(out/'summary.json',summary)
        actual=runner.read(cfg);actual.update(journal=str(out/'control.jsonl'),slo_scale=1.,slo_protocol='per-dataset-slo-v1',
            slo_attainment_target=.9,slo_ttft_s=1.,slo_tpot_s=.1,comparison_system='mixed')
        runner.write(out/'runtime_config.json',actual);return Process()
    monkeypatch.setattr(runner.asyncio,'create_subprocess_exec',create)
    return binding,row,events,samplers

def test_failed_setup_waits_for_peer_and_preserves_all_gpu_energy(monkeypatch,tmp_path):
    binding,row,events,samplers=fake_measurement_environment(monkeypatch,tmp_path,setup_fails=True)
    with pytest.raises(RuntimeError,match='invalid measurement'):
        asyncio.run(runner.run_one(None,binding,row,tmp_path/'results',object()))
    receipt=runner.read(tmp_path/'results/operations/cpu-only/receipt.json')
    assert events.index('b_setup_complete')<events.index('a_restore')
    assert receipt['measurement_valid'] is False and 'initial controls failed' in receipt['error']
    assert receipt['full_operation_energy_j']==pytest.approx(400*(receipt['operation_end_s']-receipt['operation_start_s']))
    assert receipt['full_operation_energy_j']>0 and receipt['power_evidence']['power_source_verified'] is True
    header=(tmp_path/'results/operations/cpu-only/power/power.csv').read_text().splitlines()[0].split(',')
    assert [c for c in header if c.endswith('_w')]==[f'gpu{i}_w' for i in range(8)]
    assert all(r['complete'] for r in receipt['restoration'].values())

def test_valid_incomplete_work_keeps_primary_energy_and_can_checkpoint(monkeypatch,tmp_path):
    binding,row,events,samplers=fake_measurement_environment(monkeypatch,tmp_path,setup_fails=False)
    receipt=asyncio.run(runner.run_one(None,binding,row,tmp_path/'results',object()))
    assert receipt['measurement_valid'] is True and receipt['summary']['work_complete'] is False
    assert receipt['summary']['n_expected']==10 and receipt['summary']['completed']==7
    assert receipt['summary']['good_requests']==0 and receipt['summary']['energy_j']==1234
    assert receipt['full_operation_energy_j']>0 and len(samplers[0].samples[0][1])==8

def test_real_readiness_gate_waits_for_second_power_row_after_100ms(monkeypatch,tmp_path):
    import ecopadg.measure.power as power
    binding,row,events,samplers=fake_measurement_environment(monkeypatch,tmp_path,setup_fails=False)
    def slow_start(self):
        self.append(time.time())
        async def second():
            await asyncio.sleep(.12)
            self.append(time.time());events.append('second_power_row')
        asyncio.create_task(second())
    monkeypatch.setattr(power.PowerSampler,'start',slow_start)
    receipt=asyncio.run(runner.run_one(None,binding,row,tmp_path/'results',object()))
    assert receipt['measurement_valid'] is True
    assert events.index('second_power_row')<events.index('a_setup_complete')
    assert samplers[0].samples[1][0]-samplers[0].samples[0][0]>=.1

@pytest.mark.parametrize('dataset,scale,ttft,tpot',[
    ('alpaca',.5,.5,.05),('alpaca',1.,1.,.1),('alpaca',2.,2.,.2),
    ('sharegpt',.5,2.5,.075),('sharegpt',1.,5.,.15),('sharegpt',2.,10.,.3),
    ('longbench',.5,7.5,.1),('longbench',1.,15.,.2),('longbench',2.,30.,.4)])
def test_child_args_reach_real_100_second_cell_configuration(monkeypatch,tmp_path,dataset,scale,ttft,tpot):
    import ecopadg.serving.cell as current
    from benchmarks.scripts import bench_vllm as bench
    overlay=load('ecopadg.serving.peer_actual_five_cell',ROOT.parent/'five-system-fixed-window-v1/host-overlay/ecopadg/serving/cell.py')
    trace=dict(protocol_id=runner.PROTOCOL,measurement_schema=3,arrival_window_s=100,duration_s=100,seed=701,
        dataset=dataset,split='development',comparison_systems=list(overlay.COMPARISON_SYSTEMS),
        request_hard_timeout_s=120,post_window_drain_allowance_s=120,
        requests=[{'arrival_s':0}],prompts=['x'],n_requests=1)
    config=dict(strategy='mixed',evaluation_protocol='evaluation-v3',measurement_window_protocol=runner.PROTOCOL,arrival_window_s=100)
    runner.write(tmp_path/'trace.json',trace);runner.write(tmp_path/'config.json',config)
    row=dict(trace=str(tmp_path/'trace.json'),dataset=dataset,load='cpu',seed=701,slo_scale=scale,slo_ttft_s=ttft,slo_tpot_s=tpot)
    job=dict(row=row,config=str(tmp_path/'config.json'),out=str(tmp_path/'out'),latest_arrival_epoch_s=time.time()+60,engine_ports=[30000])
    runner.write(tmp_path/'job.json',job);seen={};original_headers=bench.evaluation_headers
    async def run_cell(args):
        c=runner.read(args.config);w=overlay.configure_fixed_window(args,c,runner.read(args.trace));seen.update(c=c,w=w,args=args)
        now=time.time();bench.evaluation_headers('http://127.0.0.1:18080','evaluation-v3',now,now)
        return dict(measurement_valid=True,post_measurement_cleanup={'cleanup_complete':True})
    monkeypatch.setattr(current,'run_cell',run_cell)
    assert asyncio.run(child.execute(tmp_path/'job.json')) is True
    assert seen['w']['arrival_window_s']==100 and seen['args'].timeout==120 and seen['args'].strategy is None
    assert (seen['c']['slo_ttft_s'],seen['c']['slo_tpot_s'])==(ttft,tpot)
    assert runner.read(tmp_path/'actual-epoch.json')['checked_before_dispatch'] is True
    assert bench.evaluation_headers is original_headers
