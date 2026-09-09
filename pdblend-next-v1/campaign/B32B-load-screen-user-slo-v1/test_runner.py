import asyncio
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

ROOT=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
b=load('b32_screen',ROOT/'run.py');x=load('b32_execution',ROOT/'execution.py')


def state(index=0,generation=4,tokens=8192):
    return dict(id=b.IDS[index],generation=generation,acknowledged_generation=generation,acknowledged_generations=[generation],
        observed_control_generation=generation,scheduler_budget_pending=None,
        scheduler_budget_effective=dict(max_num_batched_tokens=tokens,max_num_seqs=32),
        active=0,running=0,waiting=0,kv_allocations={},transfer_allocations={},transfer_buffered_tensors=0,
        transfer_inflight_receives=0,transfer_inflight_sends=0,transfer_inflight_sends_observed=True,
        transfer_send_counters_observed=True,transfer_send_healthy=True,transfer_send_started=3,
        transfer_send_completed=3,transfer_send_failed=0,transport_healthy=True,timestamp=time.time(),
        transfer_observed_s=time.time(),role='mixed',mode='continuous',accepting=True,admit_prefill=True,admit_decode=True,
        scheduler_io=[dict(controls=dict(runtime=dict(generation=generation,error=None)))])


def barrier():
    return dict(drained=True,accepting=False,generation=5,drain_proof_type='synchronous_put_owner_barrier',send_counters_verified=True,
        transfers=[dict(buffered_tensors=0,inflight_receives=0,inflight_sends=0,listener_alive=True,
            allocations={},buffered_gpu_bytes=0,send_counters_observed=True,send_healthy=True,
            send_started=3,send_completed=3,send_failed=0) for _ in range(2)])


def test_eighteen_byte_exact_traces_and_original_pilot_order():
    spec=b.read(ROOT/'runspec.json');manifest=b.read(ROOT/'inputs/source-sweep-manifest.json')
    original={c['cell_id']:c for c in manifest['cells'] if c['cell_id'] in {x['cell_id'] for x in spec['cells']}}
    assert len(spec['cells'])==len(original)==3
    assert [(c['dataset'],c['rate_rps']) for c in spec['cells'][:3]]==[('alpaca',1.),('sharegpt',.25),('longbench',.125)]
    for c in spec['cells']:
        assert c['source_cell']==original[c['cell_id']] and b.sha(c['trace'])==original[c['cell_id']]['trace_sha256']
        trace=b.read(c['trace']);assert trace['split']=='development' and trace['arrival_seed']==701
        assert len(trace['requests'])==64 and not trace['formal_eligible'] and not trace['saturation_verified']
    for d in ('alpaca','sharegpt','longbench'):
        rates=[c['rate_rps'] for c in spec['cells'][3:] if c['dataset']==d];assert rates==sorted(rates)


def test_one_verified_policy_no_outputcap_and_96_baseline_files():
    source=b.read(ROOT/'inputs/controller.source.json');actual=b.read(b.CONFIG)
    source['controller_source_release']=str(b.HOST);assert source==actual
    assert actual['scheduler_budget_ablation']==dict(schema_version=1,max_num_batched_tokens=8192,max_num_seqs=32)
    assert not actual.get('prediction_cap_to_max_tokens') and actual['output_prior']==211
    protected=b.read(ROOT/'inputs/protected-baselines.json');assert len(protected['files'])==91
    assert {c['controller_config'] for c in b.read(ROOT/'runspec.json')['cells']}=={str(b.CONFIG)}


def test_v3_native_idle_accepts_actual_completed_historical_sends():
    raw=state();b.native_idle(raw,b.IDS[0],tokens=8192,accepting=True);b.native_barrier(raw,barrier())
    assert raw['transfer_send_started']==3


@pytest.mark.parametrize('fault',['legacy_unknown','runtime_error','error','pending','wrong_budget','missing_counter','bad_cache','waiting','kv','stale','send_failed'])
def test_bad_or_legacy_owner_observation_cannot_pass(fault):
    raw=state()
    if fault=='legacy_unknown':raw.update(transfer_inflight_sends=None,transfer_inflight_sends_observed=False)
    elif fault in ('runtime_error','error'):raw[fault]='failed'
    elif fault=='pending':raw['scheduler_budget_pending']={'generation':5}
    elif fault=='wrong_budget':raw['scheduler_budget_effective']['max_num_batched_tokens']=1024
    elif fault=='missing_counter':del raw['transfer_send_counters_observed']
    elif fault=='bad_cache':raw['scheduler_io'][0]['controls']['runtime']['generation']=3
    elif fault=='waiting':raw['waiting']=1
    elif fault=='kv':raw['kv_allocations']={'r':128}
    elif fault=='stale':raw['transfer_observed_s']-=2
    else:raw['transfer_send_failed']=1
    with pytest.raises(RuntimeError):b.native_idle(raw,b.IDS[0])


@pytest.mark.parametrize('fault',['legacy','pending_send','failed_send','wrong_rank','cached','bad_generation'])
def test_native_proof_cannot_be_cached_or_missing_actual_rank_sends(fault):
    proof=barrier()
    if fault=='legacy':proof['send_counters_verified']=False
    elif fault=='pending_send':proof['transfers'][0]['send_completed']=2
    elif fault=='failed_send':proof['transfers'][0]['send_failed']=1
    elif fault=='wrong_rank':proof['transfers']*=2
    elif fault=='cached':proof['drain_proof_type']='telemetry'
    else:proof['generation']=4
    with pytest.raises(RuntimeError):b.native_barrier(state(),proof)


def test_bad_proof_still_restores_8192_and_actual_ack(monkeypatch):
    async def run():
        raw=state(tokens=2048);calls=[]
        async def http(session,index,path,body=None,**kwargs):
            calls.append((path,body))
            if path=='/runtime':return copy.deepcopy(raw)
            if path=='/drain':
                raw.update(generation=5,acknowledged_generation=5,accepting=False,admit_prefill=False)
                p=barrier();p['send_counters_verified']=False;return p
            assert path=='/control' and body['generation']==6 and body['scheduler_budget']['max_num_batched_tokens']==8192
            raw.clear();raw.update(state(generation=6));return body
        monkeypatch.setattr(b,'http',http)
        result=await x.restore_one(b,None,0)
        assert not result['complete'] and result['after']['scheduler_budget_effective']['max_num_batched_tokens']==8192
        assert result['after']['accepting'] and any(p=='/control' for p,_ in calls)
    asyncio.run(run())


def test_foreign_identity_refuses_restore_control(monkeypatch):
    calls=[]
    async def http(session,index,path,body=None,**kwargs):calls.append(path);return dict(state(),id='foreign')
    monkeypatch.setattr(b,'http',http)
    result=asyncio.run(x.restore_one(b,None,0));assert not result['complete'] and set(calls)=={'/runtime'}


def test_dispatch_parser_only_cancels_exact_durable_owned_target(tmp_path):
    path=tmp_path/'dispatch.jsonl';rid='a'*32
    path.write_bytes((json.dumps(dict(port=33500,request_id=rid))+'\n').encode()+b'{"request_id":')
    assert x.dispatch_ids(path,b.PORTS)=={(0,rid)}
    path.write_text(json.dumps(dict(port=33599,request_id=rid))+'\n')
    with pytest.raises(ValueError):x.dispatch_ids(path,b.PORTS)


def test_own_host_timeout_terminates_then_kills_only_that_child():
    class Child:
        returncode=None
        def __init__(self):self.calls=[]
        def terminate(self):self.calls.append('terminate')
        def kill(self):self.calls.append('kill');self.returncode=-9
        async def wait(self):
            if self.returncode is None:raise asyncio.TimeoutError()
            return self.returncode
    child=Child();asyncio.run(x.stop_child(child));assert child.calls==['terminate','kill']


def test_source_control_requires_actual_owner_ack_after_post(monkeypatch):
    async def run():
        raw=state();counts={'reads':0}
        async def http(session,index,path,body=None,**kwargs):
            if path=='/control':raw.update(state(generation=5,tokens=2048));raw['acknowledged_generation']=4;return body
            counts['reads']+=1
            if counts['reads']>=3:raw['acknowledged_generation']=raw['generation']
            return copy.deepcopy(raw)
        monkeypatch.setattr(b,'http',http)
        result=await x.set_one(b,None,0,2048)
        assert counts['reads']==3 and result['after']['acknowledged_generation']==5
    asyncio.run(run())


@pytest.mark.parametrize('fault',[None,'ownership_log','cancel','one_proof','clock'])
def test_outer_cleanup_failures_keep_both_restore_paths_and_clock_attempt(tmp_path,monkeypatch,fault):
    async def run():
        from ecopadg.serving import backend
        calls=[];cell=x.Cell.__new__(x.Cell)
        operation=tmp_path/'op';operation.mkdir()
        (operation/'dispatch.jsonl').write_text('{bad}\n' if fault=='ownership_log' else json.dumps(dict(port=b.PORTS[0],request_id='a'*32))+'\n')
        async def http(*args,**kwargs):
            calls.append('cancel')
            if fault=='cancel':raise TimeoutError('cancel failed')
            return {}
        async def restore(base,session,index):calls.append(('restore',index));return dict(complete=not (fault=='one_proof' and index==0))
        class Clock:
            def __init__(self,hardware,gpus):assert gpus==tuple(range(8));calls.append('clock-acquired')
            async def close(self):
                calls.append('clock-close')
                if fault=='clock':raise RuntimeError('clock failed')
        base=SimpleNamespace(PORTS=b.PORTS,IDS=b.IDS,http=http,write=lambda *args:None)
        cell.b=base;cell.session=None;cell.operation=operation;cell.receipt={};cell.child=None
        cell.verified=True;cell.mutated=True;cell.row=dict(cell_id='cpu');cell.hardware=object()
        monkeypatch.setattr(x,'restore_one',restore);monkeypatch.setattr(backend,'ClockOwner',Clock)
        async def fake_thread(fn,*args,**kwargs):return fn(*args,**kwargs)
        monkeypatch.setattr(asyncio,'to_thread',fake_thread)
        result=await cell.cleanup()
        assert ('restore',0) in calls and ('restore',1) in calls and calls[-1]=='clock-close'
        assert result['complete'] is (fault is None)
    asyncio.run(run())


def test_failed_child_stop_cannot_restore_engines_or_reset_live_host_clocks(tmp_path,monkeypatch):
    async def run():
        calls=[];cell=x.Cell.__new__(x.Cell)
        cell.b=SimpleNamespace(IDS=b.IDS,PORTS=b.PORTS,write=lambda *args:None);cell.session=None;cell.operation=tmp_path
        cell.receipt={};cell.child=object();cell.verified=True;cell.mutated=True;cell.row=dict(cell_id='cpu');cell.hardware=object()
        async def stop(child):raise RuntimeError('cannot confirm child exit')
        async def restore(*args):calls.append('restore');return dict(complete=True)
        monkeypatch.setattr(x,'stop_child',stop);monkeypatch.setattr(x,'restore_one',restore)
        result=await cell.cleanup()
        assert not calls and not result['complete'] and not result['clock_release_complete']
    asyncio.run(run())


def test_identity_or_sampler_failure_before_mutation_sends_no_cleanup_control(tmp_path):
    async def run():
        cell=x.Cell.__new__(x.Cell);cell.b=SimpleNamespace(write=lambda *args:None)
        cell.operation=tmp_path;cell.row=dict(cell_id='cpu');cell.child=None;cell.verified=False;cell.mutated=False
        result=await cell.cleanup();assert result['complete'] and not result['native'] and not result['mutations_started']
    asyncio.run(run())


def test_original_selection_uses_user_slos_and_uniform_configuration():
    contract=load('b32_contract',ROOT/'contract.py')
    spec=b.read(ROOT/'runspec.json');source=b.read(ROOT/'inputs/source-sweep-manifest.json')
    assert len(contract.validate(spec,source,b.read(ROOT/'source-contract.json'),b.read,b.sha))==3
    assert [(r['slo_ttft_s'],r['slo_tpot_s']) for r in spec['cells'][:3]]==[(1.,.1),(5.,.15),(15.,.2)]


def test_independently_frozen_anchor_contract_can_select_exact_original_three():
    mod=load('b32_contract',ROOT/'contract.py')
    spec=b.read(ROOT/'runspec.json');source=b.read(ROOT/'inputs/source-sweep-manifest.json')
    contract=b.read(ROOT/'source-contract.json');spec['cells']=spec['cells'][:3]
    contract['source_selection']['cell_ids']=[r['cell_id'] for r in spec['cells']];contract['expected_cells']=3
    assert len(mod.validate(spec,source,contract,b.read,b.sha))==3
    contract['source_selection']['cell_ids'][0]='missing'
    with pytest.raises(ValueError):mod.validate(spec,source,contract,b.read,b.sha)


@pytest.mark.parametrize('dataset',['alpaca','sharegpt','longbench'])
def test_original_run_cell_injects_user_slos_before_real_controller_creation(tmp_path,monkeypatch,dataset):
    from ecopadg.serving import cell
    child=load('b32_child',ROOT/'child.py')
    row=next(r for r in b.read(ROOT/'runspec.json')['cells'] if r['dataset']==dataset)
    args=child.cell_arguments(row);args.out=tmp_path/'cell'
    observed=[]
    class BeforeAnyNetwork(Exception):pass
    def constructor(config):observed.append(copy.deepcopy(config));raise BeforeAnyNetwork()
    monkeypatch.setattr(cell,'Controller',constructor)
    with pytest.raises(BeforeAnyNetwork):asyncio.run(cell.run_cell(args))
    actual=b.read(args.out/'runtime_config.json')
    assert observed==[actual]
    assert (actual['slo_ttft_s'],actual['slo_tpot_s'])==(row['slo_ttft_s'],row['slo_tpot_s'])
    assert actual['scheduler_budget_ablation']['max_num_batched_tokens']==8192


def expanded_cpu_fixture():
    """Synthetic metadata only; no trace artifact or independent corpus claim."""
    old=b.read(ROOT/'runspec.json');contract=b.read(ROOT/'source-contract.json')
    contract.update(expected_cells=9,requests_min=1000,requests_exact=None,duration_min_s=300.)
    contract['source_selection']['arrival_seeds']=[701,702,703];contract['source_selection'].pop('cell_ids',None)
    rows=[];source=[];traces={}
    for seed in (701,702,703):
        for original in old['cells']:
            row=copy.deepcopy(original);origin=row['source_cell'];key=row['cell_id']+f'-cpu{seed}'
            row.update(cell_id=key,seed=seed,n_requests=1000,trace_duration_s=300.,trace=key,trace_sha256=key)
            origin.update(cell_id=key,arrival_seed=seed,n_requests=1000,trace_duration_s=300.,trace_sha256=key)
            traces[key]=dict(split='development',model='32b',arrival_seed=seed,
                dataset=row['dataset'],load=row['load'],rate=row['rate_rps'],duration_s=300.,
                requests=[dict(arrival_s=i*300/999) for i in range(1000)],formal_eligible=False,saturation_verified=False,
                repeat_from_existing_dev_pool=True,corpus_independence_across_arrival_seeds=False)
            rows.append(row);source.append(origin)
    old['cells']=rows
    return old,dict(cells=source),contract,traces


def test_new_frozen_contract_accepts_thousand_requests_three_seeds_and_300_seconds():
    mod=load('b32_contract',ROOT/'contract.py');spec,source,contract,traces=expanded_cpu_fixture()
    assert len(mod.validate(spec,source,contract,traces.__getitem__,lambda key:key))==9


@pytest.mark.parametrize('fault',['short_span','few_requests','foreign_slo','baseline','formal'])
def test_new_contract_rejects_invalid_extended_work_and_protocol(fault):
    mod=load('b32_contract',ROOT/'contract.py');spec,source,contract,traces=expanded_cpu_fixture()
    trace=traces[spec['cells'][0]['trace']]
    if fault=='short_span':trace['duration_s']=299.
    elif fault=='few_requests':trace['requests'].pop()
    elif fault=='foreign_slo':spec['cells'][0]['slo_ttft_s']=5.
    elif fault=='baseline':source['cells'][0]['system']='distserve'
    else:trace['split']='formal'
    with pytest.raises(ValueError):mod.validate(spec,source,contract,traces.__getitem__,lambda key:key)
