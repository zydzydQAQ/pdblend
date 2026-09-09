import asyncio
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from collections import deque

import pytest
from ecopadg.serving.types import InstanceState, RuntimeSnapshot, RequestBudget
from ecopadg.serving.profiles import ProfilePoint, ProfileStore, OutputPredictor
from ecopadg.serving.planner import JointPlanner, TransferCost
from ecopadg.serving.frequency import FrequencyCost
from ecopadg.serving.state import StateManager, StalePlan


def point(role,frequency,power,iteration=.04):
    return ProfilePoint(role,1,frequency,4096,8192,32,.1,iteration,power,30,.1,10,"fixture",.12)


def system():
    profiles=ProfileStore([point(role,f,w,it) for role in ('mixed','prefill','decode')
        for f,w,it in [(900,160,.09),(1500,120,.04),(2520,230,.035)]])
    instances=tuple(InstanceState(str(j),role,1,(j,),10,0,2520,20000,0,0,
                    free_transfer_bytes=2*1024**3,transfer_bytes_per_token=206848)
                    for j,role in enumerate(('mixed','prefill','decode')))
    snapshot=RuntimeSnapshot(1,10,instances)
    request=RequestBudget('r0',10,128,64,2,.1,output_limit=256)
    planner=JointPlanner(profiles,[TransferCost(1,1,4096,.01,1,'fixture',True,profile_batch=32)])
    return planner,snapshot,request


def test_chooses_energy_minimum_not_lowest_frequency():
    planner,snapshot,request=system()
    plan=planner.plan(snapshot,(request,),now=10,joint=False)
    assert plan.feasible
    assert any(f.frequency_mhz==1500 for f in plan.frequencies)
    assert not any(f.frequency_mhz==900 for f in plan.frequencies)


def test_admission_charges_measured_frequency_time_and_energy():
    analytical,snapshot,request=system()
    def planner(delay,energy):
        costs=[FrequencyCost(1,2520,f,delay,energy,'measured-fixture') for f in (900,1500)]
        return JointPlanner(analytical.profiles,analytical.transfers,frequency_costs=costs)
    strict=JointPlanner(analytical.profiles,analytical.transfers,frequency_costs=[])
    assert all(a.frequency_mhz==2520 for p in strict.candidates(snapshot,request,10) for a in p.frequencies)
    assert any(a.frequency_mhz==1500 for a in planner(.01,1).plan(snapshot,(request,),now=10).frequencies)
    # A cheaper steady decode is not economical after an expensive switch.
    assert all(a.frequency_mhz==2520 for a in planner(.01,100000).plan(snapshot,(request,),now=10).frequencies)
    # A three-second transition cannot fit this request's two-second TTFT.
    assert all(a.frequency_mhz==2520 for p in planner(3,1).candidates(snapshot,request,10) for a in p.frequencies)


def test_short_prefill_cost_is_independent_of_an_old_long_decode_context():
    short=ProfilePoint('mixed',1,2520,128,640,1,.05,.035,230,30,.05,1,'short')
    long=ProfilePoint('mixed',1,2520,7168,7680,4,2.,.045,250,30,.05,1,'long',2.)
    profiles=ProfileStore([short,long],interference_points=[dict(tp=1,frequency_mhz=2520,
        input_tokens=128,context_tokens=7680,background_batch=3,delay_s=.06,source_sha256='measured-cross')])
    old=RequestBudget('old',8,7168,512,5,.1,emitted=30,first_token_s=9,last_token_s=10)
    instance=InstanceState('m','mixed',1,(0,),10,0,2520,20000,1,0,requests=(old,))
    request=RequestBudget('short',10,64,128,.3,.1)
    planner=JointPlanner(profiles,allow_pd=False,dvfs=False)
    plan=planner.plan(RuntimeSnapshot(1,10,(instance,)),(request,),now=10)
    assert plan.feasible and plan.routes[0].predicted_ttft_s<.2
    # If the old long request is still in prefill, its queue delay is real.
    queued=replace(instance,requests=(replace(old,emitted=0,first_token_s=None),))
    assert not planner.plan(RuntimeSnapshot(1,10,(queued,)),(request,),now=10).feasible


def test_no_tp_extrapolation_and_stale_telemetry_restores_frequency():
    planner,snapshot,request=system()
    assert planner.profiles.lookup('decode',2,1500,128,256,1) is None
    plan=planner.plan(snapshot,(request,),now=12)
    assert not plan.feasible
    assert {f.frequency_mhz for f in plan.frequencies}=={2520}


def test_existing_per_request_deadline_cannot_be_spent_twice():
    planner,snapshot,request=system()
    existing=replace(request,request_id='old',emitted=10,first_token_s=9,last_token_s=9.8)
    snapshot=replace(snapshot,instances=tuple(replace(i,requests=(existing,)) for i in snapshot.instances))
    assert not planner.plan(snapshot,(request,),now=10).feasible


def test_pending_mixed_prefill_already_commits_decode_prefix_credit():
    measured=ProfilePoint('mixed',1,2520,128,512,8,.2,.05,200,30,0,1,'fixture',.2)
    planner=JointPlanner(ProfileStore([measured]),allow_pd=False,dvfs=False)
    old=RequestBudget('old',8,128,64,5,.1,emitted=5,first_token_s=9.85,last_token_s=9.99)
    queued=RequestBudget('queued',9.99,128,64,5,.1)
    incoming=RequestBudget('new',10,128,64,5,.1)
    instance=InstanceState('m','mixed',1,(0,),10,0,2520,10000,1,0,requests=(old,))
    snapshot=RuntimeSnapshot(1,10,(instance,))
    assert old.next_token_remaining(10)==pytest.approx(.35)
    first=planner.plan(snapshot,(queued,),now=10)
    assert first.feasible  # One .20 s prefill plus a .05 s decode fits.
    # The beam must use the same outstanding-work account as online admission.
    advanced=planner.advance(snapshot,first,queued)
    assert not planner.plan(advanced,(incoming,),now=10).feasible
    live=replace(snapshot,instances=(replace(instance,waiting=1,requests=(old,queued)),))
    assert not planner.plan(live,(incoming,),now=10).feasible  # .20 + .20 + .05 > .35


def test_pd_source_capacity_and_beam_match_block_aligned_reservations():
    async def run():
        planner,snapshot,request=system()
        planner.dvfs=False
        snapshot=replace(snapshot,instances=tuple(i for i in snapshot.instances if i.role!='mixed'))
        # A 128-token prefill generates one token and requires nine 16-token blocks.
        short=replace(snapshot,instances=tuple(replace(i,free_kv_tokens=143)
            if i.role=='prefill' else i for i in snapshot.instances))
        assert not planner.candidates(short,request,10)
        plan=planner.plan(snapshot,(request,),now=10)
        assert plan.routes[0].prefill_reserve_tokens==144
        state=StateManager(snapshot)
        await state.reserve(plan,10,request)
        advanced=planner.advance(snapshot,plan,request)
        predicted=next(i for i in advanced.instances if i.role=='prefill')
        actual=next(i for i in state.snapshot.instances if i.role=='prefill')
        from ecopadg.serving.tails import admission_budget
        assert predicted.requests==actual.requests==(admission_budget(plan,request),)
        assert predicted.reserved_kv_tokens==actual.reserved_kv_tokens==144
        assert predicted.waiting==actual.waiting==1
    asyncio.run(run())


def test_kv_reserved_before_dispatch_and_stale_plan_rejected():
    async def run():
        planner,snapshot,request=system()
        plan=planner.plan(snapshot,(request,),now=10)
        state=StateManager(snapshot)
        assert await state.reserve(plan,10)
        assert state.reservations['r0'].reserve_tokens==384
        with pytest.raises(StalePlan):
            await state.reserve(plan,10)
        await state.release('r0')
        await state.release('r0')
        assert not state.reservations
        assert all(i.reserved_kv_tokens==0 for i in state.snapshot.instances)
    asyncio.run(run())


def test_decode_kv_capacity_rejects_both_paths():
    planner,snapshot,request=system()
    snapshot=replace(snapshot,instances=tuple(replace(i,free_kv_tokens=200) for i in snapshot.instances))
    assert not planner.plan(snapshot,(request,),now=10).feasible


def test_beam_search_never_uses_future_arrivals_and_budget_falls_back():
    planner,snapshot,request=system()
    planner.decision_budget_s=0
    plan=planner.plan(snapshot,(request,replace(request,request_id='r1')),now=10)
    assert plan.routes[0].request_id=='r0'
    assert 'fallback' in plan.reason


def test_predictor_observes_only_completed_history():
    predictor=OutputPredictor(128)
    assert predictor.predict(100)==128
    predictor.observe_completed(100,32)
    assert predictor.predict(100)==32
    assert predictor.predict(1000)==128


def runtime_module():
    path=Path(__file__).resolve().parents[3]/'vllm-pd-fork/vllm/pdblend_runtime.py'
    # parents[3] is workspace for tests/serving nested tree.
    if not path.exists():
        path=Path('/root/workspace/vllm-pd-fork/vllm/pdblend_runtime.py')
    spec=importlib.util.spec_from_file_location('tested_runtime',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_temporal_control_enforces_engine_admission_and_rejects_stale_version(tmp_path,monkeypatch):
    import json
    runtime=runtime_module()
    control=tmp_path/'control.json'
    monkeypatch.setenv('PDBLEND_RUNTIME_PATH',str(control))
    scheduler=SimpleNamespace(waiting=deque(['prefill']),scheduler_config=SimpleNamespace(chunked_prefill_enabled=True))
    scheduler._schedule_default=lambda:list(scheduler.waiting)
    control.write_text(json.dumps(dict(generation=2,role='mixed',mode='temporal',admit_prefill=False)))
    state=runtime.read_runtime(scheduler)
    assert runtime.schedule(scheduler,state)==[]
    assert list(scheduler.waiting)==['prefill']
    assert not scheduler.scheduler_config.chunked_prefill_enabled
    control.write_text(json.dumps(dict(generation=1,role='mixed',mode='temporal',admit_prefill=True)))
    assert not runtime.read_runtime(scheduler)['admit_prefill']
    assert scheduler._pdblend_runtime_error=='stale generation'


def test_transfer_identity_has_no_network_address_or_arbitrary_target():
    runtime=runtime_module()
    rid=runtime.transfer_id('abc','p','i0','i1')
    assert runtime.parse_transfer(rid)['target']=='i1'
    with pytest.raises(ValueError):
        runtime.transfer_id('abc','p','127.0.0.1:80','i1')


def test_tpot_ledger_uses_accumulated_prefix_credit_once():
    budget=RequestBudget('r',0,128,64,5,.1,emitted=10,first_token_s=1,last_token_s=1.36)
    assert budget.next_token_remaining(1.4)==pytest.approx(.6)
    # A pause consumes the same account; admission and DVFS do not receive
    # independent copies of credit. Crossing the prefix deadline exhausts it.
    assert budget.next_token_remaining(1.9)==pytest.approx(.1)
    assert budget.next_token_remaining(2.1)<0


def test_profile_decode_hold_preserves_existing_queues_and_new_prefill():
    runtime=runtime_module()
    scheduler=SimpleNamespace(waiting=deque(['new']),running=deque(['old']),swapped=deque(['swapped']),
        scheduler_config=SimpleNamespace(chunked_prefill_enabled=False))
    def prefill():
        assert not scheduler.running and not scheduler.swapped
        scheduler.running.append(scheduler.waiting.popleft())
        return ['actual-prefill']
    scheduler._schedule_default=prefill
    result=runtime.schedule(scheduler,dict(role='decode',mode='temporal',admit_prefill=True,admit_decode=False))
    assert result==['actual-prefill']
    assert list(scheduler.running)==['old','new']
    assert list(scheduler.swapped)==['swapped']
