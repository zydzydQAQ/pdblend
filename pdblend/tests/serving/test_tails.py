"""Synthetic analytic energy identities; no claim of measured GPU behavior."""
import asyncio
from dataclasses import replace

import pytest

from ecopadg.serving.frequency import FrequencyPlanner,FrequencyCost
from ecopadg.serving.planner import JointPlanner,TransferCost
from ecopadg.serving.profiles import ProfilePoint,ProfileStore
from ecopadg.serving.state import StateManager
from ecopadg.serving.tails import TailModel,admission_budget
from ecopadg.serving.types import RequestBudget,InstanceState,RuntimeSnapshot


NOW=1000.


def point(role='mixed',n=4096,context=4352,batch=1,prefill=2,iteration=.01,
          frequency=2520,power=100,residency=100):
    return ProfilePoint(role,1,frequency,n,context,batch,prefill,iteration,power,residency,0,1,'cpu-identity')


def request(rid,n=4096,predicted=101,**kwargs):
    return RequestBudget(rid,NOW,n,predicted,5,.1,output_limit=predicted,**kwargs)


def snapshot(roles=('mixed','mixed')):
    return RuntimeSnapshot(1,NOW,tuple(InstanceState('i'+str(g),role,1,(g,),NOW,0,2520,
        1_000_000,0,0,free_transfer_bytes=1024**3,transfer_bytes_per_token=1024) for g,role in enumerate(roles)))


def planner(points,links=()):
    return JointPlanner(ProfileStore(points,gpu_count=8,idle_unallocated_gpu_w=10),links,dvfs=False)


def route(est,state,req,destination,source=None):
    return next(p for p in est.candidates(state,req,NOW) if p.routes[0].decode_id==destination
        and (source is None or p.routes[0].prefill_id==source))


def test_parallel_mixed_prefills_charge_node_residency_once_and_same_instance_extends_tail():
    est=planner([point(batch=b) for b in (1,2,4)]);state=snapshot()
    a=request('a');b=request('b');c=request('c')
    first=route(est,state,a,'i0');after=est.advance(state,first,a)
    assert first.routes[0].incremental_j==pytest.approx(260*3)
    parallel=route(est,after,b,'i1')
    assert parallel.routes[0].incremental_j==pytest.approx(0)
    same=route(est,after,b,'i0')
    assert same.routes[0].incremental_j==pytest.approx(260*2)
    both=est.advance(after,parallel,b)
    third=route(est,both,c,'i0')
    assert third.routes[0].incremental_j==pytest.approx(260*2)
    # An independently observed unmarked pending-prefill state has the same
    # complete remaining work, not only the request's decode duration.
    live=replace(after,instances=tuple(replace(i,requests=(a,)) if i.instance_id=='i0' else i for i in after.instances))
    assert route(est,live,b,'i1').routes[0].incremental_j==pytest.approx(0)


def test_known_prefill_queue_is_included_in_energy_not_only_ttft():
    est=planner([point(n=128,context=256,prefill=.1,iteration=.001)] +
        [point(n=7168,context=8192,batch=b,prefill=1.2,iteration=.001) for b in (1,2)])
    old=request('pending',7168,2)
    decoding=request('decoding',128,1032,emitted=32,first_token_s=NOW-1,last_token_s=NOW-.01)
    state=snapshot();state=replace(state,instances=(
        replace(state.instances[0],requests=(old,),waiting=1),
        replace(state.instances[1],requests=(decoding,),running=1)))
    incoming=request('new',128,101);plan=route(est,state,incoming,'i0')
    assert plan.routes[0].predicted_ttft_s==pytest.approx(1.301)
    assert max(TailModel(est,state,NOW).tails.values())==pytest.approx(1.201)
    assert max(TailModel(est,est.advance(state,plan,incoming),NOW).tails.values())==pytest.approx(1.4)
    assert plan.routes[0].incremental_j==pytest.approx(260*.199)


def test_pending_readiness_ages_and_runtime_reservation_matches_advance():
    async def run():
        est=planner([point(batch=b) for b in (1,2)]);state=snapshot();req=request('a')
        plan=route(est,state,req,'i0');manager=StateManager(state)
        await manager.reserve(plan,NOW,req)
        predicted=est.advance(state,plan,req)
        assert predicted.instances==manager.snapshot.instances
        admitted=manager.snapshot.instances[0].requests[0]
        assert admitted==admission_budget(plan,req)
        assert admitted.pending_ready_s==pytest.approx(NOW+2)
        assert admitted.pending_frequency_mhz==2520
        assert TailModel(est,manager.snapshot,NOW+1).tails['i0']==pytest.approx(2)
        emitted=replace(admitted,emitted=1,first_token_s=NOW+2,last_token_s=NOW+2)
        await manager.update_budget(emitted)
        assert TailModel(est,manager.snapshot,NOW+2).tails['i0']==pytest.approx(1)
    asyncio.run(run())


def test_external_clock_change_invalidates_pending_readiness_without_rewriting_request_history():
    est=planner([point(prefill=2),point(prefill=4,iteration=.02,frequency=1500)])
    state=snapshot();req=request('a');plan=route(est,state,req,'i0');admitted=est.advance(state,plan,req)
    changed=replace(admitted,instances=tuple(replace(i,frequency_mhz=1500) if i.instance_id=='i0' else i for i in admitted.instances))
    assert TailModel(est,admitted,NOW+1).tails['i0']==pytest.approx(2)
    assert TailModel(est,changed,NOW+1).tails['i0']==pytest.approx(6)  # conservative full uncompleted P + new D
    assert changed.instances[0].requests[0].pending_frequency_mhz==2520


def test_pd_clock_change_after_source_completion_uses_transport_upper_bound():
    async def run():
        est,state=pd_system();req=request('a',128,11);plan=route(est,state,req,'i2','i0')
        manager=StateManager(state);await manager.reserve(plan,NOW,req);await manager.prefill_complete('a')
        extra=point('decode',128,256,1,prefill=0,iteration=.02,frequency=1500)
        store=ProfileStore(est.profiles.points+(extra,),gpu_count=8,idle_unallocated_gpu_w=10)
        links=est.transfers+(replace(est.transfers[0],seconds_upper=.4,import_seconds_upper=.3,decode_frequency_mhz=1500),)
        changed_est=JointPlanner(store,links,dvfs=False)
        changed=replace(manager.snapshot,instances=tuple(replace(i,frequency_mhz=1500) if i.instance_id=='i2' else i for i in manager.snapshot.instances))
        assert TailModel(changed_est,changed,NOW+1).tails['i2']==pytest.approx(.4+.2)
    asyncio.run(run())


def test_commit_time_starts_the_phase_estimate_and_slack_bounds_plan_expiry():
    async def run():
        est=planner([point(batch=b) for b in (1,2)]);state=snapshot();req=request('a')
        plan=route(est,state,req,'i0');manager=StateManager(state)
        await manager.reserve(plan,NOW+.05,req)
        assert manager.snapshot.instances[0].requests[0].pending_ready_s==pytest.approx(NOW+2.05)
        tight=replace(req,ttft_s=plan.routes[0].predicted_ttft_s+.005)
        assert route(est,state,tight,'i0').expires_s==pytest.approx(NOW+.005)
        old_telemetry=replace(state,instances=tuple(replace(i,timestamp_s=NOW-.999) for i in state.instances))
        assert route(est,old_telemetry,req,'i0').expires_s==pytest.approx(NOW+.001)
    asyncio.run(run())


def pd_system():
    points=[point('prefill',128,129,b,prefill=b) for b in (1,2,4)] + [
        point('decode',128,256,b,prefill=0) for b in (1,2,4)]
    links=[TransferCost(1,1,128,.1,0,'cpu-identity',True,import_seconds_upper=.05,
        profile_batch=4,source_gpus=(p,),target_gpus=(d,)) for p in (0,1) for d in (2,3)]
    return planner(points,links),snapshot(('prefill','prefill','decode','decode'))


def test_pd_parallel_sources_share_residency_but_source_queue_and_import_serialize():
    est,state=pd_system();a=request('a',128,11);b=request('b',128,11)
    first=route(est,state,a,'i2','i0');after=est.advance(state,first,a)
    assert TailModel(est,after,NOW).tails['i2']==pytest.approx(1.2)
    assert first.routes[0].incremental_j==pytest.approx(440*1.2)
    parallel=route(est,after,b,'i3','i1')
    assert parallel.routes[0].incremental_j==pytest.approx(0)
    queued=route(est,after,b,'i3','i0')
    assert queued.routes[0].predicted_ttft_s==pytest.approx(4.11)
    assert queued.routes[0].incremental_j==pytest.approx(440*3)
    import_wait=route(est,after,b,'i2','i1')
    assert import_wait.routes[0].incremental_j==pytest.approx(440*.05)


def test_pd_prefill_completion_does_not_erase_already_promised_transfer_horizon():
    async def run():
        est,state=pd_system();req=request('a',128,11);plan=route(est,state,req,'i2','i0')
        manager=StateManager(state);await manager.reserve(plan,NOW,req)
        await manager.prefill_complete(req.request_id)
        assert not manager.snapshot.instances[0].requests
        assert TailModel(est,manager.snapshot,NOW+1).tails['i2']==pytest.approx(.2)
    asyncio.run(run())


def test_existing_decode_and_source_prefill_credit_bound_admission_expiry():
    est,state=pd_system()
    old=request('old',128,11,emitted=1,first_token_s=NOW-.035,last_token_s=NOW-.01)
    active=replace(state,instances=tuple(replace(i,requests=(old,),running=1) if i.instance_id=='i2' else i for i in state.instances))
    assert route(est,active,request('new',128,11),'i2','i0').expires_s==pytest.approx(NOW+.005)
    a=request('a',128,11);first=route(est,state,a,'i2','i0');after=est.advance(state,first,a)
    after=replace(after,instances=tuple(replace(i,requests=tuple(replace(r,ttft_s=4.015) for r in i.requests))
        if i.instance_id=='i0' else i for i in after.instances))
    assert route(est,after,request('b',128,11),'i3','i0').expires_s==pytest.approx(NOW+.005)


@pytest.mark.parametrize('role',['mixed','decode'])
def test_pending_phase_keeps_committed_clock_until_every_first_token(role):
    from test_planner import system
    est,state,req=system()
    instance=next(i for i in state.instances if i.role==role)
    first=next(p for p in est.candidates(state,req,10) if p.routes[0].decode_id==instance.instance_id)
    after=est.advance(state,first,req)
    target=next(i for i in after.instances if i.instance_id==instance.instance_id)
    candidates=[p for p in est.candidates(after,replace(req,request_id='b'),10) if p.routes[0].decode_id==target.instance_id]
    assert candidates
    assert all(next(a.frequency_mhz for a in p.frequencies if a.instance_id==target.instance_id)==target.frequency_mhz for p in candidates)
    emitted=replace(target.requests[0],emitted=10,first_token_s=9.5,last_token_s=10)
    after=replace(after,instances=tuple(replace(i,requests=(emitted,),running=1,waiting=0)
        if i.instance_id==target.instance_id else replace(i,requests=(),waiting=0) if i.role=='prefill' else i for i in after.instances))
    candidates=[p for p in est.candidates(after,replace(req,request_id='b'),10) if p.routes[0].decode_id==target.instance_id]
    assert len({next(a.frequency_mhz for a in p.frequencies if a.instance_id==target.instance_id) for p in candidates})>1


def test_frequency_uses_other_instances_pending_prefill_to_avoid_false_residency_extension():
    points=[point(n=128,context=256,prefill=.1,power=200),
        point(n=128,context=256,prefill=.2,iteration=.02,frequency=1500,power=80,residency=50),
        point(n=4096,context=4352,prefill=3)]
    est=planner(points);state=snapshot()
    active=request('a',128,110,emitted=10,first_token_s=NOW-.5,last_token_s=NOW-.01)
    background=request('b',4096,11)
    state=replace(state,instances=(replace(state.instances[0],requests=(active,),running=1),
        replace(state.instances[1],requests=(background,),waiting=1)))
    freq=FrequencyPlanner(est,[FrequencyCost(1,2520,1500,.01,0,'cpu-identity')])
    assert [(a.instance_id,a.frequency_mhz) for a in freq.plan(state,NOW).frequencies]==[('i0',1500)]
    done_prefill=replace(background,emitted=10,first_token_s=NOW-.5,last_token_s=NOW-.01)
    state=replace(state,instances=(state.instances[0],replace(state.instances[1],requests=(done_prefill,),running=1,waiting=0)))
    assert not freq.plan(state,NOW).frequencies


def test_frequency_commit_expiry_uses_prefix_slack_but_capacity_recovery_remains_available():
    from test_frequency import active_system
    est,state=active_system();instance=state.instances[0]
    costs=[FrequencyCost(1,2520,1500,.01,0,'cpu-identity')]
    measured=est.point(instance,instance.requests[0],1500,1)
    delay=measured.iteration_s*measured.bound+.01
    old=replace(instance.requests[0],first_token_s=10+delay+.005-instance.requests[0].emitted*instance.requests[0].tpot_s)
    state=replace(state,instances=(replace(instance,requests=(old,)),))
    plan=FrequencyPlanner(est,costs).plan(state,10)
    assert plan.frequencies and plan.expires_s==pytest.approx(10.005)
    stale=replace(state,instances=(replace(state.instances[0],frequency_mhz=1500,timestamp_s=8),))
    recovery=FrequencyPlanner(est,costs).plan(stale,10)
    assert recovery.frequencies[0].frequency_mhz==2520 and recovery.expires_s==11


def test_sequential_frequency_actions_spend_shared_wall_time_without_precrediting_tokens():
    from test_frequency import active_system
    est,state=active_system();first=state.instances[0]
    point=est.point(first,first.requests[0],1500,1)
    delay=.03
    def instance(name,gpu,extra):
        r=replace(first.requests[0],request_id=name,
            first_token_s=10+point.iteration_s*point.bound+delay+extra-first.requests[0].emitted*first.requests[0].tpot_s)
        return replace(first,instance_id=name,gpus=(gpu,),requests=(r,))
    state=replace(state,instances=(instance('a',0,.05),instance('b',1,.01),instance('c',2,.04)))
    freq=FrequencyPlanner(est,[FrequencyCost(1,2520,1500,delay,0,'cpu-identity')])
    plan=freq.plan(state,10)
    assert [a.instance_id for a in plan.frequencies]==['a','c']
    assert plan.expires_s==pytest.approx(10.01)  # c's .04 credit minus a's .03 switch
    # An unrelated capacity recovery is issued alone and keeps the ordinary
    # control TTL even if an energy proposal's slack is almost exhausted.
    bad=replace(state.instances[-1],frequency_mhz=1500,timestamp_s=8)
    recovery=freq.plan(replace(state,instances=state.instances[:-1]+(bad,)),10)
    assert [(a.instance_id,a.frequency_mhz) for a in recovery.frequencies]==[('c',2520)]
    assert recovery.expires_s==11
