from dataclasses import replace
import asyncio
import time

import pytest

from ecopadg.serving import admission_validation as check
from ecopadg.serving.planner import JointPlanner
from ecopadg.serving.state import StateManager
from test_planner import system


def test_hold_limit_uses_real_free_kv_and_never_more_than_six_requests():
    assert check.can_add_hold(dict(free_kv_tokens=7184),5)
    assert not check.can_add_hold(dict(free_kv_tokens=7183),5)
    assert not check.can_add_hold(dict(free_kv_tokens=100000),6)
    with pytest.raises(ValueError):check.can_add_hold(dict(free_kv_tokens=100000,waiting=1),2)
    assert check.INPUT+check.HELD_OUTPUT<=8192


def test_actual_accepting_backend_snapshot_is_required_for_capacity_attribution():
    planner,snapshot,r=system()
    planner=JointPlanner(planner.profiles,allow_pd=False,dvfs=False)
    occupied=replace(snapshot.instances[0],running=1,free_kv_tokens=16,kv_allocations=(('held',7168),))
    snapshot=replace(snapshot,instances=(occupied,))
    result=check.capacity_rejection(snapshot,r,planner,10)
    assert not result.feasible
    with pytest.raises(ValueError,match='live accepting'):
        check.capacity_rejection(replace(snapshot,instances=(replace(occupied,accepting=False),)),r,planner,10)
    with pytest.raises(ValueError,match='live accepting'):
        check.capacity_rejection(replace(snapshot,instances=(replace(occupied,free_kv_tokens=20000),)),r,planner,10)
    emptied=replace(snapshot,instances=(replace(occupied,running=0,free_kv_tokens=20000,kv_allocations=()),))
    assert planner.plan(emptied,(r,),now=10,joint=False).feasible


def valid_raw():
    control=dict(role='mixed',mode='continuous',admit_prefill=True,admit_decode=True)
    held=dict(accepting=True,admit_decode=True,free_kv_tokens=100,kv_allocations={'held':7184})
    return dict(errors=[],cleanup_errors=[],original_control=control,
        restoration=dict(control,generation=3,acknowledged_generation=3,running=0,waiting=0,active=0,
            kv_allocations={},transfer_allocations={}),reject_before=held,reject_after=held,
        probe_request_id='probe',rejected_plan=dict(feasible=False,routes=[]),
        rejected_reserved=False,reservations_after_reject={},recovered_plan=dict(feasible=True),recovered_reserved=True,
        reference=dict(token_ids=list(range(16))),recovered_output=dict(token_ids=list(range(16)),usage=dict(completion_tokens=16)),
        holds=[dict(request_id='held',after=dict(free_kv_tokens=100,kv_allocations={'held':7184}))],
        held_events=[dict(prefill=1,tokens=7168,request_ids=['held'])],clocks_written=False)


def test_raw_hardware_evidence_does_not_claim_a_full_distserve_pd_boundary():
    result=check.verify_raw(valid_raw())
    assert result['real_mixed_kv_boundary'] and result['shared_joint_planner_kv_guard']
    assert not result['distserve_pd_boundary']


@pytest.mark.parametrize('change',('decode_held','not_full','new_kv','reserved','missing_prefill','wrong_output','cleanup'))
def test_raw_passed_label_cannot_hide_missing_actual_boundary_or_cleanup(change):
    raw=valid_raw()
    if change=='decode_held':raw['reject_before']=dict(raw['reject_before'],admit_decode=False)
    elif change=='not_full':raw['reject_before']=dict(raw['reject_before'],free_kv_tokens=20000)
    elif change=='new_kv':raw['reject_after']=dict(raw['reject_after'],kv_allocations={'held':7184,'probe':7184})
    elif change=='reserved':raw['reservations_after_reject']={'probe':1}
    elif change=='missing_prefill':raw['held_events']=[]
    elif change=='wrong_output':raw['recovered_output']['token_ids']=[0]*16
    else:raw['restoration']['kv_allocations']={'held':7184}
    with pytest.raises(ValueError):check.verify_raw(raw)


def test_target_kv_and_staging_reservation_precedes_actual_producer_callback():
    planner,snapshot,r=system();now=time.time()
    snapshot=replace(snapshot,instances=tuple(replace(i,timestamp_s=now) for i in snapshot.instances))
    r=replace(r,arrival_s=now)
    plan=next(p for p in planner.candidates(snapshot,r,now) if p.routes[0].prefill_id!=p.routes[0].decode_id)
    state=StateManager(snapshot);calls=[]
    async def producer():
        target=next(i for i in state.snapshot.instances if i.instance_id==plan.routes[0].decode_id)
        assert target.reserved_kv_tokens>=plan.routes[0].reserve_tokens
        assert target.reserved_transfer_bytes>=plan.routes[0].transfer_reserve_bytes>0
        calls.append('producer');return dict(token_ids=[7])
    observation,result=asyncio.run(check.reserve_then_produce(state,plan,r,producer))
    assert calls==['producer'] and result['token_ids']==[7]
    assert observation['reserved_transfer_bytes']==plan.routes[0].transfer_reserve_bytes
    assert state.reservations[r.request_id]==plan.routes[0]


def test_failed_actual_reservation_never_invokes_producer():
    planner,snapshot,r=system();now=time.time()
    snapshot=replace(snapshot,instances=tuple(replace(i,timestamp_s=now) for i in snapshot.instances));r=replace(r,arrival_s=now)
    plan=replace(planner.plan(snapshot,(r,),now=now),feasible=False,routes=())
    state=StateManager(snapshot);calls=[]
    async def producer():calls.append('producer')
    with pytest.raises(RuntimeError,match='without confirmed target'):
        asyncio.run(check.reserve_then_produce(state,plan,r,producer))
    assert not calls and not state.reservations


def valid_pd():
    control=dict(role='mixed',mode='continuous',admit_prefill=True,admit_decode=True)
    held=dict(accepting=True,admit_decode=True,role='decode',free_kv_tokens=100,kv_allocations={'held-d':7184})
    return dict(reject_before=held,reject_after=held,new_prefill_id='new-p',rejected_until_s=2.5,
        rejected_plan=dict(feasible=False),rejected_reserved=False,reservations_after_reject={},
        producer_events=[dict(request_ids=['held-p'],started_s=1,prefill=1,tokens=7168),
            dict(request_ids=['new-p'],started_s=4,prefill=1,tokens=7168)],
        holds=[dict(request_id='held-d',producer_id='held-p',after=dict(free_kv_tokens=100,kv_allocations={'held-d':7184}))],
        recovered_plan=dict(feasible=True,routes=[dict(prefill_id='p',decode_id='d',reserve_tokens=7184,transfer_reserve_bytes=1024)]),
        target_reservation=dict(at_s=3,reserved_kv_tokens=7184,reserved_transfer_bytes=1024),
        reference_tokens=list(range(16)),recovered_output=dict(token_ids=list(range(16)),usage=dict(completion_tokens=16)),
        producer_original=control,producer_restoration=dict(control,generation=2,acknowledged_generation=2))


def test_full_pd_proof_requires_real_target_boundary_and_preproducer_reservation():
    result=check.verify_pd(valid_pd())
    assert result['distserve_pd_boundary'] and result['target_staging_reserved_before_producer']


@pytest.mark.parametrize('change',('early_producer','late_reserve','missing_staging','missing_real_hold','wrong_output','unrestored_source'))
def test_pd_proof_rejects_bad_order_unexecuted_hold_and_failed_restore(change):
    pd=valid_pd()
    if change=='early_producer':pd['producer_events'][1]['started_s']=2
    elif change=='late_reserve':pd['target_reservation']['at_s']=5
    elif change=='missing_staging':pd['target_reservation']['reserved_transfer_bytes']=0
    elif change=='missing_real_hold':pd['producer_events'].pop(0)
    elif change=='wrong_output':pd['recovered_output']['token_ids']=[9]*16
    else:pd['producer_restoration']['role']='prefill'
    with pytest.raises(ValueError):check.verify_pd(pd)


class CleanupProfiler:
    def __init__(self,*,failed_cancel=None,blocked_cancel=None,blocked_restore=None):
        self.failed_cancel=failed_cancel;self.blocked_cancel=blocked_cancel;self.blocked_restore=blocked_restore
        self.calls=[];self.controls=[]
        self.state={i:dict(role='mixed',mode='continuous',admit_prefill=True,admit_decode=True,
            generation=1,acknowledged_generation=1,active=0,running=0,waiting=0,
            kv_allocations={},transfer_allocations={}) for i in ('d','p')}
    async def call(self,instance,path,body=None):
        self.calls.append((instance['id'],path,body))
        if path=='/cancel':
            rid=body['request_id']
            if rid==self.blocked_cancel:await asyncio.Event().wait()
            if rid==self.failed_cancel:raise RuntimeError('injected first cancel failure')
            return dict(cancelled=rid,transfers=[])
        return dict(self.state[instance['id']])
    async def control(self,instance,**fields):
        self.controls.append(instance['id'])
        if instance['id']==self.blocked_restore:await asyncio.Event().wait()
        self.state[instance['id']].update(fields,generation=2,acknowledged_generation=2)


def test_first_cancel_failure_still_cancels_tasks_and_restores_both_instances():
    async def run():
        profiler=CleanupProfiler(failed_cancel='r0');d,p=dict(id='d'),dict(id='p')
        owned={'r0':d,'r1':p};local=asyncio.create_task(asyncio.Event().wait());await asyncio.sleep(0)
        result=await check.cleanup_instances(profiler,owned,[local],[(d,profiler.state['d']),(p,profiler.state['p'])],
            deadline=time.monotonic()+.5,rpc_timeout=.05,restore_timeout=.1)
        assert {body['request_id'] for _,path,body in profiler.calls if path=='/cancel'}=={'r0','r1'}
        assert local.cancelled() and set(profiler.controls)=={'d','p'}
        assert set(result['restored'])=={'d','p'} and result['errors']
        assert set(owned)=={'r0'}  # retain unconfirmed ownership for another cleanup attempt
        raw=valid_raw();raw['cleanup_errors']=result['errors']
        with pytest.raises(ValueError,match='restoration incomplete'):check.verify_raw(raw)
    asyncio.run(run())


def test_cancel_and_one_restore_timeout_do_not_block_other_instance_or_deadline():
    async def run():
        profiler=CleanupProfiler(blocked_cancel='r0',blocked_restore='d');d,p=dict(id='d'),dict(id='p')
        local=asyncio.create_task(asyncio.Event().wait());await asyncio.sleep(0)
        started=time.monotonic()
        result=await check.cleanup_instances(profiler,{'r0':d,'r1':p},[local],
            [(d,profiler.state['d']),(p,profiler.state['p'])],deadline=started+.18,rpc_timeout=.02,restore_timeout=.03)
        assert time.monotonic()-started<.18
        assert local.cancelled() and set(profiler.controls)=={'d','p'}
        assert set(result['restored'])=={'p'}
        assert any('r0' in e and 'TimeoutError' in e for e in result['errors'])
        assert any('d restoration' in e and 'TimeoutError' in e for e in result['errors'])
    asyncio.run(run())


def test_local_task_cleanup_runs_even_if_all_remote_cancellations_fail():
    async def run():
        profiler=CleanupProfiler(failed_cancel='r0');local=asyncio.create_task(asyncio.Event().wait())
        await asyncio.sleep(0)
        result=await check.cancel_requests(profiler,{'r0':dict(id='d')},[local],rpc_timeout=.02,task_timeout=.02)
        assert local.done() and local.cancelled() and len(result['errors'])==1
    asyncio.run(run())


def test_local_cancellation_wait_is_bounded_for_a_task_that_delays_first_cancel():
    async def run():
        async def delay_one_cancel():
            try:await asyncio.Event().wait()
            except asyncio.CancelledError:await asyncio.Event().wait()
        task=asyncio.create_task(delay_one_cancel());await asyncio.sleep(0)
        started=time.monotonic()
        result=await check.cancel_requests(CleanupProfiler(),{},[task],task_timeout=.01)
        await asyncio.sleep(0)
        assert time.monotonic()-started<.15 and task.cancelled()
        assert any('local request tasks' in e for e in result['errors'])
    asyncio.run(run())
