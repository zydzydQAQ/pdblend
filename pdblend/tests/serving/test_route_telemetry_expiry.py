"""Do not let a stopped, drained replica expire unrelated healthy routes."""
import asyncio
from dataclasses import replace
import pytest

from ecopadg.serving.dynamo import DynamoScheduler
from ecopadg.serving.planner import JointPlanner,TransferCost
from ecopadg.serving.profiles import ProfilePoint,ProfileStore
from ecopadg.serving.state import StateManager,StalePlan
from ecopadg.serving.types import InstanceState,RuntimeSnapshot,RequestBudget


def fixture():
    store=ProfileStore([ProfilePoint(role,tp,2520,4096,8192,32,.04,.02,200,30,.04,1,'synthetic',.04)
        for tp in (1,2) for role in ('mixed','prefill','decode')])
    request=RequestBudget('arriving',10,512,64,5,.1,output_limit=192)
    mixed=InstanceState('healthy','mixed',1,(2,),10,0,2520,40000,0,0)
    stopped=InstanceState('replacing','mixed',2,(0,1),8,0,2520,40000,0,0,accepting=False)
    return store,request,mixed,stopped


def test_dynamo_healthy_ll_admitted_while_drained_lm_is_physically_replaced():
    store,request,mixed,stopped=fixture()
    scheduler=DynamoScheduler(store,{'healthy':'LL','replacing':'LM'})
    snapshot=RuntimeSnapshot(17,10,(mixed,stopped))
    plan=scheduler.plan(snapshot,(request,),now=10)
    assert plan.feasible and plan.routes[0].decode_id=='healthy'
    assert plan.expires_s==pytest.approx(11)
    state=StateManager(snapshot)
    assert asyncio.run(state.reserve(plan,10.01,request)) is True
    assert state.reservations['arriving'].reserve_tokens==704


def test_healthy_route_expiry_ignores_unrelated_drained_replica_timestamp():
    store,request,mixed,stopped=fixture();planner=JointPlanner(store,allow_pd=False,dvfs=False)
    for accepting in (False,True):
        # Even a stale "accepting" flag cannot create a path for the old replica.
        plan=planner.plan(RuntimeSnapshot(1,10,(mixed,replace(stopped,accepting=accepting))),
                          (request,),now=10)
        assert plan.feasible and plan.routes[0].decode_id=='healthy' and plan.expires_s>10


def test_unrelated_active_request_still_constrains_shared_remaining_work_horizon():
    store,request,mixed,stopped=fixture();planner=JointPlanner(store,allow_pd=False,dvfs=False)
    old=RequestBudget('old',9,512,64,5,.1,emitted=2,first_token_s=9.8,last_token_s=9.99)
    # We cannot classify a replica with live work as a harmless stopped replica.
    assert not planner.candidates(RuntimeSnapshot(1,10,(mixed,replace(stopped,requests=(old,)))),request,10)
    fresh=replace(stopped,timestamp_s=9.2,requests=(old,))
    plans=planner.candidates(RuntimeSnapshot(1,10,(mixed,fresh)),request,10)
    assert plans and all(p.expires_s<=10.2 for p in plans)


def test_existing_mixed_request_slack_remains_an_independent_expiry_limit():
    store,request,mixed,stopped=fixture();planner=JointPlanner(store,allow_pd=False,dvfs=False)
    old=RequestBudget('old',9,512,64,5,.1,emitted=2,first_token_s=9.5,last_token_s=9.9)
    mixed=replace(mixed,requests=(old,),running=1)
    # Prefix allowance 9.5 + (emitted+1)*.1 - now is exhausted.
    assert not planner.candidates(RuntimeSnapshot(1,10,(mixed,stopped)),request,10)


@pytest.mark.parametrize('stale_side',('prefill','decode'))
def test_pd_never_ignores_stale_source_or_destination(stale_side):
    store,request,_,stopped=fixture();planner=JointPlanner(store,[TransferCost(1,1,4096,.01,1,'synthetic',True,profile_batch=32)],dvfs=False)
    source=InstanceState('p','prefill',1,(2,),10,0,2520,40000,0,0,free_transfer_bytes=2**30,transfer_bytes_per_token=4096)
    target=replace(source,instance_id='d',role='decode',gpus=(3,))
    instances=tuple(replace(i,timestamp_s=8) if i.role==stale_side else i for i in (source,target))
    assert not planner.candidates(RuntimeSnapshot(1,10,instances+(stopped,)),request,10)


@pytest.mark.parametrize('limiting_side',('prefill','decode'))
def test_pd_uses_both_endpoint_ttl_and_reservation_rejects_expired_commit(limiting_side):
    store,request,_,stopped=fixture();planner=JointPlanner(store,[TransferCost(1,1,4096,.01,1,'synthetic',True,profile_batch=32)],dvfs=False)
    source=InstanceState('p','prefill',1,(2,),10,0,2520,40000,0,0,free_transfer_bytes=2**30,transfer_bytes_per_token=4096)
    target=replace(source,instance_id='d',role='decode',gpus=(3,))
    instances=tuple(replace(i,timestamp_s=9.25) if i.role==limiting_side else i for i in (source,target))
    snapshot=RuntimeSnapshot(1,10,instances+(stopped,))
    plan=planner.plan(snapshot,(request,),now=10)
    assert plan.feasible and plan.expires_s==pytest.approx(10.25)
    with pytest.raises(StalePlan):asyncio.run(StateManager(snapshot).reserve(plan,10.3,request))


def test_live_path_snapshot_version_race_still_rejects():
    store,request,mixed,stopped=fixture();planner=JointPlanner(store,allow_pd=False,dvfs=False)
    snapshot=RuntimeSnapshot(1,10,(mixed,stopped));plan=planner.plan(snapshot,(request,),now=10)
    with pytest.raises(StalePlan):asyncio.run(StateManager(replace(snapshot,version=2)).reserve(plan,10.01,request))
