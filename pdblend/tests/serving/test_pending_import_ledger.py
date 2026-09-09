import asyncio
from dataclasses import replace

import pytest

from ecopadg.serving.planner import JointPlanner, TransferCost
from ecopadg.serving.profiles import ProfilePoint, ProfileStore
from ecopadg.serving.state import StateManager
from ecopadg.serving.types import InstanceState, RequestBudget, RuntimeSnapshot


def fixture():
    points=[ProfilePoint('prefill',1,2520,128,129,1,.1,0,200,30,0,1,'fixture'),
            ProfilePoint('decode',1,2520,128,1024,8,0,.05,180,30,0,1,'fixture')]
    planner=JointPlanner(ProfileStore(points),[TransferCost(1,1,128,.2,1,'fixture',True,.2)],
                         dvfs=False,frequency_costs=[])
    p=InstanceState('p','prefill',1,(0,),10,0,2520,10000,0,0)
    d=InstanceState('d','decode',1,(1,),10,0,2520,10000,0,0,
                    free_transfer_bytes=2**30,transfer_bytes_per_token=206848)
    request=RequestBudget('new',10,128,64,5,.1)
    return planner,RuntimeSnapshot(1,10,(p,d)),request


def test_two_pending_imports_cannot_spend_the_same_decode_credit():
    planner,snapshot,request=fixture()
    old=replace(request,request_id='old',emitted=4,first_token_s=10)
    pending=replace(request,request_id='queued',pending_import_s=.2)
    d=replace(snapshot.instances[1],requests=(old,pending),running=1,waiting=1)
    snapshot=replace(snapshot,instances=(snapshot.instances[0],d))
    assert old.next_token_remaining(10)==pytest.approx(.4)
    # .2 s promised import + .2 s new import + .05 s token exceeds .4 s.
    assert not planner.plan(snapshot,(request,),now=10).feasible
    emitted=replace(pending,emitted=35,first_token_s=10)
    snapshot=replace(snapshot,instances=(snapshot.instances[0],replace(d,requests=(old,emitted),running=2,waiting=0)))
    assert planner.plan(snapshot,(request,),now=10).feasible


def test_online_and_lookahead_keep_the_same_measured_import_debt():
    async def run():
        planner,snapshot,request=fixture()
        plan=planner.plan(snapshot,(request,),now=10)
        assert plan.routes[0].import_block_s==pytest.approx(.2)
        predicted=planner.advance(snapshot,plan,request)
        manager=StateManager(snapshot)
        await manager.reserve(plan,10,request)
        assert [i.requests for i in predicted.instances]==[i.requests for i in manager.snapshot.instances]
        assert manager.snapshot.instances[1].requests[0].pending_import_s==pytest.approx(.2)
        await manager.update_budget(replace(request,emitted=1,first_token_s=10.3,pending_import_s=0))
        assert manager.snapshot.instances[1].requests[0].pending_import_s==0
    asyncio.run(run())
