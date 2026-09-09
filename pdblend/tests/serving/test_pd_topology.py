import asyncio
from dataclasses import replace
import time
from types import SimpleNamespace

from ecopadg.serving.pd_topology import PDBlendTopologyPlanner,MeasuredCapacity
from ecopadg.serving.profiles import ProfilePoint,ProfileStore
from ecopadg.serving.planner import JointPlanner
from ecopadg.serving.dynamo_topology import TopologyCost
from ecopadg.serving.forecast import RoleForecast
from ecopadg.serving.runtime import Controller
from ecopadg.serving.topology import InstanceSpec
from ecopadg.serving.types import InstanceState,RuntimeSnapshot,RequestBudget


def fixture():
    profiles=ProfileStore([ProfilePoint('mixed',tp,2520,1024,2048,b,.1,step,watts,80*tp,0,3,'measured')
        for tp,watts,step in ((1,300,.09),(2,350,.01)) for b in (1,4,8)],idle_unallocated_gpu_w=35)
    snapshot=RuntimeSnapshot(1,60,tuple(InstanceState(chr(97+g),'mixed',1,(g,),60,0,2520,20000,0,0)
                                      for g in range(3)))
    requests=tuple(RequestBudget(str(i),60,128,64,5,.1,output_limit=64) for i in range(3))
    forecast=RoleForecast(requests,100,300,60,1,2)
    planner=PDBlendTopologyPlanner(JointPlanner(profiles),
        [TopologyCost((1,1),(2,),1,1,'measured')],
        [MeasuredCapacity(2,20000,4*1024**3,100000,'measured')],budget_s=2)
    return planner,snapshot,forecast


def test_slow_joint_search_requires_amortization_and_keeps_transition_capacity():
    planner,snapshot,forecast=fixture()
    proposal=planner.choose(snapshot,forecast,60)
    assert proposal and proposal['savings_lower_j']>proposal['cost_upper_j']
    assert proposal['remove_ids']==('a','b')
    assert proposal['replacements']==(dict(tp=2,gpus=(0,1),role='mixed'),)
    assert planner.choose(replace(snapshot,instances=snapshot.instances[:2]),forecast,60) is None
    planner.costs=(TopologyCost((1,1),(2,),1,1e9,'measured'),)
    assert planner.choose(snapshot,forecast,60) is None


def test_live_kv_and_unmeasured_tp_are_never_replaced():
    planner,snapshot,forecast=fixture()
    busy=replace(snapshot,instances=tuple(replace(i,transfer_allocations=(('live',256),))
                                         for i in snapshot.instances))
    assert planner.choose(busy,forecast,60) is None
    planner.costs=(TopologyCost((1,1),(4,),1,1,'measured'),)
    assert planner.choose(snapshot,forecast,60) is None


def test_failed_slow_recovery_does_not_unfreeze_from_stale_backend_state(tmp_path):
    async def run():
        planner,snapshot,forecast=fixture();now=time.time()
        snapshot=replace(snapshot,timestamp_s=now,instances=tuple(replace(i,timestamp_s=now) for i in snapshot.instances))
        forecast=replace(forecast,requests=tuple(replace(r,arrival_s=now) for r in forecast.requests))
        c=Controller(dict(strategy='mixed',journal=str(tmp_path/'journal')))
        c.pd_topology=planner;c.state.snapshot=snapshot
        events=[]
        async def emit(event): events.append(event)
        async def refresh(): pass
        async def failed(*args,**kwargs): raise RuntimeError('replacement and rollback failed')
        c.refresh=refresh;c.journal=SimpleNamespace(emit=emit)
        specs={i.instance_id:InstanceSpec(i.instance_id,i.tp,i.gpus,18100+k,19000+k*16)
               for k,i in enumerate(snapshot.instances)}
        c.topology_manager=SimpleNamespace(specs=specs,node_gpus=tuple(range(8)),version=0,reconfigure=failed)
        c.backend=SimpleNamespace(instances={i:dict(accepting=True) for i in specs})
        await c.pd_slow(forecast)
        assert c.frozen_instances=={'a','b'}
        assert any(e['kind']=='pd_topology_failure' for e in events)
    asyncio.run(run())
