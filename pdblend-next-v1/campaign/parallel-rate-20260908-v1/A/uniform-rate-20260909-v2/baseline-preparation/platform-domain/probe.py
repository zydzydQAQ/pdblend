"""CPU-only policy and clock-writer probes; no NVIDIA access."""
import argparse
import asyncio
from dataclasses import asdict,replace
import json
from pathlib import Path
import sys
import tempfile

parser=argparse.ArgumentParser();parser.add_argument('--host',required=True);parser.add_argument('--profile',required=True)
parser.add_argument('--maximum',type=int);args=parser.parse_args()
sys.path[:0]=[str(Path(args.host)/'src'),args.host,'/root/workspace/pdblend/.runtime-deps']
from ecopadg.serving.profiles import ProfileStore
from ecopadg.serving.types import RequestBudget,InstanceState,RuntimeSnapshot
from ecopadg.serving.planner import JointPlanner,TransferCost
from ecopadg.serving.ecoserve import EcoServeScheduler
from ecopadg.serving.distserve import DistServeScheduler
from ecopadg.serving.dynamo import DynamoScheduler
from ecopadg.serving.frequency import FrequencyPlanner
from ecopadg.serving.backend import ClockOwner,HttpEngineBackend
from ecopadg.serving.planning_executor import PlanningExecutor
from ecopadg.serving.runtime import Controller

maximum=args.maximum or 2520
kw={} if args.maximum is None else dict(max_frequency=args.maximum)
profiles=ProfileStore.load(args.profile)
request=RequestBudget('r',100.,128,64,100.,10.,output_limit=64)
def instance(i,role='mixed',**overrides):
    x=InstanceState(i,role,1,(int(i[-1]),),100.,1,maximum,1000000,0,0,
        free_transfer_bytes=1000000000,transfer_bytes_per_token=4096)
    return replace(x,**overrides)
mixed=RuntimeSnapshot(1,100.,(instance('r0'),instance('r1')))
pd=RuntimeSnapshot(1,100.,(instance('r0','prefill'),instance('r1','decode')))
transfer=TransferCost(1,1,128,.01,1.,'cpu-test-unmeasured',validated=True)
joint=JointPlanner(profiles,allow_pd=False,dvfs=False,**kw)
eco=EcoServeScheduler(profiles,['r0','r1'],**kw)
dist=DistServeScheduler(profiles,[transfer],prefill_batch=1,decode_batch=32,**kw)
dyn=DynamoScheduler(profiles,{'r0':'SS','r1':'LL'},**kw)
results={}
for label,planner,snapshot in [('mixed',joint,mixed),('ecoserve',eco,mixed),('distserve',dist,pd),('dynamollm',dyn,mixed)]:
    for case,snap,req in [('normal',snapshot,request),('tight',snapshot,replace(request,ttft_s=.0001)),
        ('no-kv',replace(snapshot,instances=tuple(replace(i,free_kv_tokens=0) for i in snapshot.instances)),request)]:
        result=planner.plan(snap,[req],now=100.)
        results[label+'-'+case]=asdict(result)
        if args.maximum:
            assert all(a.frequency_mhz<=maximum for a in result.frequencies),(label,result)
        if case=='normal':assert result.feasible,(label,result)


class FakeHardware:
    def __init__(self):self.writes=[];self.resets=[];self.observed=900
    def set_clock(self,gpu,mhz):self.writes.append((gpu,mhz))
    def current_freq(self,gpu):return self.observed
    def clock_idle(self,gpu):return False
    def reset_clock(self,gpu):self.resets.append(gpu)


async def run():
    with tempfile.TemporaryDirectory() as locks:
        hardware=FakeHardware();clock=ClockOwner(hardware,[0],lock_dir=locks,settle_timeout_s=0,**kw)
        try:
            await clock.set([0],1500)
            assert clock.applied[0]==maximum
            assert hardware.writes==[(0,1500),(0,maximum)]
            clock.applied[0]=1500;clock.deferred[0]=dict(target=1500,gpus=(0,),active_since=0)
            await clock.verify_deferred()
            assert clock.applied[0]==maximum and hardware.writes[-1]==(0,maximum)
            results['clock-fallback-writes']=hardware.writes
        finally:await clock.close()
        assert hardware.resets==[0]
    backend=HttpEngineBackend([dict(id='r0')],None,**kw)
    assert backend.frequency=={'r0':maximum}
    results['initial-backend-frequency']=backend.frequency
    worker=PlanningExecutor()
    try:
        threaded=await worker.run(eco.plan,mixed,[request],now=100.)
        assert asdict(threaded)==results['ecoserve-normal']
        assert all(a.frequency_mhz==maximum for a in threaded.frequencies)
    finally:await worker.close()
    for strategy in ('mixed','distserve','ecoserve','dynamollm-resident'):
        config=dict(strategy=strategy,profiles=args.profile,journal='/tmp/cpu-unused-journal',instances=[dict(id='r0'),dict(id='r1')],
            distserve_prefill_batch=1,distserve_decode_batch=32,dynamo_assignments={'r0':'SS','r1':'LL'})
        if args.maximum:config['max_service_frequency_mhz']=args.maximum
        controller=Controller(config)
        try:
            assert controller.planner.max_frequency==maximum
            if controller.distserve_scheduler:assert controller.distserve_scheduler.estimator.max_frequency==maximum
            if controller.dynamo_scheduler:assert controller.dynamo_scheduler.estimator.max_frequency==maximum
            if controller.eco_scheduler:
                p=controller.eco_scheduler.plan(mixed,[request],now=100.)
                assert all(a.frequency_mhz==maximum for a in p.frequencies)
        finally:await controller.planning_executor.close()
    results['controller-plumbing']=True

asyncio.run(run())
print(json.dumps(results,sort_keys=True))
