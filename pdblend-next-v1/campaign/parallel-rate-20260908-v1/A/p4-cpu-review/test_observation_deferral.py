"""Independent reproduction of A's actual busy retained-clock failure."""
import asyncio,json,sys,time
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import pytest
R=Path('/root/workspace/pdblend-next-v1')
H=R/'campaign/parallel-rate-20260908-v1/hosts/14b-fixed-p4'
sys.path[:0]=[str(H/'src'),'/root/workspace/pdblend/.runtime-deps']
from ecopadg.serving.backend import ClockOwner,HttpEngineBackend,ClockEligibilityExpired,ClockWriteUncertain
from ecopadg.serving.types import ControlPlan,FrequencyAction,RouteAction,RuntimeSnapshot,InstanceState,RequestBudget
from ecopadg.serving.runtime import Controller
from ecopadg.serving.state import StateManager

class Hardware:
    def __init__(self,observed):self.observed=observed;self.writes=[];self.reads=[];self.idle=set()
    def current_freq(self,g):self.reads.append(g);return self.observed[g]
    def clock_idle(self,g):return g in self.idle
    def set_clock(self,g,f):self.writes.append((g,f))
    def reset_clock(self,g):pass

def make(tmp_path,observed):
    hw=Hardware(observed);clock=ClockOwner(hw,tuple(observed),lock_dir=str(tmp_path),settle_timeout_s=.001)
    clock.applied={g:2520 for g in observed}
    clock.write_guard=lambda g,f,r,b=None:dict(allowed='fallback' not in r,error='pending admission retains route and clock ownership')
    return hw,clock

def plan(instances=('i',),routes=True):
    now=time.time()
    return ControlPlan(1,now,now+5,routes=(RouteAction('r','i','i',100,.2,.05,1.),) if routes else (),
                       frequencies=tuple(FrequencyAction(i,2520) for i in instances))

def backend(clock,groups):
    b=HttpEngineBackend([dict(id=i,gpus=list(g),tp=len(g),role='mixed') for i,g in groups],None,clock)
    b.exact_frequency_confirmation=True;b.unconfirmed_retained_admission_deferral=True
    return b

@pytest.mark.parametrize('observed',[{6:2415},{6:2520,7:2400}])
def test_zero_write_tp_observation_defers_and_old_plan_cannot_confirm(tmp_path,observed):
    async def run():
        hw,c=make(tmp_path,observed);b=backend(c,[('i',tuple(observed))]);p=plan()
        try:
            with pytest.raises(ClockEligibilityExpired):await b.execute(p)
            assert hw.writes==[] and c.transaction_writes==[] and c.fallbacks=={}
            event=c.coverage_limits[-1]
            assert event['measurement_confirmation'] is False and event['safely_replannable'] is True
            assert set(event['gpus'])==set(observed) and await b.confirm(p) is False
            for g in observed:hw.observed[g]=2520
            successor=plan();await b.execute(successor)
            assert await b.confirm(successor) is True and await b.confirm(p) is False
        finally:await c.close()
    asyncio.run(run())

def test_same_tp_transaction_write_prevents_deferral(tmp_path):
    async def run():
        hw,c=make(tmp_path,{6:2520,7:2415});c.applied[6]=2100
        try:
            with pytest.raises(ClockWriteUncertain):await c.set((6,7),2520,admission_observation_deferral=True)
            assert hw.writes==[(6,2520)] and not any(e['kind']=='frequency_observation_deferred' for e in c.coverage_limits)
        finally:await c.close()
    asyncio.run(run())

def test_earlier_plan_instance_write_upgrades_later_zero_write_deferral(tmp_path):
    async def run():
        hw,c=make(tmp_path,{6:2520,7:2415});c.applied[6]=2100;b=backend(c,[('first',(6,)),('i',(7,))])
        try:
            with pytest.raises(ClockWriteUncertain,match='earlier plan instance'):await b.execute(plan(('first','i')))
            assert hw.writes==[(6,2520)]
        finally:await c.close()
    asyncio.run(run())

def test_nonadmission_retains_original_hard_failure(tmp_path):
    async def run():
        hw,c=make(tmp_path,{6:2415});b=backend(c,[('i',(6,))])
        try:
            with pytest.raises(ClockWriteUncertain):await b.execute(plan(routes=False))
            assert not any(e['kind']=='frequency_observation_deferred' for e in c.coverage_limits)
        finally:await c.close()
    asyncio.run(run())

@pytest.mark.parametrize('kind',['pending','uncertain','deferred'])
def test_unresolved_or_deferred_physical_state_never_becomes_retryable(tmp_path,kind):
    async def run():
        hw,c=make(tmp_path,{6:2520,7:2415});future=asyncio.get_running_loop().create_future()
        if kind=='pending':c.pending_physical_commands[99]=future
        elif kind=='uncertain':c.physical_command_uncertainty.append(dict(physical_state_unknown=True))
        else:hw.observed[6]=300;hw.idle.add(6)
        try:
            with pytest.raises(ClockWriteUncertain):await c.set((6,7),2520,admission_observation_deferral=True)
            assert not any(e['kind']=='frequency_observation_deferred' for e in c.coverage_limits)
            if kind!='deferred':assert hw.reads==[]
        finally:future.set_result(None);await c.close()
    asyncio.run(run())

def test_actual_dispatch_rolls_back_unissued_reservation_retains_budget_and_requeues(tmp_path):
    async def run():
        config=json.loads((R/'campaign/parallel-rate-20260908-v1/A/p3/fixed-release-001/configs/alpaca.json').read_text())
        config.update(journal=str(tmp_path/'journal.jsonl'),evaluation_protocol='evaluation-v3',unconfirmed_retained_admission_deferral_v1=True)
        controller=Controller(config);original_executor=controller.planning_executor
        hw,c=make(tmp_path,{6:2415});controller.backend=backend(c,[('i',(6,))])
        now=time.time();old=RequestBudget('prior',now-1,10,30,1.,.1,30,emitted=5,first_token_s=now-.5,last_token_s=now-.01,hard_deadline_s=now+119)
        state=InstanceState('i','mixed',1,(6,),now,1,2520,100000,1,0,requests=(old,))
        controller.state=StateManager(RuntimeSnapshot(1,now,(state,)));c.state_lock=controller.state.lock
        request=RequestBudget('r',now,18,30,1.,.1,30,hard_deadline_s=now+120)
        future=asyncio.get_running_loop().create_future();controller.active['r']=dict(budget=request,future=future,route=None,client_request_id='66',timing={})
        controller.pending.put_nowait('r');deferred=asyncio.Event();pending_defer=controller.pending.defer
        def defer(rid):pending_defer(rid);deferred.set()
        controller.pending.defer=defer
        async def planning(*args,**kwargs):return replace(plan(),snapshot_version=controller.state.snapshot.version)
        controller.planning_executor=SimpleNamespace(run=planning)
        task=asyncio.create_task(controller.dispatch())
        try:
            await asyncio.wait_for(deferred.wait(),2);task.cancel()
            with pytest.raises(asyncio.CancelledError):await task
            assert controller.active['r']['route'] is None and controller.active['r']['budget'] is request
            assert request.hard_deadline_s==now+120 and not future.done() and controller.failure is None
            assert controller.state.reservations=={} and controller.state.snapshot.instances[0].requests==(old,)
            assert controller.state.snapshot.instances[0].reserved_kv_tokens==0 and controller.state.snapshot.instances[0].waiting==0
            assert hw.writes==[] and c.coverage_limits[-1]['measurement_confirmation'] is False
        finally:
            if not task.done():task.cancel();await asyncio.gather(task,return_exceptions=True)
            await original_executor.close();await c.close()
    asyncio.run(run())
