"""Pre-action expiry is retryable; no hardware is used by these regressions."""
import asyncio
from dataclasses import replace
import time
from types import SimpleNamespace

import pytest
from ecopadg.serving import backend as backend_module
from ecopadg.serving.backend import HttpEngineBackend
from ecopadg.serving.controller import Controller
from ecopadg.serving.state import StateManager,ExpiredPlan,StalePlan
from ecopadg.serving.types import ControlPlan,FrequencyAction,RouteAction
from test_planner import system


async def stop(c,task):
    task.cancel()
    try:await task
    except asyncio.CancelledError:pass
    await c.planning_executor.close()


@pytest.mark.parametrize('pd',[False,True])
@pytest.mark.parametrize('published',[False,True])
def test_unissued_rollback_restores_reservations_without_removing_raw_waiting(pd,published):
    async def run():
        _,snapshot,request=system()
        snapshot=replace(snapshot,instances=tuple(replace(i,waiting=2) for i in snapshot.instances))
        state=StateManager(snapshot)
        source,target=('1','2') if pd else ('0','0')
        route=RouteAction(request.request_id,source,target,384,.2,.04,1,
                          prefill_reserve_tokens=144 if pd else 0,transfer_reserve_bytes=100 if pd else 0)
        plan=ControlPlan(snapshot.version,10,11,routes=(route,))
        await state.reserve(plan,10,request)
        if published:
            # Real engine queue grows after reservation. The max(raw,local)
            # projection need not contain an additive synthetic waiting unit.
            raw=tuple(replace(i,waiting=4) for i in state.snapshot.instances)
            await state.publish(raw,10.1,engine_waiting={i.instance_id:4 for i in raw})
        await state.release(request.request_id,unissued=True)
        assert not state.reservations
        for i in state.snapshot.instances:
            assert i.waiting==(4 if published else 2)
            assert not i.requests and i.reserved_kv_tokens==i.reserved_transfer_bytes==0
        await state.release(request.request_id,unissued=True) # cancellation is idempotent
        assert all(i.waiting==(4 if published else 2) for i in state.snapshot.instances)
    asyncio.run(run())


def test_publish_preserves_real_waiting_floor_below_local_pending_projection():
    async def run():
        _,snapshot,request=system();state=StateManager(snapshot)
        route=RouteAction(request.request_id,'0','0',384,.2,.04,1)
        await state.reserve(ControlPlan(snapshot.version,10,11,routes=(route,)),10,request)
        rows=tuple(replace(i,waiting=3) if i.instance_id=='0' else i for i in state.snapshot.instances)
        await state.publish(rows,10.1,engine_waiting={'0':2,'1':0,'2':0})
        await state.release(request.request_id,unissued=True)
        assert state.snapshot.instances[0].waiting==2
    asyncio.run(run())


def test_backend_marks_only_expiry_before_any_action_as_retryable(monkeypatch):
    async def run():
        clock=[2.];calls=[]
        monkeypatch.setattr(backend_module,'time',SimpleNamespace(time=lambda:clock[0]))
        class Clock:
            applied={0:2520,1:2520};fallbacks={}
            async def set(self,gpus,desired):
                calls.append((tuple(gpus),desired));self.applied[gpus[0]]=desired
                if len(calls)==2:raise RuntimeError('second action failed after first applied')
        backend=HttpEngineBackend([dict(id=str(i),gpus=[i]) for i in range(2)],None,Clock())
        plan=ControlPlan(1,0,1,frequencies=(FrequencyAction('0',1500),FrequencyAction('1',1500)))
        with pytest.raises(ExpiredPlan):await backend.execute(plan)
        assert issubclass(ExpiredPlan,StalePlan) and not calls and backend.inflight_actions==0
        clock[0]=0
        with pytest.raises(RuntimeError,match='second action failed') as caught:await backend.execute(plan)
        assert not isinstance(caught.value,StalePlan)
        assert len(calls)==2 and backend.frequency['0']==1500 and backend.inflight_actions==0
    asyncio.run(run())


@pytest.mark.parametrize('pd',[False,True])
@pytest.mark.parametrize('first_feasible',[False,True])
def test_dispatch_replans_expiry_after_reserve_without_duplicate_issue(tmp_path,monkeypatch,pd,first_feasible):
    async def run():
        _,snapshot,request=system();now=time.time();request=replace(request,arrival_s=now,ttft_s=10)
        c=Controller(dict(strategy='mixed_dvfs',journal=str(tmp_path/'journal')))
        c.state=StateManager(replace(snapshot,timestamp_s=now))
        clock=[now];attempts=[];clock_writes=[];plans=[]
        monkeypatch.setattr(backend_module,'time',SimpleNamespace(time=lambda:clock[0]))
        source,target=('1','2') if pd else ('0','0')
        route=RouteAction(request.request_id,source,target,384,.2,.04,1,
            prefill_reserve_tokens=144 if pd else 0,transfer_reserve_bytes=100 if pd else 0)
        def plan(state,pending,**kwargs):
            plans.append((state,pending[0]))
            assert pending[0]==request # failed committed slack must not persist
            if len(plans)>1:
                assert c.active[request.request_id]['route'] is None
                assert not state.instances[0].waiting and not c.state.reservations
                assert all(not i.requests and not i.reserved_kv_tokens and not i.reserved_transfer_bytes for i in state.instances)
            feasible=first_feasible if len(plans)==1 else True
            return ControlPlan(state.version,now,now+100,routes=(route,) if feasible else (),
                frequencies=(FrequencyAction(target,1500),),feasible=feasible)
        class Clock:
            applied={i:2520 for i in range(3)};fallbacks={}
            async def set(self,gpus,desired):clock_writes.append((tuple(gpus),desired));self.applied[gpus[0]]=desired
        class Backend(HttpEngineBackend):
            async def execute(self,plan):
                attempts.append(plan)
                if len(attempts)==1:
                    if plan.feasible:assert request.request_id in c.state.reservations
                    clock[0]=now+101 # exactly the reserve→backend check race
                else:clock[0]=now
                return await super().execute(plan)
        c.backend=Backend([dict(id=str(i),gpus=[i]) for i in range(3)],None,Clock())
        c.planner=SimpleNamespace(plan=plan)
        future=asyncio.get_running_loop().create_future()
        c.active[request.request_id]=dict(budget=request,route=None,future=future,client_request_id='test')
        c.pending.put_nowait(request.request_id);task=asyncio.create_task(c.dispatch())
        try:
            issued=await asyncio.wait_for(future,1)
            assert issued==route and len(attempts)==len(plans)==2 and len(clock_writes)==1
            assert c.failure is None and list(c.state.reservations)==[request.request_id]
            assert c.active[request.request_id]['budget'].pending_ready_s is not None
            assert c.planning_stats.summary()['counts']['discarded_stale']==1
            # Issuance is the single future completion, never first stale plan.
            assert sum(i.waiting for i in c.state.snapshot.instances)==(2 if pd else 1)
        finally:await stop(c,task)
    asyncio.run(run())


def test_true_backend_failure_releases_unissued_state_but_is_not_retried(tmp_path):
    async def run():
        _,snapshot,request=system();now=time.time();request=replace(request,arrival_s=now)
        c=Controller(dict(strategy='mixed_dvfs',journal=str(tmp_path/'journal')));c.state=StateManager(snapshot)
        calls=[]
        def plan(state,pending,**kwargs):
            calls.append(1)
            return ControlPlan(state.version,now,now+10,routes=(RouteAction(request.request_id,'0','0',384,.2,.04,1),))
        class Backend:
            async def execute(self,plan):raise RuntimeError('actual hardware failure')
        c.backend=Backend();c.planner=SimpleNamespace(plan=plan)
        future=asyncio.get_running_loop().create_future();active=dict(budget=request,route=None,future=future,client_request_id='test')
        c.active[request.request_id]=active;c.pending.put_nowait(request.request_id);task=asyncio.create_task(c.dispatch())
        try:
            with pytest.raises(RuntimeError,match='actual hardware failure'):await asyncio.wait_for(future,1)
            assert len(calls)==1 and active['route'] is None and active['budget']==request
            assert not c.state.reservations and all(not i.waiting and not i.requests for i in c.state.snapshot.instances)
            assert not c.pending.qsize()
        finally:await stop(c,task)
    asyncio.run(run())


def test_http_cancellation_before_route_drops_unissued_waiting_and_no_retry(tmp_path,monkeypatch):
    async def run():
        _,snapshot,request=system();c=Controller(dict(strategy='mixed_dvfs',journal=str(tmp_path/'journal'),slo_ttft_s=5,slo_tpot_s=.1))
        c.state=StateManager(snapshot);entered=asyncio.Event();release=asyncio.Event();attempts=[]
        def plan(state,pending,**kwargs):
            r=pending[0];now=time.time()
            return ControlPlan(state.version,now,now+1,routes=(RouteAction(r.request_id,'0','0',384,.2,.04,1),))
        class Backend:
            frequency={};parked=set();frequency_outcomes=[]
            async def execute(self,plan):
                attempts.append(plan);entered.set();await release.wait()
                raise ExpiredPlan('no actions started')
        class Request:
            headers={}
            async def json(self):return dict(prompt=[1,2],max_tokens=2)
        c.planner=SimpleNamespace(plan=plan);c.backend=Backend()
        dispatcher=asyncio.create_task(c.dispatch());handler=asyncio.create_task(c.completions(Request()))
        try:
            await asyncio.wait_for(entered.wait(),1)
            assert c.state.snapshot.instances[0].waiting==1
            handler.cancel()
            with pytest.raises(asyncio.CancelledError):await handler
            assert not c.active and not c.state.reservations and c.state.snapshot.instances[0].waiting==0
            release.set();await asyncio.sleep(.02)
            assert len(attempts)==1 and c.failure is None and not c.pending.qsize()
        finally:
            release.set();await stop(c,dispatcher)
    asyncio.run(run())


@pytest.mark.parametrize('risk',[False,True])
@pytest.mark.parametrize('expired',[False,True])
def test_telemetry_expiry_skips_only_no_action_plan(tmp_path,risk,expired):
    async def run():
        _,snapshot,_=system();now=time.time();snapshot=replace(snapshot,instances=tuple(
            replace(i,timestamp_s=now-2 if risk else now) for i in snapshot.instances))
        c=Controller(dict(strategy='mixed_dvfs',journal=str(tmp_path/'journal'),park_idle=False))
        c.state=StateManager(snapshot);ticks=iter([True,False]);called=[]
        async def tick(_):return next(ticks)
        async def refresh():pass
        c.control_tick=tick;c.refresh=refresh
        class Backend:
            instances={i.instance_id:{} for i in snapshot.instances}
            async def verify_clocks(self):pass
            async def execute(self,plan):
                called.append(plan)
                raise ExpiredPlan('before actions') if expired else RuntimeError('actual failure')
        c.backend=Backend();c.last_frequency_optimization_s=0
        c.frequency_planner=SimpleNamespace(plan=lambda snap,t:ControlPlan(snap.version,t,t+1,frequencies=(FrequencyAction('0',1500),)))
        try:
            await c.telemetry()
            assert len(called)==1
            assert c.failure is None if expired else c.failure=='actual failure'
        finally:await c.planning_executor.close()
    asyncio.run(run())


def test_dynamo_expired_frequency_cycle_records_no_executed_action(tmp_path):
    async def run():
        c=Controller(dict(strategy='mixed',journal=str(tmp_path/'journal')));ticks=iter([True,False]);events=[]
        async def tick(_):return next(ticks)
        async def emit(event):events.append(event)
        class Backend:
            async def execute(self,plan):raise ExpiredPlan('before actions')
        c.control_tick=tick;c.journal=SimpleNamespace(emit=emit);c.backend=Backend();c.slow_task=None
        c.dynamo_scheduler=SimpleNamespace(hierarchy=SimpleNamespace(due=lambda now:['ScaleFreq'],PERIODS={'ScaleFreq':5}),
            frequency_plan=lambda snap,now:ControlPlan(snap.version,now,now+1))
        await c.dynamo_control()
        assert c.failure is None and len(events)==1
        assert events[0]['operation']=='ScaleFreq' and events[0]['period_s']==5 and events[0]['executed'] is False
        await c.planning_executor.close()
    asyncio.run(run())


def test_eco_expired_membership_restores_uncommitted_logical_groups(tmp_path):
    from ecopadg.serving.ecoserve import EcoServeScheduler
    async def run():
        _,snapshot,_=system();now=time.time()
        snapshot=replace(snapshot,instances=tuple(replace(i,role='mixed',timestamp_s=now) for i in snapshot.instances))
        c=Controller(dict(strategy='mixed',journal=str(tmp_path/'journal'),slo_ttft_s=1));c.state=StateManager(snapshot)
        c.eco_scheduler=EcoServeScheduler(c.profiles,['0','1']);before=(list(c.eco_scheduler.groups),dict(c.eco_scheduler.selected),c.eco_scheduler.version)
        c.ttft_history.append((now,2));ticks=iter([True,False])
        async def tick(_):return next(ticks)
        class Backend:
            async def execute(self,plan):raise ExpiredPlan('before actions')
        c.control_tick=tick;c.backend=Backend()
        await c.eco_resize()
        assert c.failure is None
        assert (c.eco_scheduler.groups,c.eco_scheduler.selected,c.eco_scheduler.version)==before
        await c.planning_executor.close()
    asyncio.run(run())
