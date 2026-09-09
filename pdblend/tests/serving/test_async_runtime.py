import asyncio
from dataclasses import replace
import time
import threading
from types import SimpleNamespace

import pytest
from ecopadg.measure.backends import FakeBackend
from ecopadg.serving.backend import ClockOwner
from ecopadg.serving.observe import Journal
from ecopadg.serving.state import StateManager
from ecopadg.serving.controller import Controller
from ecopadg.serving.tails import admission_budget
from ecopadg.serving.types import ControlPlan, FrequencyAction, RouteAction
from test_planner import system


def test_clock_owner_is_exclusive_and_blocking_work_leaves_loop_responsive(tmp_path):
    class SlowHardware(FakeBackend):
        def set_clock(self,gpu,freq):
            time.sleep(.03)
            super().set_clock(gpu,freq)
    async def run():
        hw=SlowHardware()
        owner=ClockOwner(hw,[0],tmp_path)
        with pytest.raises(RuntimeError):
            ClockOwner(hw,[0],tmp_path)
        done=[]
        async def heartbeat():
            await asyncio.sleep(.005)
            done.append(True)
        setter=asyncio.create_task(owner.set([0],1500))
        await heartbeat()
        assert done and not setter.done()
        await setter
        assert hw.locked[0]==1500
        await owner.close()
        assert not hw.locked
        next_owner=ClockOwner(hw,[0],tmp_path)
        await next_owner.close()
    asyncio.run(run())


def test_journal_flushes_before_shutdown(tmp_path):
    async def run():
        log=Journal(tmp_path/'events.jsonl',capacity=2)
        task=asyncio.create_task(log.run())
        for i in range(5):
            await log.emit(dict(sequence=i))
        await log.close()
        await task
        return (tmp_path/'events.jsonl').read_text().splitlines()
    assert len(asyncio.run(run()))==5


def test_failed_journal_wakes_a_producer_waiting_on_its_full_queue(tmp_path):
    class BrokenJournal(Journal):
        def write(self,batch):
            time.sleep(.02)
            raise OSError('disk unavailable')
    async def run():
        journal=BrokenJournal(tmp_path/'events.jsonl',capacity=1)
        await journal.emit({'id':1})
        writer=asyncio.create_task(journal.run())
        await asyncio.sleep(.005)
        await journal.emit({'id':2})
        with pytest.raises(RuntimeError,match='journal failed'):
            await asyncio.wait_for(journal.emit({'id':3}),.2)
        with pytest.raises(RuntimeError,match='journal failed'): await journal.flush()
        with pytest.raises(OSError): await writer
    asyncio.run(run())


def test_actual_engine_allocations_are_not_reserved_twice():
    async def run():
        planner,snapshot,request=system()
        plan=planner.plan(snapshot,(request,),now=10,joint=False)
        state=StateManager(snapshot)
        await state.reserve(plan,10,request)
        route=plan.routes[0]
        instances=tuple(replace(i,free_kv_tokens=i.free_kv_tokens-128,
                                kv_allocations=((request.request_id,128),))
                        if i.instance_id==route.decode_id else i for i in snapshot.instances)
        updated=await state.publish(instances,10.1)
        d=next(i for i in updated.instances if i.instance_id==route.decode_id)
        assert d.reserved_kv_tokens==route.reserve_tokens-128
        assert d.free_kv_tokens-d.reserved_kv_tokens==20000-route.reserve_tokens
        await state.release(request.request_id)
        assert all(i.reserved_kv_tokens==0 for i in state.snapshot.instances)
    asyncio.run(run())


def test_new_clock_intent_invalidates_queued_idle_parking(tmp_path):
    class SlowReset(FakeBackend):
        def reset_clock(self,gpu):
            time.sleep(.03)
            super().reset_clock(gpu)
    async def run():
        hw=SlowReset(gpu_count=2)
        owner=ClockOwner(hw,[0,1],tmp_path)
        await owner.set([0,1],1500)
        expected=dict(owner.epochs)
        parking=asyncio.create_task(owner.park([0,1],expected))
        await asyncio.sleep(.005)
        restore=asyncio.create_task(owner.set([1],2520))
        assert not await parking
        await restore
        assert hw.locked[1]==2520
        assert 0 not in hw.locked
        await owner.close()
    asyncio.run(run())


def test_refresh_does_not_double_count_running_request_without_first_token(tmp_path):
    async def run():
        planner,snapshot,request=system()
        plan=planner.plan(snapshot,(request,),now=10)
        route=plan.routes[0]
        instances=tuple(replace(i,running=1,waiting=0,
                        kv_allocations=((request.request_id,128),))
                        if i.instance_id==route.decode_id else i for i in snapshot.instances)
        class Backend:
            async def read_state(self):
                return replace(snapshot,instances=instances)
        c=Controller(dict(strategy='mixed',journal=str(tmp_path/'journal'),decision_budget_s=.02))
        assert c.planner.decision_budget_s==.02
        c.backend=Backend()
        c.active[request.request_id]=dict(budget=request,route=route)
        await c.refresh()
        d=next(i for i in c.state.snapshot.instances if i.instance_id==route.decode_id)
        assert d.running+d.waiting==1
        assert d.requests==(request,)
    asyncio.run(run())


def test_prefill_completion_releases_only_source_reservation_immediately():
    async def run():
        planner,snapshot,request=system()
        snapshot=replace(snapshot,instances=tuple(i for i in snapshot.instances if i.role!='mixed'))
        plan=planner.plan(snapshot,(request,),now=10)
        state=StateManager(snapshot)
        await state.reserve(plan,10,request)
        await state.prefill_complete(request.request_id)
        p,d=state.snapshot.instances
        assert p.role=='prefill' and not p.requests and not p.reserved_kv_tokens and not p.waiting
        assert d.requests==(admission_budget(plan,request),) and d.reserved_kv_tokens>0
    asyncio.run(run())


def test_uncertain_sm_counter_restores_capacity_without_node_outage(tmp_path):
    class PowerCapped(FakeBackend):
        def current_freq(self,gpu):
            return 1200
    async def run():
        hw=PowerCapped()
        owner=ClockOwner(hw,[0],tmp_path,settle_timeout_s=.001)
        await owner.set([0],2100)
        assert owner.applied[0]==2520 and hw.locked[0]==2520
        assert owner.fallbacks[0]['requested']==2100
        await owner.close()
    asyncio.run(run())


def test_measurement_close_wakes_control_timer_and_waits_started_change(tmp_path):
    async def run():
        c=Controller(dict(strategy='mixed',journal=str(tmp_path/'journal')))
        c.backend=SimpleNamespace(last_action_finished_s=0)
        finished=[]
        async def change():
            await asyncio.sleep(.03)
            finished.append(time.time())
        c.role_task=asyncio.create_task(c.control_tick(1800))
        c.slow_task=asyncio.create_task(change())
        start=time.time()
        boundary=await asyncio.wait_for(c.quiesce_controls(),1)
        assert finished and boundary>=finished[0]>=start
        assert c.role_task.result() is False
        assert not c.slow_pending
        # An idle control period adds no artificial 30-minute energy tail.
        assert await c.quiesce_controls() is None
    asyncio.run(run())


def test_idle_clock_is_verified_under_work_and_power_cap_restores_whole_group(tmp_path):
    class IdleHardware(FakeBackend):
        idle=True
        capped=False
        def current_freq(self,gpu):
            return 300 if self.idle else 1200 if self.capped else super().current_freq(gpu)
        def clock_idle(self,gpu):
            return self.idle
    async def run():
        hw=IdleHardware(gpu_count=2)
        owner=ClockOwner(hw,[0,1],tmp_path,settle_timeout_s=.001)
        try:
            await owner.set([0,1],1500)
            assert owner.applied=={0:1500,1:1500} and set(owner.deferred)=={0,1}
            assert not await owner.verify_deferred() and not owner.fallbacks
            hw.idle=False
            assert not await owner.verify_deferred() and not owner.deferred
            hw.idle=True
            await owner.park([0,1]);await owner.set([0,1],2100)
            hw.idle=False;hw.capped=True
            await owner.verify_deferred();await asyncio.sleep(.003)
            failed=await owner.verify_deferred()
            assert set(failed)=={0,1} and not owner.deferred
            assert owner.applied=={0:2520,1:2520}
            hw.idle=True
            await owner.park([0,1]);await owner.set([0,1],1500)
            await owner.park([0,1])
            assert not owner.deferred and not await owner.verify_deferred()
        finally: await owner.close()
    asyncio.run(run())


@pytest.mark.parametrize('feasible',[True,False])
@pytest.mark.parametrize('cancel_request',[True,False])
def test_dispatch_keeps_loop_responsive_and_rejects_abandoned_plans(tmp_path,feasible,cancel_request):
    async def run():
        _,snapshot,request=system()
        now=time.time()
        request=replace(request,arrival_s=now,ttft_s=10)
        snapshot=replace(snapshot,timestamp_s=now,instances=tuple(
            replace(i,timestamp_s=now) for i in snapshot.instances))
        c=Controller(dict(strategy='pdblend-joint',journal=str(tmp_path/'journal')))
        c.state=StateManager(snapshot)
        entered=threading.Event();release=threading.Event();again=threading.Event()
        calls=[];executed=[]
        def blocking_plan(state,pending,**kwargs):
            calls.append(threading.get_ident())
            if len(calls)>1:
                again.set()
                return None
            entered.set()
            assert release.wait(2)
            return ControlPlan(state.version,now,now+10,
                routes=(RouteAction(request.request_id,'0','0',384,.2,.04,1),) if feasible else (),
                frequencies=(FrequencyAction('0',2520),),feasible=feasible)
        class Backend:
            frequency={};parked=set()
            async def execute(self,plan): executed.append(plan)
        c.planner=SimpleNamespace(plan=blocking_plan)
        c.backend=Backend()
        future=asyncio.get_running_loop().create_future()
        c.active[request.request_id]=dict(budget=request,future=future,client_request_id='client')
        c.pending.put_nowait(request.request_id)
        dispatch=asyncio.create_task(c.dispatch())
        async def until(predicate):
            while not predicate(): await asyncio.sleep(.001)
        try:
            # This can finish only if planning leaves the event loop available.
            await asyncio.wait_for(until(entered.is_set),1)
            assert calls[0]!=threading.get_ident()
            if cancel_request:
                future.cancel()
                c.active.pop(request.request_id)
            else:
                await c.state.publish(snapshot.instances,time.time())
            release.set()
            await asyncio.wait_for(until(lambda:c.pending.qsize()==0 if cancel_request else again.is_set()),1)
            assert not executed and not c.state.reservations and c.failure is None
            stats=c.planning_stats.summary()
            assert stats['counts']['discarded_cancelled_request' if cancel_request else 'discarded_stale']==1
            assert stats['counts']['feasible' if feasible else 'infeasible']==1
            assert stats['elapsed_sum_s']>=stats['elapsed_max_s']>=stats['elapsed_mean_s']>0
        finally:
            release.set();dispatch.cancel()
            try: await dispatch
            except asyncio.CancelledError: pass
            future.cancel()
            await c.planning_executor.close()
    asyncio.run(run())


def test_startup_failure_closes_planning_worker(tmp_path):
    async def run():
        c=Controller(dict(strategy='mixed',journal=str(tmp_path/'journal')))
        async def initialize(app):
            await c.planning_executor.run(lambda:None)
            c.journal_task=asyncio.create_task(c.journal.run())
            await c.journal.emit(dict(kind='startup'))
            raise RuntimeError('startup failed')
        c.initialize=initialize
        with pytest.raises(RuntimeError,match='startup failed'):
            await c.start(None)
        with pytest.raises(RuntimeError,match='closed'):
            await c.planning_executor.run(lambda:None)
        assert c.journal_task.done() and c.journal_task.exception() is None
        assert (tmp_path/'journal').read_text().count('startup')==1
        await c.planning_executor.close()
    asyncio.run(run())


def test_quiescence_prevents_a_frequency_plan_finishing_after_stop(tmp_path):
    async def run():
        _,snapshot,_=system()
        now=time.time()
        snapshot=replace(snapshot,timestamp_s=now,instances=tuple(
            replace(i,timestamp_s=now) for i in snapshot.instances))
        c=Controller(dict(strategy='mixed_dvfs',journal=str(tmp_path/'journal'),park_idle=False))
        c.state=StateManager(snapshot)
        entered=threading.Event();release=threading.Event();executed=[]
        def blocked(state,at):
            entered.set()
            assert release.wait(2)
            return ControlPlan(state.version,at,at+1,frequencies=(FrequencyAction('0',900),))
        async def noop(): pass
        async def tick(seconds): return not c.control_stop.is_set()
        async def execute(plan): executed.append(plan)
        c.frequency_planner=SimpleNamespace(plan=blocked)
        c.backend=SimpleNamespace(verify_clocks=noop,execute=execute)
        c.refresh=noop;c.control_tick=tick
        telemetry=asyncio.create_task(c.telemetry())
        async def until_entered():
            while not entered.is_set(): await asyncio.sleep(.001)
        try:
            await asyncio.wait_for(until_entered(),1)
            c.control_stop.set();release.set()
            await asyncio.wait_for(telemetry,1)
            assert not executed and c.failure is None
        finally:
            release.set();telemetry.cancel()
            try: await telemetry
            except asyncio.CancelledError: pass
            await c.planning_executor.close()
    asyncio.run(run())


def test_cleanup_failure_still_restores_clocks_closes_session_and_joins_worker(tmp_path):
    async def run():
        c=Controller(dict(strategy='mixed',journal=str(tmp_path/'journal')))
        closed=[]
        class Resource:
            def __init__(self,name): self.name=name
            async def close(self):
                await asyncio.sleep(.001)
                closed.append(self.name)
        async def broken(): raise OSError('journal failed')
        c.journal_task=asyncio.create_task(broken())
        await asyncio.sleep(0)
        c.session=Resource('session');c.clock_owner=Resource('clocks')
        await c.planning_executor.run(lambda:None)
        with pytest.raises(OSError,match='journal failed'):
            await c.close_resources()
        assert set(closed)=={'session','clocks'}
        with pytest.raises(RuntimeError,match='closed'):
            await c.planning_executor.run(lambda:None)
    asyncio.run(run())
