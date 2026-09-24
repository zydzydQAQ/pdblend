"""Deadline protection uses real requests and preserves physical ownership."""
import asyncio
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pdblend.online.controller import Controller
from pdblend.online.deadline import mixed_queue_prediction
from pdblend.online.router import RequestRecord, Router
from pdblend.online.shield import Pressure, Shield
from pdblend.planner.pool import Plan, SLO
from synthetic import fc
from test_controller import make_controller


class QueueModel:
    freqs = (900, 1200, 1500, 2100)
    kv_capacity_tokens = 1000000
    profile_key = {'test': 'bounded_queue'}

    def prefill_seconds(self, n, f): return .01 * 900 / f
    def step_seconds(self, batch, context, f):
        assert 1 <= batch <= 32
        return .01 * 900 / f
    def decode_supported(self, batch, context, f): return batch <= 32


def record(i, *, running=True, instance='m', output=101, submitted=100.):
    return RequestRecord(str(i), 'M', instance, instance, 10, output, submitted,
                         first_token_s=100.01 if running else None,
                         tokens_so_far=1 if running else 0)


def configured_router(ids=('m',)):
    router = Router(ids)
    router.configure_slo_routing(model=QueueModel(), slo=SLO(1., .1),
        frequency_provider=lambda iid: 900)
    router.configure_deadline_safety()
    return router


@pytest.mark.parametrize('owned,expected_wait', [(31, 0.), (32, 1.), (35, 1.03)])
def test_running_slots_and_waiting_queue_are_separate(owned, expected_wait):
    rows = [record(i, running=i < 32) for i in range(owned)]
    prediction = mixed_queue_prediction(rows, model=QueueModel(), frequency=900,
        max_num_seqs=32, input_tokens=10, max_tokens=10, now=100.02)
    assert prediction['source'] == 'observed_proxy_queue'
    assert not prediction['native_scheduler_observed']
    assert prediction['running_batch'] == 32
    assert prediction['running'] == min(32, owned)
    assert prediction['waiting'] == max(0, owned-32)
    assert prediction['scheduler_wait_s'] == pytest.approx(expected_wait)
    assert prediction['ttft_s'] == pytest.approx(expected_wait + .02)


def test_unobservable_preemption_cannot_be_certified_by_clamping_running_batch():
    with pytest.raises(ValueError, match='running ownership'):
        mixed_queue_prediction([record(i) for i in range(33)], model=QueueModel(),
            frequency=900, max_num_seqs=32, input_tokens=10, max_tokens=10, now=101.)


def test_routing_uses_slot_release_not_just_inflight_count():
    router = configured_router(('slow', 'fast'))
    for iid, output in [('slow', 101), ('fast', 3)]:
        for i in range(32):
            row = router.dispatch(iid+str(i), 10, output, choice=('M', iid, iid))
            router.token(row)
    new = router.dispatch('new', 10, 10)
    assert new.decode_instance == 'fast'
    prediction = new.route_estimate['slo_routing']['selected']
    assert prediction['queue_prediction']['source'] == 'observed_proxy_queue'
    assert prediction['queue_prediction']['scheduler_wait_s'] > 0
    assert router.rejected == 0


def test_deadline_event_is_coalesced_and_detects_risk_before_a_miss():
    router = configured_router()
    for i in range(31):
        row = router.dispatch(str(i), 10, 101, choice=('M', 'm', 'm'))
        router.token(row)
    assert router.deadline_event.is_set()
    router.deadline_event.clear()
    risk = router.deadline_risk_snapshot()
    assert risk['saturated'] and risk['risk']
    assert all(not row['deadline_ttft_s'] for row in risk['instances'])
    router.finish(row, 101)
    assert router.deadline_event.is_set()
    assert not router.deadline_risk_snapshot()['saturated']


def test_bootstrap_samples_cannot_release_startup_protection():
    ctl, _, router = make_controller(shield=False)
    ctl.startup_safety = True
    demand = SimpleNamespace(samples=10000)
    assert not ctl._informed(demand, 1000.)
    router.first_arrival_s, router.admitted_requests = 100., 29
    assert not ctl._informed(demand, 120.)
    router.admitted_requests = 30
    assert not ctl._informed(demand, 119.99)
    assert ctl._informed(demand, 120.)


def test_startup_requires_observed_tpot_as_well_as_ttft():
    ctl, _, router = make_controller(shield=False)
    ctl.startup_safety = True
    row = record('warm', submitted=119.)
    row.first_token_s = 119.1
    router.records.append(row)
    assert not ctl._startup_observed_safe(120.)
    row.tokens_so_far, row.last_token_s = 2, 119.2
    assert ctl._startup_observed_safe(120.)
    row.last_token_s = 119.5
    assert not ctl._startup_observed_safe(120.)


def test_startup_preserves_counts_and_uses_ceiling_with_two_down_votes():
    base, fleet, router = make_controller(shield=False)
    initial = Plan({'M': 2, 'off': 2}, 900, 900, 900, 0, 1., 1., .1)
    ctl = Controller(fleet, router, base.gpus, base.planner, initial_plan=initial,
        startup_safety=True, safety_max_freq=2100, hold_initial=True,
        period_s=.02, tick_s=.005)

    async def go():
        stop = asyncio.Event()
        task = asyncio.create_task(ctl.run(stop))
        await asyncio.sleep(.06)
        stop.set()
        await task
    asyncio.run(go())
    assert ctl.plan_now.counts == initial.counts
    assert ctl.plan_now.f_M == 2100
    assert ctl.down_plan_votes == 2
    assert all(row['f_M'] == 2100 for row in ctl._log if row['kind'] == 'plan')


def test_deadline_action_only_raises_m_and_releases_after_two_safe_windows():
    ctl, _, router = make_controller(shield=False)
    ctl.deadline_safety = True
    router.configure_slo_routing(model=QueueModel(), slo=SLO(1., .1),
        frequency_provider=lambda iid: ctl.freqs[iid])
    router.configure_deadline_safety()
    ctl.max_freq = 2100
    ctl.planner.cfg.freqs = QueueModel.freqs
    initial = Plan({'M': 2, 'P': 1, 'D': 1}, 900, 900, 900, 0, 1., 1., .1)

    async def go():
        await ctl.execute(initial)
        router.deadline_risk_snapshot = lambda **kw: dict(risk=False, unavailable=[], saturated=False)
        await ctl._deadline_clock_action({'risk': True, 'saturated': True})
    asyncio.run(go())
    assert ctl.plan_now.counts == initial.counts
    assert ctl.plan_now.f_M == 1200
    assert (ctl.plan_now.f_P, ctl.plan_now.f_D) == (900, 900)
    assert ctl._deadline_hold(initial, 100., True).f_M == 1200
    assert ctl._deadline_hold(initial, 110., True).f_M == 900


def test_admission_event_raises_clocks_without_waiting_for_controller_tick():
    ctl, _, router = make_controller(n=1, shield=False)
    ctl.deadline_safety, ctl.max_freq = True, 2100
    ctl.planner.cfg.freqs = QueueModel.freqs
    router.configure_slo_routing(model=QueueModel(), slo=SLO(1., .1),
        frequency_provider=lambda iid: ctl.freqs[iid])
    router.configure_deadline_safety()

    async def go():
        await ctl.execute(Plan({'M': 1}, 900, 900, 900, 0, 1., 1., .1))
        stop = asyncio.Event()
        worker = asyncio.create_task(ctl._deadline_worker(stop))
        try:
            for i in range(31):
                row = router.dispatch(str(i), 10, 101, choice=('M', 'i0', 'i0'))
                router.token(row)
            async def raised():
                while ctl.freqs['i0'] == 900:
                    await asyncio.sleep(.001)
            await asyncio.wait_for(raised(), .5)
            assert ctl.plan_now.counts == {'M': 1}
            assert router.rejected == 0
        finally:
            stop.set()
            router.deadline_event.set()
            await worker
    asyncio.run(go())


def test_concurrent_clock_requests_preserve_the_deadline_floor():
    ctl, _, _ = make_controller(n=1, shield=False)
    started, release = threading.Event(), threading.Event()
    original = ctl.gpus.set_clock
    calls = []
    def set_clock(gpu, frequency):
        calls.append(frequency)
        if frequency == 900:
            started.set()
            assert release.wait(2.)
        original(gpu, frequency)
    ctl.gpus.set_clock = set_clock

    async def go():
        low = asyncio.create_task(ctl._set_clock('i0', 900))
        assert await asyncio.to_thread(started.wait, 1.)
        ctl._deadline_floor_mhz = 2100
        high = asyncio.create_task(ctl._set_clock('i0', 2100))
        release.set()
        await asyncio.gather(low, high)
        await ctl._set_clock('i0', 900)
        assert calls == [900, 2100]
        assert ctl.freqs['i0'] == ctl.gpus.clocks[0] == 2100
    asyncio.run(go())


def test_worker_coalesces_admissions_and_tokens_before_model_queries():
    ctl, _, router = make_controller(shield=False)
    queries = []
    def risk():
        queries.append(time.monotonic())
        return {'risk': False}
    router.deadline_risk_snapshot = risk

    async def go():
        stop = asyncio.Event()
        worker = asyncio.create_task(ctl._deadline_worker(stop))
        try:
            for _ in range(12):
                router.deadline_event.set()
                await asyncio.sleep(.01)
        finally:
            stop.set()
            router.deadline_event.set()
            await worker
    asyncio.run(go())
    assert len(queries) >= 2
    assert all(b-a >= .045 for a,b in zip(queries, queries[1:]))


def test_urgent_clock_is_not_blocked_by_another_instances_ready_wait():
    ctl, fleet, router = make_controller(n=2, shield=False)
    ctl.deadline_safety = True
    ctl.max_freq = 2100
    ctl.planner.cfg.freqs = QueueModel.freqs
    entered, release = threading.Event(), threading.Event()
    initial = Plan({'M': 1, 'off': 1}, 900, 900, 900, 0, 1., 1., .1)

    async def go():
        await ctl.execute(initial)
        active = next(i for i,r in ctl.roles.items() if r == 'M')
        parked = next(i for i,r in ctl.roles.items() if r == 'off')
        def ready():
            entered.set()
            if not release.wait(3.):
                raise RuntimeError('test ready timeout')
        fleet[parked].wait_ready = ready
        router.deadline_risk_snapshot = lambda **kw: dict(risk=False, unavailable=[])
        transition = asyncio.create_task(ctl.execute(replace(initial, counts={'M': 2})))
        try:
            assert await asyncio.to_thread(entered.wait, 1.)
            await asyncio.wait_for(ctl._deadline_clock_action({'risk': True}), .5)
            assert not transition.done()
            assert ctl.freqs[active] == 1200
        finally:
            release.set()
            await transition
        assert ctl.plan_now.f_M == 1200
        assert set(ctl.freqs.values()) == {1200}
    asyncio.run(go())


def test_deadline_clock_does_not_overlap_a_parking_instances_drain():
    ctl, _, router = make_controller(n=2, shield=False)
    ctl.deadline_safety, ctl.max_freq = True, 2100
    ctl.planner.cfg.freqs = QueueModel.freqs
    router.deadline_risk_snapshot = lambda **kw: dict(risk=False, unavailable=[])

    async def go():
        await ctl.execute(Plan({'M': 2}, 900, 900, 900, 0, 1., 1., .1))
        entered, release = asyncio.Event(), asyncio.Event()
        draining = []
        async def drain(iid):
            draining.append(iid)
            entered.set()
            await release.wait()
            return True
        ctl._drain = drain
        transition = asyncio.create_task(ctl.execute(
            Plan({'M': 1, 'off': 1}, 900, 900, 900, 0, 1., 1., .1)))
        await asyncio.wait_for(entered.wait(), .5)
        tid = ctl._active_transition_id
        try:
            await ctl._deadline_clock_action({'risk': True})
            assert ctl.freqs[draining[0]] == 900
            kept = next(i for i in ctl.roles if i != draining[0])
            assert ctl.freqs[kept] == 1200
        finally:
            release.set()
            await transition
        phases = [r for r in ctl.transition_events if r['transition_id'] == tid]
        for iid in ctl.roles:
            owned = [r for r in phases if r['instance'] == iid]
            # Exact invariant enforced by comparison acceptance._controller.
            assert all(a['finished_s'] <= b['started_s'] for a,b in zip(owned, owned[1:]))
        plans = [r for r in ctl._log if r['kind'] == 'plan']
        completed = [r for r in ctl._log if r['kind'] == 'transition_complete']
        assert len(plans) == len(completed) == 2
        assert set(completed[-1]['affected']) == {r['instance'] for r in phases}
    asyncio.run(go())


def test_m_deadline_floor_never_changes_a_new_prefill_roles_clock():
    ctl, _, _ = make_controller(n=2, shield=False)
    ctl.deadline_safety = True

    async def go():
        await ctl.execute(Plan({'M': 2}, 900, 900, 900, 0, 1., 1., .1))
        ctl._deadline_floor_mhz = 2100
        await ctl.execute(Plan({'M': 1, 'P': 1}, 900, 900, 900, 0, 1., 1., .1))
        prefill = next(i for i,r in ctl.roles.items() if r == 'P')
        mixed = next(i for i,r in ctl.roles.items() if r == 'M')
        assert ctl.freqs[prefill] == ctl.plan_now.f_P == 900
        assert ctl.freqs[mixed] == ctl.plan_now.f_M == 2100
    asyncio.run(go())


@pytest.mark.parametrize('change', ['role', 'generation', 'target_role', 'admission', 'transition'])
def test_urgent_clock_rechecks_ownership_after_waiting_for_clock_lock(change):
    ctl, fleet, router = make_controller(n=1, shield=False)
    ctl._transition_open = True
    ctl._active_transition_id = 'expected'
    ctl._transition_target_roles = {'i0': 'M'}
    fleet['i0'].spec.generation = 7

    async def go():
        lock = ctl._clock_locks.setdefault('i0', asyncio.Lock())
        await lock.acquire()
        action = asyncio.create_task(ctl._set_clock('i0', 2100, expected_role='M',
            expected_generation=7, expected_transition_id='expected'))
        await asyncio.sleep(0)
        if change == 'role': ctl.roles['i0'] = 'P'
        elif change == 'generation': fleet['i0'].spec.generation = 8
        elif change == 'target_role': ctl._transition_target_roles['i0'] = 'off'
        elif change == 'admission': router.set_accepting('i0', False)
        else: ctl._active_transition_id = 'next'
        lock.release()
        await action
        assert not ctl.gpus.clocks
        assert not ctl.transition_events
    asyncio.run(go())


def test_final_execution_refreshes_metrics_after_frequency_override():
    ctl, _, _ = make_controller(shield=False)
    ctl.deadline_safety = True
    ctl._deadline_floor_mhz = 2100
    seen = []
    def refresh(plan, demand):
        seen.append(plan)
        return replace(plan, power_w=123., ttft_s=.123, tpot_s=.0123,
                       detail=dict(plan.detail, refreshed=True))
    ctl.planner.refresh_estimate = refresh
    initial = Plan({'M': 4}, 900, 900, 900, 0, 999., 999., 999.)
    asyncio.run(ctl.execute(initial))
    assert seen[-1].f_M == 2100
    assert ctl.plan_now.power_w == 123.
    assert ctl.plan_now.ttft_s == .123 and ctl.plan_now.tpot_s == .0123


def test_collective_clock_probe_does_not_restore_dynamic_m_floor():
    ctl, _, _ = make_controller(n=8)
    ctl.shield = Shield(SLO(5., .15), mode='budget_aware')
    ctl.dynamic_m_floor, ctl.base_m_floor, ctl._m_floor = True, 4, 2
    ctl._last_pressure = {'pressure': .1}
    ctl.plan_now = Plan({'M': 2, 'off': 6}, 900, 900, 900, 0, 1., 1., .1)
    pressure = Pressure(decode=True, decode_stalled=8, decode_stalled_fraction=1.,
                        decode_active=8, mode='budget_aware')
    ctl.shield.update(pressure, 100.)
    ctl.shield.apply(ctl.plan_now, pressure, 2520)
    assert not ctl._update_m_floor(fc(1), pressure, ctl.shield.level, 100., False)
    assert ctl._m_floor == 2


def test_frequency_snapshots_bracket_physical_execution():
    ctl, _, _ = make_controller(shield=False)
    captured = []
    def capture(*, requested, reason):
        captured.append((reason, requested.copy()))
        return []
    ctl.frequency_snapshot = capture
    async def go():
        await ctl.execute(Plan({'M': 4}, 900, 900, 900, 0, 1., 1., .1))
        for task in tuple(ctl._frequency_tasks):
            task.cancel()
        await asyncio.gather(*ctl._frequency_tasks, return_exceptions=True)
    asyncio.run(go())
    assert [reason for reason,_ in captured] == ['before_transition', 'after_transition']
    assert captured[-1][1] == dict.fromkeys(range(4), 900)


def test_normal_stop_finishes_an_inflight_deadline_transition_before_stop_receipt():
    ctl, _, router = make_controller(n=1, shield=False)
    ctl.deadline_safety = True
    router.deadline_risk_snapshot = lambda **kw: dict(
        risk='frequency' not in kw, unavailable=[], saturated=True)

    async def go():
        await ctl.execute(Plan({'M': 1}, 900, 900, 900, 0, 1., 1., .1))
        entered, release = threading.Event(), threading.Event()
        setter = ctl.gpus.set_clock
        def slow_clock(gpu, mhz):
            entered.set()
            assert release.wait(1.)
            setter(gpu, mhz)
        ctl.gpus.set_clock = slow_clock
        async def control(stop):
            router.deadline_event.set()
            assert await asyncio.to_thread(entered.wait, .5)
            stop.set()
            asyncio.get_running_loop().call_later(.02, release.set)
        ctl._run_control = control
        await ctl.run(asyncio.Event())
        plans = [r for r in ctl._log if r['kind'] == 'plan']
        completed = [r for r in ctl._log if r['kind'] == 'transition_complete']
        assert len(plans) == len(completed) == 2
        tids = {r['transition_id'] for r in completed}
        assert all(r['status'] == 'passed' and r['transition_id'] in tids
                   for r in ctl.transition_events)
        assert ctl._log[-1]['kind'] == 'stop'
        assert not ctl._frequency_tasks
        assert ctl._deadline_worker_task.done()
    asyncio.run(go())


def test_cancelled_snapshot_joins_its_reader_thread():
    ctl, _, _ = make_controller(shield=False)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    def snapshot(**kw):
        entered.set()
        assert release.wait(1.)
        finished.set()
        return []
    ctl.frequency_snapshot = snapshot
    async def go():
        task = asyncio.create_task(ctl._capture_frequency('test'))
        assert await asyncio.to_thread(entered.wait, .5)
        task.cancel()
        asyncio.get_running_loop().call_later(.02, release.set)
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
    asyncio.run(go())
