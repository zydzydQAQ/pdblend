import asyncio
import copy
import time

import pytest

from test_controller import FakeGpus, make_controller
from pdblend.planner.pool import Plan
from pdblend.online.native_control import NativeControlError, validate_state
from pdblend.online.transition_measurement import interval_energy, measure_transitions


def state():
    return dict(generation=0, tp=2, pp=1, native_evidence_complete=True, transport_healthy=True,
                native_at_s=10., all_queue=[], running=[], waiting=[], retained_kv_requests=[],
                kv_allocations={}, pending_transfers=0, transfer_allocations={},
                total_blocks=100, free_blocks=99, reserved_blocks=1,
                ranks=[dict(rank=i, generation=0, healthy=True, native_evidence_complete=True,
                            pending_transfers=0, transfer_allocations={}) for i in range(2)])


def test_native_drain_requires_actual_inventory_and_every_rank():
    validate_state(state(), generation=0, tp=2, drained=True, observed_after_s=9.)
    for key, value in [('free_blocks', 98), ('all_queue', ['r']), ('retained_kv_requests', ['r']),
                       ('generation', 1), ('native_evidence_complete', False), ('ranks', []),
                       ('pending_transfers', 1), ('native_at_s', 8.)]:
        invalid = dict(state(), **{key: value})
        with pytest.raises(NativeControlError):
            validate_state(invalid, generation=0, tp=2, drained=True, observed_after_s=9.)


def test_cancel_does_not_require_unrelated_requests_to_finish():
    value = state()
    value.update(all_queue=['other'], running=['other'], kv_allocations={'other': [[3]]}, free_blocks=98)
    validate_state(value, generation=0, tp=2, request_id='cancelled')
    value['kv_allocations']['cancelled'] = [[4]]
    with pytest.raises(NativeControlError):
        validate_state(value, generation=0, tp=2, request_id='cancelled')


def test_clock_calls_do_not_block_stream_event_loop(tmp_path, monkeypatch):
    monkeypatch.setenv('PDBLEND_CLOCK_LOCK_DIR', str(tmp_path))
    class Slow(FakeGpus):
        def set_clock(self, g, mhz):
            time.sleep(.04)
            super().set_clock(g, mhz)
    async def run():
        ctl, _, _ = make_controller(n=4, shield=False)
        ctl.gpus = Slow()
        ticks = 0
        task = asyncio.create_task(ctl.execute(Plan({'M': 4}, 2520, 2520, 1500, 0, 1, 1, 1)))
        while not task.done():
            await asyncio.sleep(.005)
            ticks += 1
        await task
        assert ticks >= 3
        assert sum(row['operation'] == 'clock_set' for row in ctl.transition_events) == 4
    asyncio.run(run())


def test_native_drain_failure_prevents_hardware_parking(tmp_path, monkeypatch):
    monkeypatch.setenv('PDBLEND_CLOCK_LOCK_DIR', str(tmp_path))
    ctl, fleet, router = make_controller(n=1, shield=False)
    class Native:
        async def drain(self, iid, timeout):
            raise NativeControlError('live KV')
    ctl.native_control = Native()
    with pytest.raises(NativeControlError):
        asyncio.run(ctl.execute(Plan({'off': 1}, 2520, 2520, 2520, 0, 1, 1, 1)))
    assert not fleet['i0'].calls
    assert not ctl.gpus.clocks
    assert not router.loads['i0'].accepting
    assert ctl.plan_now is None


def test_wake_opens_proxy_only_after_native_ack(tmp_path, monkeypatch):
    monkeypatch.setenv('PDBLEND_CLOCK_LOCK_DIR', str(tmp_path))
    ctl, _, router = make_controller(n=1, shield=False)
    ctl.roles['i0'] = 'L1'
    router.set_accepting('i0', False)
    class Native:
        async def resume(self, iid, role):
            assert not router.loads[iid].accepting
            assert ctl.gpus.clocks[0] == 1200
            raise NativeControlError('stale rank')
    ctl.native_control = Native()
    with pytest.raises(NativeControlError):
        asyncio.run(ctl.execute(Plan({'M': 1}, 2520, 2520, 1200, 0, 1, 1, 1)))
    assert not router.loads['i0'].accepting
    assert ctl.plan_now is None


def test_transition_energy_uses_boundary_samples_and_union():
    samples = [(0., [10., 20.]), (1., [20., 40.]), (2., [30., 60.])]
    assert interval_energy(samples, .5, 1.5, [0]) == pytest.approx(20.)
    assert interval_energy(samples, -.1, 1.5, [0]) is None
    events = [dict(gpus=[0], started_s=0., finished_s=1.5),
              dict(gpus=[0], started_s=.5, finished_s=2.),
              dict(gpus=[1], started_s=0., finished_s=2.)]
    result = measure_transitions(events, samples, [0, 1])
    assert result['measured_union_energy_j'] == pytest.approx(120.)
    assert result['incremental_energy_j'] is None
    assert not measure_transitions(events, samples, [0, 1], sampler_error='lost sample')['energy_complete']
