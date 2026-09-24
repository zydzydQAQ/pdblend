"""CPU cancellation and real author-controller transaction integration."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace
import time

import pytest

from pdblend_baselines.ecoserve.comparison_lifecycle import ComparisonLifecycle, MODE
from pdblend_baselines.ecoserve.controller import EcoServeController
from pdblend_baselines.ecoserve.runtime import EcoServeRuntime
from pdblend.bench.comparison_ecoserve_lifecycle import audit_lifecycle_events, bound_read_cancel


async def timeout_fixture():
    events = []
    lifecycle = None
    def emit(kind, **fields):
        if kind == 'eco_http_receipt':
            fields = lifecycle.decorate_http(fields)
        fields.setdefault('at_s', time.time())
        events.append(dict(kind=kind, **fields))
    lifecycle = ComparisonLifecycle(MODE, emit)
    outcomes = []
    entered = asyncio.Event()
    async def pending_read():
        with lifecycle.http_scope('eco0', 'GET', '/baseline/state', None):
            started = time.time()
            entered.set()
            try:
                await asyncio.Event().wait()
            except BaseException as exc:
                emit('eco_http_receipt', instance_id='eco0', method='GET', path='/baseline/state',
                     body=None, started_s=started, error=repr(exc))
                raise
    async def request():
        try:
            # Real refresh uses gather children; ContextVar ownership must
            # follow those child tasks rather than relying on Task identity.
            await asyncio.gather(pending_read())
        finally:
            outcomes.append(dict(request_id='r0', finished_s=time.time()))
    origin = time.time()
    task = lifecycle.create_request_task(request(), 'r0')
    await entered.wait()
    with pytest.raises(TimeoutError):
        await lifecycle.wait_cohort([task], .01)
    assert not task.done() and not any(r['kind'] == 'eco_http_receipt' for r in events)
    await lifecycle.cancel_cohort([task], error='TimeoutError()')
    entered.clear()
    # Invoke the unchanged real controller.close, with a background read.
    controller = EcoServeController.__new__(EcoServeController)
    controller.closed = False
    controller.failure = None
    controller.quarantined = set()
    controller.producers = set()
    controller.active = {}
    controller.resize_lock = asyncio.Lock()
    controller.journal = emit
    poll = asyncio.create_task(pending_read())
    controller.tasks = [poll]
    await entered.wait()
    runtime = SimpleNamespace(controller=controller, started=True)
    async def close():
        await controller.close()
        runtime.started = False
    runtime.close = close
    await lifecycle.close(runtime)
    native = dict(service_started_s=origin, duration_s=0., outcomes=outcomes,
                  error='TimeoutError()', comparison_lifecycle=lifecycle.summary())
    config = dict(eco_comparison_lifecycle=MODE, request_timeout_s=.01)
    return events, native, config


def test_timeout_is_explicit_and_nested_request_and_close_reads_have_exact_causes():
    events, native, config = asyncio.run(timeout_fixture())
    proof = audit_lifecycle_events(events, native, config)
    reads = [r for r in events if r['kind'] == 'eco_http_receipt']
    assert len(reads) == 2 and all(bound_read_cancel(r, proof) for r in reads)
    assert reads[0]['lifecycle_request_id'] == 'r0'
    assert reads[0]['lifecycle_cancel_id'] == 'cohort-cancel'
    assert reads[1]['lifecycle_request_id'] is None
    assert reads[1]['lifecycle_cancel_id'] == 'controller-close'
    assert not proof['policy_changed']


@pytest.mark.parametrize('damage', ['post', 'body', 'error', 'cause', 'request', 'early',
    'late', 'duplicate', 'missing_begin', 'timeout', 'runner_error', 'lock', 'close', 'rollback'])
def test_causal_audit_never_exempts_mutations_unknown_operations_or_other_failures(damage):
    events, native, config = asyncio.run(timeout_fixture())
    read = next(r for r in events if r['kind'] == 'eco_http_receipt')
    begin = next(r for r in events if r['kind'] == 'eco_comparison_cohort_cancel_begin')
    close = next(r for r in events if r['kind'] == 'eco_comparison_close_begin')
    if damage == 'post':
        read['method'] = begin['pending_http'][0]['method'] = 'POST'
        read['path'] = begin['pending_http'][0]['path'] = '/baseline/clock'
    elif damage == 'body':read['body'] = begin['pending_http'][0]['body'] = {}
    elif damage == 'error':read['error'] = 'RuntimeError()'
    elif damage == 'cause':read['lifecycle_cancel_id'] = 'invented'
    elif damage == 'request':read['lifecycle_request_id'] = 'unknown'
    elif damage == 'early':read['at_s'] = begin['at_s'] - 1
    elif damage == 'late':read['at_s'] = close['at_s'] + 1
    elif damage == 'duplicate':events.append(deepcopy(read))
    elif damage == 'missing_begin':events.remove(begin)
    elif damage == 'timeout':begin['monotonic_s'] -= 1
    elif damage == 'runner_error':native['error'] = 'RuntimeError()'
    elif damage == 'lock':close['resize_lock_held'] = False
    elif damage == 'close':events[:] = [r for r in events if r['kind'] != 'eco_comparison_close_end']
    elif damage == 'rollback':events.append(dict(kind='eco_membership_rollback', at_s=close['at_s']))
    with pytest.raises(ValueError):audit_lifecycle_events(events, native, config)


def author_runtime(tmp_path, emit):
    path = tmp_path/'profile.csv'
    path.write_text('Length,Prefill Time\n16,1\n128,8\n4096,256\n')
    entered, release = asyncio.Event(), asyncio.Event()
    class Transport:
        async def state(self, identifier):
            return dict(generation=1, acknowledged_generation=1, free_kv_tokens=8192,
                        role='mixed', mode='temporal', accepting=True, admit_prefill=True, admit_decode=True)
        async def events(self, identifier, after_seq=0):return dict(events=[], next_seq=after_seq)
        async def clock(self, gpus, frequency):
            entered.set()
            await release.wait()
            return dict(acknowledged=True, success=True)
        async def park(self, gpus):return dict(acknowledged=True, success=True)
    config = dict(instances=[dict(id='eco'+str(i), gpus=[i]) for i in range(3)],
                  eco_prefill_csv=str(path), slo_ttft_s=1., slo_tpot_s=.1,
                  eco_initial_instances=1, eco_macro_lower=1, eco_macro_upper=3)
    runtime = EcoServeRuntime(config, Transport(), emit)
    runtime.started = True
    return runtime, entered, release


def test_serial_close_finishes_actual_add_and_cancels_waiting_transaction_without_rollback(tmp_path):
    async def run():
        events = []
        def emit(kind, **fields):events.append(dict(kind=kind, **fields))
        runtime, entered, release = author_runtime(tmp_path, emit)
        life = ComparisonLifecycle(MODE, emit)
        first = asyncio.create_task(runtime.controller.add_member('eco1'))
        runtime.controller.tasks = [first]
        await entered.wait()
        closing = asyncio.create_task(life.close(runtime))
        await asyncio.sleep(0)
        queued = asyncio.create_task(runtime.controller.add_member('eco2'))
        runtime.controller.tasks.append(queued)
        await asyncio.sleep(0)
        assert not closing.done() and not runtime.controller.closed
        release.set()
        await asyncio.wait_for(closing, 1.)
        assert first.result() is True and queued.cancelled()
        assert runtime.controller.closed and not runtime.started
        assert [r['instance_id'] for r in events if r['kind'] == 'eco_membership_prepare'] == ['eco1']
        commit = next(i for i, r in enumerate(events) if r['kind'] == 'eco_membership_commit')
        close = next(i for i, r in enumerate(events) if r['kind'] == 'eco_comparison_close_begin')
        assert commit < close and not any(r['kind'] == 'eco_membership_rollback' for r in events)
    asyncio.run(run())


def test_close_lock_timeout_remains_failure_and_does_not_claim_clean_close(tmp_path):
    async def run():
        events = []
        def emit(kind, **fields):events.append(dict(kind=kind, **fields))
        runtime, entered, release = author_runtime(tmp_path, emit)
        life = ComparisonLifecycle(MODE, emit)
        transaction = asyncio.create_task(runtime.controller.add_member('eco1'))
        runtime.controller.tasks = [transaction]
        await entered.wait()
        with pytest.raises(asyncio.TimeoutError):await asyncio.wait_for(life.close(runtime), .01)
        assert not runtime.controller.closed and not life.summary()['serial_close_started']
        assert not any(r['kind'] in ('eco_closed', 'eco_comparison_close_end') for r in events)
        release.set()
        await transaction
        await runtime.close()
    asyncio.run(run())
