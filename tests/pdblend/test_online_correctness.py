"""CPU protocol regressions; synthetic receipts confer no GPU qualification."""
import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pdblend.engine.carry import CarryProtocolError, encode_event
from pdblend.online.router import DuplicateRequestError, ResidentRouter, Router
from pdblend.online.server import Proxy
from pdblend.online.shield import Shield
from pdblend.online.sse import TerminalStream
from pdblend.planner.pool import SLO


def resident(roles=None, *, capacity=1000, tp=1):
    roles = roles or {'a': 'M', 'b': 'M'}
    child = Router(roles, instance_metadata={iid: dict(tp=tp, pool_id='pool', generation=2,
                   model_id='m', profile_key='cpu-profile') for iid in roles})
    child.set_roles(roles)
    model = SimpleNamespace(freqs=(1000,), kv_capacity_tokens=capacity,
                            prefill_seconds=lambda n, f: n / 10000,
                            step_seconds=lambda b, c, f: .01,
                            transfer_seconds=lambda n: .005)
    return ResidentRouter({'pool': child}, {'pool': model})


def native_receipt(record, iid, engine_id=None):
    """Contract test double, not actual engine evidence."""
    return dict(instance_id=iid, request_id=engine_id or record.request_id,
                generation=record.generation, acknowledged=True, cancelled=True,
                native_state=dict(tp=record.tp, pp=record.pp, generation=record.generation,
                                  native_evidence_complete=True, transport_healthy=True,
                                  all_queue=[], waiting=[], running=[], retained_kv_requests=[],
                                  kv_allocations={}, native_at_s=time.time(),
                                  transfer_allocations={}, pending_transfers=0,
                                  ranks=[dict(rank=i, generation=record.generation,
                                              native_evidence_complete=True, healthy=True,
                                              pending_transfers=0, transfer_allocations={})
                                         for i in range(record.tp * record.pp)]))


def test_same_pool_full_least_sequence_instance_does_not_hide_available_capacity():
    router = resident()
    # Both have one sequence; JSQ selects a after its first token, but only b
    # has room for the next request. This used to return a false 503.
    a = router.dispatch('large', 970, 20)
    b = router.dispatch('small', 10, 10)
    assert (a.decode_instance, b.decode_instance) == ('a', 'b')
    router.first_token(a)
    assert router.pools['pool'].choose(200) == ('M', 'a', 'a')
    next_record = router.dispatch('next', 200, 20)
    assert next_record.decode_instance == 'b'


@pytest.mark.parametrize('use_resident', [False, True])
def test_duplicate_request_does_not_reserve_or_execute_twice(use_resident):
    router = resident() if use_resident else Router(['a', 'b'])
    record = router.dispatch('same', 100, 10)
    before = router.inflight()
    with pytest.raises(DuplicateRequestError):
        router.dispatch('same', 100, 10)
    assert router.inflight() == before and len(router.records) == 1
    router.finish(record, 10)
    assert router.dispatch('same', 100, 10) is not None


@pytest.mark.parametrize('use_resident', [False, True])
def test_uncertain_requests_keep_load_and_native_ack_recovery_is_idempotent(use_resident):
    router = resident({'a': 'M'}) if use_resident else Router(['a'])
    record = router.dispatch('failure', 100, 10)
    record.engine_instances.add('a')
    router.finish(record, 0, 'EOF')
    assert record.terminal_state == 'uncertain'
    assert router.loads['a'].inflight_seqs == 1
    assert router.loads['a'].inflight_prefill_tokens == 100
    assert not router.loads['a'].accepting
    with pytest.raises(DuplicateRequestError):
        router.dispatch('failure', 100, 10)
    receipts = {'a': native_receipt(record, 'a')}
    assert router.recover_cancel(record, receipts, engine_request_id='failure')
    assert record.terminal_state == 'cancelled_acknowledged'
    assert router.loads['a'].inflight_seqs == router.loads['a'].inflight_prefill_tokens == 0
    assert router.loads['a'].accepting and not router.quarantined
    assert not router.recover_cancel(record, receipts, engine_request_id='failure')


@pytest.mark.parametrize('corrupt', [
    lambda r: r.update(request_id='other'),
    lambda r: r.update(acknowledged=False),
    lambda r: r.update(generation=1),
    lambda r: r['native_state'].update(retained_kv_requests=['failed']),
    lambda r: r['native_state'].update(all_queue=['failed']),
    lambda r: r['native_state'].update(transfer_allocations={'unknown': 'buffer'}),
    lambda r: r['native_state'].update(pending_transfers=1),
    lambda r: r['native_state']['ranks'].pop(),
    lambda r: r['native_state']['ranks'][0].update(generation=1),
    lambda r: r['native_state']['ranks'][0].update(native_evidence_complete=False),
    lambda r: r['native_state']['ranks'][0].update(retained={'held_requests': ['failed']}),
])
def test_incomplete_or_stale_cancel_never_releases_ownership(corrupt):
    router = resident({'a': 'M'}, tp=2)
    record = router.dispatch('failed', 100, 10)
    record.engine_instances.add('a')
    router.finish(record, 0, 'failure')
    receipt = native_receipt(record, 'a')
    corrupt(receipt)
    with pytest.raises(ValueError):
        router.recover_cancel(record, {'a': receipt}, engine_request_id='failed')
    assert record.request_id in router.selector.active
    assert router.loads['a'].inflight_seqs == 1 and router.quarantined == {'a'}


def test_one_cleanup_does_not_clear_other_uncertain_request_quarantine():
    router = resident({'a': 'M'})
    first, second = router.dispatch('one', 10, 10), router.dispatch('two', 10, 10)
    for record in (first, second):
        record.engine_instances.add('a')
        router.finish(record, 0, 'failure')
    assert router.recover_cancel(first, {'a': native_receipt(first, 'a')}, engine_request_id='one')
    assert not router.loads['a'].accepting and router.quarantined == {'a'}
    assert router.loads['a'].inflight_seqs == 1
    assert router.recover_cancel(second, {'a': native_receipt(second, 'a')}, engine_request_id='two')
    assert router.loads['a'].accepting and not router.quarantined


def test_before_engine_rejection_releases_reservation_without_quarantine():
    router = resident({'a': 'M'})
    record = router.dispatch('invalid', 10, 10)
    router.finish(record, 0, 'unsupported', terminal_state='rejected_before_engine')
    assert not router.selector.active and not router.quarantined
    assert router.loads['a'].accepting


def frames(*, count=4, reason=None, done=True, usage=True):
    events = [dict(choices=[dict(index=0, text='token', finish_reason=reason)])]
    if usage:
        events.append(dict(choices=[], usage=dict(prompt_tokens=3, completion_tokens=count)))
    return b''.join(encode_event(e) for e in events) + (b'data: [DONE]\n\n' if done else b'')


@pytest.mark.parametrize('payload', [frames(done=False), frames(usage=False),
                                   frames(count=2), frames(count=5), frames(count=2, reason='length')])
def test_eof_done_only_and_unexplained_short_stream_are_not_success(payload):
    terminal = TerminalStream(4, prompt_tokens=3)
    terminal.feed(payload)
    with pytest.raises(CarryProtocolError):
        terminal.completion_tokens()


@pytest.mark.parametrize('count,reason', [(4, None), (4, 'length'), (2, 'stop'), (0, 'stop')])
def test_legitimate_length_or_eos_terminal_works_at_every_byte_boundary(count, reason):
    payload = frames(count=count, reason=reason)
    for cut in range(len(payload)):
        terminal = TerminalStream(4, prompt_tokens=3)
        terminal.feed(payload[:cut])
        terminal.feed(payload[cut:])
        assert terminal.completion_tokens() == count


def response_stub(monkeypatch):
    response = SimpleNamespace(prepared=True, prepare=AsyncMock(), write=AsyncMock(), write_eof=AsyncMock())
    monkeypatch.setattr('pdblend.online.server.web.StreamResponse', lambda **kwargs: response)
    return response


def req(**kw):
    body = dict(prompt=[1, 2, 3], max_tokens=4, request_id='r', ignore_eos=True)
    body.update(kw)
    return SimpleNamespace(json=AsyncMock(return_value=body))


@pytest.mark.asyncio
@pytest.mark.parametrize('id_source', ['body', 'header'])
async def test_proxy_duplicate_returns_409_before_second_engine_execution(monkeypatch, id_source):
    router = resident({'a': 'M'})
    proxy = Proxy({'a': 'http://unused'}, router)
    response_stub(monkeypatch)
    started, release = asyncio.Event(), asyncio.Event()

    async def stream(record, *_args):
        record.engine_instances.add('a')
        started.set()
        await release.wait()
        return 4

    proxy._stream_leg = AsyncMock(side_effect=stream)
    def request():
        request = req(request_id='r' if id_source == 'body' else None)
        request.headers = {'X-Request-Id': 'r'}
        return request

    first = asyncio.create_task(proxy.completions(request()))
    await started.wait()
    conflict = await proxy.completions(request())
    assert conflict.status == 409 and proxy._stream_leg.await_count == 1
    release.set()
    await first
    assert not router.selector.active


@pytest.mark.asyncio
async def test_proxy_failure_recovers_only_through_native_callback(monkeypatch):
    router = resident({'a': 'M'})

    async def cancel(record, engine_id, instance_ids):
        return {iid: native_receipt(record, iid, engine_id) for iid in instance_ids}

    proxy = Proxy({'a': 'http://unused'}, router, native_cancel=cancel)
    response_stub(monkeypatch)

    async def stream(record, *_args):
        record.engine_instances.add('a')
        raise CarryProtocolError('truncated')

    proxy._stream_leg = stream
    await proxy.completions(req())
    await asyncio.gather(*tuple(proxy._cleanup_tasks))
    assert router.records[-1].terminal_state == 'cancelled_acknowledged'
    assert not router.selector.active and router.loads['a'].accepting


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['M', 'P_ONLY'])
async def test_actual_stream_eof_quarantines_m_and_p_only(monkeypatch, path):
    router = resident({'a': 'M'} if path == 'M' else {'a': 'P', 'b': 'D'})
    proxy = Proxy({iid: 'http://unused' for iid in router.loads}, router)
    response_stub(monkeypatch)

    class Upstream:
        status = 200
        content = None

        def __init__(self):
            self.content = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            pass

        async def iter_any(self):
            yield frames(count=1, done=False)

    proxy.session = SimpleNamespace(post=lambda *_a, **_kw: Upstream())
    await proxy.completions(req(max_tokens=1 if path == 'P_ONLY' else 4))
    record = router.records[-1]
    assert record.path == path and record.terminal_state == 'uncertain'
    assert 'terminal DONE' in record.error and router.selector.active
    assert record.last_token_s is not None


@pytest.mark.asyncio
async def test_p_only_ordinary_eos_sampling_does_not_require_carry_protocol(monkeypatch):
    router = resident({'a': 'P', 'b': 'D'})
    proxy = Proxy({'a': 'http://unused', 'b': 'http://unused'}, router)
    response_stub(monkeypatch)
    proxy._stream_leg = AsyncMock(return_value=1)
    await proxy.completions(req(max_tokens=1, ignore_eos=False, stop=['END']))
    assert proxy._stream_leg.await_count == 1
    assert router.records[-1].path == 'P_ONLY'
    assert router.records[-1].terminal_state == 'completed' and not router.selector.active


def test_all_active_observation_survives_history_eviction_and_long_lifetime():
    router = Router(['a'], history=1)
    old = router.dispatch('old', 10, 1000)
    old.submitted_s = 1
    router.token(old, at_s=2, count=900)
    router.dispatch('recent', 10, 1).submitted_s = 100
    records = router.observation_records(60, now=100)
    assert old in records and len(records) == 2
    pressure = Shield(SLO(5, .2)).observe(records, 100)
    assert pressure.decode and pressure.decode_stalled == 1
    assert pressure.longest_token_gap_s == 98


def test_recent_token_stall_cannot_hide_behind_lifetime_average():
    router = Router(['a'])
    record = router.dispatch('r', 10, 10000)
    record.submitted_s = 0
    router.token(record, at_s=1, count=1000)
    router.token(record, at_s=99.0)
    pressure = Shield(SLO(5, .2)).observe([record], 100)
    assert pressure.tpot_p90 < .2 and pressure.decode and pressure.decode_stalled == 1
    router.token(record, at_s=99.99)
    assert not Shield(SLO(5, .2)).observe([record], 100).decode


@pytest.mark.asyncio
@pytest.mark.parametrize('with_control', [False, True])
async def test_live_proxy_qualification_exercises_disconnect_native_cleanup_and_reuse(with_control):
    from aiohttp import web
    from pdblend.online.native_control import NativeControl
    from pdblend.online.qualification import qualify_live_proxy
    from test_carry_protocol import engines

    async with engines() as (servers, urls):
        servers[0].native_generation = 2
        router = resident({'i0': 'M'})
        specs = {iid: SimpleNamespace(base_url=url, tp=1, pp=1, generation=2 if iid == 'i0' else 0)
                 for iid, url in urls.items()}
        native = NativeControl(specs, timeout_s=2)
        proxy = Proxy(urls, router, native_cancel=native.cancel)
        runner = web.AppRunner(proxy.app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        base_url = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'

        async def control():
            # Contract-only CPU stand-in for a real offloaded GPU operation.
            receipt = await native.drain('i1', 2)
            await asyncio.sleep(.04)
            return dict(drain=receipt, resume=await native.resume('i1', 'M'))

        try:
            result = await qualify_live_proxy(proxy, base_url, [1, 2, 3], max_tokens=16, timeout_s=3,
                                               control_action=control if with_control else None)
            assert result['functional_passed'], result
            assert not result['hardware_qualified'] and not result['energy_comparable']
            assert servers[0].native_cancelled == {result['request_id']}
            assert all(result['checks'].values())
        finally:
            await runner.cleanup()
