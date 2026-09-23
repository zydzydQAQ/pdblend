"""Admission/reroute regressions; real controller and bounded own profile, no GPU."""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from pdblend_baselines.dynamollm.policy import Request
from pdblend_baselines.dynamollm.runtime import DynamoController


class Transport:
    def __init__(self):
        self.block = asyncio.Event()
        self.started = {}
        self.unhealthy = set()
        self.accepting = True
        self.free = 48080

    async def state(self, iid):
        if iid in self.unhealthy:
            raise RuntimeError('native state unhealthy')
        return dict(role='mixed', accepting=self.accepting, free_kv_tokens=self.free,
                    generation=0, kv_allocations={})

    async def clock(self, gpus, frequency):
        assert frequency == 2520

    async def stream(self, iid, payload):
        self.started[payload['request_id']] = iid
        if payload['request_id'] in ('r7', 'r8'):
            await self.block.wait()
        yield dict(token_ids=list(range(16)), choices=[dict(finish_reason='length')])

    async def cancel(self, iid, rid):
        pass


def controller(tmp_path):
    profile = tmp_path/'profile.json'
    profile.write_text(json.dumps(dict(schema=2, measurement='hardware',
        coordinate_system='input_output_batch', points=[dict(role='mixed', tp=2,
        frequency_mhz=2520, input_tokens=2048, context_tokens=2122, batch=1,
        prefill_s=.1, iteration_s=.01, power_w=200, samples=3, source_sha256='a'*64)])))
    transport = Transport()
    rows = []
    cfg = dict(profiles=str(profile), instances=[dict(id=i, gpus=[2*n, 2*n+1],
        tp=2, shape='LL') for n, i in enumerate(('a', 'b'))],
        slo_ttft_s=5., slo_tpot_s=.15, development_allow_predictor_injection=True,
        development_predictor=SimpleNamespace(predict_text=lambda _: 74),
        development_tokenizer=SimpleNamespace(decode=lambda *a, **kw: 'visible input'))
    c = DynamoController(cfg, transport, lambda event, **kw: rows.append(dict(event=event, **kw)))
    return c, transport, rows


async def wait_for(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(.002)
    await asyncio.wait_for(wait(), 2)


async def complete(c, rid='r9'):
    return [event async for event in c.handle(dict(prompt=[1000]*2048, max_tokens=16), rid)]


@pytest.mark.asyncio
async def test_two_busy_ll_replicas_never_exclude_waiter_from_both(tmp_path):
    c, transport, rows = controller(tmp_path)
    tasks = []
    try:
        await c.startup()
        tasks.append(asyncio.create_task(complete(c, 'r7')))
        await wait_for(lambda: 'r7' in transport.started)
        tasks.append(asyncio.create_task(complete(c, 'r8')))
        await wait_for(lambda: 'r8' in transport.started)
        tasks.append(asyncio.create_task(complete(c)))
        await wait_for(lambda: 'r9' in c.pending)
        for _ in range(4):
            await c.emergency_tick(time.time())
        assert any(row.get('stage') == 4 for row in rows if row['event'] == 'dynamo_emergency')
        assert 'r9' not in transport.started  # B2 is still outside measured coverage.
        assert c.exclusions.get('r9', set()) != {'a', 'b'}
        transport.block.set()
        outputs = await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert all(out[0]['token_ids'] == list(range(16)) for out in outputs)
        assert len(transport.started) == 3
        assert not c.pending and not c.kv_reservations and not c.exclusions
    finally:
        transport.block.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await c.close()


@pytest.mark.asyncio
async def test_stale_reroute_preference_can_return_to_a_healthy_source(tmp_path):
    c, transport, _ = controller(tmp_path)
    try:
        await c.startup()
        c.exclusions['r9'] = {'a'}
        transport.unhealthy.add('b')
        result = await asyncio.wait_for(complete(c), 2)
        assert result[0]['token_ids'] == list(range(16))
        assert transport.started == {'r9': 'a'}
    finally:
        await c.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('constraint', ['unhealthy', 'engine_closed', 'transaction_closed',
                                       'capacity', 'unmeasured_batch', 'slo'])
async def test_reroute_fallback_preserves_actual_admission_constraints(tmp_path, constraint):
    c, transport, _ = controller(tmp_path)
    task = None
    try:
        await c.startup()
        c.exclusions['r9'] = {'a', 'b'}
        if constraint == 'unhealthy':
            transport.unhealthy.update(('a', 'b'))
        elif constraint == 'engine_closed':
            transport.accepting = False
        elif constraint == 'transaction_closed':
            for replica in c.replicas.values():
                replica.accepting = False
        elif constraint == 'capacity':
            transport.free = 0
        elif constraint == 'unmeasured_batch':
            for replica in c.replicas.values():
                replica.requests.append(Request('busy-'+replica.instance_id, 2048, 74,
                    time.time(), 5., .15, started=True))
        else:
            c.config['slo_tpot_s'] = .005  # Profile's .01 remains infeasible.
        task = asyncio.create_task(complete(c))
        await wait_for(lambda: 'r9' in c.pending)
        await asyncio.sleep(.04)
        assert not transport.started and not task.done()
        assert c.exclusions['r9'] == {'a', 'b'}
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await c.close()


@pytest.mark.asyncio
async def test_emergency_only_excludes_source_for_a_feasible_nonexcluded_target(tmp_path):
    c, _, _ = controller(tmp_path)
    try:
        await c.startup()
        now = time.time()
        waiter = Request('r9', 2048, 74, now, 5., .15)
        c.pending['r9'] = c.requests['r9'] = waiter
        c.replicas['a'].requests.append(Request('busy', 2048, 74, now, 5., .15, started=True))
        c.emergency_stages['a'] = 2
        await c.emergency_tick(now)
        assert c.exclusions['r9'] == {'a'}
        # Even if b later becomes busy, a is not a legal reroute destination
        # while it remains excluded; the last nonexcluded candidate survives.
        c.replicas['b'].requests.append(Request('busy2', 2048, 74, now, 5., .15, started=True))
        c.emergency_stages['b'] = 2
        await c.emergency_tick(now)
        assert c.exclusions['r9'] == {'a'}
    finally:
        await c.close()
