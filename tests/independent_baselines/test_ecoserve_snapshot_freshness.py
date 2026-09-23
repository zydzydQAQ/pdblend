import asyncio
import time

import pytest

from pdblend_baselines.ecoserve.mechanism_four import observe_states, observed_live_tokens


def state(timestamp):
    return dict(free_kv_tokens=100, total_kv_tokens=100, block_size=16, generation=0,
                acknowledged_generation=0, native_at_s=timestamp, native_evidence_complete=True,
                transport_healthy=True, all_queue=[], kv_allocations={})


@pytest.mark.asyncio
async def test_fresh_replies_checked_at_individual_arrival(monkeypatch):
    clock = [100.]
    monkeypatch.setattr(time, 'time', lambda: clock[0])
    idle_received = asyncio.Event()

    class Transport:
        async def state(self, identifier):
            if identifier == 'idle':
                idle_received.set()
                return state(clock[0])
            await idle_received.wait()
            await asyncio.sleep(0)
            clock[0] += 2  # Active prefill delays only this member's reply.
            return state(clock[0])

    states, received = await observe_states(Transport(), ['idle', 'active'])
    assert states['idle']['native_at_s'] == received['idle'] == 100
    assert states['active']['native_at_s'] == received['active'] == 102


@pytest.mark.asyncio
async def test_actually_stale_reply_still_rejected():
    class Transport:
        async def state(self, identifier):
            return state(time.time()-2)

    with pytest.raises(RuntimeError, match='stale'):
        await observe_states(Transport(), ['cached'])


def test_future_sse_and_waiting_request_do_not_fabricate_live_kv():
    value = state(100.)
    value.update(all_queue=['r'], kv_allocations={'r': [[]]})
    events = [dict(at_s=101., token_ids=[42], token_index=1, finished=False)]
    assert observed_live_tokens(value, 'r', events) is None
    value['kv_allocations']['r'] = [[1]]
    assert observed_live_tokens(value, 'r', events) is None
    value['native_at_s'] = 101.5
    events.append(dict(at_s=102., token_ids=[43], token_index=2, finished=False))
    assert observed_live_tokens(value, 'r', events) == 1
    value.update(native_at_s=103., scheduler_at_s=100.)
    assert observed_live_tokens(value, 'r', events) is None
