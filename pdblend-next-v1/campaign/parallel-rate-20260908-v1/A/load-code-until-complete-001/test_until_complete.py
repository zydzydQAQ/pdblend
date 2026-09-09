"""Only synthetic executor checks; no GPU qualification."""
import asyncio
import math
import time
import pytest
from test_capacity_executor import setup, execute

def test_none_deadline_keeps_finite_work_and_cleanup_limits(tmp_path):
    executor, backend, adapter, inventory = setup(tmp_path)
    executor.deadline_s = None
    original_start, original_stop = backend.start, backend.stop
    async def start(instance, before, limit):
        assert math.isfinite(limit) and 0 < limit-time.time() <= 90
        return await original_start(instance, before, limit)
    async def stop(instance, limit):
        assert math.isfinite(limit) and 0 < limit-time.time() <= 60
        return await original_stop(instance, limit)
    backend.start, backend.stop = start, stop
    asyncio.run(execute(executor))
    asyncio.run(executor.finish_to_initial())
    assert inventory.value['complete'] and len(adapter.instances) == 2

def test_none_deadline_failure_retains_bounded_owned_rollback(tmp_path):
    executor, backend, adapter, inventory = setup(tmp_path)
    executor.deadline_s = None
    backend.fail = 'verify'
    original = backend.stop_if_owned
    async def bounded(instance, limit):
        assert math.isfinite(limit) and 0 < limit-time.time() <= 120
        return await original(instance, limit)
    backend.stop_if_owned = bounded
    with pytest.raises(RuntimeError, match='numerical'):
        asyncio.run(execute(executor))
    assert backend.calls == ['start', 'rollback_stop'] and len(adapter.instances) == 2

def test_none_deadline_unpublished_cleanup_remains_bounded(tmp_path):
    executor, backend, adapter, inventory = setup(tmp_path)
    executor.deadline_s = None
    inventory.intent(backend.allocate((4,5)), 'start', 'cpu')
    original = backend.stop_if_owned
    async def bounded(instance, limit):
        assert math.isfinite(limit) and 0 < limit-time.time() <= 120
        return await original(instance, limit)
    backend.stop_if_owned = bounded
    asyncio.run(executor.finish_to_initial())
    assert inventory.value['complete'] and backend.calls == ['rollback_stop']
