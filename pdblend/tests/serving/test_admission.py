import asyncio
import pytest
from ecopadg.serving.admission import AdmissionQueue


def test_blocked_long_request_does_not_block_short_work_or_lose_its_priority():
    async def run():
        queue=AdmissionQueue(3)
        queue.put_nowait('long');queue.put_nowait('short')
        assert await queue.get()=='long'
        queue.defer('long',.02)
        assert await queue.get()=='short'
        queue.done('short')
        await asyncio.sleep(.025)
        queue.put_nowait('new')
        assert await queue.get()=='long'  # original arrival priority retained
        queue.done('long');assert await queue.get()=='new'
        queue.done('new');assert queue.qsize()==0
    asyncio.run(run())


def test_capacity_includes_inflight_and_cancelled_retries_do_not_reappear():
    async def run():
        queue=AdmissionQueue(1);queue.put_nowait('a')
        assert await queue.get()=='a' and queue.full()
        with pytest.raises(asyncio.QueueFull): queue.put_nowait('b')
        queue.done('a');queue.defer('a')
        assert queue.qsize()==0
        waiter=asyncio.create_task(queue.get());await asyncio.sleep(0)
        queue.put_nowait('b');assert await asyncio.wait_for(waiter,.1)=='b'
    asyncio.run(run())
