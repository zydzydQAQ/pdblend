import asyncio
import time
import pytest
from ecopadg.serving.engine import EngineService


def test_blocked_executor_quarantines_only_its_engine_and_rejects_queued_work(tmp_path):
    async def run():
        service=EngineService(dict(id='a',runtime_dir=str(tmp_path),operation_timeout_s=.01))
        queue=asyncio.Queue(maxsize=2);service.streams['r']=queue
        calls=[]
        try:
            with pytest.raises(RuntimeError,match='quarantined'):
                await service.call(time.sleep,.1)
            assert not service.accepting and service.error
            assert isinstance(queue.get_nowait(),RuntimeError)
            with pytest.raises(RuntimeError,match='quarantined'):
                await service.call(calls.append,1)
            await asyncio.sleep(.12)
            assert not calls  # Native return cannot silently re-enable admission.
        finally:
            service.worker.shutdown(wait=True,cancel_futures=True)
    asyncio.run(run())
