"""One bounded worker for pure planning over immutable runtime snapshots.

Keep mutable policy/state ownership on the event loop. Cancelling a caller
cannot interrupt Python computation, so its submission retains the sole slot
until the underlying concurrent future finishes. Always await ``close`` before
closing the owning event loop.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor


class PlanningExecutor:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdblend-planning")
        self._loop = None
        self._available = asyncio.Event()
        self._available.set()
        self._pending = None
        self._closed = False
        self._close_task = None

    def _bind_loop(self):
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("PlanningExecutor belongs to a different event loop")
        return loop

    def _finished(self, future):
        # Only this event-loop callback releases capacity, never a cancelled
        # caller's finally block. A completed worker may now accept one job.
        if self._pending is future:
            self._pending = None
            self._available.set()

    async def run(self, function, *args, **kwargs):
        loop = self._bind_loop()
        while True:
            if self._closed:
                raise RuntimeError("PlanningExecutor is closed")
            if self._pending is None:
                break
            await self._available.wait()
        # No await between checking the slot and submitting: other callers or
        # close() cannot interleave here on the owning event loop.
        future = self._executor.submit(function, *args, **kwargs)
        self._pending = future
        self._available.clear()
        future.add_done_callback(lambda done: loop.call_soon_threadsafe(self._finished, done))
        # Shield keeps an abandoned computation accountable to the capacity
        # bound; it also retrieves a late exception after caller cancellation.
        return await asyncio.shield(asyncio.wrap_future(future, loop=loop))

    async def close(self):
        self._bind_loop()
        if self._close_task is None:
            self._closed = True
            self._available.set()
            # Joining a running worker must not block SSE or other loop work.
            # Keep a single shutdown task even if its first waiter cancels.
            self._close_task = asyncio.create_task(asyncio.to_thread(
                self._executor.shutdown, wait=True, cancel_futures=False))
        await asyncio.shield(self._close_task)
