import asyncio
import gc
import threading

import pytest

from ecopadg.serving.planning_executor import PlanningExecutor


async def event_set(event):
    # Polling here exercises the real event loop while the worker is blocked.
    async def wait():
        while not event.is_set():
            await asyncio.sleep(.001)
    await asyncio.wait_for(wait(), 2)


def test_result_kwargs_and_exception_are_delivered_from_one_separate_thread():
    executor = PlanningExecutor()  # Construction does not require a loop.
    async def run():
        loop_thread = threading.get_ident()
        seen = []
        def calculate(value, *, multiplier):
            seen.append(threading.current_thread())
            return value * multiplier
        try:
            assert await executor.run(calculate, 6, multiplier=7) == 42
            assert await executor.run(calculate, 3, multiplier=4) == 12
            assert seen[0] is seen[1] and seen[0].ident != loop_thread
            error = ValueError("bad profile")
            def fail():
                raise error
            with pytest.raises(ValueError) as caught:
                await executor.run(fail)
            assert caught.value is error
            assert await executor.run(lambda: 9) == 9
        finally:
            await executor.close()
        assert not seen[0].is_alive()
    asyncio.run(run())


def test_cancelled_running_waiter_retains_slot_and_cancelled_queued_call_never_runs():
    async def run():
        executor = PlanningExecutor()
        started, release = threading.Event(), threading.Event()
        executed = []
        submitted = []
        original_submit = executor._executor.submit
        def observe_submit(function, *args, **kwargs):
            submitted.append(function)
            return original_submit(function, *args, **kwargs)
        executor._executor.submit = observe_submit
        def blocking():
            started.set()
            assert release.wait(3)
            return "first"
        first = asyncio.create_task(executor.run(blocking))
        try:
            await event_set(started)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            # Cancelling many waiting callers must not fill ThreadPoolExecutor's
            # otherwise unbounded work queue with abandoned computations.
            for number in range(30):
                task = asyncio.create_task(executor.run(lambda n=number: executed.append(n)))
                await asyncio.sleep(0)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            survivor = asyncio.create_task(executor.run(lambda: executed.append("survivor")))
            for _ in range(5):
                await asyncio.sleep(.001)
            assert not survivor.done() and executed == []
            assert len(submitted) == 1  # No cancelled work accumulated behind the worker.
            release.set()
            await asyncio.wait_for(survivor, 2)
            assert executed == ["survivor"] and len(submitted) == 2
        finally:
            release.set()
            await executor.close()
    asyncio.run(run())


def test_worker_and_asynchronous_close_leave_heartbeat_running_and_reject_waiters():
    async def run():
        executor = PlanningExecutor()
        started, release = threading.Event(), threading.Event()
        worker = []
        def blocking():
            worker.append(threading.current_thread())
            started.set()
            assert release.wait(3)
            return 17
        first = asyncio.create_task(executor.run(blocking))
        closer = None
        try:
            await event_set(started)
            waiting = asyncio.create_task(executor.run(lambda: pytest.fail("submitted after close")))
            await asyncio.sleep(0)
            closer = asyncio.create_task(executor.close())
            with pytest.raises(RuntimeError, match="closed"):
                await asyncio.wait_for(waiting, 1)
            with pytest.raises(RuntimeError, match="closed"):
                await executor.run(lambda: None)
            ticks = 0
            while ticks < 10:
                await asyncio.sleep(.001)
                ticks += 1
            assert ticks == 10 and not closer.done() and not first.done()
            release.set()
            assert await asyncio.wait_for(first, 2) == 17
            await asyncio.wait_for(closer, 2)
            assert not worker[0].is_alive()
            await executor.close()
        finally:
            release.set()
            await executor.close()
            await asyncio.gather(first, *([closer] if closer else []), return_exceptions=True)
    asyncio.run(run())


def test_cancelled_close_waiter_does_not_cancel_worker_join_or_reopen_executor():
    async def run():
        executor = PlanningExecutor()
        started, release = threading.Event(), threading.Event()
        worker = []
        def blocking():
            worker.append(threading.current_thread())
            started.set()
            assert release.wait(3)
        first = asyncio.create_task(executor.run(blocking))
        try:
            await event_set(started)
            closer = asyncio.create_task(executor.close())
            await asyncio.sleep(0)
            closer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closer
            with pytest.raises(RuntimeError, match="closed"):
                await executor.run(lambda: None)
            second_close = asyncio.create_task(executor.close())
            await asyncio.sleep(.005)
            assert not second_close.done()
            release.set()
            await asyncio.wait_for(second_close, 2)
            await first
            assert not worker[0].is_alive()
        finally:
            release.set()
            await executor.close()
    asyncio.run(run())


def test_worker_exception_after_caller_cancellation_is_retrieved():
    async def run():
        executor = PlanningExecutor()
        started, release = threading.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        unhandled = []
        loop.set_exception_handler(lambda loop, context: unhandled.append(context))
        def fail_late():
            started.set()
            assert release.wait(3)
            raise ValueError("late failed plan")
        first = asyncio.create_task(executor.run(fail_late))
        try:
            await event_set(started)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            release.set()
            assert await executor.run(lambda: "next") == "next"
            await executor.close()
            gc.collect()
            await asyncio.sleep(0)
            assert unhandled == []
        finally:
            release.set()
            await executor.close()
            loop.set_exception_handler(previous)
    asyncio.run(run())


def test_close_before_first_run_is_idempotent_and_refuses_work():
    async def run():
        executor = PlanningExecutor()
        await asyncio.gather(executor.close(), executor.close())
        with pytest.raises(RuntimeError, match="closed"):
            await executor.run(lambda: None)
    asyncio.run(run())
