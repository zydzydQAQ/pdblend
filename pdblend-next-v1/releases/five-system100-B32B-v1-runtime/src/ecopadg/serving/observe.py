"""Bounded asynchronous evidence journal with backpressure and explicit failure."""
import asyncio
from collections import Counter
import json
from pathlib import Path


class PlanningStats:
    """Constant-space, event-loop-owned accounting; no request samples stored."""
    def __init__(self):
        self.counts=Counter()
        self.elapsed_sum_s=0.
        self.elapsed_max_s=0.

    def finish(self,plan,elapsed_s):
        self.counts['completed_calls']+=1
        self.counts['no_candidate' if plan is None else 'feasible' if plan.feasible else 'infeasible']+=1
        if plan is not None and 'decision budget exceeded' in plan.reason:
            self.counts['budget_fallback']+=1
        self.elapsed_sum_s+=elapsed_s
        self.elapsed_max_s=max(self.elapsed_max_s,elapsed_s)

    def discard(self,reason):
        self.counts['discarded_'+reason]+=1

    def summary(self):
        return dict(counts=dict(self.counts),elapsed_sum_s=self.elapsed_sum_s,
            elapsed_max_s=self.elapsed_max_s,
            elapsed_mean_s=self.elapsed_sum_s/self.counts['completed_calls'] if self.counts['completed_calls'] else None,
            scope='completed admission planning calls, including worker wait; retries counted separately; cancelled computations excluded')


class Journal:
    def __init__(self,path,capacity=4096):
        self.path=Path(path)
        self.queue=asyncio.Queue(capacity)
        self.error=None
        self.failed=asyncio.Event()

    async def emit(self,event):
        if self.error:
            raise RuntimeError("evidence journal failed") from self.error
        try: self.queue.put_nowait(event)
        except asyncio.QueueFull: await self.wait_or_failure(self.queue.put(event))

    async def wait_or_failure(self,operation):
        task=asyncio.create_task(operation);failure=asyncio.create_task(self.failed.wait())
        try:
            await asyncio.wait((task,failure),return_when=asyncio.FIRST_COMPLETED)
            if self.error: raise RuntimeError('evidence journal failed') from self.error
            return await task
        finally:
            for pending in (task,failure):
                if not pending.done(): pending.cancel()
            await asyncio.gather(task,failure,return_exceptions=True)

    async def flush(self):
        await self.wait_or_failure(self.queue.join())

    def write(self,batch):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open("a") as handle:
            for event in batch:
                handle.write(json.dumps(event,allow_nan=False)+"\n")

    async def run(self):
        try:
            while True:
                first=await self.queue.get()
                if first is None:
                    self.queue.task_done()
                    break
                batch=[first]
                closing=False
                while not self.queue.empty() and len(batch)<128:
                    item=self.queue.get_nowait()
                    if item is None:
                        closing=True
                        break
                    batch.append(item)
                await asyncio.to_thread(self.write,batch)
                for _ in batch: self.queue.task_done()
                if closing:
                    self.queue.task_done()
                    return
        except BaseException as exc:
            self.error=exc
            self.failed.set()
            raise

    async def close(self):
        await self.emit(None)
