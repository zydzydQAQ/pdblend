"""Real source requests retained through quiesce/drain for transition calibration."""
from __future__ import annotations
import asyncio
import math
import random
import time


def validate(n, o, batch):
    if (any(type(v) is not int for v in (n,o,batch)) or not 1<=n<=7168
            or not 2<=o<=512 or n+o>8192 or not 1<=batch<=16):
        raise ValueError('legal explicit loaded-drain workload required')


class LoadedDrain:
    def __init__(self, transport, iid, journal, *, input_tokens, output_tokens, batch):
        validate(input_tokens,output_tokens,batch)
        self.transport,self.iid,self.journal=transport,iid,journal
        self.n,self.o,self.batch=input_tokens,output_tokens,batch
        self.tasks=[];self.outcomes=[];self.request_ids=[];self.observation=None

    async def start(self, timeout_s=30.):
        state=await self.transport.state(self.iid)
        if state['free_kv_tokens'] < self.batch*(self.n+self.o):
            raise ValueError('loaded-drain workload exceeds actual free KV capacity')
        base='dynamo-loaded-drain-'+str(time.time_ns())
        async def request(index):
            rid=self.request_ids[index];rng=random.Random(9701+index)
            payload=dict(request_id=rid,prompt=[rng.randint(1000,60000) for _ in range(self.n)],
                         max_tokens=self.o,seed=701,temperature=0,ignore_eos=True)
            row=dict(request_id=rid,input_tokens=self.n,output_tokens=self.o,submitted_s=time.time(),
                     token_ids=[],finished=False)
            try:
                async for event in self.transport.stream(self.iid,payload):
                    if event.get('token_ids'):row.setdefault('first_token_s',time.time())
                    row['token_ids'].extend(event.get('token_ids',[]));row['finished']=bool(event.get('finished'))
                row['ok']=row['finished'] and len(row['token_ids'])==self.o
            except BaseException as exc:
                row.update(ok=False,error=repr(exc));raise
            finally:
                row['finished_s']=time.time();self.outcomes.append(row)
                self.journal('dynamo_loaded_drain_outcome',instance_id=self.iid,**row)
        self.request_ids=[base+'-'+str(i) for i in range(self.batch)]
        self.tasks=[asyncio.create_task(request(i)) for i in range(self.batch)]
        deadline=time.monotonic()+timeout_s
        while time.monotonic()<deadline:
            if any(t.done() for t in self.tasks):
                raise RuntimeError('loaded-drain request ended before actual live-KV observation')
            state=await self.transport.state(self.iid)
            stamp=state.get('timestamp')
            fresh=type(stamp) in (int,float) and math.isfinite(stamp) and 0<=time.time()-stamp<=.5
            allocations=state.get('kv_allocations',{})
            if (fresh and state.get('evidence_complete') is True
                    and all(rid in allocations and allocations[rid] for rid in self.request_ids)):
                self.observation=dict(observed_s=time.time(),state=state,request_ids=self.request_ids,
                    input_tokens=self.n,output_tokens=self.o,batch=self.batch)
                self.journal('dynamo_loaded_drain_live',instance_id=self.iid,**self.observation)
                return self.observation
            await asyncio.sleep(.01)
        raise TimeoutError('native live-KV observation missing before drain')

    async def finish(self):
        await asyncio.gather(*self.tasks)
        if len(self.outcomes)!=self.batch or not all(r['ok'] for r in self.outcomes):
            raise RuntimeError('loaded-drain lost a real source output')
        return dict(measured=True,live=self.observation,outcomes=self.outcomes,
                    workload_envelope=dict(max_input_tokens=self.n,max_output_tokens=self.o,
                                           max_batch=self.batch),formal_eligible=False)

    async def close(self):
        pending=[rid for rid,task in zip(self.request_ids,self.tasks) if not task.done()]
        for task in self.tasks:
            if not task.done():task.cancel()
        await asyncio.gather(*self.tasks,return_exceptions=True)
        for rid in pending:
            await self.transport.cancel(self.iid,rid)
