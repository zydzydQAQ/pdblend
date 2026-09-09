"""Bounded admission retries without blocking unrelated ready requests."""
import asyncio
import time


class AdmissionRejected(RuntimeError):
    """Explicit capacity refusal before any engine work starts."""
    def __init__(self,code,message):
        super().__init__(message)
        if code not in ('admission_queue_full','admission_deadline'):
            raise ValueError('unknown admission rejection')
        self.code=code


class AdmissionQueue:
    def __init__(self,capacity=256):
        if capacity<1: raise ValueError('positive admission queue capacity required')
        self.capacity=capacity;self.entries={};self.inflight={};self.sequence=0
        self.changed=asyncio.Event()

    def qsize(self): return len(self.entries)+len(self.inflight)
    def full(self): return self.qsize()>=self.capacity

    def put_nowait(self,request_id):
        if request_id in self.entries or request_id in self.inflight: raise ValueError('duplicate queued request')
        if self.full(): raise asyncio.QueueFull
        self.sequence+=1;self.entries[request_id]=(self.sequence,0.)
        self.changed.set()

    async def get(self):
        while True:
            now=time.monotonic()
            ready=[(sequence,rid) for rid,(sequence,retry) in self.entries.items() if retry<=now]
            if ready:
                sequence,rid=min(ready)
                self.entries.pop(rid);self.inflight[rid]=sequence
                return rid
            self.changed.clear()
            timeout=min((retry-now for _,retry in self.entries.values()),default=None)
            try: await asyncio.wait_for(self.changed.wait(),timeout)
            except asyncio.TimeoutError: pass

    def defer(self,request_id,delay_s=.01):
        sequence=self.inflight.pop(request_id,None)
        if sequence is not None:
            self.entries[request_id]=(sequence,time.monotonic()+delay_s)
            self.changed.set()

    def done(self,request_id):
        self.entries.pop(request_id,None);self.inflight.pop(request_id,None)
        self.changed.set()
