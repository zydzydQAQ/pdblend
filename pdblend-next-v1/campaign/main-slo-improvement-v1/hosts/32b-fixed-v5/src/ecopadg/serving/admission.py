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
    def __init__(self,capacity=256, *, round_fairness=False):
        if capacity<1: raise ValueError('positive admission queue capacity required')
        self.capacity=capacity;self.entries={};self.inflight={};self.sequence=0
        self.changed=asyncio.Event()
        self.round_fairness=round_fairness is True
        self.arrival_sequence={}
        self.attempt_counts={}
        self.round_members=set()
        self.round_index=0

    def qsize(self): return len(self.entries)+len(self.inflight)
    def full(self): return self.qsize()>=self.capacity

    def put_nowait(self,request_id):
        if request_id in self.entries or request_id in self.inflight: raise ValueError('duplicate queued request')
        if self.full(): raise asyncio.QueueFull
        self.sequence+=1;self.entries[request_id]=(self.sequence,0.)
        self.arrival_sequence[request_id]=self.sequence
        self.changed.set()

    async def get(self):
        if self.round_fairness:return await self.get_fair()
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

    async def get_fair(self):
        while True:
            await asyncio.sleep(0)
            now=time.monotonic()
            self.round_members.intersection_update(self.entries)
            if not self.round_members and self.entries:
                self.round_index+=1
                self.round_members=set(self.entries)
            eligible=[(self.attempt_counts.get(rid,0)>0,sequence,rid)
                      for rid,(sequence,retry) in self.entries.items()
                      if rid in self.round_members and retry<=now]
            if not eligible:
                # A deferred member cannot hold up a new first attempt. This
                # escape never repeats an old request within the same round.
                eligible=[(False,sequence,rid) for rid,(sequence,retry) in self.entries.items()
                          if rid not in self.round_members and not self.attempt_counts.get(rid)
                          and retry<=now]
            if eligible:
                _,sequence,rid=min(eligible)
                self.round_members.discard(rid)
                self.entries.pop(rid);self.inflight[rid]=sequence
                self.attempt_counts[rid]=self.attempt_counts.get(rid,0)+1
                return rid
            self.changed.clear()
            retry_times=[retry-now for rid,(_,retry) in self.entries.items()
                         if rid in self.round_members or not self.attempt_counts.get(rid)]
            timeout=max(0.,min(retry_times)) if retry_times else None
            try:await asyncio.wait_for(self.changed.wait(),timeout)
            except asyncio.TimeoutError:pass

    def defer(self,request_id,delay_s=.01):
        sequence=self.inflight.pop(request_id,None)
        if sequence is not None:
            self.entries[request_id]=(sequence,time.monotonic()+delay_s)
            self.changed.set()

    def done(self,request_id):
        self.entries.pop(request_id,None);self.inflight.pop(request_id,None)
        self.arrival_sequence.pop(request_id,None)
        self.attempt_counts.pop(request_id,None)
        self.round_members.discard(request_id)
        self.changed.set()
