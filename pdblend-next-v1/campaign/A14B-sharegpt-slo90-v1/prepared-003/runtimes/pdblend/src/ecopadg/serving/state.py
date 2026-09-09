"""Single-owner snapshot and admission reservations; no network I/O under lock."""
import asyncio
from dataclasses import replace
from .types import RuntimeSnapshot
from .tails import admission_budget


class StalePlan(RuntimeError):
    pass


class ExpiredPlan(StalePlan):
    """Execution did not start: the deadline passed before any backend action."""


class StateManager:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.reservations = {}
        self.lock = asyncio.Lock()
        self.engine_waiting = {i.instance_id:i.waiting for i in snapshot.instances}

    async def publish(self, instances, now, *, engine_waiting=None):
        async with self.lock:
            observed_waiting = ({i.instance_id:i.waiting for i in instances}
                if engine_waiting is None else engine_waiting)
            adjusted=[]
            previous={i.instance_id:i for i in self.snapshot.instances}
            for instance in instances:
                current=previous.get(instance.instance_id)
                if current and current.generation>instance.generation:
                    adjusted.append(current)
                    continue
                self.engine_waiting[instance.instance_id]=observed_waiting[instance.instance_id]
                allocations=dict(instance.kv_allocations)
                reserved=sum(max(0,(r.reserve_tokens if r.decode_id==instance.instance_id
                                    else r.prefill_reserve_tokens)-allocations.get(r.request_id,0))
                             for r in self.reservations.values()
                             if instance.instance_id in (r.decode_id,r.prefill_id))
                staging=dict(instance.transfer_allocations)
                transfer_reserved=sum(max(0,r.transfer_reserve_bytes-staging.get(r.request_id,0))
                                      for r in self.reservations.values() if r.decode_id==instance.instance_id)
                adjusted.append(replace(instance,reserved_kv_tokens=reserved,
                                        reserved_transfer_bytes=transfer_reserved))
            self.snapshot=RuntimeSnapshot(self.snapshot.version+1,now,tuple(adjusted))
            return self.snapshot

    async def reserve(self, plan, now, request=None):
        async with self.lock:
            if plan.snapshot_version != self.snapshot.version or now > plan.expires_s:
                raise StalePlan("snapshot changed or plan expired")
            if not plan.feasible:
                return False
            if any(r.request_id in self.reservations for r in plan.routes):
                raise ValueError("duplicate admission")
            by_id={i.instance_id:i for i in self.snapshot.instances}
            for route in plan.routes:
                admitted=admission_budget(replace(plan,routes=(route,)),request,now=now) if request else None
                d=by_id[route.decode_id]
                if not d.accepting or d.free_kv_tokens-d.reserved_kv_tokens < route.reserve_tokens:
                    raise StalePlan("decode capacity changed")
                if d.free_transfer_bytes-d.reserved_transfer_bytes<route.transfer_reserve_bytes:
                    raise StalePlan('KV staging capacity changed')
                by_id[d.instance_id]=replace(d,reserved_kv_tokens=d.reserved_kv_tokens+route.reserve_tokens,
                    reserved_transfer_bytes=d.reserved_transfer_bytes+route.transfer_reserve_bytes,
                    waiting=d.waiting+1,requests=d.requests+((admitted,) if admitted else ()))
                if route.prefill_id!=route.decode_id:
                    p=by_id[route.prefill_id]
                    if not p.accepting or p.free_kv_tokens-p.reserved_kv_tokens<route.prefill_reserve_tokens:
                        raise StalePlan("prefill capacity changed")
                    by_id[p.instance_id]=replace(p,reserved_kv_tokens=p.reserved_kv_tokens+route.prefill_reserve_tokens,
                        waiting=p.waiting+1,requests=p.requests+((admitted,) if admitted else ()))
            for route in plan.routes:
                self.reservations[route.request_id]=route
            self.snapshot=replace(self.snapshot,version=self.snapshot.version+1,
                                  instances=tuple(by_id.values()))
            return True

    async def release(self, request_id, *, unissued=False):
        async with self.lock:
            route=self.reservations.pop(request_id,None)
            if route is not None:
                # Only the local reserve() increment is undone. A telemetry
                # publication may have added real queued work in the meantime.
                instances=tuple(replace(i,
                    waiting=max(self.engine_waiting.get(i.instance_id,0),i.waiting-1) if unissued else i.waiting,
                    reserved_kv_tokens=max(0,i.reserved_kv_tokens-
                    max(0,(route.reserve_tokens if i.instance_id==route.decode_id else route.prefill_reserve_tokens)
                        -dict(i.kv_allocations).get(request_id,0))),
                    reserved_transfer_bytes=max(0,i.reserved_transfer_bytes-
                        max(0,route.transfer_reserve_bytes-dict(i.transfer_allocations).get(request_id,0)))
                        if i.instance_id==route.decode_id else i.reserved_transfer_bytes,
                    requests=tuple(r for r in i.requests if r.request_id!=request_id))
                    if i.instance_id in (route.decode_id,route.prefill_id) else i for i in self.snapshot.instances)
                self.snapshot=replace(self.snapshot,version=self.snapshot.version+1,instances=instances)

    async def prefill_complete(self,request_id):
        async with self.lock:
            route=self.reservations.get(request_id)
            if route:
                self.reservations[request_id]=replace(route,prefill_reserve_tokens=0)
                if route.prefill_id==route.decode_id:
                    return
                instances=[]
                for instance in self.snapshot.instances:
                    if instance.instance_id==route.prefill_id:
                        allocated=dict(instance.kv_allocations).get(request_id,0)
                        instance=replace(instance,
                            reserved_kv_tokens=max(0,instance.reserved_kv_tokens-
                                max(0,route.prefill_reserve_tokens-allocated)),
                            waiting=max(0,instance.waiting-int(not allocated)),
                            requests=tuple(r for r in instance.requests if r.request_id!=request_id))
                    instances.append(instance)
                self.snapshot=replace(self.snapshot,version=self.snapshot.version+1,
                                      instances=tuple(instances))

    async def update_budget(self,budget):
        async with self.lock:
            route=self.reservations.get(budget.request_id)
            if route and budget.emitted and route.transfer_reserve_bytes:
                self.reservations[budget.request_id]=replace(route,transfer_reserve_bytes=0)
                self.snapshot=replace(self.snapshot,instances=tuple(replace(i,
                    reserved_transfer_bytes=max(0,i.reserved_transfer_bytes-max(0,
                        route.transfer_reserve_bytes-dict(i.transfer_allocations).get(budget.request_id,0))))
                    if i.instance_id==route.decode_id else i for i in self.snapshot.instances))
            instances=tuple(replace(i,requests=tuple(budget if r.request_id==budget.request_id else r
                              for r in i.requests)) for i in self.snapshot.instances)
            self.snapshot=replace(self.snapshot,version=self.snapshot.version+1,instances=instances)

    async def apply_clocks(self,frequencies,parked):
        async with self.lock:
            instances=tuple(replace(i,frequency_mhz=frequencies.get(i.instance_id,i.frequency_mhz),
                                    parked=i.instance_id in parked) for i in self.snapshot.instances)
            if instances!=self.snapshot.instances:
                self.snapshot=replace(self.snapshot,version=self.snapshot.version+1,instances=instances)

    async def apply_windows(self,windows):
        async with self.lock:
            actions={a.instance_id:a for a in windows}
            instances=[]
            for i in self.snapshot.instances:
                a=actions.get(i.instance_id)
                if a and i.generation<=a.expected_generation:
                    i=replace(i,generation=a.expected_generation+1,mode='temporal',admit_prefill=a.admit_prefill)
                instances.append(i)
            if tuple(instances)!=self.snapshot.instances:
                self.snapshot=replace(self.snapshot,version=self.snapshot.version+1,instances=tuple(instances))
