"""Dynamo-owned GPU weight placement and acknowledged overlapping transition.

This is a functional protocol, not a GPU-qualified backend. Abstract shard units
must be translated to real tensors and checked by a hardware transport adapter.
"""
import asyncio
from dataclasses import dataclass,asdict
import inspect
import math
import time


def weight_transfer_plan(source_layout,target_layout,model_units=None):
    from scipy.optimize import linear_sum_assignment
    degrees=[len(group) for group in (*source_layout,*target_layout)]
    if not degrees or any(tp not in (1,2,4,8) for tp in degrees):raise ValueError('supported TP layouts required')
    units=model_units or math.lcm(*degrees)
    if type(units) is not int or units<1 or any(units%tp for tp in degrees):
        raise ValueError('model units must divide all TP layouts')
    for layout in (source_layout,target_layout):
        flat=[g for group in layout for g in group]
        if len(flat)!=len(set(flat)):raise ValueError('a physical GPU cannot execute two replicas in one layout')
    resident={}
    for group in source_layout:
        width=units//len(group)
        for rank,g in enumerate(group):resident[g]=set(range(rank*width,(rank+1)*width))
    transfers=[];retained=0;placements=[]
    # Match all desired ranks against all GPUs allocated to this pool. Keeping
    # the caller's arbitrary pairing can discard weights unnecessarily (e.g.
    # adjacent TP4 quarters both belong to the same target TP2 half).
    target_gpus=[g for group in target_layout for g in group]
    slots=[]
    for group_index,group in enumerate(target_layout):
        width=units//len(group)
        for rank in range(len(group)):
            slots.append((group_index,rank,set(range(rank*width,(rank+1)*width))))
    matrix=[[-len(resident.get(g,set())&shard) for g in target_gpus] for _,_,shard in slots]
    ranks,columns=linear_sum_assignment(matrix)
    assignment={int(rank):target_gpus[int(column)] for rank,column in zip(ranks,columns)}
    for group_index,group in enumerate(target_layout):
        indices=[index for index,(gi,_,_) in enumerate(slots) if gi==group_index]
        placements.append(tuple(assignment[index] for index in indices))
        for index in indices:
            shard=slots[index][2]
            target=assignment[index];here=resident.get(target,set());retained+=len(here&shard)
            grouped={}
            for unit in sorted(shard-here):
                sources=sorted(g for g,owned in resident.items() if unit in owned)
                if not sources:raise ValueError('source GPU layout does not cover every model shard')
                grouped.setdefault(sources[0],[]).append(unit)
            for source,indices in grouped.items():
                transfers.append(dict(source_gpu=source,target_gpu=target,unit_indices=indices,units=len(indices)))
    return dict(model_units=units,retained_units=retained,transfers=transfers,
                target_rank_gpus=placements,method='maximum stationary-weight bipartite matching',
                hardware_qualified=False)


@dataclass(frozen=True)
class Transition:
    transaction_id: str
    source_ids: tuple[str,...]
    source_layout: tuple[tuple[int,...],...]
    target_layout: tuple[tuple[int,...],...]
    overlap_memory_qualified: bool=False
    timeout_s: float=120.
    savings_j: float | None=None
    overhead_j: float | None=None
    target_shapes: tuple[str,...]=()


async def _maybe(value):
    return await value if inspect.isawaitable(value) else value


class Reconfiguration:
    def __init__(self,hooks,journal):
        self.hooks=hooks;self.journal=journal;self.completed={};self.active=set();self.quarantined=set();self.lock=asyncio.Lock()
    async def retire_only(self,t,*,commit):
        if t.target_layout or not t.source_ids:raise ValueError('retire-only requires old instances and no target')
        touched={g for group in t.source_layout for g in group};phases={};started=time.perf_counter();committed=False
        async with self.lock:
            if t.transaction_id in self.completed:
                previous=self.completed[t.transaction_id]
                if previous['transition']!=asdict(t):raise ValueError('retire idempotency key reused')
                return previous
            if touched&(self.active|self.quarantined):raise RuntimeError('retiring GPU group is active or quarantined')
            self.active.update(touched)
        async def call(name):
            before=time.perf_counter()
            try:return await asyncio.wait_for(getattr(self.hooks,name)(t),max(.001,t.timeout_s-(before-started)))
            finally:phases[name+'_s']=time.perf_counter()-before
        try:
            await call('prepare_retirement');await call('freeze');await call('drain');await call('retire')
            await _maybe(commit({'instances':[]}));committed=True;await call('park_retired')
            result=dict(phase='complete',transition=asdict(t),phases=phases,duration_s=time.perf_counter()-started,
                        mechanism='retire_zero_budget_pool',hardware_qualified=False)
            self.completed[t.transaction_id]=result;await _maybe(self.journal('dynamo_transition',**result));return result
        except BaseException:
            if committed:
                self.quarantined.update(touched);await asyncio.wait_for(self.hooks.isolate(t),30.)
            else:
                try:await asyncio.wait_for(self.hooks.restore(t),max(30.,t.timeout_s))
                except BaseException:
                    self.quarantined.update(touched);await asyncio.wait_for(self.hooks.isolate(t),30.)
            raise
        finally:
            async with self.lock:self.active.difference_update(touched)

    async def execute(self,transition,*,commit):
        t=transition
        if (t.savings_j is not None and t.overhead_j is not None and t.savings_j<=t.overhead_j):
            raise ValueError('Dynamo transition does not amortize measured overhead')
        touched={g for group in (*t.source_layout,*t.target_layout) for g in group}
        async with self.lock:
            if t.transaction_id in self.completed:
                previous=self.completed[t.transaction_id]
                if previous['transition']!=asdict(t):raise ValueError('idempotency key reused with another transition')
                return previous
            if touched & self.quarantined:raise RuntimeError('Dynamo GPU group is quarantined after unconfirmed recovery')
            if touched & self.active:raise RuntimeError('overlapping Dynamo GPU transition already active')
            self.active.update(touched)
        started=time.perf_counter();prepared=None;frozen=False;committed=False;phases={}
        plan=None
        async def call(name,*args):
            before=time.perf_counter()
            try:
                fn=getattr(self.hooks,name,None)
                if fn is None:raise RuntimeError('Dynamo transport lacks topology hook: '+name)
                remaining=t.timeout_s-(time.perf_counter()-started)
                if remaining<=0:raise asyncio.TimeoutError('Dynamo transition deadline')
                return await asyncio.wait_for(fn(*args),remaining)
            finally:phases[name+'_s']=time.perf_counter()-before
        try:
            plan=weight_transfer_plan(t.source_layout,t.target_layout)
            if not t.overlap_memory_qualified:
                await call('freeze',t);frozen=True
                await call('drain',t)
                await call('release',t)
            # Only a separately qualified transport may run target CUDA/NCCL
            # during source execution. Generic memory-utilization flags are not
            # proof that overlapping source/target execution is safe.
            prepared=await call('prepare',t,plan)
            verified=await call('verify',t,prepared)
            if not verified or not verified.get('ready') or not verified.get('outputs_valid'):
                raise RuntimeError('target readiness and output ACK required')
            if not frozen:
                await call('freeze',t);frozen=True
                await call('drain',t)
            activated=await call('activate',t,prepared)
            if not activated or not activated.get('activated'):raise RuntimeError('target activation ACK required')
            await _maybe(commit(prepared));committed=True
            await call('retire',t)
            result=dict(phase='complete',transition=asdict(t),phases=phases,
                        duration_s=time.perf_counter()-started,weight_plan=plan,hardware_qualified=False)
            self.completed[t.transaction_id]=result
            await _maybe(self.journal('dynamo_transition',**result))
            return result
        except BaseException as exc:
            # Stop proof has its own recovery deadline; exhausted normal timeout
            # cannot skip abort/isolation or grant another group execution.
            if committed:
                self.quarantined.update(touched)
                await asyncio.wait_for(self.hooks.isolate(t),30.)
                await _maybe(self.journal('dynamo_transition',phase='committed_cleanup_failed',
                    transaction_id=t.transaction_id,error=repr(exc),phases=phases))
                raise
            try:stopped=await asyncio.wait_for(self.hooks.abort(t,prepared),30.)
            except BaseException:stopped=None
            if stopped and stopped.get('target_stopped'):
                if frozen or stopped.get('source_restore_required'):
                    # Stop proof stays tightly bounded. A confirmed-safe cold
                    # source rebuild has its own full transition-sized budget;
                    # real TP engines take longer than the 30-second stop gate.
                    try:await asyncio.wait_for(self.hooks.restore(t),max(30.,t.timeout_s))
                    except BaseException:
                        self.quarantined.update(touched)
                        await asyncio.wait_for(self.hooks.isolate(t),30.)
                        raise
            else:
                self.quarantined.update(touched)
                await asyncio.wait_for(self.hooks.isolate(t),30.)
            await _maybe(self.journal('dynamo_transition',phase='failed',
                transaction_id=t.transaction_id,error=repr(exc),phases=phases,
                source_restored=bool(stopped and stopped.get('target_stopped') and
                                     (frozen or stopped.get('source_restore_required')))))
            raise
        finally:
            async with self.lock:self.active.difference_update(touched)
