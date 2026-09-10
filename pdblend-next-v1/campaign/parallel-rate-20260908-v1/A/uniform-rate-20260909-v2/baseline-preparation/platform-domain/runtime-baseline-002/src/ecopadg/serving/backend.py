"""Async execution backend with exclusive clocks and serialized role changes."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import fcntl
from pathlib import Path
import time

import aiohttp
from .types import InstanceState, RuntimeSnapshot
from .state import ExpiredPlan


class ClockOwner:
    def __init__(self, hardware, gpus, lock_dir="/root/workspace/pdblend/new-results/.clock-locks",
                 settle_timeout_s=.3,max_frequency=2520):
        self.max_frequency = max_frequency
        self.hardware, self.gpus = hardware, tuple(sorted(gpus))
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="gpu-clock-owner")
        self.lock = asyncio.Lock()
        self.files = []
        Path(lock_dir).mkdir(parents=True, exist_ok=True)
        try:
            for gpu in self.gpus:
                handle = open(Path(lock_dir)/f"pdblend-gpu-{gpu}.lock", "a")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except Exception:
                    handle.close()
                    raise
                self.files.append(handle)
        except Exception:
            for handle in self.files:
                handle.close()
            self.pool.shutdown(wait=False)
            raise RuntimeError("GPU clock already has a writer")
        self.applied = {}
        self.epochs={gpu:0 for gpu in self.gpus}
        self.settle_timeout_s=settle_timeout_s
        self.fallbacks={}
        self.deferred={}
        self.last_write_finished_s=0.

    async def set(self, gpus, frequency,*,verify_rise=True):
        # Reserve clock intent before awaiting the writer. This invalidates
        # idle parking planned from an older snapshot of the same GPU.
        for gpu in gpus:
            if gpu not in self.gpus:
                raise ValueError("GPU outside clock ownership")
            self.epochs[gpu]+=1
        async with self.lock:
            loop = asyncio.get_running_loop()
            rising=[]
            for gpu in gpus:
                if gpu not in self.gpus:
                    raise ValueError("GPU outside clock ownership")
                if self.applied.get(gpu) != frequency:
                    self.deferred.pop(gpu,None)
                    if self.applied.get(gpu,0)<frequency:
                        rising.append(gpu)
                    await loop.run_in_executor(self.pool, partial(self.hardware.set_clock,gpu,frequency))
                    self.last_write_finished_s=time.time()
                    self.applied[gpu] = frequency
            # Downshifts may proceed at the conservative new performance
            # estimate. Before relying on a speed-up, observe it on hardware.
            # Operator profiling instead records commanded and observed clocks
            # under its real workload; it must never relabel a safety fallback.
            if not verify_rise: rising=[]
            deadline=time.monotonic()+self.settle_timeout_s
            while rising:
                pending=[]
                for gpu in rising:
                    observed=await loop.run_in_executor(self.pool,self.hardware.current_freq,gpu)
                    if observed<frequency-15:
                        idle=(await loop.run_in_executor(self.pool,self.hardware.clock_idle,gpu)
                              if hasattr(self.hardware,'clock_idle') else False)
                        if idle:
                            self.deferred[gpu]=dict(target=frequency,gpus=tuple(gpus),active_since=None)
                        else:
                            pending.append(gpu)
                if pending and time.monotonic()>=deadline:
                    # Instantaneous SM clocks can be capped under compute
                    # load even though the driver accepted the lock. Restore
                    # the maximum command and record uncertainty; do not turn
                    # one uncertain frequency observation into a node outage.
                    for gpu in gpus:
                        if self.applied.get(gpu)!=self.max_frequency:
                            await loop.run_in_executor(self.pool,partial(self.hardware.set_clock,gpu,self.max_frequency))
                            self.last_write_finished_s=time.time()
                        self.applied[gpu]=self.max_frequency
                        self.deferred.pop(gpu,None)
                        self.fallbacks[gpu]=dict(at_s=time.time(),requested=frequency,
                            reason='observed SM clock below target after settling bound')
                    break
                rising=pending
                if rising:
                    await asyncio.sleep(.005)

    async def verify_deferred(self):
        """Confirm idle wakeups while work runs; active failures restore capacity.

        Admission reserves the measured settling bound. Hardware idle samples
        cannot establish an active speed-up, so they remain unconfirmed until
        a later sample. SLO and stale-telemetry recovery remain independent.
        """
        failed={}
        async with self.lock:
            loop=asyncio.get_running_loop()
            for gpu in tuple(self.deferred):
                item=self.deferred.get(gpu)
                if item is None: continue
                if self.applied.get(gpu)!=item['target']:
                    self.deferred.pop(gpu,None);continue
                observed=await loop.run_in_executor(self.pool,self.hardware.current_freq,gpu)
                if observed>=item['target']-15:
                    self.deferred.pop(gpu,None);continue
                idle=await loop.run_in_executor(self.pool,self.hardware.clock_idle,gpu)
                if idle: continue
                now=time.monotonic()
                if item['active_since'] is None: item['active_since']=now
                if now-item['active_since']<self.settle_timeout_s: continue
                for member in item['gpus']:
                    if self.applied.get(member)!=item['target']: continue
                    await loop.run_in_executor(self.pool,partial(self.hardware.set_clock,member,self.max_frequency))
                    self.last_write_finished_s=time.time()
                    self.applied[member]=self.max_frequency
                    self.deferred.pop(member,None)
                    event=dict(at_s=time.time(),requested=item['target'],
                        reason='active SM clock below target after idle wakeup settling bound')
                    self.fallbacks[member]=event;failed[member]=event
        return failed

    async def close(self):
        async with self.lock:
            errors=[]
            for gpu in self.gpus:
                try:
                    await asyncio.get_running_loop().run_in_executor(self.pool,self.hardware.reset_clock,gpu)
                    self.last_write_finished_s=time.time()
                except Exception as exc:
                    errors.append(str(exc))
            for handle in self.files:
                handle.close()
            self.pool.shutdown(wait=False, cancel_futures=True)
            if errors:
                raise RuntimeError("clock restoration failed: " + "; ".join(errors))

    async def park(self,gpus,expected_epochs=None):
        expected_epochs=dict(self.epochs) if expected_epochs is None else expected_epochs
        async with self.lock:
            for gpu in gpus:
                if gpu not in self.gpus:
                    raise ValueError("GPU outside clock ownership")
                if self.epochs[gpu]!=expected_epochs[gpu]:
                    continue
                if gpu in self.applied:
                    await asyncio.get_running_loop().run_in_executor(self.pool,self.hardware.reset_clock,gpu)
                    self.last_write_finished_s=time.time()
                    self.applied.pop(gpu,None)
                    self.deferred.pop(gpu,None)
            return all(self.epochs[g]==expected_epochs[g] for g in gpus)


class HttpEngineBackend:
    def __init__(self, instances, session, clocks=None,park_grace_s=.5,max_frequency=2520):
        self.max_frequency = clocks.max_frequency if clocks is not None else max_frequency
        self.instances = {i["id"]:i for i in instances}
        self.session, self.clocks = session, clocks
        self.version = 0
        self.topology_version = 0
        self.role_locks = {i:asyncio.Lock() for i in self.instances}
        self.frequency = {i:self.max_frequency for i in self.instances}
        self.last = {}
        self.parked=set()
        self.idle_since={}
        self.park_grace_s=park_grace_s
        self.frequency_outcomes=[]
        self.dvfs_resume_s={}
        self.inflight_actions=0
        self.last_action_finished_s=0.

    async def json(self, instance_id, path, payload=None):
        instance = self.instances[instance_id]
        method = self.session.post if payload is not None else self.session.get
        kwargs = {"json":payload} if payload is not None else {}
        kwargs["timeout"]=aiohttp.ClientTimeout(total=.5 if path in ("/runtime","/health") else 35)
        async with method(instance["url"]+path,**kwargs) as response:
            if response.status != 200:
                raise RuntimeError(f"{instance_id} {path}: {response.status} {await response.text()}")
            return await response.json()

    async def read_state(self):
        async def read(i,c):
            try:
                raw = await self.json(i,"/runtime")
                self.last[i] = raw
                fresh = not (raw.get("error") or raw.get('diagnostic_recompute') or raw.get('diagnostic_transport')
                             or not raw.get('admit_decode',True))
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
                raw = self.last.get(i,{})
                fresh = False
            return InstanceState(i,raw.get("role",c["role"]),c["tp"],tuple(c["gpus"]),
                raw.get("timestamp",0),raw.get("generation",0),self.frequency.get(i,self.max_frequency),
                raw.get("free_kv_tokens",0),raw.get("running",0),raw.get("waiting",0),
                accepting=fresh and raw.get("accepting",False),
                kv_allocations=tuple((rid.split(':')[1] if rid.startswith('pdb:') else rid,n)
                                     for rid,n in raw.get('kv_allocations',{}).items()),parked=i in self.parked,
                free_transfer_bytes=raw.get('free_transfer_bytes',0),
                transfer_bytes_per_token=raw.get('transfer_bytes_per_token',0),
                transfer_allocations=tuple(raw.get('transfer_allocations',{}).items()),
                mode=raw.get('mode','continuous'),admit_prefill=raw.get('admit_prefill',True),
                dvfs_allowed=time.time()>=self.dvfs_resume_s.get(i,0))
        epoch=self.topology_version
        states=await asyncio.gather(*(read(i,c) for i,c in tuple(self.instances.items())))
        if epoch!=self.topology_version:
            return await self.read_state()
        self.version+=1
        return RuntimeSnapshot(self.version,time.time(),tuple(states))

    def replace_instances(self,remove_ids,added):
        # Called under the controller action lock after all removed requests
        # drain. Concurrent state reads retry when this topology epoch changes.
        if not set(remove_ids)<=set(self.instances):
            raise ValueError('topology source changed before commit')
        updated={i:c for i,c in self.instances.items() if i not in remove_ids}
        for config in added:
            if config['id'] in updated:
                raise ValueError('duplicate replacement instance')
            updated[config['id']]=dict(config)
            self.frequency[config['id']]=self.max_frequency
            self.role_locks[config['id']]=asyncio.Lock()
        self.instances=updated
        self.topology_version+=1
        for i in remove_ids:
            self.parked.discard(i);self.idle_since.pop(i,None)

    async def execute(self, plan):
        changes=bool(plan.frequencies or plan.roles or plan.windows)
        if changes: self.inflight_actions+=1
        try:
            await self._execute(plan)
        finally:
            if changes:
                self.inflight_actions-=1
                self.last_action_finished_s=time.time()

    async def _execute(self, plan):
        if time.time() > plan.expires_s:
            raise ExpiredPlan("expired execution plan before actions")
        # Frequency actions precede prefill dispatch; P->D dependencies belong
        # to the route executor, never concurrent fire-and-forget tasks.
        for action in plan.frequencies:
            started=time.time()
            desired=self.max_frequency if started<self.dvfs_resume_s.get(action.instance_id,0) else action.frequency_mhz
            if self.clocks:
                await self.clocks.set(self.instances[action.instance_id]["gpus"],desired)
            elif action.frequency_mhz != self.frequency[action.instance_id]:
                raise RuntimeError("no clock owner configured")
            actual=(max(self.clocks.applied[g] for g in self.instances[action.instance_id]['gpus'])
                    if self.clocks else action.frequency_mhz)
            self.frequency[action.instance_id]=actual
            uncertain=bool(self.clocks and any(self.clocks.fallbacks.get(g,{}).get('at_s',0)>=started
                                            for g in self.instances[action.instance_id]['gpus']))
            if uncertain:
                self.dvfs_resume_s[action.instance_id]=time.time()+5
            self.frequency_outcomes.append(dict(instance_id=action.instance_id,requested=action.frequency_mhz,
                commanded=actual,conservative_fallback=uncertain,at_s=time.time()))
            self.frequency_outcomes=self.frequency_outcomes[-256:]
            self.parked.discard(action.instance_id)
        for action in sorted(plan.windows,key=lambda a:a.admit_prefill):
            async with self.role_locks[action.instance_id]:
                raw=await self.json(action.instance_id,'/runtime')
                if raw['generation']!=action.expected_generation:
                    raise RuntimeError('stale window generation')
                await self.json(action.instance_id,'/control',dict(generation=raw['generation']+1,
                    role='mixed',mode='temporal',admit_prefill=action.admit_prefill))
        for action in plan.roles:
            if action.savings_lower_j <= action.switching_upper_j:
                raise ValueError("role change cannot amortize switching cost")
            async with self.role_locks[action.instance_id]:
                raw=await self.json(action.instance_id,"/runtime")
                if raw["generation"] != action.expected_generation:
                    raise RuntimeError("stale role generation")
                await self.json(action.instance_id,"/control",dict(
                    generation=raw["generation"]+1,role=action.role,
                    mode=raw["mode"],admit_prefill=True))

    async def confirm(self, plan):
        for action in plan.windows:
            raw=await self.json(action.instance_id,'/runtime')
            if (raw['generation']!=action.expected_generation+1 or raw['mode']!='temporal'
                    or raw['admit_prefill']!=action.admit_prefill):
                return False
        for action in plan.roles:
            raw=await self.json(action.instance_id,"/runtime")
            if raw["role"] != action.role or raw["generation"] != action.expected_generation+1:
                return False
        return all(self.frequency[a.instance_id]>=a.frequency_mhz for a in plan.frequencies)

    async def cancel(self, request_id):
        fields=request_id.split(':')
        targets=fields[3:5] if len(fields)==5 and fields[0]=='pdb' else tuple(self.instances)
        targets=tuple(i for i in dict.fromkeys(targets) if i in self.instances)
        async def cancel_one(i):
            return await asyncio.wait_for(self.json(i,'/cancel',dict(request_id=request_id)),2)
        results=await asyncio.gather(*(cancel_one(i) for i in targets),return_exceptions=True)
        return {i:repr(r) for i,r in zip(targets,results) if isinstance(r,BaseException)}

    async def park_idle(self,instances):
        if self.clocks is None:
            return
        expected_epochs=dict(self.clocks.epochs)
        for instance in instances:
            if not instance.accepting:
                continue
            if (not instance.requests and not instance.running and not instance.waiting
                    and not instance.reserved_kv_tokens and instance.instance_id not in self.parked):
                since=self.idle_since.setdefault(instance.instance_id,time.monotonic())
                if time.monotonic()-since<self.park_grace_s:
                    continue
                if await self.clocks.park(instance.gpus,expected_epochs):
                    self.parked.add(instance.instance_id)
            elif instance.requests or instance.running or instance.waiting or instance.reserved_kv_tokens:
                self.idle_since.pop(instance.instance_id,None)

    async def verify_clocks(self):
        if self.clocks is None: return
        failed=await self.clocks.verify_deferred()
        if not failed: return
        for instance_id,config in self.instances.items():
            if set(config['gpus']).intersection(failed):
                self.frequency[instance_id]=max(self.clocks.applied.get(g,self.max_frequency) for g in config['gpus'])
                self.dvfs_resume_s[instance_id]=time.time()+5
                self.frequency_outcomes.append(dict(instance_id=instance_id,commanded=self.max_frequency,
                    conservative_fallback=True,reason='active wakeup verification',at_s=time.time()))
        self.frequency_outcomes=self.frequency_outcomes[-256:]
