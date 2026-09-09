"""Asynchronous request lifecycle and independently scheduled control tasks."""
import asyncio
from collections import deque
from contextlib import suppress
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import uuid

import aiohttp
from aiohttp import web
from .backend import ClockOwner, HttpEngineBackend
from .observe import Journal, PlanningStats
from .planner import JointPlanner, TransferCost
from .profiles import ProfileStore, OutputPredictor
from .state import StateManager, StalePlan, ExpiredPlan
from .reconfigure import ResidentRolePlanner, RoleCost,search_roles
from .ecoserve import EcoServeScheduler
from .distserve import DistServeScheduler
from .dynamo import DynamoScheduler
from .dynamo_topology import DynamoTopologyPlanner,TopologyCost
from .topology import TopologyManager,DockerLifecycle,InstanceSpec,validate_layout
from .stream import CompletionAccumulator
from .interconnect import InterconnectTopology
from .forecast import forecast_roles
from .frequency import FrequencyCost,FrequencyPlanner
from .pending_frequency import risk_plan as measured_risk_plan, capacity_plan as pending_capacity_plan, frequency_plan as pending_frequency_plan
from .capacity_admission import plan as capacity_admission_plan
from .admission_diagnostics import attach as attach_admission_diagnostics, observed_plan
from .pd_topology import PDBlendTopologyPlanner,MeasuredCapacity,idle as topology_idle
from .admission import AdmissionQueue, AdmissionRejected
from .planning_executor import PlanningExecutor
from .tails import admission_budget
from .idle_admission import current_clock_first_plan
from .completion_policy import (PROTOCOL as EVALUATION_PROTOCOL, benchmark_timing,
                                recovery_budget, recovery_snapshot, engine_residual)
from .types import (RuntimeSnapshot, RequestBudget, ControlPlan, FrequencyAction,
                    RouteAction)

STRATEGIES=("mixed", "mixed_dvfs", "pdblend-greedy", "pdblend-joint", "pdblend-dynamic", "ecoserve", "distserve", "dynamollm-resident", "dynamollm")


def admission_response(code,message):
    return web.HTTPTooManyRequests(content_type='application/json',text=json.dumps(
        dict(error=dict(type='admission_rejection',code=code,message=message))))


class Controller:
    def __init__(self,config):
        self.config=config
        self.strategy=config["strategy"]
        if self.strategy not in STRATEGIES:
            raise ValueError("strategy has not passed its implementation gate: "+self.strategy)
        if self.config.get('measured_frequency_write_guard_v1') is True and (
                not self.strategy.startswith('pdblend') or self.config.get('allow_pd') is not False
                or self.config.get('topology') or self.config.get('slow_topology')
                or self.config.get('dynamic_pools') or any(i.get('role')!='mixed' for i in config['instances'])):
            raise ValueError('physical measured clock guard requires independent mixed PDB')
        self.state=StateManager(RuntimeSnapshot(0,time.time(),()))
        self.pending=AdmissionQueue(config.get("max_pending",256),
            round_fairness=self.strategy.startswith("pdblend") and config.get("admission_round_fairness") is True)
        self.active={}
        self.evaluation_v3=config.get('evaluation_protocol') == EVALUATION_PROTOCOL
        self.request_tasks=set()
        self.accepting_input=True
        self.predictor=OutputPredictor(config.get("output_prior",256))
        self.journal=Journal(config["journal"])
        self.planning_stats=PlanningStats()
        self.profiles=ProfileStore.load(config["profiles"]) if config.get("profiles") else ProfileStore(())
        self.interconnect=(InterconnectTopology.parse(Path(config['interconnect']).read_text())
                           if config.get('interconnect') else None)
        self.planner=JointPlanner(self.profiles,
            transfers=[TransferCost(**x) for x in config.get("transfers",[])],
            allow_pd=self.strategy.startswith("pdblend") and config.get('allow_pd',True),
            dvfs=self.strategy!="mixed" and config.get('dvfs',True),
            clock_settle_s=config.get('clock_settle_s',.3),
            frequency_costs=config.get('frequency_costs',()),
            decision_budget_s=config.get('decision_budget_s',.01),
            telemetry_ttl_s=config.get("telemetry_ttl_s",1),topology=self.interconnect,
            reserved_batch_guard=(self.strategy.startswith("pdblend") and config.get("pending_admission_capacity_guard") is True),
            protect_pending_decode=(self.strategy.startswith('pdblend')
                and config.get('protect_pending_decode',False)),
            residency_horizon=(self.strategy.startswith('pdblend')
                and config.get('residency_horizon',False) and config.get('park_idle',True)),
            park_grace_s=config.get('park_grace_s',.5),
            independent_idle_mixed_on_stale_tail=(self.strategy.startswith('pdblend')
                and config.get('independent_idle_mixed_on_stale_tail') is True))
        self.admission_diagnostics=attach_admission_diagnostics(self.planner,
            self.strategy.startswith('pdblend') and config.get('admission_round_fairness') is True)
        self.failure=None
        self.frozen_instances=set()
        self.action_lock=asyncio.Lock()
        self.control_stop=asyncio.Event()
        self.ttft_history=deque(maxlen=1024)
        self.arrival_history=deque(maxlen=4096)
        self.history_started_s=time.time()
        self.eco_scheduler=(EcoServeScheduler(self.profiles,
            [i['id'] for i in config['instances']][:config.get('eco_initial_instances',len(config['instances']))],
            lower=config.get('eco_macro_lower',2),upper=config.get('eco_macro_upper',3))
            if self.strategy=='ecoserve' else None)
        self.distserve_scheduler=(DistServeScheduler(self.profiles,self.planner.transfers,
            prefill_batch=config['distserve_prefill_batch'],decode_batch=config['distserve_decode_batch'],
            clock_settle_s=config.get('clock_settle_s',.3),topology=self.interconnect)
            if self.strategy=='distserve' else None)
        self.dynamo_scheduler=(DynamoScheduler(self.profiles,config['dynamo_assignments'],
            clock_settle_s=config.get('clock_settle_s',.3),
            frequency_costs=config.get('frequency_costs',()),
            input_cuts=tuple(config.get('dynamo_input_cuts',(255,1023))),
            output_cuts=tuple(config.get('dynamo_output_cuts',(99,349))))
            if self.strategy in ('dynamollm-resident','dynamollm') else None)
        self.topology_manager=None
        self.slow_task=None
        self.slow_pending=deque(maxlen=2)
        self.retained_weights=config.get('retained_weights')
        self.pd_topology=None
        if self.strategy=='pdblend-dynamic' and config.get('slow_topology',False):
            if not config.get('topology'):
                raise ValueError('PDBlend slow topology requires a physical execution backend')
            self.pd_topology=PDBlendTopologyPlanner(self.planner,
                [TopologyCost(**c) for c in config.get('topology_costs',[])],
                [MeasuredCapacity(**c) for c in config.get('measured_capacities',[])],
                node_gpus=tuple(config.get('node_gpus',range(8))),
                error_fraction=config.get('reconfiguration_error_fraction',.3))
        if self.strategy=='dynamollm':
            if not config.get('topology') or not config.get('topology_costs'):
                raise ValueError('DynamoLLM requires a physical topology backend and measured switch costs')
            self.dynamo_topology=DynamoTopologyPlanner(self.profiles,
                [TopologyCost(**c) for c in config['topology_costs']],
                park_grace_s=config.get('park_grace_s',.5))
        self.role_planner=ResidentRolePlanner([RoleCost(**c) for c in config.get("role_costs",[])])
        self.frequency_planner=(FrequencyPlanner(self.planner,[FrequencyCost(**c) for c in config.get('frequency_costs',[])])
            if (self.strategy=='mixed_dvfs' or self.strategy.startswith('pdblend')) and config.get('dvfs',True) else None)
        self.last_frequency_optimization_s=0.
        if self.strategy=="pdblend-dynamic" and not self.role_planner.costs:
            raise ValueError("dynamic policy requires measured role switching costs")
        self.planning_executor=PlanningExecutor()

    async def recovery_plan(self, snapshot, request, now, *, pressure_pending=None):
        """Finish work through the native strategy without rewriting its SLO."""
        if not self.evaluation_v3 or request.hard_deadline_s is None or now >= request.hard_deadline_s:
            return None
        pending=(recovery_budget(request, now),)
        recovered=recovery_snapshot(snapshot, now)
        if self.strategy == 'mixed':
            plan=self.max_mixed(request, now)
        elif self.eco_scheduler or self.dynamo_scheduler:
            plan=(self.eco_scheduler or self.dynamo_scheduler).plan(recovered, pending, now=now)
        elif self.distserve_scheduler:
            plan=await self.planning_executor.run(self.distserve_scheduler.plan, recovered, pending, now=now)
        else:
            # Preserve routing/physical/profile constraints; only this request's
            # failed latency target is replaced for completion planning.
            plan=await self.planning_executor.run(observed_plan,capacity_admission_plan, self.planner, recovered, pending, now=now, joint=False, enabled=self.pending_capacity_enabled(), pressure_pending=pressure_pending)
        if plan is None or not plan.feasible:
            return None
        return replace(plan, reason='completion recovery; original SLO retained: '+plan.reason,
                       expires_s=min(plan.expires_s, request.hard_deadline_s))

    async def start(self,app):
        self.clock_owner=None
        try:
            await self.initialize(app)
        except BaseException:
            await self.close_resources()
            raise

    async def close_resources(self):
        async def close_journal():
            if not self.journal_task.done():
                await self.journal.close()
            await self.journal_task
        closing=[self.planning_executor.close()]
        if hasattr(self,'journal_task'): closing.append(close_journal())
        if hasattr(self,'session'): closing.append(self.session.close())
        clocks=getattr(self,'clock_owner',None) or getattr(getattr(self,'backend',None),'clocks',None)
        if clocks: closing.append(clocks.close())
        # A failed log flush must not prevent clock restoration or thread join.
        results=await asyncio.gather(*closing,return_exceptions=True)
        if self.admission_diagnostics is not None:
            await asyncio.to_thread(self.admission_diagnostics.dump,Path(self.config['journal']).with_name('admission_diagnostics.json'))
        for result in results:
            if isinstance(result,BaseException): raise result

    async def initialize(self,app):
        self.tokenizer = None
        if self.config.get("tokenizer"):
            from transformers import AutoTokenizer
            self.tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained,
                                                       self.config["tokenizer"])
        self.tokenizer_slots = asyncio.Semaphore(2)
        self.session=aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120),
                                         connector=aiohttp.TCPConnector(limit=256),trust_env=False)
        clocks=None
        if self.config.get("manage_clocks",True):
            from ecopadg.measure.backends import PynvmlBackend
            hardware=await asyncio.to_thread(PynvmlBackend)
            gpus=set(self.config.get("node_gpus",range(8) if self.config.get("topology") else
                {g for i in self.config["instances"] for g in i["gpus"]}))
            clocks=await asyncio.to_thread(ClockOwner,hardware,gpus)
            self.clock_owner=clocks
            await clocks.set(sorted(gpus),2520)
            unused=gpus-{g for i in self.config['instances'] for g in i['gpus']}
            if unused: await clocks.park(sorted(unused),dict(clocks.epochs))
        self.backend=HttpEngineBackend(self.config["instances"],self.session,clocks,
                                       park_grace_s=self.config.get('park_grace_s',.5))
        if self.config.get('measured_frequency_write_guard_v1') is True:
            self.backend.exact_frequency_confirmation=True
            if clocks is not None:
                clocks.write_guard=self.clock_write_allowed
                clocks.state_lock=self.state.lock
                clocks.write_guard_journal=str(Path(self.config['journal']).with_suffix('.clock-guard.jsonl'))
        if self.config.get('topology'):
            t=self.config['topology']
            template=json.loads(await asyncio.to_thread(Path(t['engine_template']).read_text))
            specs=[InstanceSpec(i['id'],i['tp'],tuple(i['gpus']),i['port'],i['kv_port'],i['role'])
                   for i in self.config['instances']]
            self.topology_manager=TopologyManager(self.backend,
                DockerLifecycle(t['runtime_dir'],t['image'],template),specs,
                self.config.get('node_gpus',list(range(8))),self.journal,
                freeze=self.freeze_topology,commit=self.commit_topology)
        # Each independent cell restores its explicit starting roles/windows.
        # Otherwise a prior EcoServe cell can leave the next strategy paused.
        for instance_id,instance in self.backend.instances.items():
            raw=await self.backend.json(instance_id,'/runtime')
            desired=dict(role='mixed' if self.eco_scheduler or self.dynamo_scheduler
                         else instance.get('role','mixed'),
                         mode='temporal' if self.eco_scheduler else 'continuous',
                         admit_prefill=not bool(self.eco_scheduler),admit_decode=True)
            current={k:raw.get(k,True if k.startswith('admit_') else 'continuous') for k in desired}
            if current!=desired:
                if raw.get('running') or raw.get('waiting') or raw.get('active'):
                    raise RuntimeError('initial strategy configuration requires drained instances')
                await self.backend.json(instance_id,'/control',dict(desired,generation=raw['generation']+1))
        if self.dynamo_scheduler:
            self.dynamo_scheduler.hierarchy.due(time.time())
        ids=sorted(self.backend.instances)
        if self.config.get("prepare_peers",True):
            for index,instance_id in enumerate(ids):
                if ids[index+1:]:
                    await self.backend.json(instance_id,"/prepare-peers",dict(peers=ids[index+1:]))
        self.journal_task=asyncio.create_task(self.journal.run())
        await self.refresh()
        self.telemetry_task=asyncio.create_task(self.telemetry())
        self.dispatch_task=asyncio.create_task(self.dispatch())
        self.role_task=(asyncio.create_task(self.roles()) if self.strategy=="pdblend-dynamic"
                        and self.config.get('dynamic_pools',True) else
                        asyncio.create_task(self.eco_resize()) if self.eco_scheduler else
                        asyncio.create_task(self.dynamo_control()) if self.dynamo_scheduler else None)
        self.pd_slow_control_task=asyncio.create_task(self.pd_slow_control()) if self.pd_topology else None

    async def refresh(self):
        snapshot=await self.backend.read_state()
        instances=[]
        for i in snapshot.instances:
            budgets=tuple(a["budget"] for a in self.active.values()
                          if a.get("route") and (a["route"].decode_id==i.instance_id
                            or (a["route"].prefill_id==i.instance_id and not a.get("prefill_done"))))
            instances.append(replace(i,requests=budgets,
                                     waiting=max(i.waiting,sum(r.emitted==0 and
                                         r.request_id not in dict(i.kv_allocations) for r in budgets)),
                                     accepting=i.accepting and i.instance_id not in self.frozen_instances))
        await self.state.publish(instances,time.time(),
            engine_waiting={i.instance_id:i.waiting for i in snapshot.instances})

    def pending_capacity_enabled(self):
        return self.strategy.startswith('pdblend') and self.config.get('pending_admission_capacity_guard') is True

    def frequency_pending_budgets(self):
        if not (self.strategy.startswith('pdblend')
                and self.config.get('pending_admission_capacity_guard') is True):return ()
        return tuple(a['budget'] for a in self.active.values()
                     if not a.get('route') and not a['future'].done())

    def clock_write_allowed(self,gpus,frequency,reason,bootstrap=None):
        from .physical_frequency import evaluate
        return evaluate(self,gpus,frequency,reason,bootstrap)

    async def telemetry(self):
        try:
            while await self.control_tick(.1):
                await self.refresh()
                if self.control_stop.is_set(): break
                if self.config.get('measured_frequency_write_guard_v1') is True:
                    try:
                        async with self.action_lock:
                            await self.backend.verify_clocks()
                    except ExpiredPlan:
                        continue
                else:
                    await self.backend.verify_clocks()
                if self.control_stop.is_set(): break
                if self.config.get("park_idle",True):
                    if self.config.get('measured_frequency_write_guard_v1') is True:
                        try:
                            async with self.action_lock:
                                await self.backend.park_idle(self.state.snapshot.instances)
                        except ExpiredPlan:
                            continue
                    else:
                        await self.backend.park_idle(self.state.snapshot.instances)
                    await self.state.apply_clocks(self.backend.frequency,self.backend.parked)
                now=time.time()
                # Stale telemetry or a per-request miss always restores service
                # capacity, independent of the admission planner's energy goal.
                risk=[i for i in self.state.snapshot.instances
                      if i.instance_id not in self.frozen_instances and
                      (now-i.timestamp_s > self.planner.telemetry_ttl_s
                       or any(r.next_token_remaining(now)<0 for r in i.requests))]
                if risk and self.pending_capacity_enabled():
                    if self.frequency_planner is None: continue
                    async with self.action_lock:
                        if self.control_stop.is_set(): break
                        plan=await self.planning_executor.run(measured_risk_plan,self.frequency_planner,
                            self.state.snapshot,time.time(),self.frequency_pending_budgets(),tuple(self.frozen_instances))
                        if self.control_stop.is_set(): break
                        # Recheck the actual fresh ledger and new/cancelled
                        # arrivals after the worker while retaining the lock.
                        plan=measured_risk_plan(self.frequency_planner,self.state.snapshot,time.time(),
                            self.frequency_pending_budgets(),tuple(self.frozen_instances))
                        if plan.snapshot_version!=self.state.snapshot.version or time.time()>plan.expires_s: continue
                        if plan.frequencies:
                            try: await self.backend.execute(plan)
                            except ExpiredPlan: continue
                            if not await self.backend.confirm(plan): raise RuntimeError('risk frequency plan unconfirmed')
                            await self.state.apply_clocks(self.backend.frequency,self.backend.parked)
                        await self.journal.emit(dict(kind='frequency_risk_epoch',at_s=time.time(),plan=asdict(plan)))
                elif risk:
                    async with self.action_lock:
                        if self.control_stop.is_set(): break
                        try:
                            await self.backend.execute(ControlPlan(self.state.snapshot.version,now,now+1,
                                frequencies=tuple(FrequencyAction(i.instance_id,2520) for i in risk
                                                  if i.instance_id in self.backend.instances),
                                reason="stale telemetry or per-request SLO risk"))
                        except ExpiredPlan:
                            continue
                        await self.state.apply_clocks(self.backend.frequency,self.backend.parked)
                elif self.frequency_planner and now-self.last_frequency_optimization_s>=self.config.get('frequency_period_s',.5):
                    async with self.action_lock:
                        if self.control_stop.is_set(): break
                        pending=self.frequency_pending_budgets()
                        plan=await self.planning_executor.run(
                            pending_frequency_plan,self.frequency_planner,self.state.snapshot,
                            time.time(),pending,tuple(self.frozen_instances),self.pending_capacity_enabled())
                        if self.control_stop.is_set(): break
                        current_pending=self.frequency_pending_budgets()
                        if current_pending:
                            plan=pending_capacity_plan(self.frequency_planner,self.state.snapshot,
                                time.time(),current_pending,tuple(self.frozen_instances))
                        elif pending:
                            continue
                        self.last_frequency_optimization_s=now
                        if (plan.snapshot_version!=self.state.snapshot.version
                                or time.time()>plan.expires_s):
                            continue
                        if plan.frequencies:
                            try:
                                await self.backend.execute(plan)
                            except ExpiredPlan:
                                continue
                            if not await self.backend.confirm(plan): raise RuntimeError('periodic frequency plan unconfirmed')
                            await self.state.apply_clocks(self.backend.frequency,self.backend.parked)
                            await self.journal.emit(dict(kind='frequency_epoch',at_s=time.time(),plan=asdict(plan)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failure=str(exc)

    def max_mixed(self,request,now):
        snapshot=self.state.snapshot
        candidates=[i for i in snapshot.instances if i.role=="mixed" and i.accepting
                    and now-i.timestamp_s<=1 and i.free_kv_tokens-i.reserved_kv_tokens
                    >= request.input_tokens+(request.output_limit or request.predicted_output)]
        if not candidates:
            return None
        i=min(candidates,key=lambda x:(len(x.requests)+x.waiting,x.instance_id))
        return ControlPlan(snapshot.version,now,now+1,
            routes=(RouteAction(request.request_id,i.instance_id,i.instance_id,
                request.input_tokens+(request.output_limit or request.predicted_output),0,0,0),),
            frequencies=(FrequencyAction(i.instance_id,2520),),
            reason="full-frequency mixed control; predictions not used")

    def stale_admission(self, active, plan, error=None):
        self.planning_stats.discard('stale')
        if self.admission_diagnostics is not None:
            self.admission_diagnostics.mark(active['budget'].request_id,'state_expired',detail=repr(error),snapshot_version=plan.snapshot_version,current_version=self.state.snapshot.version)
        if self.strategy not in ('pdblend-joint','pdblend-dynamic'):
            return
        if isinstance(error,ExpiredPlan):
            reason='backend_expired'
        else:
            changed=plan.snapshot_version!=self.state.snapshot.version
            expired=time.time()>plan.expires_s
            reason=('version_and_expiry' if changed and expired else 'version' if changed
                    else 'expired' if expired else 'capacity')
        self.planning_stats.counts['pdb_stale_'+reason]+=1
        if self.config.get('stale_single_request_retry',True):
            # Recompute on the next fresh snapshot; never rebase an old plan.
            # This marker is owned by the request's existing active record.
            active['pdb_single_retry']=True

    async def record_idle_tail_fallback(self, snapshot, request, plan, active):
        """Keep the first decision input; admission events prove execution."""
        if (not self.strategy.startswith('pdblend')
                or not getattr(self.planner, 'independent_idle_mixed_on_stale_tail', False)
                or active.get('idle_tail_fallback_snapshot_recorded')
                or 'fresh idle mixed fallback;' not in plan.reason):
            return
        active['idle_tail_fallback_snapshot_recorded']=True
        await self.journal.emit(dict(kind='pdb_stale_tail_fallback_snapshot',
            at_s=time.time(), request_id=request.request_id,
            client_request_id=active.get('client_request_id'),
            request=asdict(request), snapshot=asdict(snapshot), plan=asdict(plan),
            completion_recovery=plan.reason.startswith('completion recovery;'),
            scope='first fallback input; original planned-arrival SLO retained; execution requires admission'))

    async def dispatch(self):
        current=None
        try:
            while True:
                current=await self.pending.get()
                rid=current
                if rid not in self.active:
                    self.pending.done(rid)
                    continue
                active=self.active[rid]
                if active["future"].cancelled():
                    self.pending.done(rid)
                    continue
                request=active["budget"]
                retry=False
                while rid in self.active:
                    now=time.time()
                    deadline=(request.hard_deadline_s if self.evaluation_v3 and request.hard_deadline_s is not None
                              else request.arrival_s+request.ttft_s)
                    if now>deadline:
                        if self.admission_diagnostics is not None:self.admission_diagnostics.mark(rid,'request_deadline_expired',deadline_s=deadline)
                        raise_for=AdmissionRejected('admission_deadline',"admission deadline expired")
                        if not active["future"].done():
                            active["future"].set_exception(raise_for)
                        break
                    pending=tuple(a["budget"] for a in self.active.values() if not a.get("route"))
                    pressure_pending=tuple(a["budget"] for a in self.active.values()
                        if not a.get("route") and not a["future"].done())
                    if self.strategy=="mixed_dvfs" or (
                            self.strategy in ('pdblend-joint','pdblend-dynamic')
                            and active.get('pdb_single_retry',False)):
                        pending=(request,)
                        if self.strategy!='mixed_dvfs':
                            self.planning_stats.counts['pdb_single_request_retries']+=1
                    snapshot=self.state.snapshot
                    completion_recovery=False
                    planning_started=time.perf_counter()
                    if self.evaluation_v3:
                        timing=active.setdefault('timing', {})
                        if 'first_planning_s' not in timing:
                            timing['first_planning_s']=time.time()
                            timing['first_snapshot_s']=snapshot.timestamp_s
                            timing['first_instance_ages_s']={i.instance_id:now-i.timestamp_s for i in snapshot.instances}
                    if self.strategy=="mixed":
                        plan=self.max_mixed(request,now)
                    elif self.eco_scheduler or self.dynamo_scheduler:
                        # These policies own mutable assignments and windows.
                        policy=self.eco_scheduler or self.dynamo_scheduler
                        plan=policy.plan(snapshot,(request,),now=now)
                    elif self.distserve_scheduler:
                        plan=await self.planning_executor.run(
                            self.distserve_scheduler.plan,snapshot,(request,),now=now)
                    else:
                        plan=await self.planning_executor.run(observed_plan,capacity_admission_plan,self.planner,snapshot,
                            (request,)+tuple(r for r in pending if r.request_id!=rid),
                            joint=self.strategy in ("mixed_dvfs","pdblend-joint","pdblend-dynamic"),now=now,enabled=self.pending_capacity_enabled(),pressure_pending=pressure_pending)
                    self.planning_stats.finish(plan,time.perf_counter()-planning_started)
                    if self.evaluation_v3 and (plan is None or not plan.feasible):
                        recovered=await self.recovery_plan(snapshot, request, time.time(),pressure_pending=pressure_pending)
                        if recovered is not None:
                            plan=recovered
                            completion_recovery=True
                            self.planning_stats.counts['completion_recovery']+=1
                    if rid not in self.active or active["future"].cancelled():
                        self.planning_stats.discard('cancelled_request')
                        if self.admission_diagnostics is not None:self.admission_diagnostics.mark(rid,'request_cancelled')
                        break
                    if plan is None:
                        self.pending.defer(rid);retry=True
                        break
                    await self.record_idle_tail_fallback(snapshot,request,plan,active)
                    if not plan.feasible:
                        async with self.action_lock:
                            if (plan.snapshot_version!=self.state.snapshot.version
                                    or time.time()>plan.expires_s):
                                self.stale_admission(active,plan)
                                self.pending.defer(rid);retry=True
                                break
                            try:
                                await self.backend.execute(plan)
                            except ExpiredPlan as exc:
                                self.stale_admission(active,plan,exc)
                                self.pending.defer(rid);retry=True
                                break
                            await self.state.apply_clocks(self.backend.frequency,self.backend.parked)
                        # Explicit unprofiled fallback is available only in
                        # development; it invalidates formal evidence.
                        if self.config.get("allow_unprofiled_fallback",False):
                            plan=self.max_mixed(request,now)
                        if plan is None or not plan.feasible:
                            self.pending.defer(rid);retry=True
                            break
                    try:
                        if self.evaluation_v3: active['timing']['action_wait_started_s']=time.time()
                        async with self.action_lock:
                            if rid not in self.active or active["future"].cancelled():
                                self.planning_stats.discard('cancelled_request')
                                break
                            plan=await current_clock_first_plan(self,plan,request,completion_recovery=completion_recovery)
                            committed_s=time.time()
                            if self.evaluation_v3: active['timing']['action_acquired_s']=committed_s
                            admitted=admission_budget(plan,request,now=committed_s)
                            await self.state.reserve(plan,committed_s,admitted)
                            active['budget']=admitted
                            active["route"]=plan.routes[0]
                            if self.evaluation_v3: active['timing']['reserved_s']=time.time()
                            try:
                                await self.backend.execute(plan)
                                if not await self.backend.confirm(plan):
                                    raise RuntimeError("unconfirmed plan")
                                await self.state.apply_clocks(self.backend.frequency,self.backend.parked)
                                await self.state.apply_windows(plan.windows)
                                clock_outcomes=(self.backend.frequency_outcomes[-len(plan.frequencies):]
                                                if plan.frequencies else [])
                                if self.eco_scheduler:
                                    self.eco_scheduler.committed(plan,time.time())
                                if self.evaluation_v3: active['timing']['backend_confirmed_s']=time.time()
                            except BaseException:
                                # No request has been forwarded yet. Clear the
                                # controller's view before releasing so refresh
                                # cannot resurrect an abandoned admission.
                                active['route']=None
                                active['budget']=request
                                await self.state.release(rid,unissued=True)
                                raise
                    except StalePlan as exc:
                        self.stale_admission(active,plan,exc)
                        self.pending.defer(rid);retry=True
                        break
                    except Exception as exc:
                        if not active["future"].done():
                            active["future"].set_exception(exc)
                        break
                    try:
                        active.pop('pdb_single_retry',None)
                        await self.journal.emit(dict(kind="admission",request_id=rid,
                            client_request_id=active['client_request_id'],
                            plan=asdict(plan),arrival_s=request.arrival_s,
                            clock_outcomes=clock_outcomes,
                            input_tokens=request.input_tokens,at_s=time.time()))
                        if not active["future"].done():
                            active["future"].set_result(plan.routes[0])
                    except Exception as exc:
                        await self.state.release(rid)
                        if not active["future"].done():
                            active["future"].set_exception(exc)
                    break
                if not retry: self.pending.done(rid)
                current=None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failure=str(exc)
            if current in self.active and not self.active[current]["future"].done():
                self.active[current]["future"].set_exception(exc)

    async def completions(self,request):
        if not self.evaluation_v3:
            return await self._completions(request)
        if not self.accepting_input:
            raise web.HTTPServiceUnavailable(text='evaluation arrival window closed')
        timing=benchmark_timing(request,self.config,time.time())
        request['pdblend_timing']=timing
        remaining=timing['hard_deadline_s']-time.time()
        if remaining<=0:
            raise web.HTTPGatewayTimeout(text='original hard request deadline already expired')
        task=asyncio.current_task()
        self.request_tasks.add(task)
        try:
            return await asyncio.wait_for(self._completions(request),remaining)
        except asyncio.TimeoutError as exc:
            raise web.HTTPGatewayTimeout(text='original hard request deadline expired') from exc
        finally:
            self.request_tasks.discard(task)

    async def _completions(self,request):
        arrived=time.time()
        timing=request.get('pdblend_timing') if self.evaluation_v3 else None
        if timing is not None: arrived=timing['planned_arrival_s']
        if self.failure:
            raise web.HTTPServiceUnavailable(text=self.failure)
        if self.pending.full():
            raise admission_response('admission_queue_full',"admission queue full")
        body=await request.json()
        if not isinstance(body.get("prompt"),(str,list)) or int(body.get("max_tokens",0))<1:
            raise web.HTTPBadRequest(text="prompt and positive max_tokens required")
        if not isinstance(body["prompt"],list):
            if self.tokenizer is None:
                raise web.HTTPBadRequest(text="configure tokenizer for text prompts")
            async with self.tokenizer_slots:
                body["prompt"] = await asyncio.to_thread(self.tokenizer.encode,
                                                           body["prompt"], add_special_tokens=False)
        rid=uuid.uuid4().hex
        input_tokens=len(body["prompt"])
        budget=RequestBudget(rid,arrived,input_tokens,
            self.predictor.predict(input_tokens),
            self.config["slo_ttft_s"],self.config["slo_tpot_s"],output_limit=body["max_tokens"],
            hard_deadline_s=timing['hard_deadline_s'] if timing else None)
        self.arrival_history.append(budget)
        if self.dynamo_scheduler:
            self.dynamo_scheduler.arrival(budget)
        future=asyncio.get_running_loop().create_future()
        self.active[rid]=dict(budget=budget,future=future,route=None,
                             client_request_id=request.headers.get('X-Request-Id','')[:256])
        if timing is not None:
            timing['body_decoded_s']=time.time()
            timing['queued_s']=time.time()
            self.active[rid]['timing']=timing
        try:
            self.pending.put_nowait(rid)
        except asyncio.QueueFull:
            self.active.pop(rid,None)
            raise admission_response('admission_queue_full',"admission queue full")
        response=None
        route=None
        wire_id=rid
        completed=False
        try:
            route=await asyncio.wait_for(future,max(.000001,budget.hard_deadline_s-time.time())
                if self.evaluation_v3 else budget.ttft_s+1)
            if timing is not None: timing['forward_started_s']=time.time()
            if route.prefill_id!=route.decode_id:
                nonce=rid
                producer_id=f"pdb:{nonce}:p:{route.prefill_id}:{route.decode_id}"
                wire_id=f"pdb:{nonce}:d:{route.prefill_id}:{route.decode_id}"
                source=self.backend.instances[route.prefill_id]
                async with self.session.post(source["url"]+"/v1/completions",
                    json=dict(body,max_tokens=1,stream=False),headers={"X-Request-Id":producer_id}) as prefill:
                    if prefill.status!=200:
                        raise RuntimeError("prefill failed: "+await prefill.text())
                    await prefill.read()
                self.active[rid]["prefill_done"]=True
                await self.state.prefill_complete(rid)
                await self.journal.emit(dict(kind="prefill_complete",request_id=rid,at_s=time.time()))
            target=self.backend.instances[route.decode_id]
            client_stream=body.get("stream",False)
            aggregate=CompletionAccumulator()
            acknowledged=False
            async with self.session.post(target["url"]+"/v1/completions",json=dict(body,stream=True),
                                         headers={"X-Request-Id":wire_id}) as downstream:
                if downstream.status!=200:
                    raise RuntimeError("decode failed: "+await downstream.text())
                if client_stream:
                    response=web.StreamResponse(headers={"Content-Type":"text/event-stream"})
                    await response.prepare(request)
                async for raw in downstream.content:
                    if raw.startswith(b"data: [DONE]"):
                        acknowledged=True
                    if raw.startswith(b"data: ") and b"[DONE]" not in raw:
                        event=json.loads(raw[6:])
                        aggregate.add(event,time.time())
                        ids=event.get("token_ids") or []
                        if ids:
                            now=time.time()
                            b=self.active[rid]["budget"]
                            if not b.emitted:
                                if timing is not None: timing['first_token_s']=now
                                await self.journal.emit(dict(kind="first_token",request_id=rid,at_s=now))
                                self.ttft_history.append((now,now-b.arrival_s))
                            self.active[rid]["budget"]=replace(b,emitted=b.emitted+len(ids),
                                first_token_s=b.first_token_s or now,last_token_s=now,
                                pending_import_s=0.,pending_ready_s=None,pending_frequency_mhz=None)
                            await self.state.update_budget(self.active[rid]["budget"])
                        if event.get("usage"):
                            self.predictor.observe_completed(input_tokens,event["usage"]["completion_tokens"])
                    if response is not None:
                        await response.write(raw)
                result=aggregate.result()
                if not acknowledged:
                    raise RuntimeError('downstream closed without completion acknowledgement')
                completed=True
                if timing is not None: timing['stream_end_s']=time.time()
                if response is None:
                    return web.json_response(result)
                with suppress(ConnectionError): await response.write_eof()
                return response
        except AdmissionRejected as exc:
            raise admission_response(exc.code,str(exc)) from exc
        except (asyncio.TimeoutError,TimeoutError,RuntimeError) as exc:
            if response is not None:
                with suppress(ConnectionError):
                    await response.write(("data: "+json.dumps({"error":str(exc)})+"\n\n").encode())
                return response
            raise web.HTTPServiceUnavailable(text=str(exc)) from exc
        finally:
            future.cancel()
            self.pending.done(rid)
            if not completed and rid in self.active and not self.active[rid]['budget'].emitted:
                self.ttft_history.append((time.time(),time.time()-arrived))
            self.active.pop(rid,None)
            await self.state.release(rid,unissued=route is None)
            await self.journal.emit(dict(kind="request_end",request_id=rid,
                                        completed=completed,at_s=time.time()))
            # Release imported buffers after partial transfer/cancellation too.
            if not completed and (wire_id.startswith("pdb:") or self.evaluation_v3):
                unconfirmed=await self.backend.cancel(wire_id)
                if unconfirmed:
                    await self.journal.emit(dict(kind='cancel_unconfirmed',request_id=rid,
                        instances=unconfirmed,at_s=time.time()))
            if timing is not None:
                timing['cleanup_end_s']=time.time()
                await self.journal.emit(dict(kind='request_timing',request_id=rid,
                    client_request_id=request.headers.get('X-Request-Id','')[:256],
                    completed=completed,**timing))

    async def freeze_topology(self,ids,value):
        async with self.action_lock:
            if value: self.frozen_instances.update(ids)
            else: self.frozen_instances.difference_update(ids)
            await self.refresh()

    async def commit_topology(self,ids,added):
        async with self.action_lock:
            if any(a.get('route') and (a['route'].prefill_id in ids or a['route'].decode_id in ids)
                   for a in self.active.values()):
                raise RuntimeError('topology commit attempted before request drain')
            shape=((getattr(self,'_dynamo_transaction_shape',None) or
                    next((self.dynamo_scheduler.assignments[i] for i in ids
                          if i in self.dynamo_scheduler.assignments),'LL')) if self.dynamo_scheduler else None)
            old_gpus={g for i,c in self.backend.instances.items() if i in ids for g in c['gpus']}
            self.backend.replace_instances(ids,[s.endpoint() for s in added])
            unused=old_gpus-{g for c in self.backend.instances.values() for g in c['gpus']}
            if unused and self.backend.clocks:
                await self.backend.clocks.park(sorted(unused),dict(self.backend.clocks.epochs))
            if self.dynamo_scheduler:
                mapping={i:p for i,p in self.dynamo_scheduler.assignments.items() if i not in ids}
                mapping.update({s.instance_id:shape for s in added})
                self.dynamo_scheduler.assignments=mapping
                self.dynamo_scheduler.version+=1
            await self.refresh()

    async def dynamo_slow(self,operation,now):
        try:
            snapshot=self.state.snapshot
            forecasts=self.dynamo_scheduler.forecast(now)
            proposal=await asyncio.to_thread(self.dynamo_topology.choose,snapshot,
                dict(self.dynamo_scheduler.assignments),forecasts,operation,
                cached_weights=bool(self.retained_weights),
                has_unrouted_requests=any(item.get('route') is None for item in self.active.values()))
            if self.control_stop.is_set():
                return
            if not proposal:
                await self.journal.emit(dict(kind='dynamo_control_epoch',operation=operation,
                    at_s=now,executed=False,reason='no feasible measured configuration amortizes switching cost',
                    period_s=self.dynamo_scheduler.hierarchy.PERIODS[operation]))
                return
            removed=set(proposal['remove_ids'])
            occupied={g for i in snapshot.instances if i.instance_id not in removed for g in i.gpus}
            preferred=[g for i in snapshot.instances if i.instance_id in removed for g in i.gpus]
            available=preferred+[g for g in self.topology_manager.node_gpus if g not in occupied and g not in preferred]
            port=max(max(s.port,s.kv_port+s.tp) for s in self.topology_manager.specs.values())+16
            specs=[];cursor=0
            for index,tp in enumerate(proposal.get('add_tps',proposal['target_tps'])):
                specs.append(InstanceSpec('dyn'+uuid.uuid4().hex[:12],tp,tuple(available[cursor:cursor+tp]),
                    port,port+8,'mixed',self.topology_manager.version+1))
                cursor+=tp;port+=16
            # A pure expansion has no removed ID from which commit can infer
            # the logical pool. Keep its explicit shape throughout rollback too.
            previous_shape=getattr(self,'_dynamo_transaction_shape',None)
            self._dynamo_transaction_shape=proposal['shape']
            try:
                result=await self.topology_manager.reconfigure(proposal['remove_ids'],tuple(specs),
                    savings_lower_j=proposal['savings_lower_j'],cost_upper_j=proposal['cost_upper_j'],
                    retained_weights=self.retained_weights,capacity_recovery=proposal.get('capacity_recovery'))
            finally:
                self._dynamo_transaction_shape=previous_shape
            self.retained_weights=result['retained_weights']
            await self.journal.emit(dict(kind='dynamo_control_epoch',operation=operation,at_s=time.time(),
                executed=True,period_s=self.dynamo_scheduler.hierarchy.PERIODS[operation],result=result,
                retained_ids=proposal.get('retained_ids',()),target_tps=proposal['target_tps'],
                add_tps=proposal.get('add_tps',proposal['target_tps'])))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.journal.emit(dict(kind='dynamo_control_epoch',operation=operation,at_s=time.time(),
                executed=False,error=repr(exc),reason='physical transaction failed; recovery recorded separately'))
        finally:
            if operation=='ScaleInst' and not self.control_stop.is_set():
                await self.dynamo_reassign(time.time())

    async def dynamo_reassign(self,now):
        async with self.action_lock:
            before=dict(self.dynamo_scheduler.assignments)
            after=self.dynamo_scheduler.resident_reassignment(self.state.snapshot,now)
            self.dynamo_scheduler.commit_assignments(after)
            await self.journal.emit(dict(kind='dynamo_pool_fragmentation',at_s=now,
                before=before,after=after,version=self.dynamo_scheduler.version,
                reason='native whole-instance demand; fractional demand spills componentwise upward; live requests stay'))

    async def dynamo_control(self):
        try:
            while await self.control_tick(.1):
                now=time.time()
                for operation in self.dynamo_scheduler.hierarchy.due(now):
                    if self.control_stop.is_set(): break
                    if operation!='ScaleFreq' and self.topology_manager:
                        if operation not in self.slow_pending:
                            self.slow_pending.append(operation)
                        continue
                    async with self.action_lock:
                        if self.control_stop.is_set(): break
                        if operation=='ScaleFreq':
                            plan=self.dynamo_scheduler.frequency_plan(self.state.snapshot,now)
                            try:
                                await self.backend.execute(plan)
                            except ExpiredPlan:
                                details=dict(plan=asdict(plan),executed=False,reason='execution deadline passed before actions')
                            else:
                                if not await self.backend.confirm(plan):
                                    raise RuntimeError('DynamoLLM frequency change unconfirmed')
                                await self.state.apply_clocks(self.backend.frequency,self.backend.parked)
                                details=dict(plan=asdict(plan),executed=True)
                        elif operation=='ScaleInst':
                            before=dict(self.dynamo_scheduler.assignments)
                            assignments=self.dynamo_scheduler.resident_reassignment(self.state.snapshot,now)
                            self.dynamo_scheduler.commit_assignments(assignments)
                            details=dict(before=before,after=assignments,executed=True,
                                scope='resident logical-pool reassignment; physical node count unchanged')
                        else:
                            details=dict(executed=False,reason='resident variant has no physical TP backend')
                    await self.journal.emit(dict(kind='dynamo_control_epoch',operation=operation,
                        period_s=self.dynamo_scheduler.hierarchy.PERIODS[operation],at_s=now,**details))
                if (not self.control_stop.is_set() and self.slow_pending
                        and (self.slow_task is None or self.slow_task.done())):
                    self.slow_task=asyncio.create_task(self.dynamo_slow(self.slow_pending.popleft(),time.time()))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failure=str(exc)

    async def eco_resize(self):
        """Resident-instance mitosis: real groups control separate rolling windows.

        The single-node variant retains inactive model replicas. Their idle
        power remains in the eight-card integral; no cold ScaleInst claim.
        """
        try:
            while await self.control_tick(self.config.get('eco_scale_period_s',5)):
                now=time.time()
                recent=[v for at,v in self.ttft_history if now-at<=60]
                async with self.action_lock:
                    if self.control_stop.is_set(): break
                    before=tuple(self.eco_scheduler.groups)
                    previous_selected=dict(self.eco_scheduler.selected)
                    previous_version=self.eco_scheduler.version
                    assigned={x for g in before for x in g}
                    available=[i for i in self.state.snapshot.instances if i.instance_id not in assigned
                               and i.accepting and not i.requests and now-i.timestamp_s<=1]
                    if recent and sum(recent)/len(recent)>self.config['slo_ttft_s'] and available:
                        self.eco_scheduler.add_instance(available[0].instance_id)
                    else:
                        credits=[r.next_token_remaining(now) for i in self.state.snapshot.instances
                                 if i.instance_id in assigned for r in i.requests if r.emitted]
                        n=len(assigned)
                        if credits and sum(credits)/len(credits)>self.config['slo_ttft_s']*(n+1)/n:
                            self.eco_scheduler.remove_idle_instance(self.state.snapshot)
                    if tuple(self.eco_scheduler.groups)!=before:
                        plan=self.eco_scheduler.membership_plan(self.state.snapshot,now)
                        try:
                            await self.backend.execute(plan)
                        except ExpiredPlan:
                            # A pre-action expiry changed no engine window.
                            self.eco_scheduler.groups=list(before)
                            self.eco_scheduler.selected=previous_selected
                            self.eco_scheduler.version=previous_version
                            continue
                        if not await self.backend.confirm(plan):
                            raise RuntimeError('EcoServe membership window transition not acknowledged')
                        await self.state.apply_windows(plan.windows)
                if tuple(self.eco_scheduler.groups)!=before:
                    await self.journal.emit(dict(kind='eco_macro_membership',before=before,
                        after=self.eco_scheduler.groups,version=self.eco_scheduler.version,
                        scope='resident replicas; no KV migration',at_s=now))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failure=str(exc)

    async def roles(self):
        """Use queued work, or an explicitly uncertain arrival-history forecast."""
        try:
            while await self.control_tick(self.config.get('role_search_period_s',.5)):
                snapshot=self.state.snapshot
                pending=tuple(a["budget"] for a in self.active.values() if not a.get("route"))[:8]
                now=time.time()
                forecast=None
                if not pending:
                    forecast=forecast_roles(tuple(self.arrival_history),now,self.history_started_s)
                    if forecast is None: continue
                    pending=forecast.requests
                plan=await asyncio.to_thread(search_roles,self.planner,self.role_planner,snapshot,pending,now,
                    self.config.get('reconfiguration_error_fraction',.3),
                    horizon_s=forecast.horizon_s if forecast else None,
                    repetitions=forecast.repetitions if forecast else 1.)
                if self.control_stop.is_set(): break
                if plan:
                    ids={a.instance_id for a in plan.roles}
                    # Admission and role changes serialize at the same commit
                    # boundary. Never freeze an instance selected from a stale
                    # snapshot after a new request has reserved its KV.
                    async with self.action_lock:
                        if self.control_stop.is_set(): break
                        current=self.state.snapshot
                        selected=[i for i in current.instances if i.instance_id in ids]
                        if current.version!=plan.snapshot_version or any(
                                i.requests or i.running or i.waiting or i.reserved_kv_tokens
                                or not i.accepting for i in selected):
                            continue
                        self.frozen_instances.update(ids)
                        try:
                            await self.backend.execute(plan)
                            if not await self.backend.confirm(plan):
                                raise RuntimeError("role transaction not confirmed")
                            self.role_planner.confirmed(plan,time.time())
                            await self.journal.emit(dict(kind="role_commit",plan=asdict(plan),at_s=time.time(),
                                demand_source='historical_admissions' if forecast else 'queued_requests',
                                forecast=asdict(forecast) if forecast else None))
                        except Exception as exc:
                            await self.journal.emit(dict(kind="role_failed",error=str(exc),at_s=time.time()))
                        finally:
                            self.frozen_instances.difference_update(ids)
                            await self.refresh()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failure=str(exc)

    async def pd_slow_control(self):
        try:
            while await self.control_tick(self.config.get('slow_topology_period_s',300)):
                if self.slow_task and not self.slow_task.done(): continue
                forecast=forecast_roles(tuple(self.arrival_history),time.time(),self.history_started_s,
                    horizon_s=self.config.get('slow_topology_horizon_s',300))
                if forecast:
                    self.slow_task=asyncio.create_task(self.pd_slow(forecast))
        except asyncio.CancelledError: raise
        except Exception as exc: self.failure=str(exc)

    async def pd_slow(self,forecast):
        ids=()
        manager_started=False
        try:
            proposal=await asyncio.to_thread(self.pd_topology.choose,self.state.snapshot,forecast,time.time(),
                                             cached_weights=bool(self.retained_weights))
            if self.control_stop.is_set(): return
            await self.journal.emit(dict(kind='pd_topology_epoch',at_s=time.time(),
                forecast=asdict(forecast),proposal=asdict(proposal['source_cost']) if proposal else None))
            if proposal is None: return
            ids=proposal['remove_ids']
            async with self.action_lock:
                current=self.state.snapshot
                selected=[i for i in current.instances if i.instance_id in ids]
                if (time.time()>proposal['expires_s'] or len(selected)!=len(ids)
                        or any(i.generation!=proposal['source_generations'][i.instance_id] for i in selected)
                        or not all(topology_idle(i,time.time()) for i in selected)):
                    return
                self.frozen_instances.update(ids)
                await self.refresh()
            port=max(max(s.port,s.kv_port+s.tp) for s in self.topology_manager.specs.values())+16
            transaction=uuid.uuid4().hex[:10];specs=[]
            for index,item in enumerate(proposal['replacements']):
                specs.append(InstanceSpec('pdbslow-'+transaction+'-'+str(index),item['tp'],item['gpus'],
                    port,port+8,item['role'],self.topology_manager.version+1));port+=32
            validate_layout([s for i,s in self.topology_manager.specs.items() if i not in ids]+specs,
                            self.topology_manager.node_gpus)
            manager_started=True
            result=await self.topology_manager.reconfigure(ids,tuple(specs),
                savings_lower_j=proposal['savings_lower_j'],cost_upper_j=proposal['cost_upper_j'],
                retained_weights=self.retained_weights)
            self.retained_weights=result.get('retained_weights',self.retained_weights)
            await self.journal.emit(dict(kind='pd_topology_result',at_s=time.time(),
                capacity_loss_upper_j=proposal['capacity_loss_upper_j'],result=result))
        except asyncio.CancelledError: raise
        except Exception as exc:
            await self.journal.emit(dict(kind='pd_topology_failure',at_s=time.time(),error=repr(exc)))
        finally:
            if ids and not manager_started:
                async with self.action_lock:
                    self.frozen_instances.difference_update(ids)
            # Once entered, the manager exclusively owns recovery/isolation.
            # A failed rollback must never be unfrozen by stale cached state.

    async def control_tick(self,seconds):
        try:
            await asyncio.wait_for(self.control_stop.wait(),seconds)
            return False
        except asyncio.TimeoutError:
            return True

    async def quiesce_controls(self):
        """Close the measured workload and finish already-started changes.

        The caller keeps sampling all GPUs until this returns. Idle periodic
        timers wake immediately; their sleep time is not added to energy.
        """
        at=time.time()
        slow_active=self.slow_task is not None and not self.slow_task.done()
        self.control_stop.set()
        if self.role_task: await self.role_task
        if getattr(self,'pd_slow_control_task',None): await self.pd_slow_control_task
        if hasattr(self,'telemetry_task'): await self.telemetry_task
        slow_active=slow_active or (self.slow_task is not None and not self.slow_task.done())
        self.slow_pending.clear()
        if self.slow_task: await self.slow_task
        async with self.action_lock: pass
        clock_end=getattr(getattr(self.backend,'clocks',None),'last_write_finished_s',0.)
        end=time.time() if slow_active else max(self.backend.last_action_finished_s,clock_end)
        return end if end>at else None

    async def finish_measurement(self, deadline_s):
        """Prove requests, control actions and engine resources have drained.

        This is a measurement boundary, not permission to discard late work.
        The sampler remains running while this method awaits completion. A
        bounded or failed drain is retained as an incomplete measurement.
        """
        self.accepting_input=False
        result=dict(drain_complete=False, drain_end_s=None,
                    controls_end_s=None, residual={}, error=None)
        async def finish():
            while self.request_tasks or self.active:
                result['residual']={'requests':len(self.request_tasks),
                                    'active_ids':list(self.active)}
                await asyncio.sleep(.02)
            control_tail=await self.quiesce_controls()
            # Legacy callers use None to mean no additional control tail.
            # V3 still needs the actual observed quiescence boundary.
            result['controls_end_s']=control_tail or time.time()
            # A final control can change topology. Inspect the live backend
            # after it finishes, not the original configuration's instances.
            while True:
                ids=tuple(self.backend.instances)
                states=await asyncio.gather(*(self.backend.json(i,'/runtime') for i in ids),
                                           return_exceptions=True)
                residual={}
                for instance_id, raw in zip(ids, states):
                    if isinstance(raw, BaseException):
                        pending={'read_error':repr(raw)}
                    else:
                        pending=engine_residual(raw,time.time(),
                            ttl_s=float(self.config.get('telemetry_ttl_s',1.)))
                    if pending: residual[instance_id]=pending
                if not ids: residual['topology']='no live engines to verify'
                result['residual']=residual
                if not residual:
                    # Freeze all endpoints and cross their owner/rank barrier.
                    # Current PUT sends are synchronous; missing send counters
                    # remain unknown and are covered by this explicit proof.
                    barriers=await asyncio.gather(*(self.backend.json(i,'/drain',
                        {'expected_generation':raw['generation']})
                        for i,raw in zip(ids,states)),return_exceptions=True)
                    result['drain_barriers']={}
                    for i,raw,proof in zip(ids,states,barriers):
                        if isinstance(proof,BaseException):
                            raise RuntimeError('engine drain barrier failed '+i+': '+repr(proof))
                        result['drain_barriers'][i]=proof
                        if (proof.get('drained') is not True or proof.get('accepting') is not False
                                or proof.get('generation')!=raw['generation']+1
                                or proof.get('drain_proof_type')!='synchronous_put_owner_barrier'):
                            raise RuntimeError('engine drain barrier proof missing or inconsistent: '+i)
                    result.update(drain_complete=True,drain_end_s=time.time())
                    return
                await asyncio.sleep(.05)
        remaining=deadline_s-time.time()
        if remaining<=0:
            result['error']='absolute drain deadline already expired'
            return result
        try:
            await asyncio.wait_for(finish(),remaining)
        except asyncio.TimeoutError:
            result['error']='absolute drain deadline expired before completion proof'
        except Exception as exc:
            result['error']=repr(exc)
        return result

    async def stop(self,app):
        self.control_stop.set()
        tasks=[task for task in (self.dispatch_task,self.telemetry_task,self.role_task,self.slow_task,
               getattr(self,'pd_slow_control_task',None)) if task is not None]
        tasks.extend(task for task in self.request_tasks if task is not asyncio.current_task())
        for task in tasks: task.cancel()
        try:
            for result in await asyncio.gather(*tasks,return_exceptions=True):
                if isinstance(result,BaseException) and not isinstance(result,asyncio.CancelledError):
                    raise result
        finally:
            await self.close_resources()

    def application(self):
        app=web.Application(client_max_size=16*1024**2)
        app.router.add_post("/v1/completions",self.completions)
        async def models(request):
            return web.json_response(dict(object='list',data=[dict(object='model',
                id=self.config.get('served_model','Qwen2.5-14B-Instruct'),owned_by='pdblend')]))
        app.router.add_get('/v1/models',models)
        app.on_startup.append(self.start)
        app.on_cleanup.append(self.stop)
        return app
