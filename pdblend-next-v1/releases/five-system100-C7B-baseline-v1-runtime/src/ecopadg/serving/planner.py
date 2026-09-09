"""Joint route/frequency admission using one per-request deadline ledger."""
from dataclasses import replace
from functools import lru_cache
import itertools
import time
import math

from .profiles import ProfileStore
from .frequency import FrequencyCost,recovery_actions
from .transfers import TransferCost, TransferStore  # Re-export the historical import.
from .types import (ControlPlan, FrequencyAction, RequestBudget, RouteAction,
                    RuntimeSnapshot)
from .tails import TailModel, admission_budget, resident_context
from .pd_energy import ResidencyHorizon


class JointPlanner:
    def __init__(self, profiles: ProfileStore, transfers=(), *, max_frequency=2520,
                 telemetry_ttl_s=1., decision_budget_s=.01, depth=3, width=8,
                 allow_pd=True, dvfs=True,clock_settle_s=0,topology=None,frequency_costs=None,
                 protect_pending_decode=False, residency_horizon=False, park_grace_s=.5,
                 coverage_aware_recovery=False):
        if not math.isfinite(decision_budget_s) or decision_budget_s<=0:
            raise ValueError('finite positive planning decision budget required')
        self.profiles = profiles
        self.transfers = tuple(transfers)
        self.max_frequency = max_frequency
        self.telemetry_ttl_s = telemetry_ttl_s
        self.decision_budget_s = decision_budget_s
        self.depth, self.width = min(depth,3), min(width,8)
        self.allow_pd, self.dvfs = allow_pd, dvfs
        self.protect_pending_decode = protect_pending_decode
        self.coverage_aware_recovery = coverage_aware_recovery
        if residency_horizon and (not math.isfinite(park_grace_s) or park_grace_s < 0):
            raise ValueError('finite nonnegative parking grace required')
        self.residency_horizon = residency_horizon
        self.park_grace_s = park_grace_s
        self.clock_settle_s=clock_settle_s
        self.topology=topology
        self.transfer_store=TransferStore(self.transfers,topology)
        # A standalone analytical caller may omit transition measurements.
        # The serving runtime always supplies an explicit collection, for
        # which an unmeasured transition cannot admit a request.
        self.frequency_costs=(None if frequency_costs is None else
            tuple(c if isinstance(c,FrequencyCost) else FrequencyCost(**c) for c in frequency_costs))
        self._frequency_costs={}
        for cost in self.frequency_costs or ():
            key=(cost.tp,cost.source_mhz,cost.target_mhz)
            old=self._frequency_costs.get(key,(0.,0.))
            self._frequency_costs[key]=(max(old[0],cost.duration_upper_s),max(old[1],cost.energy_upper_j))
        self.transfer_point=lru_cache(maxsize=4096)(self._transfer_point)

    def frequency_cost(self,instance,target):
        wakeup=self.clock_settle_s if instance.parked else 0.
        if target==instance.frequency_mhz: return wakeup,0.
        if self.frequency_costs is None:
            return max(wakeup,self.clock_settle_s if target>instance.frequency_mhz else 0.),0.
        measured=self._frequency_costs.get((instance.tp,instance.frequency_mhz,target))
        if measured is None: return None
        return wakeup+measured[0],measured[1]

    def _transfer_point(self,source_tp,target_tp,source_gpus,target_gpus,input_tokens,batch,decode_frequency_mhz=None):
        return self.transfer_store.lookup(source_tp,target_tp,source_gpus,target_gpus,
                                          input_tokens,batch,decode_frequency_mhz)

    def point(self, instance, request, freq, batch, *, context=None):
        context = max(resident_context(instance) if context is None else context,
                      request.input_tokens+(1 if instance.role=='prefill' else request.predicted_output))
        return self.profiles.lookup_execution_phase(instance.role, instance.tp, freq,
                                   request.input_tokens, context, batch)

    def safe_for_existing(self, instance, point, now, prefill_delay=0):
        for req in instance.requests:
            delay = point.iteration_s * point.bound + prefill_delay
            if req.next_token_remaining(now) < delay:
                return False
        return True

    def pending_decode_safe(self, instance, frequency, iteration, interference, switching):
        """Reserve the first decode interval before its first token is observed.

        Any pending stream can emit while another prefill/import is running.
        Its TTFT credit cannot pay for that later interruption. Count every
        other committed phase conservatively; no execution order or eventual
        output length is assumed.
        """
        pending = [r for r in instance.requests if not r.emitted]
        blocks = []
        for request in pending:
            if instance.role == 'mixed':
                point = self.profiles.lookup('mixed', instance.tp, frequency,
                    request.input_tokens, request.input_tokens + 1, 1)
                if point is None:
                    return False
                blocks.append(point.phase_time_bound('prefill'))
            else:
                blocks.append(request.pending_import_s)
        committed = sum(blocks)
        for request, own_phase in zip(pending, blocks):
            interruption = interference + switching + max(0., committed - own_phase)
            if iteration + interruption > request.tpot_s:
                return False
        return True

    def node_residency(self,snapshot,overrides=None):
        overrides=overrides or {}
        used={g for i in snapshot.instances for g in i.gpus}
        total=max(0,self.profiles.gpu_count-len(used))*self.profiles.idle_unallocated_gpu_w
        for i in snapshot.instances:
            total+=self.instance_residency(i,overrides.get(i.instance_id))
        return total

    def instance_residency(self,instance,frequency=None):
        idle=self.profiles.idle_unallocated_gpu_w*instance.tp
        if instance.parked and frequency is None: return self.profiles.parked_residency(instance.tp)
        value=self.profiles.residency(instance.role,instance.tp,
                                     instance.frequency_mhz if frequency is None else frequency)
        return value if value is not None else idle

    def candidates(self, snapshot, request, now):
        live = [i for i in snapshot.instances if i.accepting
                and 0 <= now-i.timestamp_s <= self.telemetry_ttl_s]
        paths = [(m,m) for m in live if m.role == "mixed"]
        if self.allow_pd:
            paths += list(itertools.product([p for p in live if p.role == "prefill"],
                                            [d for d in live if d.role == "decode"]))
        plans=[]
        tails=TailModel(self,snapshot,now)
        if any(t is None for t in tails.tails.values()): return plans
        try:
            residency_model = (ResidencyHorizon(self, snapshot, tails, self.park_grace_s)
                               if self.residency_horizon else None)
        except ValueError:
            return plans  # Unmeasured source work cannot earn residency credit.
        node_tail=max(tails.tails.values(),default=0.)
        current_residency=self.node_residency(snapshot)
        resident={i.instance_id:self.instance_residency(i) for i in snapshot.instances}
        credit={i.instance_id:min((r.next_token_remaining(now) for r in i.requests),default=float('inf'))
                for i in snapshot.instances}
        remaining_by_instance={i.instance_id:max((max(r.predicted_output-r.emitted,0) for r in i.requests),default=0)
                               for i in snapshot.instances}
        # The shared remaining-work energy horizon depends on live request
        # state, including requests on a different route. A drained frozen
        # replica contributes only its known resident power; its obsolete
        # engine timestamp must not expire a healthy instance's admission.
        tail_dependencies=tuple(i for i in snapshot.instances if i.requests)
        if any(not 0 <= now-i.timestamp_s <= self.telemetry_ttl_s for i in tail_dependencies):
            return plans
        tail_expiry=min((i.timestamp_s+self.telemetry_ttl_s for i in tail_dependencies),
                        default=now+self.telemetry_ttl_s)
        points={}
        def point(instance,frequency,batch):
            key=(instance.instance_id,frequency,batch)
            if key not in points:
                points[key]=self.point(instance,request,frequency,batch,context=tails.contexts[instance.instance_id])
            return points[key]
        for p,d in paths:
            telemetry_expiry=min(tail_expiry,p.timestamp_s+self.telemetry_ttl_s,
                                 d.timestamp_s+self.telemetry_ttl_s)
            reservation = ((request.input_tokens + (request.output_limit or request.predicted_output)+15)//16)*16
            prefill_reservation=((request.input_tokens+16)//16)*16 if p is not d else 0
            if d.free_kv_tokens - d.reserved_kv_tokens < reservation:
                continue
            if p is not d and p.free_kv_tokens - p.reserved_kv_tokens < prefill_reservation:
                continue
            staging=request.input_tokens*d.transfer_bytes_per_token if p is not d else 0
            if p is not d and (not staging or d.free_transfer_bytes-d.reserved_transfer_bytes<staging):
                continue
            batch = max(1, d.running + d.waiting + 1)
            source_point=None
            if p is not d:
                source_point=point(p,self.max_frequency,max(1,p.waiting+1))
                if source_point is None: continue
            fs = (self.profiles.frequencies(d.role,d.tp) if self.dvfs and d.dvfs_allowed
                  else (self.max_frequency,))
            # A not-yet-first-token request owns a phase/transfer promise at
            # the current clock. Reuse it until that phase completes; changing
            # it would require repricing every outstanding transport contract.
            if self.dvfs and any(not r.emitted for r in d.requests):
                fs=(d.frequency_mhz,)
            for freq in fs:
                dp = point(d,freq,batch)
                # New prefill length and resident decode context are distinct
                # axes. A short arrival must not inherit a long old request's
                # prefill work merely because they share one mixed instance.
                pp=(self.profiles.lookup('mixed',p.tp,freq,request.input_tokens,
                                        request.input_tokens+1,1) if p is d else source_point)
                if dp is None or pp is None:
                    continue
                transfer_s=transfer_j=0.
                import_s=dp.iteration_s*dp.bound
                if p is not d:
                    # Admission sets the target clock before dispatching P;
                    # use that candidate clock, not the previous live clock.
                    link=self.transfer_point(p.tp,d.tp,p.gpus,d.gpus,request.input_tokens,source_point.batch,freq)
                    if link is None: continue
                    transfer_s,transfer_j=link.seconds_upper,link.incremental_j
                    import_s=link.import_seconds_upper or import_s
                pending_imports=(sum(r.pending_import_s for r in d.requests if not r.emitted)
                                 if p is not d else 0.)
                prefill=pp.phase_time_bound('prefill')
                iteration=dp.iteration_s * dp.bound
                interference=import_s
                if p is d:
                    context=tails.contexts[d.instance_id]
                    measured=self.profiles.interference(d.tp,freq,request.input_tokens,context,
                                                       max(d.running+d.waiting,len(d.requests)))
                    # Without a cross-context measurement, the fallback must
                    # cover the new prefill's measured history uncertainty.
                    interference=(max(dp.interference_s*max(pp.bound,dp.bound),prefill)
                                  if measured is None else measured*max(pp.bound,dp.bound))
                queue_delay=(math.ceil(p.waiting/max(pp.batch,1))+int(p is not d and p.running>0))*prefill
                if p is d:
                    queued=[r for r in p.requests if not r.emitted]
                    pending_prefills=[self.profiles.lookup('mixed',p.tp,freq,r.input_tokens,
                                                          r.input_tokens+1,1) for r in queued]
                    if not all(pending_prefills): continue
                    queue_delay=(sum(q.phase_time_bound('prefill') for q in pending_prefills)
                                 +max(0,p.waiting-len(queued))*prefill)
                costs=[self.frequency_cost(d,freq)]
                if p is not d: costs.append(self.frequency_cost(p,self.max_frequency))
                if any(cost is None for cost in costs): continue
                switching=sum(cost[0] for cost in costs)
                switching_j=sum(cost[1] for cost in costs)
                ttft=queue_delay+prefill+transfer_s+pending_imports+iteration+switching
                # Stability: a full predicted decode batch must fit the token
                # budget; every admitted request also passes its own deadline.
                # Pending mixed prefills have already committed part of each
                # resident request's credit. Include them again when checking
                # the total promise, not just the new prefill's interference.
                committed_delay=queue_delay if p is d else pending_imports
                d_delay=dp.iteration_s*dp.bound+committed_delay+interference+switching
                p_delay=pp.iteration_s*pp.bound+queue_delay+prefill+switching
                if (self.protect_pending_decode and not self.pending_decode_safe(
                        d, freq, iteration, interference, switching)):
                    continue
                if (ttft > request.ttft_remaining(now) or iteration > request.tpot_s
                        or credit[d.instance_id] < d_delay
                        or (p is not d and credit[p.instance_id] < p_delay)):
                    continue
                remaining=max(request.predicted_output-1,0)
                decode_time=remaining*dp.iteration_s
                existing_tail=remaining_by_instance[d.instance_id]*dp.iteration_s
                admitted=replace(request,pending_import_s=import_s if p is not d else 0.,
                    pending_ready_s=now+max(0.,ttft-iteration),pending_frequency_mhz=freq)
                after_tail=tails.after_admission(d,admitted,freq,switch_delay=costs[0][0],
                    source=p if p is not d else None,source_delay=costs[1][0] if p is not d else 0.,new_point=dp,
                    per_instance=bool(residency_model))
                if after_tail is None: continue
                new_residency=current_residency+self.instance_residency(d,freq)-resident[d.instance_id]
                if p is not d:
                    new_residency+=self.instance_residency(p,self.max_frequency)-resident[p.instance_id]
                if residency_model:
                    frequencies = {d.instance_id: freq}
                    source_windows = {}
                    if p is not d:
                        frequencies[p.instance_id] = self.max_frequency
                        source_windows[p.instance_id] = max(
                            residency_model.prefill_windows[p.instance_id],
                            queue_delay + prefill + transfer_s + switching)
                    residency_j = (residency_model.energy(after_tail, frequencies, source_windows)
                                   - residency_model.before_j)
                else:
                    residency_j = new_residency*after_tail-current_residency*node_tail
                # Instance-level shared dynamic work apportioned to the added
                # sequence; residency is incremental only when its tail grows.
                prefill_share=1 if p is d else max(1,p.waiting+1)
                joules=(max(0.,pp.phase_power("prefill")-pp.residency_w)*pp.prefill_s/prefill_share
                        + max(0.,dp.phase_power("decode")-dp.residency_w)*decode_time/batch
                        + residency_j + transfer_j
                        # The node tail above already includes resident power
                        # during the switch. Add only its remaining measured
                        # all-node cost, once per affected instance.
                        + max(0.,switching_j-new_residency*switching))
                current=point(d,d.frequency_mhz,batch)
                if current is not None and d.requests:
                    old_tail=remaining_by_instance[d.instance_id]*current.iteration_s
                    joules += ((dp.phase_power("decode")-dp.residency_w)*existing_tail
                               -(current.phase_power("decode")-current.residency_w)*old_tail)
                route=RouteAction(request.request_id,p.instance_id,d.instance_id,
                                  reservation,ttft,iteration,joules,
                                  prefill_reservation,staging,import_s if p is not d else 0.)
                frequencies=[FrequencyAction(d.instance_id,freq)]
                if p is not d:
                    frequencies.append(FrequencyAction(p.instance_id,self.max_frequency))
                expires=min(telemetry_expiry,now+request.ttft_remaining(now)-ttft,
                    now+credit[d.instance_id]-d_delay,
                    now+credit[p.instance_id]-p_delay if p is not d else float('inf'))
                plans.append(ControlPlan(snapshot.version,now,expires,
                    routes=(route,),frequencies=tuple(frequencies),
                    reason="minimum measured incremental energy among feasible paths"))
        return sorted(plans,key=lambda p:p.routes[0].incremental_j)

    def advance(self, snapshot, plan, request):
        route=plan.routes[0]
        request=admission_budget(plan,request)
        frequencies={a.instance_id:a.frequency_mhz for a in plan.frequencies}
        instances=[]
        for instance in snapshot.instances:
            changes={}
            if instance.instance_id == route.decode_id:
                changes.update(
                    reserved_kv_tokens=instance.reserved_kv_tokens+route.reserve_tokens,
                    reserved_transfer_bytes=instance.reserved_transfer_bytes+route.transfer_reserve_bytes,
                    waiting=instance.waiting+1,requests=instance.requests+(request,))
            if route.prefill_id != route.decode_id and instance.instance_id == route.prefill_id:
                changes.update(waiting=instance.waiting+1,
                    reserved_kv_tokens=instance.reserved_kv_tokens+route.prefill_reserve_tokens,
                    requests=instance.requests+(request,))
            if instance.instance_id in frequencies:
                changes.update(frequency_mhz=frequencies[instance.instance_id],parked=False)
            instances.append(replace(instance,**changes) if changes else instance)
        return replace(snapshot,instances=tuple(instances))

    def plan(self, snapshot: RuntimeSnapshot, pending: tuple[RequestBudget,...], *,
             joint=True, now=None):
        now=time.time() if now is None else now
        if not pending:
            return ControlPlan(snapshot.version,now,now+self.telemetry_ttl_s,reason="no pending admission")
        started=time.perf_counter()
        first=self.candidates(snapshot,pending[0],now)
        if not first:
            actions,expires,covered=recovery_actions(self,[(i,True) for i in snapshot.instances],
                now,now+self.telemetry_ttl_s,maximum=self.max_frequency)
            return ControlPlan(snapshot.version,now,expires,frequencies=actions,
                reason=("infeasible: measured coverage recovery and retain admission queue" if covered else
                        "infeasible or unmeasured: restore capacity and retain admission queue"),feasible=False)
        # Candidate 1 decomposes the decision: choose the layout at full
        # frequency, then minimize energy over feasible frequencies on it.
        full=[p for p in first if all(a.frequency_mhz==self.max_frequency for a in p.frequencies)]
        layout=(full or first)[0].routes[0]
        fallback=next(p for p in first if (p.routes[0].prefill_id,p.routes[0].decode_id)
                      ==(layout.prefill_id,layout.decode_id))
        if not joint:
            return replace(fallback,reason="full-frequency layout selection, then independent feasible DVFS")
        if time.perf_counter()-started>self.decision_budget_s:
            return replace(fallback,reason="decision budget exceeded; feasible greedy fallback")
        beam=[(p.routes[0].incremental_j,self.advance(snapshot,p,pending[0]),p) for p in first[:self.width]]
        for request in pending[1:self.depth]:
            expanded=[]
            for energy,state,root in beam:
                if time.perf_counter()-started > self.decision_budget_s:
                    return replace(fallback,reason="decision budget exceeded; feasible greedy fallback")
                for p in self.candidates(state,request,now)[:self.width]:
                    expanded.append((energy+p.routes[0].incremental_j,state,p,root))
                if time.perf_counter()-started>self.decision_budget_s:
                    return replace(fallback,reason="decision budget exceeded; feasible greedy fallback")
            if not expanded:
                break
            beam=[(energy,self.advance(state,p,request),root) for energy,state,p,root in
                  sorted(expanded,key=lambda x:x[0])[:self.width]]
        return replace(min(beam,key=lambda x:x[0])[2],reason="joint search over currently queued requests")
