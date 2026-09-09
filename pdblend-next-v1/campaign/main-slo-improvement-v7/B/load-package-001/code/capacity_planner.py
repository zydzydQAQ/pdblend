"""Pure mixed-replica capacity proposals. No execution, trace, or dataset API.

Bounds are model/node/source-specific calibration inputs, not measurements made
by this module. A proposal is a short-lived intent requiring executor recheck.
"""
from dataclasses import dataclass, replace
import math


def finite(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value>0 if positive else value>=0)


def sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def groups(values):
    return tuple(sorted(tuple(sorted(g)) for g in values))


@dataclass(frozen=True)
class Identity:
    model_sha256: str
    node_sha256: str
    engine_image: str
    source_sha256: str
    tp: int

    def __post_init__(self):
        if (not all(sha(v) for v in (self.model_sha256,self.node_sha256,self.source_sha256))
                or not self.engine_image.startswith('sha256:') or not sha(self.engine_image[7:])
                or self.tp not in (1,2,4,8)):
            raise ValueError('complete model/node/source/image/TP identity required')


@dataclass(frozen=True)
class Evidence:
    identity: Identity
    raw_sha256: str
    certified: bool = True

    def usable(self,identity):
        return self.certified is True and self.identity==identity and sha(self.raw_sha256)


@dataclass(frozen=True)
class LayoutBound:
    resident_groups: tuple
    demand_domain_sha256: str
    sustainable_rate_lower_rps: float
    evidence: Evidence

    def __post_init__(self):
        if not sha(self.demand_domain_sha256) or not finite(self.sustainable_rate_lower_rps,positive=True):
            raise ValueError('independent sustained whole-layout capacity bound required')
        if self.resident_groups!=groups(self.resident_groups):
            raise ValueError('layout GPU groups must be canonical')


@dataclass(frozen=True)
class TransitionBound:
    operation: str                  # remove, restore_cold, restore_warm
    gpus: tuple
    duration_upper_s: float
    energy_upper_j: float           # conservative total-node transition energy
    evidence: Evidence
    peak_memory_per_gpu_upper_bytes: int = 0
    cached_weights_sha256: str | None = None

    def __post_init__(self):
        if (self.operation not in ('remove','restore_cold','restore_warm')
                or not finite(self.duration_upper_s,positive=True)
                or not finite(self.energy_upper_j,positive=True)
                or tuple(sorted(self.gpus))!=self.gpus):
            raise ValueError('positive measured time/whole-node energy bounds required')
        if self.operation.startswith('restore') and self.peak_memory_per_gpu_upper_bytes<=0:
            raise ValueError('restore requires measured peak memory upper bound')
        if self.operation=='restore_warm' and not sha(self.cached_weights_sha256):
            raise ValueError('warm restore requires retained weight cache identity')


@dataclass(frozen=True)
class SavingsBound:
    source_groups: tuple
    target_groups: tuple
    demand_domain_sha256: str
    rate_lower_rps: float
    rate_upper_rps: float
    whole_node_saving_lower_w: float
    evidence: Evidence

    def __post_init__(self):
        if (not sha(self.demand_domain_sha256) or not finite(self.rate_lower_rps)
                or not finite(self.rate_upper_rps) or self.rate_lower_rps>self.rate_upper_rps
                or not finite(self.whole_node_saving_lower_w,positive=True)):
            raise ValueError('matched-workload whole-node saving bound required')


@dataclass(frozen=True)
class Resident:
    instance_id: str
    gpus: tuple
    observed_at_s: float
    generation: int
    last_changed_s: float
    role: str = 'mixed'
    accepting: bool = True
    transport_healthy: bool = True
    active_requests: int = 0
    queued_requests: int = 0
    kv_allocations: int = 0
    reserved_kv_tokens: int = 0
    transfer_allocations: int = 0
    inflight_transfers: int | None = 0
    transfer_counts_known: bool = True
    removable: bool = True
    isolated_transport_candidate: bool = False
    reserved_transfer_bytes: int = 0
    pending_controls: int = 0
    error: str | None = None

    def __post_init__(self):
        counts=(self.generation,self.active_requests,self.queued_requests,self.kv_allocations,
            self.reserved_kv_tokens,self.transfer_allocations,
            self.reserved_transfer_bytes,self.pending_controls)
        if (type(self.transfer_counts_known) is not bool or type(self.removable) is not bool
                or type(self.isolated_transport_candidate) is not bool
                or (self.transfer_counts_known and
                    (type(self.inflight_transfers) is not int or self.inflight_transfers < 0))
                or (not self.transfer_counts_known and self.inflight_transfers is not None)):
            raise ValueError('unknown transfer counts must remain None; known counts must be explicit nonnegative integers')
        if (not self.instance_id or any(type(v) is not int or v<0 for v in counts)
                or not finite(self.observed_at_s) or not finite(self.last_changed_s)
                or self.last_changed_s>self.observed_at_s):
            raise ValueError('valid observation, generation, and explicit nonnegative residual counts required')

    def idle(self,now,age):
        return (self.removable and (self.transfer_counts_known or self.isolated_transport_candidate)
            and self.accepting and self.transport_healthy and not self.error
            and 0<=now-self.observed_at_s<=age
            and not any((self.active_requests,self.queued_requests,self.kv_allocations,
                self.reserved_kv_tokens,self.transfer_allocations,self.inflight_transfers,
                self.reserved_transfer_bytes,self.pending_controls)))


@dataclass(frozen=True)
class Spare:
    gpus: tuple
    observed_at_s: float
    unallocated_since_s: float
    free_memory_per_gpu_bytes: int
    unallocated: bool = True
    gpu_processes: int = 0
    residual_allocations: int = 0
    cached_weights_sha256: str | None = None

    def __post_init__(self):
        if (not finite(self.observed_at_s) or not finite(self.unallocated_since_s)
                or self.unallocated_since_s>self.observed_at_s or any(type(v) is not int or v<0
                    for v in (self.free_memory_per_gpu_bytes,self.gpu_processes,self.residual_allocations))):
            raise ValueError('valid free-GPU ownership, memory, and timestamp evidence required')

    def idle(self,now,age):
        return (self.unallocated and not self.gpu_processes and not self.residual_allocations
            and 0<=now-self.observed_at_s<=age)


@dataclass(frozen=True)
class Snapshot:
    identity: Identity
    version: int
    residents: tuple
    spares: tuple
    transition_inflight: bool = False

    def __post_init__(self):
        allocations=[tuple(i.gpus) for i in self.residents+self.spares]
        flat=[g for allocation in allocations for g in allocation]
        if (len({i.instance_id for i in self.residents})!=len(self.residents)
                or any(len(g)!=self.identity.tp or g!=tuple(sorted(g)) for g in allocations)
                or len(flat)!=len(set(flat)) or any(type(g) is not int or not 0<=g<8 for g in flat)):
            raise ValueError('unique instances and disjoint, physical TP GPU groups required')


@dataclass(frozen=True)
class Demand:
    observed_at_s: float
    history_span_s: float
    rate_lower_rps: float
    rate_upper_rps: float
    domain_sha256: str
    queued_requests: int = 0
    oldest_queue_wait_s: float = 0.
    recent_rate_lower_rps: float = 0.
    rate_trend_lower_rps2: float = 0.
    pending_ttft_remaining_s: float | None = None

    def __post_init__(self):
        if (not all(finite(v) for v in (self.observed_at_s,self.history_span_s,
                self.rate_lower_rps,self.rate_upper_rps,self.oldest_queue_wait_s,
                self.recent_rate_lower_rps,self.rate_trend_lower_rps2)) or self.rate_lower_rps>self.rate_upper_rps
                or (self.pending_ttft_remaining_s is not None and not finite(self.pending_ttft_remaining_s))
                or not sha(self.domain_sha256) or type(self.queued_requests) is not int or self.queued_requests<0):
            raise ValueError('past-observation demand bounds required')


@dataclass(frozen=True)
class State:
    low_since_s: float | None = None
    high_since_s: float | None = None
    last_change_s: float | None = None
    layout_groups: tuple | None = None


@dataclass(frozen=True)
class Policy:
    down_utilization: float = .60
    up_utilization: float = .85
    low_hold_s: float = 60.
    high_hold_s: float = 5.
    min_resident_s: float = 120.
    min_off_s: float = 30.
    cooldown_s: float = 60.
    observation_max_age_s: float = 1.
    demand_history_min_s: float = 60.
    amortization_horizon_s: float = 600.
    savings_margin: float = .20
    min_residents: int = 1
    queue_pressure_s: float = 1.

    def __post_init__(self):
        if (not 0<self.down_utilization<self.up_utilization<1 or not 0<=self.savings_margin<1
                or type(self.min_residents) is not int or self.min_residents<1
                or any(not finite(getattr(self,k),positive=True) for k in ('low_hold_s','high_hold_s',
                    'min_resident_s','min_off_s','cooldown_s','observation_max_age_s',
                    'demand_history_min_s','amortization_horizon_s','queue_pressure_s'))):
            raise ValueError('ordered hysteresis and positive time bounds required')


@dataclass(frozen=True)
class Proposal:
    action: str
    reason: str
    gpus: tuple
    remove_id: str | None
    source_generation: int | None
    snapshot_version: int
    identity: Identity
    created_s: float
    expires_s: float
    duration_upper_s: float
    action_energy_upper_j: float
    round_trip_cost_upper_j: float | None
    predicted_savings_lower_j: float | None
    predicted_net_lower_j: float | None
    break_even_s: float | None
    target_capacity_lower_rps: float
    predicted_unserved_during_restore_upper: float | None
    evidence_sha256: tuple
    recovery_mode: str | None = None
    energy_claim: str = 'prediction_only'
    slo_during_transition_guaranteed: bool = False
    recovery_cache_sha256: str | None = None
    required_memory_per_gpu_bytes: int = 0


@dataclass(frozen=True)
class Result:
    proposal: Proposal | None
    state: State
    reason: str


class CapacityPlanner:
    def __init__(self,identity,layouts,transitions,savings,policy=Policy()):
        self.identity=identity;self.layouts=tuple(layouts);self.transitions=tuple(transitions)
        self.savings=tuple(savings);self.policy=policy

    def capacity(self,layout,demand):
        if not layout:return 0.,None
        matches=[c for c in self.layouts if c.resident_groups==layout
            and c.demand_domain_sha256==demand.domain_sha256 and c.evidence.usable(self.identity)]
        if not matches:return None,None
        selected=min(matches,key=lambda c:c.sustainable_rate_lower_rps)
        return selected.sustainable_rate_lower_rps,selected.evidence.raw_sha256

    def cost(self,operation,gpus,cache=None):
        matches=[c for c in self.transitions if c.operation==operation and c.gpus==gpus
            and c.evidence.usable(self.identity) and (operation!='restore_warm' or c.cached_weights_sha256==cache)]
        if not matches:return None
        if len(matches)!=1:
            raise ValueError('freeze one aggregate transition bound with all raw sources before planning')
        return matches[0]

    def growth_signals(self, snapshot, demand, cap, now):
        """Empirical causal forecast, never a guarantee of future arrivals/SLO."""
        p = self.policy
        cold = [self.cost('restore_cold', slot.gpus) for slot in snapshot.spares
                if slot.idle(now, p.observation_max_age_s)
                and now-slot.unallocated_since_s >= p.min_off_s]
        cold = [c for c in cold if c is not None]
        duration = max([c.duration_upper_s for c in cold] or [0.])
        baseline = max(demand.rate_lower_rps, demand.recent_rate_lower_rps)
        crossing = ((max(0., p.up_utilization*cap-baseline)/demand.rate_trend_lower_rps2)
                    if demand.rate_trend_lower_rps2 > 0 else None)
        trend = bool(duration and demand.history_span_s >= 10. and crossing is not None
                     and crossing <= duration)
        deadline = bool(duration and cap > 0 and demand.queued_requests >= 2
            and demand.pending_ttft_remaining_s is not None
            and 0 < demand.pending_ttft_remaining_s <= duration
            and demand.queued_requests/cap >= demand.pending_ttft_remaining_s)
        deficit = bool(demand.rate_lower_rps > cap or (demand.queued_requests > 0
                       and demand.oldest_queue_wait_s >= p.queue_pressure_s))
        return dict(capacity_deficit=deficit, pending_deadline_risk=deadline,
                    cold_headroom_risk=trend, measured_cold_upper_s=duration or None,
                    estimated_time_to_headroom_s=crossing, prediction_only=True,
                    future_trace_observed=False, slo_recovery_guaranteed=False)

    def choose(self,snapshot,demand,state,now):
        p=self.policy
        if snapshot.identity!=self.identity:return Result(None,state,'identity_mismatch')
        if snapshot.transition_inflight:return Result(None,state,'transition_inflight')
        if any(i.role!='mixed' for i in snapshot.residents):return Result(None,state,'mixed_layout_only')
        if not finite(now) or not 0<=now-demand.observed_at_s<=p.observation_max_age_s:
            return Result(None,replace(state,low_since_s=None,high_since_s=None),'stale_demand')
        current=groups(i.gpus for i in snapshot.residents)
        if state.layout_groups!=current:
            state=replace(state,low_since_s=None,high_since_s=None,layout_groups=current)
        cap,cap_proof=self.capacity(current,demand)
        if cap is None:return Result(None,state,'current_layout_capacity_unverified')
        signals=self.growth_signals(snapshot,demand,cap,now)
        deficit=signals['capacity_deficit']
        urgent=deficit or signals['pending_deadline_risk']
        high=urgent or signals['cold_headroom_risk'] or (cap>0 and demand.rate_upper_rps>=p.up_utilization*cap)
        state=replace(state,high_since_s=(state.high_since_s if state.high_since_s is not None else now)
            if high else None)
        cooldown=(state.last_change_s is not None and now-state.last_change_s<p.cooldown_s)
        if high:
            state=replace(state,low_since_s=None)
            if not urgent and (cooldown or now-state.high_since_s<p.high_hold_s):
                return Result(None,state,'growth_hysteresis')
            choices=[]
            for slot in snapshot.spares:
                if not slot.idle(now,p.observation_max_age_s):continue
                if now-slot.unallocated_since_s<p.min_off_s:continue
                target=groups(current+(slot.gpus,));target_cap,target_proof=self.capacity(target,demand)
                if target_cap is None or target_cap<=cap:continue
                options=[self.cost('restore_cold',slot.gpus),self.cost('restore_warm',slot.gpus,slot.cached_weights_sha256)]
                options=[c for c in options if c is not None and slot.free_memory_per_gpu_bytes>=c.peak_memory_per_gpu_upper_bytes]
                if not options:continue
                selected=min(options,key=lambda c:(c.duration_upper_s,c.energy_upper_j))
                evidence=tuple(v for v in (cap_proof,target_proof,selected.evidence.raw_sha256) if v)
                proposal=Proposal('restore',('capacity_deficit' if deficit else 'pending_deadline_risk'
                    if signals['pending_deadline_risk'] else 'cold_start_headroom'
                    if signals['cold_headroom_risk'] else 'capacity_headroom'),
                    slot.gpus,None,None,snapshot.version,self.identity,now,now+p.observation_max_age_s,
                    selected.duration_upper_s,selected.energy_upper_j,None,None,None,None,target_cap,
                    demand.queued_requests+max(0.,demand.rate_upper_rps-cap)*selected.duration_upper_s,
                    evidence,selected.operation,'capacity_recovery_no_energy_gain_claim',
                    recovery_cache_sha256=selected.cached_weights_sha256,
                    required_memory_per_gpu_bytes=selected.peak_memory_per_gpu_upper_bytes)
                choices.append(proposal)
            if not choices:return Result(None,state,'no_verified_free_gpu_restore')
            proposal=min(choices,key=lambda c:(max(0.,demand.rate_upper_rps-c.target_capacity_lower_rps),
                c.duration_upper_s,c.action_energy_upper_j))
            return Result(proposal,state,'restore_proposed')
        if demand.history_span_s<p.demand_history_min_s:
            return Result(None,replace(state,low_since_s=None),'insufficient_demand_history')
        if demand.queued_requests:
            return Result(None,replace(state,low_since_s=None),'queued_work_blocks_shrink')
        if len(snapshot.residents)<=p.min_residents:return Result(None,state,'minimum_resident_count')
        if any(not (i.accepting and i.transport_healthy and not i.error
                and 0<=now-i.observed_at_s<=p.observation_max_age_s) for i in snapshot.residents):
            return Result(None,replace(state,low_since_s=None),'remaining_capacity_state_unverified')
        candidates=[]
        for resident in snapshot.residents:
            if not resident.idle(now,p.observation_max_age_s):continue
            target=groups(i.gpus for i in snapshot.residents if i.instance_id!=resident.instance_id)
            target_cap,target_proof=self.capacity(target,demand)
            if target_cap is None or demand.rate_upper_rps>p.down_utilization*target_cap:continue
            remove=self.cost('remove',resident.gpus)
            restore=self.cost('restore_cold',resident.gpus)
            if remove is None or restore is None:continue
            matches=[s for s in self.savings if s.source_groups==current and s.target_groups==target
                and s.demand_domain_sha256==demand.domain_sha256 and s.evidence.usable(self.identity)
                and s.rate_lower_rps<=demand.rate_lower_rps and demand.rate_upper_rps<=s.rate_upper_rps]
            if not matches:continue
            saving=min(matches,key=lambda s:s.whole_node_saving_lower_w)
            effective_w=saving.whole_node_saving_lower_w*(1-p.savings_margin)
            transition_s=remove.duration_upper_s+restore.duration_upper_s
            useful_s=max(0.,p.amortization_horizon_s-transition_s)
            cost_j=remove.energy_upper_j+restore.energy_upper_j
            saved_j=effective_w*useful_s
            if saved_j<=cost_j:continue
            evidence=tuple(v for v in (cap_proof,target_proof,remove.evidence.raw_sha256,
                restore.evidence.raw_sha256,saving.evidence.raw_sha256) if v)
            proposal=Proposal('remove','amortized_idle_capacity',resident.gpus,resident.instance_id,
                resident.generation,snapshot.version,self.identity,now,now+p.observation_max_age_s,
                remove.duration_upper_s,remove.energy_upper_j,cost_j,saved_j,saved_j-cost_j,
                transition_s+cost_j/effective_w,target_cap,None,evidence)
            candidates.append((resident,proposal))
        if not candidates:return Result(None,replace(state,low_since_s=None),'no_safe_amortized_removal')
        state=replace(state,low_since_s=state.low_since_s if state.low_since_s is not None else now)
        if cooldown or now-state.low_since_s<p.low_hold_s:return Result(None,state,'shrink_hysteresis')
        eligible=[proposal for resident,proposal in candidates if now-resident.last_changed_s>=p.min_resident_s]
        if not eligible:return Result(None,state,'minimum_residence_time')
        return Result(max(eligible,key=lambda c:c.predicted_net_lower_j),state,'remove_proposed')


def committed(state,proposal,finished_s,*,execution_verified=False):
    """Call only after real execution/health verification, never on proposal."""
    if execution_verified is not True or not finite(finished_s) or finished_s<proposal.created_s:
        raise ValueError('actual transition completion timestamp required')
    return replace(state,last_change_s=finished_s,low_since_s=None,high_since_s=None,layout_groups=None)


def revalidate(proposal,snapshot,now,max_age_s=1.):
    """Executor must repeat this under its node/admission/control transaction."""
    if (snapshot.identity!=proposal.identity or snapshot.version!=proposal.snapshot_version
            or snapshot.transition_inflight or not proposal.created_s<=now<=proposal.expires_s):return False
    if proposal.action=='remove':
        if any(not(i.accepting and i.transport_healthy and not i.error
                and 0<=now-i.observed_at_s<=max_age_s) for i in snapshot.residents):return False
        return any(i.instance_id==proposal.remove_id and i.gpus==proposal.gpus
            and i.generation==proposal.source_generation and i.idle(now,max_age_s) for i in snapshot.residents)
    return any(s.gpus==proposal.gpus and s.idle(now,max_age_s)
        and s.free_memory_per_gpu_bytes>=proposal.required_memory_per_gpu_bytes
        and (proposal.recovery_mode!='restore_warm' or s.cached_weights_sha256==proposal.recovery_cache_sha256)
        for s in snapshot.spares)
