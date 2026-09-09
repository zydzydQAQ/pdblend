"""Cost-amortized resident role policy and serialized transactional execution."""
from dataclasses import dataclass,replace
import time
from .types import ControlPlan, RoleAction


@dataclass(frozen=True)
class RoleCost:
    tp: int
    source_role: str
    target_role: str
    time_upper_s: float
    energy_upper_j: float
    source_sha256: str


class ResidentRolePlanner:
    def __init__(self,costs,minimum_dwell_s=30):
        self.costs=tuple(costs)
        self.minimum_dwell_s=minimum_dwell_s
        self.last_commit={}

    def plan(self,snapshot,target_role,predicted_saving_w,horizon_s,error_fraction,
             capacity_loss_j=0,now=None,instance_id=None):
        now=time.time() if now is None else now
        candidates=[]
        for instance in snapshot.instances:
            if instance_id is not None and instance.instance_id!=instance_id:
                continue
            if (instance.role==target_role or not instance.accepting or instance.requests
                    or instance.running or instance.waiting or instance.reserved_kv_tokens
                    or now-instance.timestamp_s>1
                    or now-self.last_commit.get(instance.instance_id,float('-inf'))<self.minimum_dwell_s):
                continue
            resulting=[target_role if i.instance_id==instance.instance_id else i.role
                       for i in snapshot.instances]
            # An empty logical pool is allowed when the remaining layout can
            # still execute complete requests. Active requests never move.
            if not ('mixed' in resulting or ('prefill' in resulting and 'decode' in resulting)):
                continue
            for cost in self.costs:
                if (cost.tp,cost.source_role,cost.target_role)!=(instance.tp,instance.role,target_role) or not cost.source_sha256:
                    continue
                saving_lower=max(0,predicted_saving_w*(1-error_fraction))*max(0,horizon_s-cost.time_upper_s)
                total_cost=cost.energy_upper_j+capacity_loss_j
                if saving_lower>total_cost:
                    candidates.append((saving_lower-total_cost,RoleAction(instance.instance_id,
                        instance.generation,target_role,saving_lower,total_cost)))
        if not candidates:
            return None
        action=max(candidates,key=lambda c:c[0])[1]
        return ControlPlan(snapshot.version,now,now+1,roles=(action,),
                           reason='predicted saving lower bound exceeds measured switching cost')

    def confirmed(self,plan,now):
        for action in plan.roles:
            self.last_commit[action.instance_id]=now


def search_roles(planner,roles,snapshot,pending,now,error_fraction=.3,budget_s=.05,
                 *,horizon_s=None,repetitions=1.):
    """Replay already-arrived admissions, carrying shared reservations forward.

    A role change must amortize against known queued work. No future arrivals
    or completed output lengths are needed, and node residency is not charged
    independently for every queued request from the same initial snapshot.
    """
    started=time.perf_counter()
    pending=tuple(pending[:3])
    if not pending: return None
    def cost(state):
        energy=0.
        for request in pending:
            candidates=planner.candidates(state,request,now)
            if not candidates: return None
            selected=candidates[0]
            energy+=selected.routes[0].incremental_j
            state=planner.advance(state,selected,request)
        return energy
    original=cost(snapshot)
    if original is None: return None
    horizon=max(r.ttft_remaining(now) for r in pending) if horizon_s is None else horizon_s
    if horizon<=0: return None
    best=None
    for instance in snapshot.instances:
        if instance.requests or instance.running or instance.waiting or not instance.accepting:
            continue
        for role in ('mixed','prefill','decode'):
            if time.perf_counter()-started>budget_s: return best[1] if best else None
            if role==instance.role: continue
            hypothetical=replace(snapshot,instances=tuple(replace(i,role=role)
                if i.instance_id==instance.instance_id else i for i in snapshot.instances))
            changed=cost(hypothetical)
            if changed is None: continue
            plan=roles.plan(snapshot,role,(original-changed)*repetitions/horizon,horizon,error_fraction,
                            now=now,instance_id=instance.instance_id)
            if plan:
                gain=plan.roles[0].savings_lower_j-plan.roles[0].switching_upper_j
                if best is None or gain>best[0]: best=(gain,plan)
    return best[1] if best else None
