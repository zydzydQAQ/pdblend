"""Measured frequency actions for current ledger work; no blind max recovery."""
from dataclasses import replace
import math
from .types import ControlPlan,FrequencyAction
from .capacity_admission import evaluate


def assessed(estimator,instance,target,now,elapsed=0.):
    if (instance.role not in ('mixed','decode') or not instance.accepting or instance.parked
            or not instance.dvfs_allowed or not 0<=now-instance.timestamp_s<=estimator.telemetry_ttl_s
            or not instance.requests or any(r.emitted<=0 or r.first_token_s is None
                or r.first_token_s>now for r in instance.requests)):
        return None
    batch=max(1,len(instance.requests),instance.running+instance.waiting)
    points=[estimator.point(instance,r,target,batch) for r in instance.requests]
    if not all(points):return None
    cost=estimator.frequency_cost(instance,target)
    if cost is None:return None
    delay=cost[0]
    steps=[p.iteration_s*p.bound for p in points]
    credits=[r.next_token_remaining(now) for r in instance.requests]
    totals=[elapsed+delay+step for step in steps]
    if (any(not math.isfinite(v) or v<0 for v in steps+credits+totals)
            or any(step>r.tpot_s or total>credit for r,step,total,credit
                   in zip(instance.requests,steps,totals,credits))):return None
    expires=min(instance.timestamp_s+estimator.telemetry_ttl_s,
                now+min(credit-total for credit,total in zip(credits,totals)))
    return dict(delay=delay,max_step=max(steps),expires_s=expires,batch=batch)


def safe_energy_plan(planner,snapshot,now,decision,excluded=()):
    if not getattr(decision,'frequencies',()):return decision
    instances={i.instance_id:i for i in snapshot.instances}
    accepted=[];elapsed=0.;expires=decision.expires_s
    for action in decision.frequencies:
        instance=instances.get(action.instance_id)
        proof=(assessed(planner.estimator,instance,action.frequency_mhz,now,elapsed)
               if instance is not None and action.instance_id not in excluded else None)
        if proof is None:continue
        if action.frequency_mhz!=instance.frequency_mhz:
            accepted.append(action);elapsed+=proof['delay'];expires=min(expires,proof['expires_s'])
    return replace(decision,frequencies=tuple(accepted),expires_s=expires,
        reason=decision.reason+'; every clock change verified against full admitted ledger coverage and prefix credit')


def risk_plan(planner,snapshot,now,pending,excluded=()):
    estimator=planner.estimator
    if pending:
        return ControlPlan(snapshot.version,now,now+estimator.telemetry_ttl_s,frequencies=(),
            reason='risk recovery held: full pending queue keeps route/clock ownership in atomic admission')
    actions=[];elapsed=0.;expires=now+estimator.telemetry_ttl_s
    blocked=[]
    for instance in snapshot.instances:
        if instance.instance_id in excluded or not instance.requests:continue
        choices=[]
        for target in estimator.profiles.frequencies(instance.role,instance.tp):
            proof=assessed(estimator,instance,target,now,elapsed)
            if proof is not None:choices.append((proof['delay']+proof['max_step'],-target,target,proof))
        if not choices:
            blocked.append(instance.instance_id);continue
        _,_,target,proof=min(choices)
        if target!=instance.frequency_mhz:
            actions.append(FrequencyAction(instance.instance_id,target))
            elapsed+=proof['delay'];expires=min(expires,proof['expires_s'])
    return ControlPlan(snapshot.version,now,expires,frequencies=tuple(actions),
        reason='measured risk recovery; stale, pending-phase, uncovered or exhausted-prefix instances hold clocks: '+str(tuple(blocked)))


def capacity_plan(planner,snapshot,now,pending,excluded=(),enabled=True):
    if not enabled:return planner.plan(snapshot,now)
    decision,pressure=evaluate(planner.estimator,snapshot,pending,now=now,joint=False,enabled=True)
    if not pressure:return safe_energy_plan(planner,snapshot,now,planner.plan(snapshot,now),excluded)
    target=tuple((a.instance_id,a.frequency_mhz) for a in decision.frequencies) if decision.feasible else ()
    return ControlPlan(snapshot.version,now,now+planner.estimator.telemetry_ttl_s,frequencies=(),
        reason='causal pending pressure: admission owns route and clocks atomically; feasible targets '+str(target))


def frequency_plan(planner,snapshot,now,pending,excluded=(),enabled=True):
    if not enabled:return planner.plan(snapshot,now)
    return (capacity_plan(planner,snapshot,now,pending,excluded) if pending
            else safe_energy_plan(planner,snapshot,now,planner.plan(snapshot,now),excluded))
