"""Shared post-prefill DVFS, with measured switching cost and prefix slack."""
from dataclasses import dataclass
import math
import json
from pathlib import Path

from .types import ControlPlan,FrequencyAction
from .tails import TailModel


@dataclass(frozen=True)
class FrequencyCost:
    tp: int
    source_mhz: int
    target_mhz: int
    duration_upper_s: float
    energy_upper_j: float
    source_sha256: str

    def __post_init__(self):
        if (self.tp not in (1,2,4,8) or self.source_mhz==self.target_mhz
                or min(self.source_mhz,self.target_mhz)<=0 or not self.source_sha256
                or any(not math.isfinite(v) or v<0 for v in (self.duration_upper_s,self.energy_upper_j))):
            raise ValueError('finite measured frequency transition required')


def coverage_recovery(estimator, instance, now, *, elapsed_s=0.):
    """Fastest measured next decode step, never an energy-saving claim.

    Only already-decoding requests can authorize this alternative to max
    recovery. Actual clocks, freshness and prefix budgets must all agree.
    """
    if not getattr(estimator,'coverage_aware_recovery',False): return None
    if (not estimator.dvfs or not instance.dvfs_allowed or not instance.accepting
            or instance.parked or instance.role not in ('mixed','decode')
            or not math.isfinite(now) or not math.isfinite(elapsed_s)
            or elapsed_s<0 or not 0<=now-instance.timestamp_s<=estimator.telemetry_ttl_s
            or not instance.requests or instance.waiting
            or instance.running>len(instance.requests)
            or any(r.emitted<=0 or r.first_token_s is None or not math.isfinite(r.first_token_s)
                   or r.first_token_s>now for r in instance.requests)):
        return None
    batch=max(1,len(instance.requests),instance.running)
    choices=[]
    for frequency in estimator.profiles.frequencies(instance.role,instance.tp):
        points=[estimator.point(instance,r,frequency,batch) for r in instance.requests]
        if not all(points): continue
        if frequency==instance.frequency_mhz:
            delay=0.
        else:
            measured=[c for c in estimator.frequency_costs or ()
                if c.tp==instance.tp and c.source_mhz==instance.frequency_mhz
                and c.target_mhz==frequency and c.source_sha256]
            if not measured: continue
            delay=max(c.duration_upper_s for c in measured)
        steps=[p.iteration_s*p.bound for p in points]
        credits=[r.next_token_remaining(now) for r in instance.requests]
        totals=[elapsed_s+delay+step for step in steps]
        if any(not math.isfinite(x) or x<0 for x in steps+credits+totals): continue
        if any(not math.isfinite(r.tpot_s) or r.tpot_s<=0 or step>r.tpot_s or total>credit
               for r,step,total,credit in zip(instance.requests,steps,totals,credits)): continue
        expires=min(instance.timestamp_s+estimator.telemetry_ttl_s,
                    now+min(credit-total for credit,total in zip(credits,totals)))
        if expires<now: continue
        choices.append((delay+max(steps),-frequency,frequency,delay,expires))
    if not choices: return None
    _,_,frequency,delay,expires=min(choices)
    return frequency,delay,expires


def recovery_actions(estimator, instances, now, expires, *, maximum=2520):
    """Account for sequential clock commands before later bounded recovery."""
    if not getattr(estimator,'coverage_aware_recovery',False):
        return tuple(FrequencyAction(i.instance_id,maximum) for i,_ in instances),expires,False
    actions=[];elapsed=0.;used=False
    for instance,eligible in instances:
        covered=coverage_recovery(estimator,instance,now,elapsed_s=elapsed) if eligible else None
        if covered is not None:
            frequency,delay,limit=covered;expires=min(expires,limit);used=True
        else:
            frequency=maximum
            costs=[c.duration_upper_s for c in estimator.frequency_costs or ()
                if c.tp==instance.tp and c.source_mhz==instance.frequency_mhz
                and c.target_mhz==frequency and c.source_sha256]
            delay=(math.inf if instance.parked else 0. if frequency==instance.frequency_mhz
                   else max(costs) if costs else math.inf)
        actions.append(FrequencyAction(instance.instance_id,frequency));elapsed+=delay
    return tuple(actions),expires,used


class FrequencyPlanner:
    def __init__(self,estimator,costs):
        self.estimator=estimator;self.costs=tuple(costs)

    def plan(self,snapshot,now):
        actions=[];recoveries=[];proposals=[];estimator=self.estimator
        expires=now+estimator.telemetry_ttl_s
        telemetry_expiry=min((i.timestamp_s+estimator.telemetry_ttl_s for i in snapshot.instances),default=expires)
        residency=estimator.node_residency(snapshot)
        tail_model=TailModel(estimator,snapshot,now)
        tails=tail_model.tails
        for i in snapshot.instances:
            if not i.accepting or not i.requests: continue
            if i.role=='prefill':
                if i.frequency_mhz!=2520: recoveries.append((i,False))
                continue
            if not i.dvfs_allowed or not 0<=now-i.timestamp_s<=estimator.telemetry_ttl_s:
                recoveries.append((i,False));continue
            # Admission owns the prefill/import budget. Optimize the now-pure
            # decode phase only once every admitted request has a first token.
            if any(not r.emitted for r in i.requests): continue
            if any(tail is None for tail in tails.values()):
                recoveries.append((i,True));continue
            other_tail=max((tail for name,tail in tails.items() if name!=i.instance_id),default=0.)
            batch=max(1,len(i.requests),i.running+i.waiting);choices=[]
            for frequency in estimator.profiles.frequencies(i.role,i.tp):
                points=[tail_model.point(i,r,frequency,batch) for r in i.requests]
                if not all(points): continue
                candidates=[c for c in self.costs if c.tp==i.tp and c.source_mhz==i.frequency_mhz
                            and c.target_mhz==frequency and c.source_sha256]
                if frequency!=i.frequency_mhz and not candidates: continue
                delay=max((c.duration_upper_s for c in candidates),default=0.)
                switch_j=max((c.energy_upper_j for c in candidates),default=0.)
                if any(p.iteration_s*p.bound>r.tpot_s or
                       p.iteration_s*p.bound+delay>r.next_token_remaining(now) for r,p in zip(i.requests,points)):
                    continue
                duration=max(max(r.predicted_output-r.emitted,1)*p.iteration_s for r,p in zip(i.requests,points))
                power=max(p.phase_power_bound('decode') for p in points)
                other_residency=max(0,residency-estimator.instance_residency(i))
                energy=power*duration+other_residency*max(0,duration-other_tail)+switch_j
                slack=min(r.next_token_remaining(now)-p.iteration_s*p.bound-delay for r,p in zip(i.requests,points))
                choices.append((energy,frequency,slack,delay))
            if not choices and getattr(estimator,'coverage_aware_recovery',False):
                recoveries.append((i,True));continue
            chosen=min(choices)[1] if choices else 2520
            if chosen!=i.frequency_mhz:
                action=FrequencyAction(i.instance_id,chosen)
                if chosen==2520:
                    recoveries.append((i,False))
                else:
                    _,_,slack,delay=next(item for item in choices if item[1]==chosen)
                    proposals.append((action,slack,delay))
        if recoveries:
            # Restore service first. A separate subsequent optimization plan
            # cannot prevent recovery by expiring while waiting for its turn.
            actions,expires,covered=recovery_actions(estimator,recoveries,now,expires)
            return ControlPlan(snapshot.version,now,expires,frequencies=actions,
                reason=('measured coverage recovery before frequency optimization' if covered else
                        'restore capacity before the next frequency optimization'))
        elapsed=0.
        for action,slack,delay in proposals:
            remaining=slack-elapsed
            if remaining<0 or telemetry_expiry<now: continue
            actions.append(action)
            expires=min(expires,telemetry_expiry,now+remaining)
            # Backend clock actions execute sequentially. Do not presume that
            # tokens emitted meanwhile will provide the later instance credit.
            elapsed+=delay
        return ControlPlan(snapshot.version,now,expires,frequencies=tuple(actions),
            reason='shared post-prefill energy minimum; measured switching cost and per-request prefix credit')


def verify_frozen_costs(config,profiles,freeze):
    """Every reachable optimization transition must have frozen hardware data."""
    from .evidence import sha256
    from .measurement import power_evidence
    costs=[FrequencyCost(**c) for c in config.get('frequency_costs',[])]
    if not costs: raise ValueError('formal DVFS requires measured frequency switching costs')
    raw_by_hash={}
    for path in config.get('frequency_evidence',[]):
        path=Path(path).resolve();digest=sha256(path)
        if str(path) not in freeze['groups']['profiles'] or freeze['files'].get(str(path))!=digest:
            raise ValueError('frequency switching evidence is not frozen')
        raw=json.loads(path.read_text())
        if (not raw.get('complete') or raw.get('sampling_error') or not raw.get('prefix_matches_reference')
                or not raw.get('frequency_samples') or not raw.get('engine_provenance')
                or any(e['image_id']!=freeze['identities']['engine_image'] for e in raw['engine_provenance'])):
            raise ValueError('invalid clock-transition hardware evidence')
        if config.get('power_mode')=='instant' and not power_evidence(
                raw.get('power_samples',[]),raw.get('power_source'),raw.get('power_metadata'))['power_source_verified']:
            raise ValueError('clock-transition costs require verified instantaneous power')
        raw_by_hash[digest]=raw
    for cost in costs:
        raw=raw_by_hash.get(cost.source_sha256,{})
        measurements=[r for r in raw.get('switches',[]) if
                      (r['tp'],r['source_mhz'],r['target_mhz'])==(cost.tp,cost.source_mhz,cost.target_mhz)]
        if (not measurements or cost.duration_upper_s<max(r['finished_s']-r['started_s'] for r in measurements)
                or cost.energy_upper_j<max(r['energy_j'] for r in measurements)):
            raise ValueError('frequency cost is missing or below its measured evidence')
    reachable={i['tp'] for i in config.get('instances',[])}
    reachable.update(tp for c in config.get('topology_costs',[]) for tp in c['target_tps'])
    pairs={(c.tp,c.source_mhz,c.target_mhz) for c in costs}
    for tp in reachable:
        frequencies={p['frequency_mhz'] for p in profiles['points'] if p['tp']==tp}
        if any((tp,a,b) not in pairs for a in frequencies for b in frequencies if a!=b):
            raise ValueError('unmeasured reachable frequency transition')
