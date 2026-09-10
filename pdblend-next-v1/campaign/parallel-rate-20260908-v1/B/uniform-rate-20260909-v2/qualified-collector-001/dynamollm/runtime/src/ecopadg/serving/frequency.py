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
                if i.frequency_mhz!=2520: recoveries.append(FrequencyAction(i.instance_id,2520))
                continue
            if not i.dvfs_allowed or not 0<=now-i.timestamp_s<=estimator.telemetry_ttl_s:
                recoveries.append(FrequencyAction(i.instance_id,2520));continue
            # Admission owns the prefill/import budget. Optimize the now-pure
            # decode phase only once every admitted request has a first token.
            if any(not r.emitted for r in i.requests): continue
            if any(tail is None for tail in tails.values()):
                recoveries.append(FrequencyAction(i.instance_id,2520));continue
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
            chosen=min(choices)[1] if choices else 2520
            if chosen!=i.frequency_mhz:
                action=FrequencyAction(i.instance_id,chosen)
                if chosen==2520:
                    recoveries.append(action)
                else:
                    _,_,slack,delay=next(item for item in choices if item[1]==chosen)
                    proposals.append((action,slack,delay))
        if recoveries:
            # Restore service first. A separate subsequent optimization plan
            # cannot prevent recovery by expiring while waiting for its turn.
            return ControlPlan(snapshot.version,now,expires,frequencies=tuple(recoveries),
                reason='restore capacity before the next frequency optimization')
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
