"""Fixed spatial PD, independent stage batch admission and latency routing.

Placement is selected by DistServeSearch before startup. No DVFS or dynamic
layout adaptation is added to the paper-mechanism baseline.
"""
from dataclasses import replace

from .planner import JointPlanner
from .types import ControlPlan,FrequencyAction


class DistServeScheduler:
    def __init__(self,profiles,transfers,*,prefill_batch,decode_batch,clock_settle_s=.3,topology=None):
        if min(prefill_batch,decode_batch)<1:
            raise ValueError('independent positive stage batch limits required')
        self.limits={'prefill':prefill_batch,'decode':decode_batch}
        self.estimator=JointPlanner(profiles,transfers,dvfs=False,clock_settle_s=clock_settle_s,topology=topology)

    def plan(self,snapshot,pending,*,now):
        states=tuple(replace(i,accepting=i.accepting and len(i.requests)<self.limits[i.role])
                     for i in snapshot.instances if i.role in self.limits)
        candidates=self.estimator.candidates(replace(snapshot,instances=states),pending[0],now)
        if not candidates:
            return ControlPlan(snapshot.version,now,now+1,feasible=False,
                frequencies=tuple(FrequencyAction(i.instance_id,2520) for i in states),
                reason='DistServe: stage batch, KV or latency capacity unavailable')
        occupancy={i.instance_id:len(i.requests) for i in states}
        return replace(min(candidates,key=lambda p:(p.routes[0].predicted_ttft_s,
                       occupancy[p.routes[0].decode_id],p.routes[0].decode_id)),
                       reason='DistServe: fixed placement and minimum predicted first-token latency')
