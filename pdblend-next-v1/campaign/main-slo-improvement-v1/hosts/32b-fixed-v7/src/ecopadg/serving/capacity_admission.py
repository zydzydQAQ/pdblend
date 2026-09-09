"""Original search cohort and full causal queue pressure are independent inputs."""
from dataclasses import replace
import time

def evaluate(planner,snapshot,pending,*,now=None,joint=True,enabled=False,pressure_pending=None):
    now=time.time() if now is None else now
    original=planner.plan(snapshot,pending,now=now,joint=joint)
    if not enabled or not pending:return original,False
    if not original.feasible and original.frequencies:
        # An unsuccessful route search cannot independently push an occupied
        # instance to an uncovered maximum clock. Only a feasible atomic route
        # or the separate measured current-work recovery may change clocks.
        original=replace(original,frequencies=(),reason=original.reason+'; infeasible admission holds clocks')
    pressure_pending=tuple(pending if pressure_pending is None else pressure_pending)
    if not pressure_pending:return original,False
    oldest=min(pressure_pending,key=lambda r:(r.arrival_s,r.request_id))
    fresh=sum(i.accepting and 0<=now-i.timestamp_s<=planner.telemetry_ttl_s for i in snapshot.instances)
    queue_wait=max(0.,now-oldest.arrival_s)>=.1*oldest.ttft_s
    backlog=len(pressure_pending)>fresh
    actual_request=next((r for r in pressure_pending if r.request_id==pending[0].request_id),pending[0])
    remaining=actual_request.ttft_remaining(now)
    predicted_risk=bool(original.feasible and original.routes
        and original.routes[0].predicted_ttft_s>=.5*remaining)
    if not (queue_wait or backlog or predicted_risk):return original,False
    candidates=planner.candidates(snapshot,pending[0],now)
    if not candidates:return original,True
    chosen=min(candidates,key=lambda p:(p.routes[0].predicted_tpot_s,
        p.routes[0].predicted_ttft_s,p.routes[0].incremental_j,
        p.routes[0].decode_id,tuple((a.instance_id,a.frequency_mhz) for a in p.frequencies)))
    reasons=','.join(name for name,yes in [('wait10pct',queue_wait),('backlog',backlog),('ttft50pct',predicted_risk)] if yes)
    return replace(chosen,reason='causal capacity pressure ('+reasons+'); original measured admission feasibility'),True

def plan(planner,snapshot,pending,*,now=None,joint=True,enabled=False,pressure_pending=None):
    return evaluate(planner,snapshot,pending,now=now,joint=joint,enabled=enabled,pressure_pending=pressure_pending)[0]
