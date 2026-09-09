"""Pure pending-work ranking over the unchanged admission feasibility checks."""
from dataclasses import replace
import time

def plan(planner,snapshot,pending,*,now=None,joint=True,enabled=False):
    if not enabled or not pending:
        return planner.plan(snapshot,pending,now=now,joint=joint)
    now=time.time() if now is None else now
    candidates=planner.candidates(snapshot,pending[0],now)
    if not candidates:
        # Preserve explicit original infeasibility / measured coverage recovery.
        return planner.plan(snapshot,pending,now=now,joint=joint)
    chosen=min(candidates,key=lambda p:(p.routes[0].predicted_tpot_s,
        p.routes[0].predicted_ttft_s,p.routes[0].incremental_j,
        p.routes[0].decode_id,tuple((a.instance_id,a.frequency_mhz) for a in p.frequencies)))
    return replace(chosen,reason='pending-work capacity ranking within original measured admission feasibility')
