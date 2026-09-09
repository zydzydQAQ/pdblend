"""Conservative role demand forecast from admission-visible history only."""
from dataclasses import dataclass,replace
import math


@dataclass(frozen=True)
class RoleForecast:
    requests: tuple
    repetitions: float
    horizon_s: float
    observations: int
    rate_lower_rps: float
    rate_upper_rps: float


def forecast_roles(history,now,started_s,*,window_s=60,horizon_s=30):
    """Three shape strata, with a conservative arrival-count lower bound.

    Inputs are copies of budgets at admission, so completed lengths cannot
    leak into the prediction. This estimates demand; admission still checks
    actual queues and per-request deadlines after any role change.
    """
    span=min(window_s,now-started_s)
    if span<30: return None
    observed=[r for r in history if now-span<=r.arrival_s<=now]
    n=len(observed)
    if n<20: return None
    # A conservative normal count bound, explicitly a heuristic forecast,
    # separate from the independent-run confidence intervals in evaluation.
    rate=max(0.,n-1.96*math.sqrt(n))/span
    if rate<=0: return None
    ordered=sorted(observed,key=lambda r:(r.input_tokens,r.predicted_output))
    samples=tuple(replace(ordered[min(n-1,int((i+.5)*n/3))],
        request_id='role-forecast-'+str(i),arrival_s=now,emitted=0,
        first_token_s=None,last_token_s=None) for i in range(3))
    return RoleForecast(samples,rate*horizon_s/3,horizon_s,n,rate,(n+1.96*math.sqrt(n))/span)
