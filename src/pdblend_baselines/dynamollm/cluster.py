"""Whole-cluster ScaleInst target inventory, distinct from pool ScaleShard."""
from collections import defaultdict
import math


def plan_reallocation(current,desired):
    available=defaultdict(list)
    for instance in sorted(current,key=lambda i:i.instance_id):available[(instance.shape,instance.tp)].append(instance)
    retained=[];targets=[]
    for shape,choices in sorted(desired.items()):
        for choice,count in choices:
            for _ in range(count):
                matches=available[(shape,choice.tp)]
                if matches:retained.append(matches.pop(0).instance_id)
                else:targets.append(dict(shape=shape,tp=choice.tp,batch=choice.batch,
                                         capacity_rps=choice.capacity_rps,power_w=choice.power_w))
    sources=[i for i in current if i.instance_id not in retained]
    return dict(retained_ids=sorted(retained),source_ids=sorted(i.instance_id for i in sources),
        source_tps=sorted(i.tp for i in sources),targets=targets,target_tps=sorted(r['tp'] for r in targets),
        retire_only=bool(sources) and not targets,changed=bool(sources or targets))


def admission_reason(*,required_rate,current_capacity,savings_j,overhead_j):
    if any(not math.isfinite(x) for x in (required_rate,current_capacity,savings_j,overhead_j)):
        raise ValueError('finite measured capacity and cost required')
    if required_rate>current_capacity+1e-9:return 'required_capacity'
    if savings_j>overhead_j:return 'amortized_energy'
    return None


def pool_forecasts(rates,active_shapes):
    """Keep zero-budget class arrivals in the first compatible allocated pool.

    Native pools retain their own forecast; absent pools carry the same ordered
    componentwise spill used by the cluster allocation into an existing pool.
    This is a prediction of offered work, so it is never clamped to old capacity.
    """
    from .policy import SHAPES,dominates
    result={shape:0. for shape in SHAPES if shape in active_shapes}
    for shape,rate in rates.items():
        if rate<0 or not math.isfinite(rate):raise ValueError('finite nonnegative class forecast required')
        if not rate:continue
        pool=next((s for s in result if dominates(s,shape)),None)
        if pool is None:raise ValueError('class forecast has no compatible allocated pool')
        result[pool]+=rate
    return result
