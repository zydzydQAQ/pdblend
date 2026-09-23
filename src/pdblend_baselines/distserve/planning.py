"""Ports of upstream config enumeration and goodput bisection (Apache-2.0).

Modified: explicit model geometry, injected simulator and diagnostics; caller
owns calibration-only history. No PDblend optimizer, profile index or power model.
"""
import itertools
import math
import numpy as np

from .policy import positive_int


def enumerate_configs(*, layers, attention_heads, num_nodes, gpus_per_node,
                      high_affinity=False, allowed_tps=None):
    for name,value in [('layers',layers),('attention_heads',attention_heads),('num_nodes',num_nodes),('gpus_per_node',gpus_per_node)]:
        positive_int(value,name)
    total=num_nodes*gpus_per_node
    possible_pps=[p for p in range(1,layers+1) if layers%p==0]
    possible_tps=[t for t in range(1,attention_heads+1) if attention_heads%t==0]
    if allowed_tps is not None:possible_tps=[t for t in possible_tps if t in allowed_tps]
    limit=total if high_affinity else gpus_per_node
    tps=[t for t in possible_tps if t<=limit];pps=[p for p in possible_pps if p<=limit]
    cpps=[1] if high_affinity else [p for p in possible_pps if p<=num_nodes]
    configs=[]
    for c,tp,pp,td,pd in itertools.product(cpps,tps,pps,tps,pps):
        segment=tp*pp+td*pd
        if (not high_affinity and segment>gpus_per_node) or c*segment>total:continue
        if c*pp not in possible_pps or c*pd not in possible_pps:continue
        configs.append((c,tp,pp,td,pd))
    return tuple(configs)


def gpu_count(config):
    cross,tp,pp,td,pd=config
    return cross*(tp*pp+td*pd)


def capability_matrix(configs, *, supported_pairs, measured_pairs):
    """Absent data are never renamed to structural incompatibility."""
    supported=set(map(tuple,supported_pairs));measured=set(map(tuple,measured_pairs))
    rows=[]
    for config in configs:
        cross,tp,pp,td,pd=config
        pairs={(tp,cross*pp),(td,cross*pd)}
        unsupported=sorted(pairs-supported);missing=sorted((pairs & supported)-measured)
        status='unsupported_engine' if unsupported else 'missing_profile' if missing else 'covered'
        rows.append(dict(config=tuple(config),status=status,unsupported_pairs=unsupported,
                         missing_profile_pairs=missing,gpu_count=gpu_count(config)))
    return rows


def binary_goodput(config, simulate, *, ttft_s, tpot_s, ttft_percentage=90,
                   tpot_percentage=90, max_per_gpu_rate=5., epsilon=.25):
    for name,value in [('ttft_s',ttft_s),('tpot_s',tpot_s),('max_per_gpu_rate',max_per_gpu_rate),('epsilon',epsilon)]:
        if not math.isfinite(value) or value<=0:raise ValueError(name+' must be finite and positive')
    if not 0<ttft_percentage<=100 or not 0<tpot_percentage<=100:raise ValueError('invalid percentile')
    low=0.;high=float(max_per_gpu_rate);best=0.;trials=[]
    result=dict(config=tuple(config),status='complete',predicate='separate_ttft_tpot_quantiles_strict_less',
                trials=trials,best_per_gpu_rate=None,observation='simulation_only',gpu_qualified=False)
    while high-low>epsilon:
        per_gpu=(low+high)/2;rate=per_gpu*gpu_count(config)
        try:
            values=simulate(tuple(config),rate)
            ttft=np.asarray(values['ttft_s'],dtype=float);tpot=np.asarray(values['tpot_s'],dtype=float)
            if ttft.ndim!=1 or not len(ttft) or ttft.shape!=tpot.shape:
                raise ValueError('per-offered-request latency vectors required')
            if np.any(np.isnan(ttft)) or np.any(np.isnan(tpot)) or np.any(ttft<0) or np.any(tpot<0):
                raise ValueError('invalid simulated latency')
            qttft=float(np.quantile(ttft,ttft_percentage/100));qtpot=float(np.quantile(tpot,tpot_percentage/100))
            passed=qttft<ttft_s and qtpot<tpot_s
        except Exception as exc:
            result.update(status='simulation_failed',error=f'{type(exc).__name__}: {exc}')
            return result
        trials.append(dict(per_gpu_rate=per_gpu,rate_rps=rate,ttft_quantile_s=qttft,tpot_quantile_s=qtpot,
                           joint_attainment=float(np.mean((ttft<=ttft_s)&(tpot<=tpot_s))),
                           offered=len(ttft),passed=bool(passed)))
        if passed:low=per_gpu;best=per_gpu
        else:high=per_gpu
    result['best_per_gpu_rate']=best
    result['search_upper_bound']=max_per_gpu_rate
    return result


def best_config(config_to_goodput):
    valid=[(c,r) for c,r in config_to_goodput.items() if r is not None and math.isfinite(r) and r>=0]
    if not valid:return None,0.
    return max(valid,key=lambda item:(item[1],-gpu_count(item[0])))


def select_placement(*, layers, attention_heads, allowed_tps, supported_pairs,
                     measured_pairs, simulator, rate_rps, ttft_s, tpot_s,
                     num_nodes=1, gpus_per_node=8, high_affinity=False,
                     max_per_gpu_rate=5., epsilon=.25, require_profile_coverage=True):
    """Official per-GPU search followed by replica allocation for declared rate.

    The fixed-budget overload rule deploys maximum replicas and records the
    shortfall. It never invents goodput or treats simulation as qualification.
    """
    if not math.isfinite(rate_rps) or rate_rps<=0:raise ValueError('positive target rate required')
    configs=enumerate_configs(layers=layers,attention_heads=attention_heads,num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,high_affinity=high_affinity,allowed_tps=allowed_tps)
    matrix=capability_matrix(configs,supported_pairs=supported_pairs,measured_pairs=measured_pairs)
    if require_profile_coverage and any(r['status']=='missing_profile' for r in matrix):
        raise ValueError('supported DistServe configuration has missing profile; complete calibration first')
    searches=[binary_goodput(row['config'],simulator,ttft_s=ttft_s,tpot_s=tpot_s,
                 max_per_gpu_rate=max_per_gpu_rate,epsilon=epsilon)
              for row in matrix if row['status']=='covered']
    chosen,goodput=best_config({r['config']:r['best_per_gpu_rate'] for r in searches})
    if chosen is None:raise ValueError('no successfully simulated measured DistServe placement')
    total=num_nodes*gpus_per_node;per_replica=gpu_count(chosen)
    capacity=goodput*per_replica
    available=total//per_replica
    requested=math.ceil(rate_rps/capacity) if capacity>0 else available+1
    replicas=min(requested,available)
    return dict(upstream_revision='82831f1604cc6b10bebd360f6c437a07790dde9f',
        selection='maximum per-GPU goodput; ties use fewer GPUs; replicas cover declared rate',
        matrix=matrix,searches=searches,complete_search_space=all(r['status']=='covered' for r in matrix),
        gpu_qualified=False,measurement='predicted_initial_configuration',
        selected=dict(config=chosen,per_gpu_goodput=goodput,replicas=replicas,
            requested_replicas=requested,total_gpu_count=replicas*per_replica,
            predicted_capacity_rps=capacity*replicas,
            predicted_capacity_shortfall=capacity*replicas<rate_rps))
