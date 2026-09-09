"""Deterministic next-rate decisions from one frozen system/version at a time.

This module plans experiments only. It does not relabel failed work, pool seeds,
infer a throughput plateau, or claim that an unmeasured boundary is certified.
"""
from collections import defaultdict
from decimal import Decimal

SEEDS=(1701,2701,3701)
SYSTEMS=('pdblend','mixed','distserve','dynamollm','ecoserve')

def decimal(rate):return Decimal(str(rate))

def classify(rows):
    if not rows:return 'unmeasured'
    if any(not x.get('measurement_valid') or not x.get('work_complete') for x in rows):
        return 'incomplete_or_invalid'
    passed=[x['slo_attainment']>=.90 for x in rows]
    return 'pass' if all(passed) else 'fail' if not any(passed) else 'uncertain'

def next_boundary(observations):
    versions={x['implementation_id'] for x in observations}
    if len(versions)>1:raise ValueError('boundary observations mix implementation versions')
    groups=defaultdict(list)
    for row in observations:groups[decimal(row['rate_rps'])].append(row)
    states={r:classify(v) for r,v in groups.items()}
    passes=sorted(r for r,s in states.items() if s=='pass')
    fails=sorted(r for r,s in states.items() if s=='fail')
    unknown=sorted(r for r,s in states.items() if s not in ('pass','fail'))
    result=dict(implementation_id=next(iter(versions),None),certified=False,
        service_boundary_only=True,throughput_platform_established=False,
        observed_states={str(r):states[r] for r in sorted(states)},
        incomplete_or_uncertain_rates=[str(r) for r in unknown],
        independent_seeds=list(SEEDS),jobs=[])
    if not passes:
        return dict(result,action='need_complete_service_pass_anchor',lower=None,upper=None)
    lower=max(passes);upper=min((r for r in fails if r>lower),default=None)
    result.update(lower=str(lower),upper=str(upper) if upper else None)
    if upper is None:
        if unknown and max(unknown)>lower:
            return dict(result,action='diagnose_or_resolve_higher_rate_uncertainty')
        rate=lower*Decimal('1.25')
        result.update(action='expand_1_25',jobs=[dict(rate=str(rate),seed=701)])
        return result
    width=(upper-lower)/lower
    result['relative_width']=str(width)
    if width>Decimal('.10'):
        midpoint=(lower+upper)/2
        if midpoint in unknown:
            return dict(result,action='retain_uncertain_interval')
        return dict(result,action='bisect',jobs=[dict(rate=str(midpoint),seed=701)])
    jobs=[]
    confirmed=True
    for endpoint,expected in ((lower,'pass'),(upper,'fail')):
        for seed in SEEDS:
            rows=[r for r in groups[endpoint] if r['seed']==seed]
            if not rows:
                jobs.append(dict(rate=str(endpoint),seed=seed));confirmed=False
            elif classify(rows)!=expected:confirmed=False
    # Nonmonotonic observations or a threshold-crossing repeated seed remain
    # visible uncertainty, even if a nearby pair happens to bracket 90%.
    contradictory=any(f<lower for f in fails) or any(lower<=u<=upper for u in unknown)
    if jobs:return dict(result,action='confirm_independent_seeds',jobs=jobs)
    if not confirmed or contradictory:return dict(result,action='retain_uncertain_interval')
    return dict(result,action='service_interval_confirmed',certified=True)

def paired_jobs(decisions):
    """New rates and seeds always materialize an identical trace for all five."""
    union=sorted({(j['rate'],j['seed']) for d in decisions for j in d['jobs']},
                 key=lambda x:(decimal(x[0]),x[1]))
    return [dict(rate=rate,seed=seed,systems=list(SYSTEMS),arrival_window_s=100,
        reuse_only_exact_frozen_trace=True) for rate,seed in union]
