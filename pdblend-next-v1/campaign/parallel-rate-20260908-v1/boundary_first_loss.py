"""Latest user scope: stop after the first complete SLO<90% observation."""
from decimal import Decimal
from collections import defaultdict

def next_boundary(observations):
    versions={x['implementation_id'] for x in observations}
    if len(versions)>1:raise ValueError('mixed versions cannot establish a first-loss endpoint')
    groups=defaultdict(list)
    for x in observations:groups[Decimal(str(x['rate_rps']))].append(x)
    pass_rates=[];fail_rates=[];uncertain=[]
    for rate,rows in sorted(groups.items()):
        if any(not r.get('measurement_valid') or not r.get('work_complete') for r in rows):
            uncertain.append(rate);continue
        success=[r['slo_attainment']>=.90 for r in rows]
        if all(success):pass_rates.append(rate)
        elif not any(success):fail_rates.append(rate)
        else:uncertain.append(rate)
    end=min(fail_rates,default=None)
    lower=max((r for r in pass_rates if end is None or r<end),default=None)
    answer=dict(implementation_id=next(iter(versions),None),jobs=[],
        last_complete_service_pass=str(lower) if lower is not None else None,
        first_complete_service_loss=str(end) if end is not None else None,
        uncertain_or_incomplete_rates=[str(r) for r in uncertain],
        below_90_is_not_relative_baseline_failure=True,
        independent_seed_confirmation_required=False,interval_bisection_required=False,
        throughput_plateau_established=False)
    if end is not None:
        return dict(answer,action='stop_higher_rates',requested_stop_condition_observed=True)
    if not pass_rates:return dict(answer,action='need_complete_control',requested_stop_condition_observed=False)
    if uncertain and max(uncertain)>=max(pass_rates):
        return dict(answer,action='diagnose_before_expanding',requested_stop_condition_observed=False)
    return dict(answer,action='expand_1_25',requested_stop_condition_observed=False,
        jobs=[dict(rate=str(max(pass_rates)*Decimal('1.25')),seed=701)])

def paired_jobs(decisions):
    groups=sorted({(x['rate'],x['seed']) for d in decisions for x in d['jobs']},key=lambda v:(Decimal(v[0]),v[1]))
    return [dict(rate=rate,seed=seed,systems=['pdblend','mixed','distserve','dynamollm','ecoserve'],
        arrival_window_s=100,exploration_precedes_endpoint_pairing=True) for rate,seed in groups]
