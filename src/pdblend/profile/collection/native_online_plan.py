"""Predeclared PD32 online-observation discovery, separate from static energy.

The first design observes fixed clocks and both real clock transitions. It
does not fit an online energy model or qualify the original PD controller.
"""
from __future__ import annotations
from dataclasses import asdict,replace
from pathlib import Path
import time

from pdblend.bench.client import Request,poisson_trace,SLOS
from pdblend.bench.run import offline_forecast
from pdblend.profile.query.native_query_replay import tuning_groups
from .native_timing_plan import binding,read_bound,digest
from .native_timing_replay import Resolver
from .native_timing_audit import need
from .native_frequency_domain import validate_domain,domain_fields
from .native_runtime_collect import write_new

SCHEMA='pdblend-native-online-discovery-plan/v1'
REVISION='pdblend_native32_causal_clock_discovery_v1'
MODEL='Qwen2.5-32B-Instruct'
CANDIDATE='pdblend-native-online-discovery-coverage-freeze/v1'


def build_online_plan(ledger_ref,provenance_ref,frequency_domain_ref):
    domain=validate_domain(read_bound(frequency_domain_ref))
    need(domain['model_id']==MODEL and domain['tp']==2 and domain['frequencies_mhz']==[1500,2100],
         'online discovery requires declared native32 TP2 1500/2100 domain')
    groups=tuning_groups(ledger_ref,provenance_ref,Resolver(),MODEL,2,domain)
    points=[]
    for phase,scales,duration,seed in [('training',(.25,1.),60.,9901),('holdout',(.25,.5,.75,1.),150.,9902)]:
        for group in groups:
            if group['rate_scale'] not in scales:continue
            for name,frequencies in [('fixed_low',[1500]),('fixed_high',[2100]),
                                     ('low_to_high',[1500,2100]),('high_to_low',[2100,1500])]:
                points.append(dict(purpose=phase,dataset=group['dataset'],rate_scale=group['rate_scale'],
                    target_rate_rps=group['target_rate_rps'],parent_trace=group['trace_binding'],
                    confirmation=group['confirmation'],duration_s=duration,seed=seed,schedule=name,
                    actions=[dict(offset_s=0. if i==0 else duration/2,frequency_mhz=f) for i,f in enumerate(frequencies)],
                    frequency_domain_sha256=digest(domain),arrival_family='poisson'))
    return dict(schema=SCHEMA,revision=REVISION,model_id=MODEL,tp=2,pp=1,replicas=4,gpu_count=8,
        slots=4,min_m_instances=4,max_num_seqs=32,roles={'M':4},query_ledger=ledger_ref,query_provenance=provenance_ref,
        frequency_domain_ref=frequency_domain_ref,**domain_fields(domain),points=points,
        query_period_s=1.,native_state_period_s=.5,clock_settle_s=2.,tail_timeout_s=120.,
        initial_prior='exact independent tuning forecast with group target rate; frozen before each window',
        content_partition='SHA256(prompt,max_tokens) parity: even training, odd holdout',
        hardware_policy='predeclared fixed-frequency or one clock transition using real Controller.execute',
        planner_decisions_executed=False,online_model_fitted=False,online_policy_qualified=False,
        static_rate_model_consumed=False,formal_eligible=False,evaluation_used=False,
        training_service_s=sum(p['duration_s'] for p in points if p['purpose']=='training'),
        holdout_service_s=sum(p['duration_s'] for p in points if p['purpose']=='holdout'),
        scope='discovery of causal queries and actual clock-action cost; no online prediction qualification')


def validate_online_plan(plan):
    need(plan==build_online_plan(plan['query_ledger'],plan['query_provenance'],plan['frequency_domain_ref']),
         'online discovery plan differs from frozen non-evaluation inputs')
    return plan


def online_trace(plan,point):
    need(plan.get('schema')==SCHEMA and point in plan['points'],'online point outside declared discovery design')
    parent=read_bound(point['parent_trace']);records={};training=point['purpose']=='training'
    for row in parent['requests']:
        need(row['source']==point['dataset'],'online parent source differs')
        key=digest(dict(prompt=row['prompt'],max_tokens=row['max_tokens']))
        if int(key,16)%2==(0 if training else 1):records[key]=dict(prompt=row['prompt'],output_tokens=row['max_tokens'])
    need(records,'independent online content partition empty')
    requests=poisson_trace([records[k] for k in sorted(records)],point['target_rate_rps'],point['duration_s'],point['seed'],point['dataset'])
    need(requests and all(len(r.prompt)+r.max_tokens<=8192 and 2<=r.max_tokens<=512 for r in requests),
         'online real request domain missing')
    return dict(schema='pdblend-native-online-discovery-trace/v1',model_id=MODEL,dataset=point['dataset'],
        selection_split='calibration_'+point['purpose'],duration_s=point['duration_s'],seed=point['seed'],
        arrival_family='poisson',rate_rps=point['target_rate_rps'],requests=[asdict(r) for r in requests],
        parent_trace=point['parent_trace'],slo=dict(zip(('ttft_s','tpot_s'),SLOS[point['dataset']])),evaluation_used=False)


def build_prior(point):
    """The bootstrap comes from bound tuning, never the generated GPU cohort."""
    parent=read_bound(point['parent_trace']);confirmation=read_bound(point['confirmation'])
    need(confirmation.get('split')=='tuning' and confirmation.get('dataset')==point['dataset']
         and confirmation.get('trace_sha256')==point['parent_trace']['sha256']
         and confirmation.get('metrics',{}).get('passed') is True,
         'online bootstrap lacks confirmed independent tuning input')
    requests=[Request(**r) for r in parent['requests']]
    forecast=replace(offline_forecast(requests),rate_rps=point['target_rate_rps'])
    return dict(schema='pdblend-native-online-tuning-prior/v1',model_id=MODEL,dataset=point['dataset'],
        parent_trace=point['parent_trace'],confirmation=point['confirmation'],rate_scale=point['rate_scale'],
        target_rate_rps=point['target_rate_rps'],forecast=asdict(forecast),evaluation_used=False,
        cohort_outcomes_used=False,online_energy_model_used=False)


def freeze_prior(point,out):
    return write_new(Path(out),dict(build_prior(point),frozen_s=time.time()))


def validate_prior(ref,point,started_s):
    value=read_bound(ref)
    need({k:v for k,v in value.items() if k!='frozen_s'}==build_prior(point)
         and value['frozen_s']<=started_s,'online bootstrap differs or was frozen after observation')
    return value


def freeze_discovery_candidate(plan,training_ref,out):
    """Freeze training inventory, not a prediction or selected online policy."""
    from .native_online_collect import replay_online_collection
    result=replay_online_collection(training_ref,plan,phase='training')
    need(result['raw_complete'],'complete raw training required before discovery holdout')
    return write_new(Path(out),dict(schema=CANDIDATE,revision=REVISION,plan_sha256=digest(plan),
        training=training_ref,training_finished_s=result['finished_s'],frozen_s=time.time(),
        training_query_count=result['query_count'],model_fitted=False,policy_selected=False,
        coverage_semantics='observed training query inventory only; min/max is not a qualified hull',
        formal_eligible=False,online_policy_qualified=False))


def validate_discovery_candidate(ref,plan,*,started_s):
    from .native_online_collect import replay_online_collection
    value=read_bound(ref)
    need(value.get('schema')==CANDIDATE and value.get('revision')==REVISION
         and value.get('plan_sha256')==digest(plan) and value.get('model_fitted') is False
         and value.get('policy_selected') is False and value.get('formal_eligible') is False
         and value['training_finished_s']<=value['frozen_s']<=started_s,
         'holdout requires frozen training-only discovery inventory')
    result=replay_online_collection(value['training'],plan,phase='training')
    need(result['raw_complete'] and result['finished_s']==value['training_finished_s']
         and result['query_count']==value['training_query_count'],'frozen discovery training does not replay')
    return value
