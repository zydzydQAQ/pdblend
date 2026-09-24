"""Independent PDblend CUDA raw audit and timing-only holdout fit.

No profile registry flag, power component or deployment is promoted here.
"""
from __future__ import annotations
import math
import statistics
from .native_timing_plan import digest
from .native_frequency_domain import check_rows


def need(condition, message):
    if not condition:raise ValueError(message)


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def measured_events(sample, *, tp=1):
    ranks=sorted(sample.get('ranks',[]),key=lambda r:r.get('rank',-1))
    need(len(ranks)==tp and [r.get('rank') for r in ranks]==list(range(tp)), 'complete physical rank inventory required')
    selected=[]
    for rank in ranks:
        events=[]
        for row in rank.get('samples',[]):
            need(row.get('system')=='pdblend' and row.get('measurement_scope')=='runner'
                 and row.get('rank')==rank['rank'] and row.get('tp')==tp and row.get('pp')==1
                 and not row.get('failed') and finite(row.get('gpu_elapsed_ms')) and row['gpu_elapsed_ms']>0
                 and finite(row.get('at_s')), 'PDblend native CUDA timing identity missing')
            if row.get('role') not in ('prefill','decode'):continue
            batch=row.get('batch');vectors=[row.get(k) for k in ('prompt_lengths','context_lengths','scheduled_lengths','request_ids')]
            need(type(batch)is int and 1<=batch<=32 and all(isinstance(v,list) and len(v)==batch for v in vectors),
                 'native per-request shape vectors missing')
            prompts,contexts,scheduled,ids=vectors
            need(len(ids)==len(set(ids)) and all(type(v)is int and 1<=v<=8192 for vector in vectors[:3] for v in vector),
                 'native shape or request identity invalid')
            if row['role']=='prefill' and not prompts==contexts==scheduled:continue
            if row['role']=='decode':
                need(scheduled==[1]*batch,'decode must schedule one real token per request')
                if len(set(contexts))!=1:continue  # Explicitly not a homogeneous scalar-context observation.
            events.append(row)
        selected.append(events)
    def key(row):return tuple((k,tuple(row[k]) if isinstance(row[k],list) else row[k]) for k in
        ('role','batch','request_ids','prompt_lengths','context_lengths','scheduled_lengths'))
    need(all(list(map(key,rs))==list(map(key,selected[0])) for rs in selected), 'rank sequence/shape alignment differs')
    return [dict(role=row['role'],batch=row['batch'],context_tokens=max(row['context_lengths']),
            prompt_tokens=max(row['prompt_lengths']),at_s=row['at_s'],request_ids=row['request_ids'],
            latency_ms=max(rs[i]['gpu_elapsed_ms'] for rs in selected)) for i,row in enumerate(selected[0])]


def audit_window(raw, *, identity):
    point=raw['point'];cap=raw['capability'];start=raw['start_s'];end=raw['end_s']
    need(raw.get('schema')=='pdblend-native-timing-window-v1' and raw.get('status')=='measured', 'window did not complete')
    need(raw.get('system')=='pdblend' and all(cap.get(k)==v for k,v in identity.items()), 'window model/source/UUID identity differs')
    need(finite(start) and finite(end) and end-start>=5. and raw['settle_finished_s']-raw['settle_started_s']>=2.
         and raw['measurement_started_s']<=raw['settle_started_s']<=raw['settle_finished_s']<=start,
         'idle measurement arm, settle, or service duration incomplete')
    need(raw['measurement_start'].get('acknowledged') is True,'measurement start ACK missing')
    stopped=raw['measurement_stop'].get('ranks',[])
    need(len(stopped)==identity['tp'] and {r.get('rank') for r in stopped}==set(range(identity['tp']))
         and all(r.get('acknowledged') is True for r in stopped),'measurement stop rank ACK missing')
    need(not raw.get('cleanup_errors') and not raw.get('sampler_error'), 'cleanup or sampler failed')
    state=raw['drain']
    from pdblend.bench.comparison_acceptance import _state
    need(state.get('acknowledged') is True and state.get('drained') is True,'native final drain ACK missing')
    _state(state,identity['tp'],state.get('response_at_s'))
    clock=raw['clock_receipt']
    need(clock.get('acknowledged') is True and clock.get('success') is True
         and clock.get('requested_frequency_mhz')==point['frequency_mhz']
         and {g.get('gpu_uuid') for g in clock.get('gpus',[])}==set(identity['gpu_uuids']), 'physical clock ACK differs')
    clocks=[(t,v) for t,v in raw['frequency_samples'] if start<=t<=end]
    need(clocks and clocks[0][0]<=start+1. and clocks[-1][0]>=end-1.
         and all(0<b[0]-a[0]<=1. for a,b in zip(clocks,clocks[1:]))
         and all(len(v)==identity['tp'] and all(finite(f) and abs(f-point['frequency_mhz'])<=30 for f in v) for _,v in clocks),
         'observed frequency coverage differs')
    clients=raw['client_requests'];ids={r['request_id'] for r in clients}
    need(clients and len(ids)==len(clients) and all(r.get('terminal') is True and not r.get('error')
         and r['completion_tokens']==point['output_tokens'] and finite(r['submitted_s']) and finite(r['finished_s'])
         and r['finished_s']>=r['submitted_s'] for r in clients), 'complete client workload receipt missing')
    events=[r for r in measured_events(raw['sample'],tp=identity['tp']) if r['role']==point['role'] and start<=r['at_s']<end
            and r['batch']==point['batch'] and r['prompt_tokens']==point['prompt_tokens']]
    need(len(events)>=(8 if point['role']=='decode' else 1), 'actual requested shape CUDA steps missing')
    need(all(set(r['request_ids'])<=ids for r in events),'CUDA sample refers to unbound request')
    return [dict(r,**({'frequency_domain_sha256':point['frequency_domain_sha256']} if 'frequency_domain_sha256' in point else {}),frequency_mhz=point['frequency_mhz'],purpose=point['purpose'],window_id=raw['window_id'],
                 point_sha256=digest({k:point[k] for k in ('role','batch','prompt_tokens','output_tokens','frequency_mhz')})) for r in events]


def fit_component(training,holdout,*,identity,raw_bindings,measurement_qualification,limits):
    """Fit actual shape CUDA times; independent windows decide qualification."""
    import numpy as np
    from scipy.optimize import nnls
    from scipy.spatial import ConvexHull
    need(training and holdout,'training and independent holdout required')
    frequencies=check_rows([*training,*holdout],identity)
    need(not {r['window_id'] for r in training}&{r['window_id'] for r in holdout},'holdout window leakage')
    need(not {r['point_sha256'] for r in training}&{r['point_sha256'] for r in holdout},'holdout point leakage')
    def coordinates(row):return [float(row['batch']),row['context_tokens']/8192.] if row['role']=='decode' else [row['prompt_tokens']/8192.]
    def features(row):
        if row['role']=='decode':
            b,c=coordinates(row);return [1.,b,b*c]
        l=coordinates(row)[0];return [1.,l,l*l]
    models=[];valid=measurement_qualification.get('qualified') is True
    for f in frequencies:
        for role in ('prefill','decode'):
            train=[r for r in training if r['frequency_mhz']==f and r['role']==role]
            hold=[r for r in holdout if r['frequency_mhz']==f and r['role']==role]
            need(train and hold,'missing frequency/role training or holdout')
            x=np.asarray([features(r) for r in train]);y=np.asarray([r['latency_ms'] for r in train])
            coef,_=nnls(x,y);vertices=np.unique(np.asarray([coordinates(r) for r in train]),axis=0)
            hull=ConvexHull(vertices) if role=='decode' else None
            errors=[];outside=[]
            for row in hold:
                target=np.asarray(coordinates(row))
                covered=(bool(np.all(hull.equations[:,:-1]@target+hull.equations[:,-1]<=1e-9)) if hull is not None
                         else bool(vertices.min()-1e-9<=target[0]<=vertices.max()+1e-9))
                if not covered:outside.append(dict(batch=row['batch'],context_tokens=row['context_tokens']))
                errors.append(abs(float(np.dot(features(row),coef))-row['latency_ms'])/row['latency_ms'])
            ordered=sorted(errors);checks=dict(mean_relative_error=statistics.mean(errors),
                p95_relative_error=ordered[math.ceil(.95*len(ordered))-1],max_relative_error=max(errors))
            passed=not outside and all(checks[k]<=v for k,v in limits.items());valid &= passed
            models.append(dict(frequency_mhz=f,role=role,coefficients=coef.tolist(),training_events=len(train),
                holdout_events=len(hold),coverage_vertices=vertices[hull.vertices].tolist() if hull is not None else vertices.tolist(),holdout=checks,qualified=passed,
                uncovered_holdout_shapes=list({digest(r):r for r in outside}.values())))
    return dict(schema='pdblend-native-timing-component-v1',system='pdblend',identity=identity,
        component_qualified=bool(valid),formal_eligible=False,full_profile_qualified=False,energy_comparable=False,
        scope='native_cuda_timing_component_only',models=models,raw_bindings=raw_bindings,
        measurement_qualification=measurement_qualification,holdout_limits=limits,
        remaining_gates=['power_domain_and_holdout','runtime_components','full_profile_composition',
                         'actual_tuning_query_coverage','formal_workload_energy'])
