"""Model-owned timing designs for a future capacity-aware collector.

This module only prepares immutable points. It does not alter the frozen v1
collector, schedule GPUs, fit observations or qualify a deployable profile.
"""
from __future__ import annotations

import math
from copy import deepcopy

from .native_timing_plan import binding,read_bound,digest,FREQUENCIES
from .native_timing_audit import need,finite
from .native_frequency_domain import validate_domain,domain_fields,point_fields

MODEL_TP={'Qwen2.5-7B-Instruct':1,'Qwen2.5-14B-Instruct':1,'Qwen2.5-32B-Instruct':2}
DATASETS=('alpaca','sharegpt','longbench')
BATCHES=(1,2,4,8,16,24,32)
CAPACITY_POLICY=dict(schema='pdblend-native-timing-capacity-policy/v2',
    reservation='batch * ceil((prompt_tokens + output_tokens) / native_block_size) * native_block_size',
    strict_limit='10 * reserved_tokens < 9 * actual_total_kv_tokens',require_idle_full_free=True,
    unsupported_scope='configured_native_KV_capacity_guard_not_measured_timing',
    never_downgrade=['unhealthy_or_incomplete_state','stale_receipt','identity_mismatch','request_error',
                    'timeout','partial_completion','measurement_error','cleanup_failure'])


def _quantile(values,q):
    ordered=sorted(values)
    return ordered[math.floor(q*(len(ordered)-1))]


def _independent_interior(lo,hi,excluded,count):
    """Deterministic distinct shapes selected before any GPU observation."""
    result=[]
    for index in range(1,count+1):
        target=round(lo+(hi-lo)*index/(count+1))
        candidates=sorted(range(lo+1,hi),key=lambda n:(abs(n-target),n))
        selected=next((n for n in candidates if n not in excluded and n not in result),None)
        need(selected is not None,'model-owned shape interval lacks distinct independent holdouts')
        result.append(selected)
    return result


def build_plan(ledger_ref,provenance_ref,corpus_refs,*,model_id,frequency_domain_ref=None):
    need(model_id in MODEL_TP,'unsupported independent model topology')
    tp=MODEL_TP[model_id];ledger=read_bound(ledger_ref);provenance=read_bound(provenance_ref)
    domain=validate_domain(read_bound(frequency_domain_ref)) if frequency_domain_ref is not None else None
    frequencies=tuple(domain['frequencies_mhz']) if domain else FREQUENCIES
    if domain:
        need(domain['model_id']==model_id and domain['tp']==tp,'frequency domain model differs')
        fields=domain_fields(domain)
        need(all(ledger.get(k)==v and provenance.get(k)==v for k,v in fields.items())
             and ledger.get('frequency_domain_ref')==provenance.get('frequency_domain_ref')
             and ledger.get('frequency_domain_ref',{}).get('sha256')==frequency_domain_ref['sha256'],
             'new domain needs freshly executed bound planner ledger/provenance')
    else:
        need('frequency_domain' not in ledger and 'frequency_domain_ref' not in ledger,
             'new frequency ledger requires its explicit domain binding')
    need(ledger.get('schema')=='pdblend-offline-query-ledger/v2' and ledger.get('evaluation_read') is False
         and ledger.get('min_m_floor_overridden') is False and provenance.get('evaluation_read') is False
         and any(r.get('sha256')==ledger_ref['sha256'] for r in provenance.get('outputs',{}).values()),
         'bound actual non-evaluation canonical-policy query ledger required')
    owned=[r for r in ledger.get('ledgers',[]) if r.get('model_id')==model_id]
    blocked=[r for r in owned if r.get('status')=='blocked_missing_tuning_anchor']
    active=[r for r in owned if r not in blocked]
    need(active and all(r.get('selection_split')=='tuning' and r.get('tp')==tp and r.get('pp')==1
         and r.get('min_m_instances')==4 and r.get('engine_max_num_seqs')==32
         and r.get('frequency_scope')==list(frequencies)
         and (not domain or r.get('frequency_domain_sha256')==digest(domain)) for r in active),
         'model-owned TP/min-M/native32 tuning query domain differs')
    need(all(not r.get('queries') for r in blocked),'blocked tuning entry cannot supply successful queries')
    need(set(corpus_refs)==set(DATASETS),'three explicit model-owned corpus bindings required')
    prompts=[];contexts=[];corpora=[]
    for dataset in DATASETS:
        ref=corpus_refs[dataset];corpus=read_bound(ref)
        need(corpus.get('model_name')==model_id and corpus.get('dataset')==dataset,
             'timing corpus model/dataset identity differs')
        need(corpus.get('calibration') and corpus.get('tuning'),'independent calibration/tuning corpus splits required')
        rows=corpus['calibration']+corpus['tuning']
        for row in rows:
            n,m=row.get('input_tokens'),row.get('output_tokens')
            need(type(n)is int and type(m)is int and 1<=n<=7168 and 1<=m and n+m<=8192,
                 'invalid non-evaluation request shape')
            prompts.append(n);contexts.append(n+(m+1)//2)
        corpora.append(dict(dataset=dataset,**ref,splits=['calibration','tuning'],
            calibration_requests=len(corpus['calibration']),tuning_requests=len(corpus['tuning'])))
    demands={};unsupported=[];zero=[]
    for entry in active:
        for query in entry.get('queries',[]):
            if query.get('finite_arguments') is not True:continue
            method,args=query['method'],query['args']
            if method in ('decode_supported','step_seconds','token_energy_j'):
                b,c,f=args;shape=dict(role='decode',batch=b,context_tokens=c,frequency_mhz=f)
            elif method in ('prefill_seconds','prefill_marginal_seconds','prefill_energy_j'):
                n,f=args
                if n==0 and method=='prefill_marginal_seconds':
                    zero.append(dict(dataset=entry['dataset'],rate_scale=entry['rate_scale'],method=method,args=args));continue
                shape=dict(role='prefill',input_tokens=n,frequency_mhz=f)
            else:continue
            need(all(finite(v) and v>0 for k,v in shape.items() if k!='role') and shape['frequency_mhz'] in frequencies,
                 'finite timing query arguments differ from the exact-frequency scope')
            outside=(not 1<=shape.get('batch',1)<=32 or shape.get('context_tokens',0)>8192 or shape.get('input_tokens',0)>8191)
            key=digest(shape);item=demands.setdefault(key,dict(shape=shape,methods=[],datasets=[],rate_scales=[]))
            for field,value in [('methods',method),('datasets',entry['dataset']),('rate_scales',entry['rate_scale'])]:
                if value not in item[field]:item[field].append(value)
            if outside and key not in {r['shape_sha256'] for r in unsupported}:
                unsupported.append(dict(shape_sha256=key,shape=shape,reason='query_outside_native32_or_context_domain',
                                        sampled=False,clamped=False))
    actual=[item['shape'] for item in demands.values() if digest(item['shape']) not in {r['shape_sha256'] for r in unsupported}]
    decode=[s['context_tokens'] for s in actual if s['role']=='decode']
    prefill=[s['input_tokens'] for s in actual if s['role']=='prefill']
    need(decode,'actual model-owned decode timing demands absent')
    # Rounded integer requests bracket fractional model queries. Observed CUDA
    # context coordinates, not these requested prompts, define any future hull.
    pre_train={min(prompts),max(prompts),*(_quantile(prompts,q) for q in (.25,.5,.75,.95))}
    for n in prefill:pre_train.update((math.floor(n),math.ceil(n)))
    need(all(1<=n<=8191 for n in pre_train),'prefill query cannot form a legal complete request')
    pre_train=sorted(pre_train)
    # Keep a bounded design while retaining all queried extrema and the
    # model's corpus quantiles. Exact demands remain in the immutable ledger.
    pre_train=sorted({pre_train[0],pre_train[-1],*(_quantile(pre_train,q) for q in (.25,.5,.75))})
    pre_hold=_independent_interior(pre_train[0],pre_train[-1],set(pre_train),5)
    target_contexts={min(decode),max(decode),min(contexts),max(contexts),
                     *(_quantile(contexts,q) for q in (.25,.5,.75,.95))}
    ctx_low=min(target_contexts);ctx_high=max(target_contexts)
    target_contexts.update(n for n in (4096,6144,7168) if ctx_low<=n<=ctx_high)
    decode_train=sorted({max(1,math.floor(c)-1) for c in target_contexts}|
                        {max(1,math.ceil(max(target_contexts))-1)})
    unsupported_contexts=[n for n in decode_train if n+64>8192]
    # A design limitation remains explicit; it is not silently clamped into
    # the engine or presented as supported timing coverage.
    decode_train=[n for n in decode_train if n+64<=8192]
    need(len(decode_train)>=2,'model-owned decode interval lacks distinct training contexts')
    hold_contexts=_independent_interior(decode_train[0],decode_train[-1],set(decode_train),7)
    hold_decode=list(zip((1,3,6,12,20,28,32),hold_contexts))
    points=[]
    for f in frequencies:
        for purpose,shapes in [('training',[(b,c) for b in BATCHES for c in decode_train]),('holdout',hold_decode)]:
            points.extend(dict(frequency_mhz=f,role='decode',purpose=purpose,batch=b,prompt_tokens=c,output_tokens=64,
                repeats=3,seed=9701 if purpose=='training' else 9702) for b,c in shapes)
        for purpose,lengths in [('training',pre_train),('holdout',pre_hold)]:
            points.extend(dict(frequency_mhz=f,role='prefill',purpose=purpose,batch=1,prompt_tokens=n,output_tokens=1,
                repeats=3,seed=9701 if purpose=='training' else 9702) for n in lengths)
    if domain:
        for point in points:point['frequency_domain_sha256']=digest(domain)
    result=dict(schema='pdblend-native-timing-plan/v2',system='pdblend',model_id=model_id,tp=tp,pp=1,
        fleet_gpu_count=8,resident_instances=8//tp,query_ledger=ledger_ref,query_provenance=provenance_ref,
        corpus_bindings=corpora,corpus_refs=corpus_refs,points=points,actual_queries=list(demands.values()),
        unsupported_queries=unsupported,zero_increment_identities=zero,
        unlaunchable_decode_prompts=[dict(prompt_tokens=n,output_tokens=64,reason='requested_output_exceeds_context',
            sampled=False,clamped=False) for n in unsupported_contexts],
        blocked_ledger_entries=blocked,query_coverage_complete=False,capacity_policy=deepcopy(CAPACITY_POLICY),
        settle_s=2.,measure_s=5.,max_num_seqs=32,max_model_len=8192,max_num_batched_tokens=8192,
        holdout_limits=dict(mean_relative_error=.10,p95_relative_error=.20,max_relative_error=.25),
        selection_split='model_owned_calibration_tuning_and_independent_timing_holdout',
        evaluation_used_for_selection=False,min_m_floor_overridden=False,formal_eligible=False,
        full_profile_qualified=False,hardware_executed=False,collector_integration_required=True,
        scope='native_cuda_timing_component_only',
        coverage_policy='only audited measured CUDA shapes enter fitting; capacity exclusions remain bound raw receipts',
        required_remaining=['capacity_aware_collector_and_full_raw_replay','independent_timing_holdout',
            'power_and_runtime_component_qualification','complete_tuning_query_replay','formal_workload_energy'])
    if domain:result.update(frequency_domain_ref=frequency_domain_ref,**domain_fields(domain))
    return result


def validate_plan(plan):
    expected=build_plan(plan['query_ledger'],plan['query_provenance'],plan['corpus_refs'],model_id=plan['model_id'],frequency_domain_ref=plan.get('frequency_domain_ref'))
    need(plan==expected,'v2 timing plan differs from model-owned bound non-evaluation inputs')
    return plan
