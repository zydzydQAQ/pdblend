"""Actual-query-bound native power pilot; no fit, holdout, or formal promotion."""
from __future__ import annotations
import math
from .native_timing_plan import binding, read_bound, digest
from .native_timing_plan_v2 import MODEL_TP

FREQUENCIES = (1500, 2520)
PILOT = (('decode',1,16), ('decode',4,256), ('decode',1,7168), ('decode',32,7168),
         ('prefill',1,16), ('prefill',1,7936))


def build_plan(ledger_ref, provenance_ref, *, model_id='Qwen2.5-7B-Instruct'):
    ledger, provenance = read_bound(ledger_ref), read_bound(provenance_ref)
    if (ledger.get('schema') != 'pdblend-offline-query-ledger/v2' or ledger.get('evaluation_read') is not False
            or ledger.get('min_m_floor_overridden') is not False or provenance.get('evaluation_read') is not False
            or not any(ref.get('sha256') == ledger_ref['sha256'] for ref in provenance['outputs'].values())):
        raise ValueError('bound actual non-evaluation canonical-policy query ledger required')
    tp=MODEL_TP.get(model_id)
    selected = [row for row in ledger['ledgers'] if row['model_id'] == model_id]
    if tp is None or not selected or any(row['selection_split'] != 'tuning' or row['tp'] != tp or row['pp'] != 1
            or row['engine_max_num_seqs'] != 32 or row['min_m_instances'] != 4
            or row['frequency_scope'] != list(FREQUENCIES) for row in selected):
        raise ValueError('power pilot requires the model-owned TP1/TP2 native32 tuning domain')
    queries = {}; zero_marginal_queries=[]
    for row in selected:
        for query in row['queries']:
            if not query['finite_arguments']:continue
            method, args, kwargs = query['method'], query['args'], query['kwargs']
            if method in ('decode_supported','decode_power_supported','step_seconds','token_energy_j'):
                batch, context, frequency = args
                shape = dict(role='decode', batch=batch, context_tokens=context, frequency_mhz=frequency)
            elif method == 'decode_power_w':
                batch, frequency = args
                shape = dict(role='decode',batch=batch,context_tokens=kwargs['ctx'],frequency_mhz=frequency)
            elif method in ('prefill_seconds','prefill_power_w','prefill_energy_j','prefill_marginal_seconds'):
                length, frequency = args
                if method=='prefill_marginal_seconds' and length==0:
                    zero_marginal_queries.append(dict(dataset=row['dataset'],rate_scale=row['rate_scale'],
                        method=method,args=args,scope='zero_increment_identity_not_a_gpu_sampling_point'))
                    continue
                shape = dict(role='prefill',input_tokens=length,frequency_mhz=frequency)
            else:continue
            if any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0
                   for k,v in shape.items() if k != 'role'):
                raise ValueError('finite ledger shape differs from numeric query arguments')
            key = digest(shape)
            item = queries.setdefault(key, dict(shape=shape, methods=[], datasets=[], rate_scales=[]))
            for field, value in [('methods',method),('datasets',row['dataset']),('rate_scales',row['rate_scale'])]:
                if value not in item[field]:item[field].append(value)
    points = [dict(role=role,batch=batch,prompt_tokens=length,frequency_mhz=f,repeats=3,
        purpose='coverage_feasibility_pilot',seed=9701,
        output_tokens=8192-length if role=='decode' else 1)
        for f in FREQUENCIES for role,batch,length in PILOT]
    return dict(schema='pdblend-native-power-pilot-plan/v1',system='pdblend',model_id=model_id,tp=tp,pp=1,
        query_ledger=ledger_ref,query_provenance=provenance_ref,actual_queries=list(queries.values()),
        zero_marginal_queries=zero_marginal_queries,points=points,
        settle_s=2.,measure_s=5.,max_num_seqs=32,execution='serial_on_existing_exclusive_eight_gpu_fleet',
        evaluation_used_for_selection=False,parameters_fitted=False,formal_eligible=False,
        power_component_qualified=False,full_profile_qualified=False,
        requested_context_is_not_measured_power_domain=True,
        remaining_gates=['calibration_grid_from_actual_pilot_contexts','freeze_candidate_before_independent_holdout',
            'pure_decode_context_and_batch_interpolation_holdout','prefill_request_cycle_vs_cuda_energy_composition',
            'native_worker_source_compatibility','strict_runtime_timing_power_composition',
            'complete_tuning_query_coverage','formal_workload_energy'])


def validate_plan(plan):
    if plan != build_plan(plan['query_ledger'],plan['query_provenance'],model_id=plan['model_id']):
        raise ValueError('power pilot differs from immutable actual-query design')
    return plan


def short_context_limit(target, prompt=16, settle_s=2.,measure_s=5.):
    """Necessary constant-step speed bound, not proof about an unmeasured engine.

    Even omitting prefill/barrier lead, effective context is prompt+(2+2.5)/step.
    At target121/prompt16, a step faster than about42.9ms cannot reach that mean.
    Actual variable-speed windows must use their raw native context records.
    """
    if target <= prompt:raise ValueError('target must exceed real prompt length')
    return dict(target_context=target,prompt_tokens=prompt,
        necessary_constant_step_s=(settle_s+measure_s/2)/(target-prompt),
        assumption='constant step, no extra prefill/barrier lead',native_impossibility_proved=False)
