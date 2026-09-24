"""Non-evaluation PDblend CUDA timing supplement, with explicit holdout shapes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

FREQUENCIES = (1500, 2520)
BATCHES = (1, 2, 4, 8, 16, 24, 32)
CONTEXTS = (64, 256, 2048, 4096, 7424)
DECODE_HOLDOUT = ((1,121),(3,482),(6,1281),(12,1868),(20,5745),(28,6781),(32,7270))
PREFILL_TRAIN = (16,64,128,512,2048,4096,7168,7936)
PREFILL_HOLDOUT = (33,38,47,352,1122,1688,2005,5788,6755)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def binding(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read_bound(ref):
    path=Path(ref['path'])
    if binding(path)['sha256']!=ref['sha256']:raise ValueError('input checksum differs: '+str(path))
    return json.loads(path.read_text())


def build_plan(ledger_ref, bindings_ref, *, model_id='Qwen2.5-7B-Instruct'):
    ledger=read_bound(ledger_ref);inputs=read_bound(bindings_ref)
    if (ledger.get('schema')!='pdblend-offline-query-ledger/v2' or ledger.get('evaluation_read') is not False
            or ledger.get('min_m_floor_overridden') is not False):
        raise ValueError('actual non-evaluation canonical-policy ledger required')
    owned=[r for r in ledger['ledgers'] if r['model_id']==model_id]
    if not owned or any(r.get('selection_split')!='tuning' or r['tp']!=1 or r['pp']!=1
                       or r['min_m_instances']!=4 or r['frequency_scope']!=list(FREQUENCIES) for r in owned):
        raise ValueError('supplement is restricted to bound TP1 two-frequency tuning queries')
    if not any(v.get('sha256')==ledger_ref['sha256'] for v in inputs['outputs'].values()):
        raise ValueError('ledger is not an output of bound input provenance')
    points=[]
    for f in FREQUENCIES:
        for purpose,shapes in [('training',[(b,c) for b in BATCHES for c in CONTEXTS]),
                               ('holdout',list(DECODE_HOLDOUT))]:
            for batch,context in shapes:
                points.append(dict(frequency_mhz=f,role='decode',purpose=purpose,batch=batch,
                    prompt_tokens=context,output_tokens=64,repeats=3,seed=9701 if purpose=='training' else 9702))
        for purpose,lengths in [('training',PREFILL_TRAIN),('holdout',PREFILL_HOLDOUT)]:
            for n in lengths:
                points.append(dict(frequency_mhz=f,role='prefill',purpose=purpose,batch=1,
                    prompt_tokens=n,output_tokens=1,repeats=3,seed=9701 if purpose=='training' else 9702))
    return dict(schema='pdblend-native-timing-plan-v1',system='pdblend',model_id=model_id,tp=1,pp=1,
        selection_split='calibration_and_independent_holdout',evaluation_used_for_selection=False,
        query_ledger=ledger_ref,query_bindings=bindings_ref,points=points,settle_s=2.,measure_s=5.,
        max_num_seqs=32,max_model_len=8192,max_num_batched_tokens=8192,
        holdout_limits=dict(mean_relative_error=.10,p95_relative_error=.20,max_relative_error=.25),
        scope='native_cuda_timing_component_only',formal_eligible=False,full_profile_qualified=False,
        power_scope='prefill_decode_burst_auxiliary_not_pure_decode_power',
        required_remaining=['power_domain_and_independent_holdout','runtime_capacity_static_wake_transfer_dvfs',
            'scoped_profile_composition','complete_tuning_query_coverage','pdblend_window_raw_acceptance'])


def validate_plan(plan):
    if (plan.get('schema')!='pdblend-native-timing-plan-v1' or plan.get('system')!='pdblend'
            or plan.get('tp')!=1 or plan.get('pp')!=1 or plan.get('evaluation_used_for_selection') is not False
            or plan.get('settle_s')!=2. or plan.get('measure_s')!=5. or plan.get('max_num_seqs')!=32):
        raise ValueError('PDblend TP1 timing protocol differs')
    expected=build_plan(plan['query_ledger'],plan['query_bindings'],model_id=plan['model_id'])
    if plan!=expected:raise ValueError('plan differs from non-evaluation domain and frozen holdout design')
    return plan
