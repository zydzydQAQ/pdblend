"""Strict v2 capacity exclusions and measured-only timing partitioning.

No hardware action occurs here. A future collector must obtain the recorded
fresh idle receipts before deciding that a predeclared point is unsupported.
"""
from __future__ import annotations

from copy import deepcopy

from .native_timing_audit import audit_window,fit_component,finite,need
from .native_timing_plan import digest
from .native_frequency_domain import check_rows,plan_frequencies
from .native_timing_plan_v2 import CAPACITY_POLICY,MODEL_TP

IDENTITY=('model_id','model_hash','tokenizer_hash','engine_revision','source_revision','image_digest','tp','pp','gpu_uuids')


def _point(plan,point):
    need(plan.get('schema')=='pdblend-native-timing-plan/v2' and plan.get('capacity_policy')==CAPACITY_POLICY,
         'explicit immutable v2 capacity policy required')
    need(type(plan.get('tp'))is int and plan.get('tp')==MODEL_TP.get(plan.get('model_id')) and plan.get('pp')==1,
         'model-owned v2 topology required')
    frequencies=plan_frequencies(plan)
    need(point.get('frequency_mhz') in frequencies and point.get('frequency_domain_sha256')==plan.get('frequency_domain_sha256'),
         'capacity point frequency domain differs')
    candidate=dict(point);repeat=candidate.pop('repeat',None)
    need(candidate in plan['points'] and type(repeat)is int and 0<=repeat<candidate['repeats'],
         'capacity receipt is not a predeclared point/repeat')
    need(all(type(point.get(k))is int for k in ('batch','prompt_tokens','output_tokens'))
         and 1<=point['batch']<=32 and 1<=point['prompt_tokens']
         and 1<=point['output_tokens'] and point['prompt_tokens']+point['output_tokens']<=8192,
         'planned request does not fit the native logical context domain')


def capacity_decision(plan,point,*,identity,capability,capability_received_s,drain,drain_received_s,observed_s):
    """Recompute strict reservation from fully free measured native capacity."""
    from pdblend.bench.comparison_acceptance import _state
    _point(plan,point)
    need(set(IDENTITY)<=set(identity) and identity['model_id']==plan['model_id']
         and identity['tp']==plan['tp'] and identity['pp']==1
         and all(isinstance(identity.get(k),str) and identity[k] and identity[k]!='unknown'
                 for k in ('model_hash','tokenizer_hash','engine_revision','source_revision','image_digest'))
         and len(identity['gpu_uuids'])==len(set(identity['gpu_uuids']))==plan['tp']
         and all(isinstance(u,str) and u.startswith('GPU-') for u in identity['gpu_uuids'])
         and all(capability.get(k)==identity[k] for k in IDENTITY) and capability.get('supported') is True,
         'capacity native model/source/physical identity differs')
    need(all(finite(t) for t in (drain_received_s,capability_received_s,observed_s))
         and drain_received_s<=capability_received_s<=observed_s
         and observed_s-drain_received_s<=1.,'fresh ordered idle capacity receipts required')
    need(drain.get('acknowledged') is True and drain.get('drained') is True,
         'capacity preflight drain acknowledgement missing')
    generation=_state(drain,identity['tp'],drain_received_s)
    state=capability.get('state',{})
    need(_state(state,identity['tp'],capability_received_s)==generation,
         'capacity generation changed between idle receipts')
    for key in ('total_kv_tokens','free_kv_tokens','block_size','max_num_seqs','max_model_len'):
        need(type(state.get(key))is int and state[key]==drain.get(key),'idle native capacity changed: '+key)
    need(state['max_num_seqs']==32 and state['max_model_len']==8192,
         'capacity native32/context8192 launch limits changed')
    block=state['block_size'];end=point['prompt_tokens']+point['output_tokens']
    need(state['total_kv_tokens']%block==0,'native token capacity does not match its observed block size')
    reserved=point['batch']*((end+block-1)//block)*block
    supported=point['batch']<=state['max_num_seqs'] and 10*reserved<9*state['total_kv_tokens']
    return dict(schema='pdblend-native-timing-capacity-decision/v2',supported=supported,
        reason='within_native_KV_capacity_guard' if supported else 'configured_native_KV_capacity_guard',
        requested_tokens=point['batch']*end,reserved_tokens=reserved,
        actual_total_kv_tokens=state['total_kv_tokens'],actual_free_kv_tokens=state['free_kv_tokens'],
        block_size=block,max_num_seqs=state['max_num_seqs'],generation=generation,
        lhs_10_reserved=10*reserved,rhs_9_actual_total=9*state['total_kv_tokens'],
        request_submitted=False,hardware_capacity_observed=True,timing_measured=False,formal_eligible=False)


def unsupported_capacity_record(plan,point,*,identity,capability,capability_received_s,drain,drain_received_s,observed_s):
    """Construct only an evidenced exclusion; never catch operational errors."""
    decision=capacity_decision(plan,point,identity=identity,capability=capability,
        capability_received_s=capability_received_s,drain=drain,drain_received_s=drain_received_s,observed_s=observed_s)
    need(decision['supported'] is False,'a supported point cannot be labeled unsupported')
    return deepcopy(dict(schema='pdblend-native-timing-unsupported-capacity/v2',system='pdblend',
        status='unsupported_capacity',point=point,plan_sha256=digest(plan),identity=identity,
        capability=capability,capability_received_s=capability_received_s,drain=drain,drain_received_s=drain_received_s,
        observed_s=observed_s,capacity_decision=decision,client_requests=[],sample=dict(ranks=[]),
        measurement_started=False,cleanup_errors=[],error=None,timing_measured=False,formal_eligible=False))


def audit_unsupported_capacity(raw,*,plan,identity):
    need(raw.get('schema')=='pdblend-native-timing-unsupported-capacity/v2' and raw.get('system')=='pdblend'
         and raw.get('status')=='unsupported_capacity' and raw.get('plan_sha256')==digest(plan)
         and raw.get('identity')==identity and raw.get('client_requests')==[]
         and raw.get('sample')==dict(ranks=[]) and raw.get('measurement_started') is False
         and raw.get('timing_measured') is False and raw.get('formal_eligible') is False
         and raw.get('error') is None and raw.get('cleanup_errors')==[]
         and raw.get('sampler_error') is None
         and not any(k in raw for k in ('start_s','end_s','measurement_start','measurement_stop','power_samples','frequency_samples')),
         'unsupported capacity cannot hide requests, measurements or failures')
    decision=capacity_decision(plan,raw['point'],identity=identity,capability=raw['capability'],
        capability_received_s=raw['capability_received_s'],drain=raw['drain'],drain_received_s=raw['drain_received_s'],
        observed_s=raw['observed_s'])
    need(decision['supported'] is False and raw.get('capacity_decision')==decision,
         'unsupported capacity inequality/receipt does not reproduce')
    return decision


def partition_windows(plan,windows,*,identities):
    """Require complete planned inventory; only measured CUDA rows enter fits.

    Entries are {instance_id, raw}; future collector/replay layers must bind
    their bytes and launch identities before using this semantic reducer.
    """
    from .native_timing_replay import _audited_window
    ids=tuple('pd-timing-'+str(i) for i in range(plan['resident_instances']))
    need(set(identities)==set(ids),'v2 complete resident engine identity inventory required')
    stable={k:identities[ids[0]].get(k) for k in IDENTITY if k!='gpu_uuids'}
    need(stable.get('model_id')==plan['model_id'] and stable.get('tp')==plan['tp'] and stable.get('pp')==1
         and all(all(identity.get(k)==v for k,v in stable.items()) and len(identity.get('gpu_uuids',[]))==plan['tp']
                 for identity in identities.values()),'v2 resident model/source/topology identities differ')
    gpus=[u for identity in identities.values() for u in identity['gpu_uuids']]
    need(len(gpus)==len(set(gpus))==8 and all(isinstance(u,str) and u.startswith('GPU-') for u in gpus),
         'v2 resident physical fleet is incomplete or overlaps')
    expected={(digest(point),r):ids[i%len(ids)] for i,point in enumerate(plan['points']) for r in range(point['repeats'])}
    seen=set();training=[];holdout=[];unsupported=[];measured=[]
    for entry in windows:
        raw=entry['raw'];point=dict(raw['point']);repeat=point.pop('repeat',None);key=(digest(point),repeat)
        need(key in expected and key not in seen and entry.get('instance_id')==expected[key],
             'v2 timing point/repeat/owner inventory differs')
        seen.add(key);identity=identities[expected[key]];_point(plan,raw['point'])
        if raw.get('status')=='unsupported_capacity':
            decision=audit_unsupported_capacity(raw,plan=plan,identity=identity)
            unsupported.append(dict(point_sha256=key[0],repeat=repeat,instance_id=expected[key],decision=decision))
            continue
        # Exceptions and failed/partial rows are fatal. They cannot be
        # converted into benign capacity exclusions by this reducer.
        rows=_audited_window(raw,identity)
        (training if point['purpose']=='training' else holdout).extend(rows)
        measured.append(dict(point_sha256=key[0],repeat=repeat,instance_id=expected[key],events=len(rows)))
    need(seen==set(expected),'v2 timing windows are incomplete; unmeasured planned points cannot disappear')
    return dict(schema='pdblend-native-timing-window-partition/v2',training=training,holdout=holdout,
        measured=measured,unsupported=unsupported,planned_windows=len(expected),formal_eligible=False,
        full_profile_qualified=False,unsupported_points_in_fit=False)


def fit_measured_partition(partition,*,identity,raw_bindings,measurement_qualification,limits):
    """Incomplete/rank-deficient domains stay unqualified without invented fits."""
    import numpy as np
    train,hold=partition['training'],partition['holdout'];missing=[]
    for frequency in check_rows([*train,*hold],identity):
        for role in ('prefill','decode'):
            rows=[r for r in train if (r['frequency_mhz'],r['role'])==(frequency,role)]
            tests=[r for r in hold if (r['frequency_mhz'],r['role'])==(frequency,role)]
            if not rows or not tests:
                missing.append(dict(frequency_mhz=frequency,role=role,reason='missing_training_or_independent_holdout'));continue
            features=([[1.,r['batch'],r['batch']*r['context_tokens']/8192.] for r in rows] if role=='decode'
                else [[1.,r['prompt_tokens']/8192.,(r['prompt_tokens']/8192.)**2] for r in rows])
            if np.linalg.matrix_rank(features)<3:
                missing.append(dict(frequency_mhz=frequency,role=role,reason='rank_deficient_measured_training'))
            elif role=='decode':
                coordinates=np.asarray([[r['batch'],r['context_tokens']/8192.] for r in rows])
                if np.linalg.matrix_rank(coordinates-coordinates[0])<2:
                    missing.append(dict(frequency_mhz=frequency,role=role,reason='degenerate_measured_decode_hull'))
    result=dict(schema='pdblend-native-timing-supported-fit/v2',component_qualified=False,component=None,
        missing_domains=missing,unsupported=deepcopy(partition['unsupported']),
        formal_eligible=False,full_profile_qualified=False,energy_comparable=False)
    if missing:return result
    component=fit_component(train,hold,identity=identity,raw_bindings=raw_bindings,
        measurement_qualification=measurement_qualification,limits=limits)
    result.update(component=component,component_qualified=component['component_qualified'])
    return result
