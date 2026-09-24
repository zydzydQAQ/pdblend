"""Replay native role/shape and physical board power; never grant profile coverage."""
from __future__ import annotations
import math
from pathlib import Path
from pdblend.bench.comparison_metering import summarize_comparison
from pdblend.online.native_control import validate_state
from .native_timing_audit import need, finite, measured_events


class PowerFrequencyQualificationError(ValueError):
    """Complete physical observations failed only their requested-clock limit.

    This is an invalid power-model sample, not a successful audit. The caller
    may continue unrelated sampling only after separate full-fleet restoration.
    Missing/malformed observations and every other audit failure stay ordinary
    exceptions and can never enter this typed continuation path.
    """
    def __init__(self, audited, frequency_evidence):
        message = 'observed target physical board frequency differs from requested clock'
        super().__init__(message)
        self.audit = dict(audited, passed=False, error=message,
            error_kind='expected_measurement_qualification_gap',
            qualification_gap='observed_frequency_mismatch',
            non_frequency_checks_passed=True, frequency_data_complete=True,
            frequency_evidence=frequency_evidence)


def _frequency_evidence(power, spec, point, start, end):
    """Separate acquisition/coverage failure from observed value mismatch."""
    samples = power['frequency_samples']
    need(samples and all(isinstance(row, (list, tuple)) and len(row) == 2
         and finite(row[0]) and isinstance(row[1], (list, tuple)) and len(row[1]) == 8
         and all(finite(value) and value > 0 for value in row[1]) for row in samples),
         'physical frequency samples are missing, malformed or nonfinite')
    need(all(b[0] > a[0] for a, b in zip(samples, samples[1:])),
         'physical frequency timestamps are not strictly increasing')
    need(finite(point['frequency_mhz']) and point['frequency_mhz'] > 0,
         'requested physical frequency is invalid')
    columns = [power['gpus'].index(g) for g in spec['gpus']]
    clocks = [r for r in samples if start <= r[0] <= end]
    need(clocks and clocks[0][0] <= start+1 and clocks[-1][0] >= end-1
         and all(0 < b[0]-a[0] <= 1 for a, b in zip(clocks, clocks[1:])),
         'target physical board frequency observation coverage is incomplete')
    mismatches = [dict(at_s=at, gpu=power['gpus'][i], observed_mhz=values[i],
                       requested_mhz=point['frequency_mhz'])
                  for at, values in clocks for i in columns if abs(values[i]-point['frequency_mhz']) > 30]
    return dict(requested_mhz=point['frequency_mhz'], tolerance_mhz=30,
        max_allowed_gap_s=1., observation_count=len(clocks), observed_gpu_ids=list(spec['gpus']),
        first_s=clocks[0][0], last_s=clocks[-1][0],
        max_gap_s=max((b[0]-a[0] for a,b in zip(clocks,clocks[1:])), default=0.),
        mismatch_count=len(mismatches), mismatches=mismatches, data_complete=True)


def context_window(events, start, end):
    """Time-weight actual native decode context; nominal prompt is not a node."""
    rows = sorted(events, key=lambda r:r['at_s'])
    before = [r for r in rows if r['at_s'] <= start]
    need(before and end > start, 'context lacks service-start bracketing observation')
    rows = [before[-1]]+[r for r in rows if start < r['at_s'] < end]
    need(end-rows[-1]['at_s'] <= 1. and all(0 < b['at_s']-a['at_s'] <= 1.
         for a,b in zip(rows,rows[1:])), 'native shape observation gap exceeds one second')
    integral = 0.
    for i,row in enumerate(rows):
        right = rows[i+1]['at_s'] if i+1<len(rows) else end
        integral += (right-max(start,row['at_s']))*row['context_tokens']
    return dict(context_min=min(r['context_tokens'] for r in rows),
        context_max=max(r['context_tokens'] for r in rows),effective_context_tokens=integral/(end-start),
        context_semantics='time_weighted_actual_native_scheduled_context',native_steps=len(rows))


def _continuation_protocol(raw, expected_capability):
    """Recheck receipts before treating an invalid sample as recoverable.

    The native clock ACK has no epoch field. Epoch proof therefore comes from
    the full-rank scheduler states bracketing the request and its final drain.
    The live collector supplies its already source/image-validated capability.
    """
    spec, cap, point = raw['spec'], raw['capability'], raw['point']
    keys = ('model_id', 'model_hash', 'tokenizer_hash', 'engine_revision',
            'image_digest', 'source_revision', 'tp', 'pp', 'gpu_uuids')
    need(all(cap.get(k) for k in keys), 'continuation native implementation identity missing')
    need(Path(spec['model']).name == cap['model_id'], 'continuation launch model differs')
    if expected_capability is not None:
        need(all(cap[k] == expected_capability.get(k) for k in keys),
             'continuation capability differs from initial validated resident identity')
    before = raw['before']
    validate_state(before, generation=spec['generation'], tp=spec['tp'], pp=spec['pp'], drained=True)
    need(before.get('acknowledged') is True and before.get('drained') is True,
         'continuation initial native drain ACK missing')
    need(before['native_at_s'] <= raw['settle_started_s'], 'continuation initial drain is out of order')
    validate_state(raw['drain'], generation=spec['generation'], tp=spec['tp'], pp=spec['pp'],
                   drained=True, observed_after_s=raw['end_s'])
    armed = raw['measurement_start']['ranks']
    need(len(armed) == spec['tp'] and {r.get('rank') for r in armed} == set(range(spec['tp']))
         and all(r.get('acknowledged') is True and r.get('system') == 'pdblend'
                 and r.get('scope') == 'runner' and r.get('tp') == spec['tp']
                 and r.get('pp') == spec['pp'] for r in armed),
         'continuation measurement arm lacks exact all-rank protocol ACK')
    clock = raw['clock']; ack = clock['ack']
    need(ack.get('acknowledged') is True and ack.get('success') is True
         and ack.get('requested_frequency_mhz') == point['frequency_mhz'],
         'continuation requested clock ACK differs')
    need(sorted(r.get('gpu_uuid', '') for r in ack.get('gpus', [])) == sorted(cap['gpu_uuids']),
         'continuation clock ACK physical UUID inventory differs')
    observations = clock['observations']
    need(observations and all(finite(r.get('at_s')) and before['native_at_s'] <= r['at_s']
         <= raw['settle_started_s'] and len(r['frequencies_mhz']) == spec['tp']
         and all(finite(f) and f > 0 for f in r['frequencies_mhz']) for r in observations),
         'continuation initial clock observation missing or out of order')
    need(all(abs(f-point['frequency_mhz']) <= 15 for f in observations[-1]['frequencies_mhz']),
         'continuation requested clock was never initially observed')


def audit_power_window(raw, *, expected_capability=None):
    point, spec, lease = raw['point'],raw['spec'],raw['lease']
    start,end = raw['start_s'],raw['end_s']
    need(raw.get('status') == 'measured' and not raw.get('cleanup_errors')
         and not raw.get('error') and not raw.get('error_kind'), 'raw power window did not complete')
    need(raw.get('system')=='pdblend' and raw.get('schema')=='pdblend-native-power-window/v1', 'wrong native power scope')
    tp=spec['tp']
    from .native_timing_plan_v2 import MODEL_TP
    need(tp==MODEL_TP.get(raw['capability'].get('model_id')) and spec['pp']==1 and spec['max_num_seqs']==32,
         'pilot requires model-owned TP1/TP2 native32 launch')
    need(finite(start) and finite(end) and end-start>=5. and start-raw['settle_started_s']>=2.,
         'continuous settle/service interval incomplete')
    cap=raw['capability'];expected_uuid=[lease['gpu_uuids'][lease['gpu_ids'].index(g)] for g in spec['gpus']]
    need(cap.get('supported') is True and cap.get('tp')==tp and cap.get('pp')==1
         and cap.get('gpu_uuids')==expected_uuid,'native capability differs from measured physical devices')
    need(raw['measurement_start'].get('acknowledged') is True,'idle measurement arm lacks ACK')
    stopped=raw['measurement_stop'].get('ranks',[])
    need(len(stopped)==tp and {r.get('rank') for r in stopped}==set(range(tp))
         and all(r.get('acknowledged') is True for r in stopped),
         'native measurement stop rank ACK missing')
    validate_state(raw['drain'],generation=spec['generation'],tp=tp,pp=1,drained=True)
    need(raw['drain'].get('drained') is True and raw['drain'].get('acknowledged') is True,'final native drain missing')
    allrows=[r for rank in raw['sample'].get('ranks',[]) for r in rank.get('samples',[])
             if raw['settle_started_s'] <= r.get('at_s',-1) < end]
    need(allrows and all(r.get('role')==point['role'] for r in allrows), 'prefill/mixed work contaminates requested power role')
    events=[r for r in measured_events(raw['sample'],tp=tp) if raw['settle_started_s']<=r['at_s']<end]
    ids={r['request_id'] for r in raw['client_requests']}
    need(events and all(set(r['request_ids'])<=ids and r['batch']==point['batch']
         and r['prompt_tokens']==point['prompt_tokens'] for r in events),'actual native batch/prompt/request ownership differs')
    # measured_events aligns the full physical rank sequences before assigning
    # the shared logical event its coordinator (rank-0) wall clock. Rank-local
    # launch timestamps can straddle a window boundary by a few microseconds;
    # independently counting those timestamps would reject a complete TP step.
    # Compare the unfiltered coordinator inventory; rank shape disagreement is
    # still rejected by measured_events, never silently dropped.
    coordinator=next(r for r in raw['sample']['ranks'] if r['rank']==0)
    coordinator_rows=[r for r in coordinator['samples'] if raw['settle_started_s']<=r.get('at_s',-1)<end]
    need(len(events)==len(coordinator_rows),'unqualified native shapes cannot be silently filtered')
    for row in raw['client_requests']:
        need(not row.get('error') or row.get('cancel_expected') is True,'unexpected client request failure')
    if point['role']=='decode':
        need(len(ids)==point['batch'] and all(r['submitted_s']<=raw['settle_started_s']
             and r['observed_running_through_s']>=end for r in raw['client_requests']),
             'decode cohort changed or terminated before the power window ended')
        need(all(len(r['request_ids'])==len(ids) and set(r['request_ids'])==ids for r in events),
             'decode window is not full batch')
        live=raw['service_end_state']
        validate_state(live,generation=spec['generation'],tp=tp,pp=1,observed_after_s=end)
        need(set(live['all_queue'])==ids and set(live['running'])==ids,
             'native decode cohort ended before the service boundary')
        contexts=context_window(events,start,end)
        need(contexts['native_steps']>=8,'insufficient actual native decode steps')
        cancellation=raw.get('cancel_receipts',{})
        need(set(cancellation)==ids and all(row.get('acknowledged') is True and row.get('cancelled') is True
             and row.get('request_id')==rid and row.get('generation')==spec['generation']
             for rid,row in cancellation.items()),
             'incomplete measured-request cancellation receipts')
        for rid,row in cancellation.items():
            validate_state(row['native_state'],generation=spec['generation'],tp=tp,pp=1,request_id=rid)
        scope='continuous_pure_decode_window_at_actual_growing_context'
    else:
        need(all(r.get('terminal') is True and r.get('completion_tokens')==1 and not r.get('error')
                 for r in raw['client_requests']), 'prefill request stream incomplete')
        prefill_ids=[r['request_ids'][0] for r in events]
        need(len(prefill_ids)==len(set(prefill_ids)) and set(prefill_ids)==ids,
             'prefill-only cycle lacks one complete native forward per real request')
        need(any(start<=r['at_s']<end for r in events),'service interval lacks prefill CUDA events')
        contexts=dict(input_tokens=point['prompt_tokens'],native_steps=sum(start<=r['at_s']<end for r in events))
        scope='prefill_only_request_cycle_mean_includes_admission_and_idle_gaps'
    power=raw['power']
    need(power.get('error') is None, 'physical sampler reported an error')
    summary=summarize_comparison(power,gpu_uuids=lease['gpu_uuids'],origin_s=start,duration_s=end-start,
        tail_end_s=end,gpu_uuid_binding_verified=lease['gpu_uuid_binding_verified'])
    need(summary['energy_comparable'] is True,'physical eight-board power integration incomplete')
    boards=summary['service']['power']['per_gpu']
    audited=dict(schema='pdblend-native-power-window-audit/v1',passed=True,**contexts,
        role=point['role'],batch=point['batch'],frequency_mhz=point['frequency_mhz'],power_scope=scope,
        power_w=sum(boards[u]['mean_power_w'] for u in expected_uuid),
        measured_board_energy_j=sum(boards[u]['energy_j'] if 'energy_j' in boards[u] else boards[u]['integral'] for u in expected_uuid),
        public_eight_board_metering=summary,formal_eligible=False,power_component_qualified=False,
        full_profile_qualified=False,coverage_from_nominal_prompt=False,
        native_group_timestamp_basis='rank0_wall_clock_after_full_rank_sequence_alignment',
        active_prefill_kernel_power_qualified=False)
    # This runs last: even an observed low clock cannot hide a later non-clock
    # role/request/rank/power/identity failure behind a continuation exception.
    frequency = _frequency_evidence(power, spec, point, start, end)
    if frequency['mismatch_count']:
        _continuation_protocol(raw, expected_capability)
        raise PowerFrequencyQualificationError(audited, frequency)
    return audited
