"""One measurement boundary shared by development, calibration and formal cells."""
import csv
import json
import math
from pathlib import Path

import numpy as np
from ecopadg.metrics import bench_time_bounds, clip_power_window, summarize_bench, classify_row_slo
from ecopadg.measure.power import trapezoid_energy, trapezoid_mean_power
from ecopadg.measure.backends import INSTANT_POWER_SOURCE_ID
from ecopadg.types import SloSpec


def power_evidence(power, source=None, metadata=None):
    """Verify exact field provenance for every raw eight-card power sample."""
    source = source or {}; metadata = metadata or []
    errors = []
    instant = (source.get('mode') == 'instant' and source.get('source_id') == INSTANT_POWER_SOURCE_ID
               and source.get('field_id') == 186 and source.get('scope_id') == 0)
    ages = []; spans = []; previous = [0]*8
    if not instant:
        errors.append('power source is not explicit NVML instant field 186 / GPU scope 0')
    if not power or len(metadata) != len(power):
        errors.append('power metadata does not cover every raw sample')
    if instant and len(metadata) == len(power):
        for (t, watts), row in zip(power, metadata):
            expected = dict(mode='instant', source_id=INSTANT_POWER_SOURCE_ID, field_id=186,
                            scope_id=0, value_type=1, return_code=0)
            if (len(watts) != 8 or row.get('t_s') != t or row.get('gpus') != list(range(8))
                    or any(row.get(k) != [v]*8 for k, v in expected.items())):
                errors.append('inconsistent field source, GPU identity or sample timestamp'); break
            vectors = [row.get(k, []) for k in ('nvml_timestamp_us', 'nvml_latency_us', 'read_started_s', 'read_finished_s')]
            if any(len(values) != 8 for values in vectors):
                errors.append('missing per-GPU NVML timestamp or read interval'); break
            for gpu, (stamp, latency, started, finished) in enumerate(zip(*vectors)):
                if (type(stamp) is not int or stamp <= 0 or stamp < previous[gpu]
                        or type(latency) is not int or latency < 0
                        or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in (started, finished))
                        or not started <= finished or not -.05 <= t-finished <= .25
                        or not -.05 <= finished-stamp/1e6 <= .25):
                    errors.append('invalid, stale or regressed NVML timestamp/read interval'); break
                previous[gpu] = stamp; ages.append(finished-stamp/1e6); spans.append(finished-started)
            if errors: break
    return dict(power_mode=source.get('mode', 'unspecified'), power_source_id=source.get('source_id'),
        power_field_id=source.get('field_id'), power_source_verified=instant and not errors,
        power_metadata_schema=1, power_metadata_samples=len(metadata), power_source_errors=errors,
        power_nvml_age_max_s=max(ages) if ages else None,
        power_read_duration_max_s=max(spans) if spans else None,
        power_timebase='host row-end epoch seconds; per-GPU NVML CPU timestamps retained in power_metadata.jsonl')


def summarize_cell(trace,rows,power,utilization,slo,*,sampling_error=None,reconfiguration_end_s=None,
                   power_source=None,power_metadata=None,require_power_mode=None,
                   evaluation_protocol=None,drain_result=None,slo_attainment_target=.9,
                   arrival_lateness_limit_s=None):
    n=len(trace['requests'])
    if not n:
        raise ValueError('empty workload')
    summary=summarize_bench(rows,SloSpec(*slo),n_expected=n)
    summary.pop('buckets')
    provenance=power_evidence(power,power_source,power_metadata)
    if require_power_mode not in (None,'instant'):
        raise ValueError('only explicit instant provenance may be required for serving cells')
    identified=(len(rows)==n and len({r['request_id'] for r in rows})==n
        and all(str(r['request_id'])==str(index) for index,r in enumerate(rows)))
    complete_work=[bool(r['success'] and r['token_count_source']=='server_usage' and r.get('token_ids_verified')
        and r['input_tokens']==q['prompt_len'] and r['generated_tokens']==q['output_len'])
        for r,q in zip(rows,trace['requests'])]
    rejected=[bool(not r['success'] and r.get('http_status')==429
        and r.get('admission_rejection') in ('admission_queue_full','admission_deadline')
        and r['generated_tokens']==0 and r['token_count_source']=='missing'
        and not r.get('token_ids_verified') and r.get('ttft_s') is None and not r.get('n_text_chunks',0)
        and r.get('prompt_len')==q['prompt_len'] and r.get('output_len')==q['output_len'])
        for r,q in zip(rows,trace['requests'])]
    valid=identified and not sampling_error and all(complete_work)
    start,client_end=bench_time_bounds(rows)
    if reconfiguration_end_s is not None and not math.isfinite(reconfiguration_end_s):
        raise ValueError('nonfinite reconfiguration completion boundary')
    v3=evaluation_protocol=='evaluation-v3'
    drain_result=dict(drain_result or {})
    end=max(client_end,reconfiguration_end_s or client_end)
    if v3:
        if not 0 < slo_attainment_target <= 1:
            raise ValueError('SLO attainment target must be in (0,1]')
        for key in ('drain_end_s','controls_end_s','observed_end_s'):
            boundary=drain_result.get(key)
            if isinstance(boundary,(int,float)) and math.isfinite(boundary):
                end=max(end,boundary)
    window_error=None
    try:
        if v3 and (len(power)<2 or len(utilization)<2):
            raise ValueError('insufficient samples for the entire measurement window')
        power=clip_power_window(power,start,end,pad_s=0)
        utilization=clip_power_window(utilization,start,end,pad_s=0)
    except ValueError as exc:
        if not v3: raise
        window_error=str(exc); power=[]; utilization=[]
    if any(len(ws)!=8 or any(not np.isfinite(w) or w<0 for w in ws) for _,ws in power) or any(len(us)!=8 or
            any(not np.isfinite(u) or not 0<=u<=100 for u in us) for _,us in utilization):
        if not v3:
            raise ValueError('requires actual power and utilization on all eight GPUs')
        window_error='requires actual power and utilization on all eight GPUs'; power=[]; utilization=[]
    token_itl=[x for r in rows if r['token_itl_exact'] for x in json.loads(r['token_itl_s'])]
    validity=('invalid_power_source' if require_power_mode=='instant' and not provenance['power_source_verified']
              else 'ok' if valid else 'invalid_work')
    summary.update(validity=validity,measurement_schema=2,**provenance,
        capacity_observation_valid=bool(identified and not sampling_error and provenance['power_source_verified']
            and all(ok or refusal for ok,refusal in zip(complete_work,rejected))),
        admission_rejections=sum(rejected),
        generated_tokens=sum(r['generated_tokens'] for r in rows),
        expected_generated_tokens=sum(q['output_len'] for q in trace['requests']),
        energy_j=trapezoid_energy(power),gpu_count=8,
        gpu_util=trapezoid_mean_power(utilization)/800,
        gpu_util_per_gpu=[trapezoid_mean_power([(t,[us[g]]) for t,us in utilization])/100 for g in range(8)],
        token_itl_count=len(token_itl),token_itl_exact_requests=sum(bool(r['token_itl_exact']) for r in rows),
        token_itl_p50_s=float(np.quantile(token_itl,.5)) if token_itl else None,
        token_itl_p95_s=float(np.quantile(token_itl,.95)) if token_itl else None,
        token_itl_p99_s=float(np.quantile(token_itl,.99)) if token_itl else None,
        measurement_start_s=start,measurement_end_s=end,client_completion_end_s=client_end,
        reconfiguration_tail_s=end-client_end,sampling_error=sampling_error)
    if v3:
        def finite_number(value):
            return isinstance(value,(int,float)) and math.isfinite(value)
        timing_errors=[]
        for row in rows:
            planned=row.get('planned_arrival_s'); dispatch=row.get('actual_dispatch_s')
            first=row.get('first_token_s'); last=row.get('last_token_s'); stream=row.get('stream_end_s')
            finish=row.get('finish_s')
            if (row.get('evaluation_protocol')!='evaluation-v3' or not finite_number(planned)
                    or row.get('arrival_s')!=planned or not finite_number(finish) or finish<planned
                    or (dispatch is not None and (not finite_number(dispatch) or dispatch<planned))):
                timing_errors.append('invalid planned arrival/dispatch/finish provenance'); continue
            if row.get('success'):
                if (not all(finite_number(v) for v in (dispatch,first,last,stream))
                        or not planned<=dispatch<=first<=last<=stream<=finish
                        or not math.isclose(first-planned,row.get('ttft_s',-1),abs_tol=1e-5)
                        or not math.isclose(last-planned,row.get('latency_s',-1),abs_tol=1e-5)):
                    timing_errors.append('successful row lacks ordered exact token/stream timing')
        drain_complete=(drain_result.get('drain_complete') is True
            and all(finite_number(drain_result.get(k)) for k in ('drain_end_s','controls_end_s'))
            and not drain_result.get('error'))
        dispatch_delays=[r['actual_dispatch_s']-r['planned_arrival_s'] for r in rows
            if finite_number(r.get('actual_dispatch_s')) and finite_number(r.get('planned_arrival_s'))]
        arrival_fidelity_valid=bool(len(dispatch_delays)==n and all(delay>=0 for delay in dispatch_delays))
        independent=all(r.get('open_loop_independent') is True for r in rows)
        if arrival_lateness_limit_s is not None and (not finite_number(arrival_lateness_limit_s)
                or arrival_lateness_limit_s<0):
            raise ValueError('declared arrival lateness diagnostic limit must be nonnegative')
        measurement_valid=bool(identified and not sampling_error and provenance['power_source_verified']
            and not window_error and not timing_errors and drain_complete)
        work_complete=bool(identified and all(complete_work))
        good=[bool(ok and classify_row_slo(row,SloSpec(*slo))) for row,ok in zip(rows,complete_work)]
        good_count=sum(good); completed_count=sum(complete_work)
        energy=summary['energy_j'] if not window_error else None
        summary.update(evaluation_protocol='evaluation-v3',evaluation_schema=3,measurement_schema=3,
            measurement_valid=measurement_valid,work_complete=work_complete,
            slo_feasible=bool(measurement_valid and good_count/n>=slo_attainment_target),
            slo_attainment_target=slo_attainment_target,slo_attainment=good_count/n,
            good_requests=good_count,completed_work_requests=completed_count,
            failed_requests=n-completed_count,offered_requests=n,
            request_timeouts=sum(bool(r.get('request_timeout')) for r in rows),
            arrival_fidelity_valid=arrival_fidelity_valid,open_loop_verified=bool(independent and arrival_fidelity_valid),
            dispatch_lateness_limit_s=arrival_lateness_limit_s,
            dispatch_lateness_within_declared_limit=(all(delay<=arrival_lateness_limit_s for delay in dispatch_delays)
                if arrival_lateness_limit_s is not None and arrival_fidelity_valid else None),
            dispatch_delay_max_s=max(dispatch_delays) if dispatch_delays else None,
            dispatch_delay_p99_s=float(np.quantile(dispatch_delays,.99)) if dispatch_delays else None,
            energy_j=energy,
            energy_per_offered_request_j=energy/n if measurement_valid else None,
            energy_per_completed_request_j=energy/completed_count if measurement_valid and completed_count else None,
            energy_per_good_request_j=energy/good_count if measurement_valid and good_count else None,
            good_requests_per_j=good_count/energy if measurement_valid and energy and energy>0 else None,
            goodput_measurement_rps=good_count/(end-start) if end>start else None,
            measurement_duration_s=end-start,measurement_window_error=window_error,
            timestamp_errors=timing_errors,incomplete_drain=not drain_complete,
            drain_complete=drain_complete,drain_end_s=drain_result.get('drain_end_s'),
            controls_end_s=drain_result.get('controls_end_s'),drain_result=drain_result,
            drain_tail_s=max(0,end-client_end),
            capacity_observation_valid=bool(measurement_valid and
                all(ok or refusal for ok,refusal in zip(complete_work,rejected))),
            validity=('invalid_measurement' if not measurement_valid else
                      'ok' if work_complete else 'invalid_work'))
        if window_error:
            summary.update(gpu_util=None,gpu_util_per_gpu=[None]*8)
    return finite_json(summary)


def finite_json(value):
    if isinstance(value,float) and not math.isfinite(value):
        return None
    if isinstance(value,dict):
        return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [finite_json(v) for v in value]
    return value


def save_raw(directory,rows,power,utilization,*,power_source=None,power_metadata=None):
    directory=Path(directory)
    with (directory/'bench.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]) if rows else ['request_id','success','error'])
        writer.writeheader();writer.writerows(rows)
    util=dict(utilization)
    with (directory/'power.csv').open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['t_s']+[f'gpu{i}_w' for i in range(8)]+[f'gpu{i}_util_pct' for i in range(8)])
        # Keep power evidence if a later utilization read failed. A missing
        # counter remains empty; summarize_cell still rejects sampling errors.
        writer.writerows([t]+list(ws)+list(util.get(t,[None]*8)) for t,ws in power)
    if power_source is not None:
        (directory/'power_source.json').write_text(json.dumps(power_source,indent=2,allow_nan=False))
    if power_metadata is not None:
        with (directory/'power_metadata.jsonl').open('w') as handle:
            for row in power_metadata:
                handle.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
