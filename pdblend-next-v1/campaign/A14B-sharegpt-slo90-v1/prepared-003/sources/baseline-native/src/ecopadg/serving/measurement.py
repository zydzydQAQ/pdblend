"""One measurement boundary shared by development, calibration and formal cells."""
import csv
import json
import math
from pathlib import Path

import numpy as np
from ecopadg.metrics import bench_time_bounds, clip_power_window, summarize_bench
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
                   power_source=None,power_metadata=None,require_power_mode=None):
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
    end=max(client_end,reconfiguration_end_s or client_end)
    power=clip_power_window(power,start,end,pad_s=0)
    utilization=clip_power_window(utilization,start,end,pad_s=0)
    if any(len(ws)!=8 for _,ws in power) or any(len(us)!=8 or
            any(not np.isfinite(u) or not 0<=u<=100 for u in us) for _,us in utilization):
        raise ValueError('requires actual power and utilization on all eight GPUs')
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
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
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
