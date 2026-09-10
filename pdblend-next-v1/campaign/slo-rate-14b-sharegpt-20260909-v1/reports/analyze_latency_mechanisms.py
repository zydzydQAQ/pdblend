"""Read-only mechanism evidence from frozen benchmark, controller, configuration and clocks."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/baseline-mechanism-analysis-001'
POINTS = [('A', .75), ('B', 1.25), ('C', 1.5), ('C', 1.75), ('C', 2.)]
SYSTEMS = ['pdblend', 'mixed', 'distserve', 'dynamollm', 'ecoserve']
CATEGORIES = ['good', 'completed_ttft_only', 'completed_tpot_only', 'completed_both', 'timeout', 'admission_rejection', 'other_failure']
csv.field_size_limit(16 * 1024**2)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked(reference):
    assert sha(reference['path']) == reference['sha256'], reference['path']
    return read(reference['path'])


def truth(value):
    return str(value).lower() in ('true', '1')


def number(value):
    if value in (None, ''):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def quantile(values, fraction):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    position = fraction * (len(values) - 1)
    left = int(position)
    return values[left] + (position - left) * (values[min(left + 1, len(values) - 1)] - values[left])


def delta(timing, first, last):
    a, b = number(timing.get(first)), number(timing.get(last))
    return b - a if a is not None and b is not None else None


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def main():
    assert not any((OUT / name).exists() for name in ('latency-mechanisms.csv', 'request-mechanisms.csv',
        'instance-mechanisms.csv', 'sampled-clock-dwell.csv', 'evidence.json')), 'fresh evidence version required'
    nodes = {host: read(ROOT / host / 'observations.json') for host in ('A', 'C')}
    sources = [ref(ROOT / host / 'observations.json') for host in ('A', 'C')]
    b_ref = ref(ROOT / 'B/reference.json')
    b = checked(b_ref)
    old = checked(b['source'])
    selected_b = set(b.get('cell_ids', []))
    nodes['B'] = [row for row in old['observations'] if
        (row.get('measurement_host'), row.get('model'), row.get('dataset'), row.get('slo_scale')) ==
        ('B', '14b', 'sharegpt', 1.) and (not selected_b or row['cell_id'] in selected_b)]
    sources += [b_ref, b['source']]
    results, requests_out, instances_out, cells, clocks_out = [], [], [], [], []
    for host, rate in POINTS:
        for system in SYSTEMS:
            row = next(row for row in nodes[host] if row['rate_rps'] == rate and row['system'] == system and row['repeat'] == 1)
            assert row['measurement_valid'] and row['strict_slo_recomputed']
            cp = checked(row['checkpoint'])
            directory = Path(row['raw_requests']['path']).parent
            config_ref = dict(path=str(directory / 'runtime_config.json'), sha256=cp['artifacts'][str(directory / 'runtime_config.json')])
            config = checked(config_ref)
            journal_ref = dict(path=str(directory / 'control.jsonl'), sha256=cp['artifacts'][str(directory / 'control.jsonl')])
            assert sha(journal_ref['path']) == journal_ref['sha256']
            events = [json.loads(line) for line in Path(journal_ref['path']).read_text().splitlines()]
            admissions = {str(e['client_request_id']): e for e in events if e['kind'] == 'admission'}
            timings = {str(e['client_request_id']): e for e in events if e['kind'] == 'request_timing'}
            assert sha(row['raw_requests']['path']) == row['raw_requests']['sha256']
            with Path(row['raw_requests']['path']).open() as stream:
                requests = list(csv.DictReader(stream))
            trace, summary = checked(row['trace_reference']), checked(row['summary'])
            assert len(requests) == row['n_expected'] == trace['n_requests']
            ttft_s, tpot_s = config['slo_ttft_s'], config['slo_tpot_s']
            assert summary['fixed_window']['effective_slo_s'] == dict(ttft=ttft_s, tpot=tpot_s)
            counts = collections.Counter({name: 0 for name in CATEGORIES})
            local = []
            for request in requests:
                rid = request['request_id']
                expected = trace['requests'][int(rid)]
                assert int(request['prompt_len']) == expected['prompt_len'] and int(request['output_len']) == expected['output_len']
                completed = truth(request['success']) and not request['error'] and not truth(request['request_timeout'])
                if completed:
                    assert request['token_count_source'] == 'server_usage' and truth(request['token_ids_verified'])
                    assert int(request['generated_tokens']) == expected['output_len']
                    ft, pt = float(request['ttft_s']), float(request['tpot_s'])
                    category = ('completed_both' if ft > ttft_s and pt > tpot_s else
                                'completed_ttft_only' if ft > ttft_s else
                                'completed_tpot_only' if pt > tpot_s else 'good')
                else:
                    category = ('timeout' if truth(request['request_timeout']) else
                                'admission_rejection' if request['admission_rejection'] else 'other_failure')
                counts[category] += 1
                timing, admission = timings.get(rid, {}), admissions.get(rid, {})
                routes = admission.get('plan', {}).get('routes', [])
                route = routes[0] if len(routes) == 1 else {}
                decode = route.get('decode_id')
                value = dict(cell_id=row['cell_id'], host=host, rate_rps=rate, system=system, request_id=rid,
                    category=category, completed=completed, input_tokens=int(request['prompt_len']),
                    requested_output_tokens=int(request['output_len']), generated_tokens=int(request['generated_tokens']),
                    ttft_s=number(request['ttft_s']), tpot_s=number(request['tpot_s']),
                    dispatch_delay_s=number(request['dispatch_delay_s']),
                    pre_forward_wait_s=delta(timing, 'planned_arrival_s', 'forward_started_s'),
                    before_first_planning_s=delta(timing, 'planned_arrival_s', 'first_planning_s'),
                    planning_and_admission_wait_s=delta(timing, 'first_planning_s', 'action_wait_started_s'),
                    action_lock_wait_s=delta(timing, 'action_wait_started_s', 'action_acquired_s'),
                    reservation_and_setup_s=delta(timing, 'action_acquired_s', 'forward_started_s'),
                    backend_first_token_s=delta(timing, 'forward_started_s', 'first_token_s'),
                    prefill_id=route.get('prefill_id'), decode_id=decode,
                    configured_dynamo_shape_pool=config.get('dynamo_assignments', {}).get(decode),
                    admission_reason=admission.get('plan', {}).get('reason'),
                    planned_frequencies=admission.get('plan', {}).get('frequencies'),
                    forward_started_s=timing.get('forward_started_s'), stream_end_s=timing.get('stream_end_s'),
                    cleanup_end_s=timing.get('cleanup_end_s'))
                local.append(value)
            assert counts['good'] == row['good_requests'] and sum(counts.values()) == row['n_expected']
            assert sum(value['completed'] for value in local) == row['completed_work_requests']
            bad_ttft = [value for value in local if value['category'] in ('completed_ttft_only', 'completed_both')]
            record = dict(cell_id=row['cell_id'], host=host, rate_rps=rate, system=system,
                reference_only=host == 'B', n_requests=len(local), **counts,
                good_fraction=counts['good'] / len(local), actual_ttft_threshold_s=ttft_s, actual_tpot_threshold_s=tpot_s,
                completed_ttft_p50_s=quantile([v['ttft_s'] for v in local if v['completed']], .5),
                completed_ttft_p95_s=quantile([v['ttft_s'] for v in local if v['completed']], .95),
                completed_tpot_p95_s=quantile([v['tpot_s'] for v in local if v['completed']], .95),
                ttft_miss_pre_forward_mean_s=mean([v['pre_forward_wait_s'] for v in bad_ttft]),
                ttft_miss_planning_and_admission_mean_s=mean([v['planning_and_admission_wait_s'] for v in bad_ttft]),
                ttft_miss_backend_first_token_mean_s=mean([v['backend_first_token_s'] for v in bad_ttft]),
                ttft_miss_action_lock_mean_s=mean([v['action_lock_wait_s'] for v in bad_ttft]),
                actual_dispatch_delay_max_s=max(v['dispatch_delay_s'] for v in local),
                configured_instance_count=len(config['instances']),
                routed_decode_instance_count=len({v['decode_id'] for v in local if v['decode_id']}),
                energy_j=row['energy_j'], measurement_duration_s=row['measurement_duration_s'],
                average_all8_power_w=row['energy_j'] / row['measurement_duration_s'],
                end_of_bench_after_arrival_window_s=summary['fixed_window']['hold_started_s'] - summary['fixed_window']['arrival_window_end_s'],
                energy_per_good_request_j=row['energy_per_good_request_j'])
            instance_evidence = []
            for instance in config['instances']:
                values = [value for value in local if value['decode_id'] == instance['id']]
                misses = [value for value in values if value['category'] in ('completed_ttft_only', 'completed_both')]
                instance_value = dict(cell_id=row['cell_id'], host=host, rate_rps=rate, system=system,
                    instance_id=instance['id'], gpus=instance['gpus'], tp=instance['tp'],
                    configured_dynamo_shape_pool=config.get('dynamo_assignments', {}).get(instance['id']),
                    decode_request_count=len(values), completed_ttft_miss_count=len(misses),
                    input_tokens=sum(v['input_tokens'] for v in values),
                    generated_tokens=sum(v['generated_tokens'] for v in values),
                    ttft_miss_pre_forward_mean_s=mean([v['pre_forward_wait_s'] for v in misses]),
                    ttft_miss_backend_first_token_mean_s=mean([v['backend_first_token_s'] for v in misses]))
                instance_evidence.append(instance_value)
                instances_out.append(instance_value)
            clock_paths = [path for path in cp['artifacts'] if path.endswith('/clocks.csv')]
            assert len(clock_paths) == 1
            clock_ref = dict(path=clock_paths[0], sha256=cp['artifacts'][clock_paths[0]])
            assert sha(clock_ref['path']) == clock_ref['sha256']
            with Path(clock_ref['path']).open() as stream:
                samples = list(csv.DictReader(stream))
            start = summary['fixed_window']['arrival_epoch_s']
            end = max(float(value['cleanup_end_s']) for value in local if value['cleanup_end_s'] is not None)
            assert float(samples[0]['t_s']) <= start and float(samples[-1]['t_s']) >= end
            clock_dwell = []
            for gpu in range(8):
                durations = collections.Counter()
                for left, right in zip(samples, samples[1:]):
                    interval = max(0., min(end, float(right['t_s'])) - max(start, float(left['t_s'])))
                    if interval:
                        durations[float(left['gpu' + str(gpu) + '_sm_mhz'])] += interval
                assert math.isclose(sum(durations.values()), end - start, abs_tol=1e-5)
                for mhz, seconds in sorted(durations.items()):
                    value = dict(cell_id=row['cell_id'], host=host, rate_rps=rate, system=system,
                                 gpu=gpu, observed_sm_mhz=mhz, sampled_dwell_s=seconds,
                                 observed_request_window_s=end-start, dwell_fraction=seconds/(end-start))
                    clock_dwell.append(value)
                    clocks_out.append(value)
            preserved_config = {key: value for key, value in config.items() if key in
                ('strategy', 'slo_ttft_s', 'slo_tpot_s', 'slo_scale', 'output_prior', 'max_pending', 'manage_clocks',
                 'park_idle', 'max_service_frequency_mhz', 'allow_pd', 'dvfs', 'dynamic_pools', 'slow_topology',
                 'stale_single_request_retry', 'protect_pending_decode', 'residency_horizon', 'candidate_slo_margin',
                 'dynamo_assignments', 'dynamo_input_cuts', 'dynamo_output_cuts', 'controller_source_release')}
            preserved_config['instances'] = [{k: i.get(k) for k in ('id', 'tp', 'gpus', 'role', 'service_budget_tokens')}
                                              for i in config['instances']]
            sources_cell = dict(checkpoint=row['checkpoint'], raw_requests=row['raw_requests'], summary=row['summary'],
                                trace=row['trace_reference'], runtime_config=config_ref, controller_journal=journal_ref,
                                sampled_clocks=clock_ref, raw_power=row['raw_power'])
            sources.extend(sources_cell.values())
            if row.get('audit_reference'):
                checked(row['audit_reference'])
                sources_cell['original_audited'] = row['audit_reference']
                sources.append(row['audit_reference'])
            cells.append(dict(metrics=record, sources=sources_cell, runtime_config=preserved_config,
                event_counts=dict(collections.Counter(event['kind'] for event in events)),
                epoch_operations=dict(collections.Counter(event.get('operation') for event in events if event['kind'] == 'dynamo_control_epoch')),
                admission_plan_reason_counts=dict(collections.Counter(v['admission_reason'] for v in local)),
                admission_planned_frequency_counts=dict(collections.Counter(str(f['frequency_mhz']) for event in events
                    if event['kind'] == 'admission' for f in event.get('plan', {}).get('frequencies', []))),
                instance_routing=instance_evidence, sampled_clock_dwell=clock_dwell))
            results.append(record)
            requests_out.extend(local)
    unique_sources = list({(r['path'], r['sha256']): r for r in sources}.values())
    assert all(sha(r['path']) == r['sha256'] for r in unique_sources), 'source changed during analysis'
    OUT.mkdir(exist_ok=True)
    table(OUT / 'latency-mechanisms.csv', results)
    table(OUT / 'request-mechanisms.csv', requests_out)
    table(OUT / 'instance-mechanisms.csv', instances_out)
    table(OUT / 'sampled-clock-dwell.csv', clocks_out)
    evidence = dict(schema='slo-rate-baseline-latency-mechanisms-v1', created_s=time.time(), passed=True,
        analysis_code=ref(__file__), representative_points=[dict(host=h, rate_rps=r) for h, r in POINTS],
        analyzed_cells=len(results), analyzed_requests=len(requests_out), sources=unique_sources, cells=cells,
        source_hashes_verified_before_and_after=True, CPU_only=True, GPU_operations=False,
        limitations=['Representative within-host comparisons, not a causal ablation or independent-seed estimate',
            'pre_forward_wait includes controller queue, planning and admission; it is not isolated GPU compute time',
            'Controller frequency plan counts are decisions, not clock transitions or dwell; sampled SM clock CSV is separate',
            'Clock dwell is a left-sample approximation clipped from arrival epoch to last controller cleanup; not setup or whole-operation energy',
            'Energy values are previously audited all-eight-GPU primary windows, not attributed per-request energy',
            'PDB and baselines have different frozen runtime families and resident placements; no controlled attribution to a single mechanism',
            'B is a fixed historical reference; comparisons are paired within host, not across hosts'])
    save(OUT / 'evidence.json', evidence)
    print(json.dumps(dict(passed=True, cells=len(results), requests=len(requests_out), sources=len(unique_sources),
                         evidence=ref(OUT / 'evidence.json')), indent=2))


if __name__ == '__main__':
    main()
