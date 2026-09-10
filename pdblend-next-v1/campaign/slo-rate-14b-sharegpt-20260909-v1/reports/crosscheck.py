"""Independent local CSV arithmetic checks of newly published terminal cells.

No SSH, GPU operations, qualification replay, or service-failure classification.
Partial work remains in the denominator. A producer's service audit is separate.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

csv.field_size_limit(16 * 1024**2)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def checked(reference):
    path = Path(reference['path'])
    need(sha(path) == reference['sha256'], 'changed immutable file: ' + str(path))
    return path


def truth(value):
    return str(value).lower() in ('1', 'true')


def close(actual, expected, field):
    if expected is None:
        need(actual is None or isinstance(actual, float) and math.isnan(actual), 'undefined metric changed: ' + field)
    else:
        need(isinstance(actual, (int, float)) and math.isfinite(actual) and
             math.isclose(actual, expected, abs_tol=1e-8, rel_tol=1e-8), 'raw metric differs: ' + field)


def integrate(power, start, end):
    times = [float(row['t_s']) for row in power]
    need(len(times) >= 2 and all(math.isfinite(t) for t in times) and
         all(b > a for a, b in zip(times, times[1:])), 'invalid power timestamps')
    need(times[0] <= start < end <= times[-1], 'power does not bracket primary window')
    energy, utilization = [], []
    for gpu in range(8):
        totals = []
        for key, limit in ((f'gpu{gpu}_w', None), (f'gpu{gpu}_util_pct', 100)):
            values = [float(row[key]) for row in power]
            need(all(math.isfinite(v) and v >= 0 and (limit is None or v <= limit) for v in values),
                 'invalid power/utilization field: ' + key)
            total = 0.
            for i, (t0, t1) in enumerate(zip(times, times[1:])):
                lo, hi = max(start, t0), min(end, t1)
                if lo >= hi:
                    continue
                v0, v1 = values[i], values[i + 1]
                left = v0 + (v1 - v0) * (lo - t0) / (t1 - t0)
                right = v0 + (v1 - v0) * (hi - t0) / (t1 - t0)
                total += (left + right) * .5 * (hi - lo)
            totals.append(total)
        energy.append(totals[0])
        utilization.append(totals[1] / (end - start) / 100)
    return energy, utilization


def check_row(row):
    paths = {key: checked(row[key]) for key in ('raw_requests', 'raw_power', 'summary',
        'receipt', 'checkpoint', 'binding', 'audit_reference', 'trace_reference')}
    cp, summary = read(paths['checkpoint']), read(paths['summary'])
    config_path = paths['raw_requests'].parent / 'runtime_config.json'
    config_ref = dict(path=str(config_path), sha256=cp['artifacts'][str(config_path)])
    config = read(checked(config_ref))
    trace = read(paths['trace_reference'])
    with paths['raw_requests'].open() as stream:
        requests = list(csv.DictReader(stream))
    with paths['raw_power'].open() as stream:
        power = list(csv.DictReader(stream))
    scale = {'A': .5, 'C': 2.}[row['measurement_host']]
    thresholds = (5 * scale, .15 * scale)
    need((config['slo_ttft_s'], config['slo_tpot_s'], config['slo_scale'], config['arrival_window_s']) ==
         (*thresholds, scale, 100.), 'actual runtime SLO/window differs')
    fixed = summary['fixed_window']
    need(fixed['effective_slo_s'] == dict(ttft=thresholds[0], tpot=thresholds[1]) and
         fixed['arrival_window_s'] == 100 and fixed['slo_scale'] == scale and
         summary['trace_sha256'] == row['trace_sha256'] and summary['comparison_system'] == row['system'],
         'summary actual SLO/window/trace/system differs')
    n = len(trace['requests'])
    need(n > 0 and len(requests) == n == row['n_expected'], 'request denominator differs')
    actual = {item['request_id']: item for item in requests}
    need(len(actual) == n and set(actual) == {str(i) for i in range(n)}, 'request identity differs')
    epoch = float(actual['0']['planned_arrival_s'])
    good = completed = generated = 0
    offset_errors = []
    for index, prescribed in enumerate(trace['requests']):
        request = actual[str(index)]
        need(int(request['prompt_len']) == prescribed['prompt_len'] and
             int(request['output_len']) == prescribed['output_len'], 'requested work changed')
        offset_errors.append(abs(float(request['planned_arrival_s']) - epoch - prescribed['arrival_s']))
        tokens = int(request['generated_tokens'])
        need(tokens >= 0, 'negative generated tokens')
        generated += tokens
        done = (truth(request['success']) and not request['error'] and
            request['token_count_source'] == 'server_usage' and truth(request['token_ids_verified']) and
            int(request['input_tokens']) == prescribed['prompt_len'] and tokens == prescribed['output_len'] and
            not truth(request.get('request_timeout')))
        completed += done
        good += bool(done and float(request['ttft_s']) < thresholds[0] and float(request['tpot_s']) < thresholds[1])
    need(max(offset_errors) <= 1e-5, 'frozen arrival offsets differ')
    start, end = summary['measurement_start_s'], summary['measurement_end_s']
    need(start == epoch and end >= epoch + 100, 'primary measurement window is shorter than arrival window')
    energy, utilization = integrate(power, start, end)
    derived = dict(n_expected=n, completed_work_requests=completed, good_requests=good,
        slo_attainment=good / n, completion_fraction=completed / n,
        measurement_duration_s=end - start, energy_j=sum(energy), gpu_util=statistics.fmean(utilization),
        goodput_measurement_rps=good / (end - start),
        energy_per_good_request_j=sum(energy) / good if good else None,
        failed_requests=n - completed, request_timeouts=sum(truth(r.get('request_timeout')) for r in requests))
    for field in ('ttft_s', 'tpot_s'):
        values = [float(r[field]) for r in requests if r[field] not in ('', None)]
        need(all(math.isfinite(v) and v >= 0 for v in values), 'invalid raw latency')
        derived[field.replace('_s', '_avg_s')] = statistics.fmean(values) if values else None
    if row.get('generated_token_count_complete') is True:
        derived['generated_tokens'] = generated
    for field, expected in derived.items():
        close(row.get(field), expected, field)
    need(row['work_complete'] == (completed == n), 'work completion differs')
    need(row['energy_measured_gpu_count'] == 8, 'energy scope is not eight GPUs')
    for index in range(8):
        close(row['energy_per_gpu_j'][index], energy[index], 'energy_per_gpu_j')
        close(row['gpu_util_per_gpu'][index], utilization[index], 'gpu_util_per_gpu')
    return dict(metrics=derived, actual_slo_ttft_s=thresholds[0], actual_slo_tpot_s=thresholds[1],
        actual_arrival_window_s=100., frozen_offset_max_error_s=max(offset_errors), energy_gpu_count=8,
        runtime_config=config_ref, energy_per_gpu_j=energy, gpu_util_per_gpu=utilization,
        token_count_exact=row.get('generated_token_count_complete') is True,
        full_qualification_context_reverified=False, service_failure_classification_replayed=False)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def build(root, rows=None):
    root = Path(root)
    rows = rows if rows is not None else read(root / 'reports/current/results.json')['observations']
    entries, pairs, issues = [], {}, []
    for row in rows:
        rate = format(float(row.get('rate_rps_decimal', row.get('rate_rps'))), '.12g')
        if row.get('measurement_valid') is True:
            pairs.setdefault(rate, []).append(row)
        if row.get('reference_only'):
            continue
        entry = dict(cell_id=row['cell_id'], engineering_attempt=row.get('engineering_attempt', 1),
            measurement_host=row['measurement_host'], system=row['system'], rate_rps=row['rate_rps'],
            repeat=row.get('repeat', 1), audit_reference=row.get('audit_reference'), checked_s=time.time())
        if row.get('measurement_valid') is not True:
            entry.update(status='engineering_invalid_preserved', error=None)
        elif row.get('metric_source_integrity') != 'verified':
            entry.update(status='awaiting_minimal_mirror', error=None)
        else:
            try:
                entry.update(check_row(row), status='passed', error=None)
            except FileNotFoundError as exc:
                entry.update(status='awaiting_minimal_mirror', error=str(exc))
            except (KeyError, ValueError, TypeError, IndexError, OSError) as exc:
                entry.update(status='failed', error=str(exc))
                issues.append(dict(cell_id=row['cell_id'], error=str(exc)))
        entries.append(entry)
    pairing = []
    for rate, values in sorted(pairs.items(), key=lambda item: float(item[0])):
        hashes = sorted({row['trace_sha256'] for row in values if row.get('trace_sha256')})
        complete = all(row.get('trace_sha256') for row in values)
        matched = complete and len(hashes) == 1
        pairing.append(dict(rate_rps=float(rate), cell_ids=[row['cell_id'] for row in values],
            physical_hosts=sorted({row['measurement_host'] for row in values}),
            systems=sorted({row['system'] for row in values}), trace_hashes=hashes,
            observed_trace_pairing_matches=matched, expected_grid_complete=False))
        if not matched:
            issues.append(dict(rate_rps=float(rate), error='cross-host/system observed trace SHA mismatch or missing'))
    result = dict(schema='slo-rate-local-cpu-crosscheck-ledger-v1', updated_s=time.time(),
        checker=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        scope='New A/C terminal raw arithmetic; observed trace equality includes immutable B reference',
        limitations='No qualification replay or service-failure classification; no GPU or remote call; pairing does not establish cross-host causal comparability',
        entries=entries, trace_pairing=pairing, issues=issues,
        passed_count=sum(e['status'] == 'passed' for e in entries),
        failed_count=sum(e['status'] == 'failed' for e in entries),
        pending_count=sum(e['status'] == 'awaiting_minimal_mirror' for e in entries),
        complete=False)
    save(root / 'reports/crosscheck-ledger.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = build(args.root)
    print(json.dumps({key: result[key] for key in ('passed_count', 'failed_count', 'pending_count', 'issues')}))
