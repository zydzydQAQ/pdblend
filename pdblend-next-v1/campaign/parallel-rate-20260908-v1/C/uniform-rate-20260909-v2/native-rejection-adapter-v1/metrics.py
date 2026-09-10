"""Read-only audit of original request/power evidence for the uniform-rate campaign."""
import bisect
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

ROOT = Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def checked(reference):
    need(sha(reference['path']) == reference['sha256'], 'changed evidence: ' + reference['path'])
    return read(reference['path'])


def truth(value):
    return str(value).lower() in ('1', 'true')


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8)


def integrate(rows, start, end, columns):
    times = [float(row['t_s']) for row in rows]
    need(len(times) > 1 and times[0] <= start < end <= times[-1], 'physical samples do not cover measurement')
    need(all(b > a for a, b in zip(times, times[1:])), 'physical timestamps are not strictly increasing')
    values = [[float(row[key]) for key in columns] for row in rows]
    need(all(math.isfinite(v) and v >= 0 for row in values for v in row), 'invalid physical sample')

    def at(t):
        i = min(max(bisect.bisect_right(times, t) - 1, 0), len(times) - 2)
        fraction = (t - times[i]) / (times[i + 1] - times[i])
        return [a + fraction * (b - a) for a, b in zip(values[i], values[i + 1])]

    samples = [(start, at(start))] + [(t, v) for t, v in zip(times, values) if start < t < end] + [(end, at(end))]
    return [sum((tb - ta) * (a[i] + b[i]) / 2 for (ta, a), (tb, b) in zip(samples, samples[1:]))
            for i in range(len(columns))]


def received_output_count(request):
    """Reconstruct indexed stream prefixes independently of producer flags."""
    if str(request.get('token_evidence_schema')) != '2' or not truth(request.get('received_token_count_exact')):
        return None
    need(truth(request.get('token_stream_opened')) and truth(request.get('token_sequence_verified'))
         and not truth(request.get('token_stream_parse_error')), 'invalid token prefix provenance')
    ids = json.loads(request['received_token_ids'])
    events = json.loads(request['received_token_events'])
    need(isinstance(ids, list) and isinstance(events, list)
         and all(type(v) is int for v in ids), 'invalid received token IDs')
    rebuilt = []
    for event in events:
        values = event.get('token_ids')
        need(isinstance(values, list) and values and all(type(v) is int for v in values), 'invalid token event')
        need(type(event.get('token_index')) is int and event['token_index'] == len(rebuilt) + len(values),
             'missing, duplicate or unordered token event')
        rebuilt.extend(values)
    need(rebuilt == ids and int(request['received_token_count']) == len(ids), 'token count does not match event evidence')
    need(len(ids) <= int(request['output_len']), 'received more tokens than prescribed')
    expected_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest() if ids else ''
    need(request['output_token_sha256'] == expected_hash, 'received token hash differs')
    if not ids:
        need(int(request.get('n_text_chunks') or 0) == 0, 'zero prefix has text output')
    if request.get('token_count_source') == 'server_usage':
        need(int(request['generated_tokens']) == len(ids), 'terminal usage and stream prefix differ')
    return len(ids)


def checkpoint_host(checkpoint):
    binding = checkpoint.get('binding')
    if binding:
        reference = binding if isinstance(binding, dict) else dict(path=binding, sha256=checkpoint['binding_sha256'])
        bound = checked(reference)
        host = {'iZwz9gfq11hx1sbob59yrgZ': 'C', 'iZwz9i5bte3xkpmcoes3t2Z': 'B',
                'iZwz9274emxme9019d2sjgZ': 'Anew20260909'}.get(bound.get('hostname'))
    else:
        host = None
    claimed = checkpoint['row'].get('measurement_host', checkpoint['row'].get('node'))
    need(not host or not claimed or host == claimed, 'physical host and row differ')
    # Early checkpoints keep hardware identity in the separately pinned reuse
    # audit. Their numeric audit must not invent a host from a model or path.
    return host or claimed


def diagnosed_zero_output(checkpoint_reference):
    if not checkpoint_reference:
        return set(), None
    cp = checked(checkpoint_reference)
    row = cp['row']
    if (checkpoint_host(cp), row['model'], row['system']) != ('C', '7b', 'ecoserve'):
        return set(), None
    receipt = cp['receipt']
    receipt = checked(receipt) if isinstance(receipt, dict) else read(receipt)
    if receipt['summary'].get('work_complete') is True:
        return set(), None
    receipt_path = cp['receipt']['path'] if isinstance(cp['receipt'], dict) else cp['receipt']
    cell = Path(receipt_path).parents[2] / 'cells' / row['cell_id']
    with (cell / 'bench.csv').open() as stream:
        requests = list(csv.DictReader(stream))
    if not any(r.get('http_status') == '503' for r in requests):
        return set(), None
    path = Path(__file__).parent / 'native_queue.py'
    need(sha(path) == '451c5da42165936f9e949895e2c4ea024e19417597ef7b30a27ec388db7ce055', 'native capacity auditor changed')
    spec = importlib.util.spec_from_file_location('_C_native_capacity_partition', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    proof = module.audit(checkpoint_reference)
    need(proof['passed'] and proof['independently_recomputed']
        and proof['checkpoint'] == checkpoint_reference and proof['no_unknown_errors'],
        'native refusal diagnosis failed')
    return set(proof['native_rejection_request_ids']), proof



def request_metrics(rows, trace, row, *, zero_output_ids=()):
    expected = {str(request.get('idx', index)): request for index, request in enumerate(trace['requests'])}
    need(len(expected) == len(rows) == row['n_requests'] > 0, 'request denominator differs')
    actual = {str(r['idx']): r for r in rows}
    need(len(actual) == len(rows) and set(actual) == set(expected), 'request identities differ')
    complete = good = generated = actual_output = 0
    latencies = {'ttft_s': [], 'tpot_s': []}
    exact_tokens = True
    for key, prescribed in expected.items():
        request = actual[key]
        need(str(request['request_id']) == key, 'request ID differs from trace index')
        need(int(request['prompt_len']) == prescribed['prompt_len'] and int(request['output_len']) == prescribed['output_len'], 'prescribed lengths changed')
        emitted = int(request['generated_tokens'])
        need(emitted >= 0, 'negative output tokens')
        generated += emitted
        terminal_usage = emitted > 0 and request.get('token_count_source') == 'server_usage' and truth(request.get('token_ids_verified'))
        rejected_without_output = (not truth(request.get('success')) and emitted == 0
            and int(request.get('n_text_chunks') or 0) == 0
            and request.get('first_token_s') in (None, '') and request.get('last_token_s') in (None, '')
            and request.get('admission_rejection') not in (None, '', '0', 'False', 'false'))
        received = received_output_count(request)
        diagnosed_zero = key in zero_output_ids
        if diagnosed_zero:
            need(not truth(request.get('success')) and emitted == 0
                 and int(request.get('n_text_chunks') or 0) == 0, 'refused request has output')
        exact = terminal_usage or received is not None or rejected_without_output or diagnosed_zero
        exact_tokens = exact_tokens and exact
        actual_output += emitted if terminal_usage else received if received is not None else 0
        work = (truth(request.get('success')) and not request.get('error') and terminal_usage
            and not truth(request.get('request_timeout')) and int(request['input_tokens']) == prescribed['prompt_len']
            and emitted == prescribed['output_len'])
        complete += int(work)
        for name in latencies:
            if request.get(name) not in (None, ''):
                value = float(request[name])
                need(math.isfinite(value) and value >= 0, 'invalid recorded latency')
                latencies[name].append(value)
        good += int(bool(work and request.get('ttft_s') not in (None, '') and request.get('tpot_s') not in (None, '')
            and float(request['ttft_s']) < row['slo_ttft_s'] and float(request['tpot_s']) < row['slo_tpot_s']))
    n = len(expected)
    return dict(n_expected=n, completed_work_requests=complete, good_requests=good, generated_tokens=generated,
        expected_generated_tokens=sum(x['output_len'] for x in expected.values()),
        work_complete=complete == n, completion_fraction=complete / n, slo_attainment=good / n,
        failed_requests=n - complete, request_timeouts=sum(truth(x.get('request_timeout')) for x in rows),
        ttft_avg_s=statistics.fmean(latencies['ttft_s']) if latencies['ttft_s'] else None,
        tpot_avg_s=statistics.fmean(latencies['tpot_s']) if latencies['tpot_s'] else None,
        generated_token_count_complete=exact_tokens,
        actual_output_tokens=actual_output if exact_tokens else None,
        observed_output_tokens_lower_bound=actual_output,
        token_count_semantics='terminal usage or independently verified received token prefix; rejected output needs proof')


def audit_receipt(receipt_path, row, *, receipt_reference=None, checkpoint_reference=None):
    receipt_path = Path(receipt_path)
    receipt = checked(receipt_reference) if receipt_reference else read(receipt_path)
    directory = receipt_path.parents[2] / 'cells' / row['cell_id']
    summary = read(directory / 'summary.json')
    need(receipt.get('summary') == summary, 'receipt and producer summary differ')
    need(receipt.get('measurement_valid') is True and summary.get('measurement_valid') is True
        and summary.get('fixed_window_valid') is True and summary.get('gpu_count') == 8
        and summary.get('power_source_verified') is True, 'measurement is not valid')
    need(receipt.get('clock_restore_complete') is True and receipt.get('child_stopped') is True
        and not receipt.get('outer_cleanup_errors') and receipt.get('restoration')
        and all(v.get('complete') is True for v in receipt['restoration'].values()), 'measurement cleanup is incomplete')
    need(summary.get('trace_sha256') == row['trace_sha256'] == sha(row['trace']), 'trace digest differs')
    trace = read(row['trace'])
    with (directory / 'bench.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    zero_ids, zero_proof = diagnosed_zero_output(checkpoint_reference)
    result = request_metrics(rows, trace, row, zero_output_ids=zero_ids)
    for key in ('n_expected', 'completed_work_requests', 'generated_tokens', 'expected_generated_tokens', 'work_complete'):
        need(summary.get(key) == result[key], 'producer/raw mismatch: ' + key)
    for key in ('ttft_avg_s', 'tpot_avg_s'):
        raw = summary.get(key)
        need((result[key] is None and (raw is None or isinstance(raw, float) and math.isnan(raw)))
            or (result[key] is not None and raw is not None and close(raw, result[key])), 'latency average differs: ' + key)
    start, end = summary['measurement_start_s'], summary['measurement_end_s']
    duration = end - start
    need(duration >= 100 and close(duration, summary['measurement_duration_s']), 'measurement denominator differs')
    with (directory / 'power.csv').open() as stream:
        powers = list(csv.DictReader(stream))
    energy = integrate(powers, start, end, [f'gpu{i}_w' for i in range(8)])
    util = [v / duration / 100 for v in integrate(powers, start, end, [f'gpu{i}_util_pct' for i in range(8)])]
    need(close(sum(energy), summary['energy_j']), 'eight GPU energy differs')
    need(len(summary['gpu_util_per_gpu']) == 8 and all(close(a, b) for a, b in zip(util, summary['gpu_util_per_gpu']))
        and close(statistics.fmean(util), summary['gpu_util']), 'eight GPU utilization differs')
    arrival_path = ROOT / 'audit_cooperative_arrivals_v1.py'
    spec = importlib.util.spec_from_file_location('_uniform_arrival_audit', arrival_path)
    arrival = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arrival)
    arrival_proof = arrival.recompute(rows, row['n_requests'])
    result.update(measurement_valid=True, arrival_qualification=dict(passed=True, **arrival_proof),
        measurement_duration_s=duration, energy_j=sum(energy), energy_per_gpu_j=energy,
        gpu_util=statistics.fmean(util), gpu_util_per_gpu=util,
        request_throughput_rps=result['completed_work_requests'] / duration,
        token_throughput_tps=result['actual_output_tokens'] / duration if result['actual_output_tokens'] is not None else None,
        token_throughput_is_exact=result['generated_token_count_complete'],
        goodput_measurement_rps=result['good_requests'] / duration,
        energy_per_good_request_j=sum(energy) / result['good_requests'] if result['good_requests'] else None,
        producer_slo_attainment=summary.get('slo_attainment'),
        strict_slo_recomputed=True, latency_denominator='all recorded per-request values',
        primary_window_only=True, receipt=ref(receipt_path), summary=ref(directory / 'summary.json'),
        raw_requests=ref(directory / 'bench.csv'), raw_power=ref(directory / 'power.csv'))
    if zero_proof:
        result['zero_output_diagnosis'] = zero_proof
    return result


def audit_checkpoint(checkpoint_path):
    checkpoint = read(checkpoint_path)
    for path, digest in checkpoint.get('artifacts', {}).items():
        need(sha(path) == digest, 'checkpoint artifact changed: ' + path)
    receipt = checkpoint['receipt']
    if isinstance(receipt, str):
        receipt = {'path': receipt, 'sha256': checkpoint['receipt_sha256']}
    row = checkpoint['row']
    result = audit_receipt(receipt['path'], row, receipt_reference=receipt, checkpoint_reference=ref(checkpoint_path))
    result.update({key: row[key] for key in ('cell_id', 'model', 'dataset', 'rate_rps', 'system', 'repeat',
        'seed', 'trace_sha256', 'content_pairing_sha256', 'slo_ttft_s', 'slo_tpot_s')})
    result['measurement_host'] = checkpoint_host(checkpoint)
    result['checkpoint'] = ref(checkpoint_path)
    return result
