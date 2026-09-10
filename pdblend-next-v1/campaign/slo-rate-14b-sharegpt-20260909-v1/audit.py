"""Raw evidence audit, with explicit service failures distinct from engineering faults."""
import csv
import json
import math
from pathlib import Path

import slo_support as p


def truth(value):
    return str(value).lower() in ('true', '1')


def close(a, b, tolerance=1e-5):
    return math.isclose(float(a), float(b), abs_tol=tolerance, rel_tol=0)


def classify_requests(requests, trace, row, events):
    """Only exact declared deadlines and independently mapped queue refusals qualify.

    A missing/foreign HTTP response, incomplete journal, early deadline, native
    fault or unknown rejection fails closed. This function does not certify
    the power sampler, runtime configuration, native health, or cleanup; audit
    must verify those independently before setting service_terminal_valid.
    """
    expected = {str(i): r for i, r in enumerate(trace['requests'])}
    actual = {str(r['request_id']): r for r in requests}
    p.need(len(actual) == len(requests) == len(expected) == row['n_requests'], 'request denominator differs')
    p.need(set(actual) == set(expected), 'request identities differ')
    p.need(not any(e.get('kind') in ('cancel_unconfirmed', 'runtime_error') for e in events),
           'native cancellation/runtime fault in journal')
    timing_rows = [e for e in events if e.get('kind') == 'request_timing']
    ending_rows = [e for e in events if e.get('kind') == 'request_end']
    timings = {str(e['client_request_id']): e for e in timing_rows}
    endings = {e['request_id']: e for e in ending_rows}
    p.need(len(timings) == len(timing_rows) and len(endings) == len(ending_rows), 'duplicate controller terminal records')
    p.need(set(timings) <= set(actual), 'foreign controller request')
    p.need({e['request_id'] for e in timing_rows} == set(endings), 'controller timing/end mismatch')
    failures = []
    zero_output = []
    for rid, request in actual.items():
        prescribed = expected[rid]
        p.need(int(request['prompt_len']) == prescribed['prompt_len'] and
               int(request['output_len']) == prescribed['output_len'], 'changed requested work')
        completed = (truth(request.get('success')) and not request.get('error') and
            request.get('token_count_source') == 'server_usage' and truth(request.get('token_ids_verified')) and
            int(request['input_tokens']) == prescribed['prompt_len'] and
            int(request['generated_tokens']) == prescribed['output_len'] and not truth(request.get('request_timeout')))
        timing = timings.get(rid)
        if timing:
            ending = endings[timing['request_id']]
            for a, b in [('planned_arrival_s', 'planned_arrival_s'), ('actual_dispatch_s', 'actual_dispatch_s'),
                         ('hard_deadline_s', 'request_deadline_s')]:
                p.need(close(timing[a], request[b]), 'controller/client time mismatch')
            p.need(timing['completed'] == ending['completed'] and
                   float(ending['at_s']) <= float(timing['cleanup_end_s']), 'inconsistent request cleanup')
        if completed:
            p.need(timing and timing['completed'] and 'forward_started_s' in timing and 'stream_end_s' in timing,
                   'successful request lacks controller/native terminal evidence')
            continue
        p.need(not truth(request.get('success')), 'partial or malformed successful response')
        rejection = request.get('admission_rejection')
        if rejection in ('admission_queue_full', 'admission_deadline'):
            p.need(request.get('http_status') == '429' and not truth(request.get('request_timeout')),
                   'capacity rejection has foreign status')
            prefix = 'RuntimeError: HTTP 429: '
            error = request.get('error', '')
            p.need(error.startswith(prefix), 'unrecognized capacity response')
            body = json.loads(error[len(prefix):])
            message = 'admission queue full' if rejection == 'admission_queue_full' else 'admission deadline expired'
            p.need(body == dict(error=dict(type='admission_rejection', code=rejection, message=message)),
                   'unrecognized capacity response body')
            p.need(int(request['generated_tokens']) == 0 and int(request.get('n_text_chunks') or 0) == 0
                   and not request.get('first_token_s') and not request.get('last_token_s')
                   and not truth(request.get('token_stream_opened')), 'rejected request has output')
            if rejection == 'admission_queue_full':
                p.need(not timing, 'queue-full request was admitted')
                p.need(not any(str(e.get('client_request_id', '')) == rid for e in events),
                       'queue-full response has admitted journal evidence')
            else:
                p.need(timing and not timing['completed'] and 'forward_started_s' not in timing,
                       'deadline refusal lacks unforwarded controller proof')
                p.need(float(timing['cleanup_end_s']) >= float(request['request_deadline_s']) - .1,
                       'admission refused before hard deadline')
            zero_output.append(rid)
            failures.append(dict(request_id=rid, classification=rejection))
            continue
        p.need(rejection in (None, '', '0', 'False', 'false'), 'unknown capacity rejection')
        p.need(truth(request.get('request_timeout')) and request.get('error') == 'request_hard_timeout'
               and request.get('http_status', '') in ('', '200'), 'failure is not an isolated request deadline')
        p.need(timing and not timing['completed'], 'timeout lacks admitted terminal evidence')
        p.need(close(float(request['request_deadline_s']) - float(request['planned_arrival_s']),
                     row['request_hard_timeout_s']), 'request timeout policy changed')
        p.need(float(timing['cleanup_end_s']) >= float(request['request_deadline_s']) - .1,
               'timeout occurred before declared deadline')
        if 'forward_started_s' not in timing:
            p.need(int(request['generated_tokens']) == 0 and int(request.get('n_text_chunks') or 0) == 0
                   and not truth(request.get('token_stream_opened')), 'unforwarded timeout has output')
            zero_output.append(rid)
        failures.append(dict(request_id=rid, classification='request_hard_timeout'))
    return dict(passed=True, independently_recomputed=True, service_terminal_valid=True,
                failed_request_ids=[x['request_id'] for x in failures], failures=failures,
                zero_output_ids=zero_output, n_expected=len(expected),
                controller_terminal_count=len(timings), full_request_denominator_preserved=True)


def audit(checkpoint):
    cp = p.checked(checkpoint)
    for path, digest in cp['artifacts'].items():
        p.need(p.sha(path) == digest, 'raw artifact changed: ' + path)
    row = cp['row']; release = p.checked(cp['release']); binding = p.checked(cp['binding'])
    p.need(row in release['rows'] and cp['declaration'] == release['declaration'], 'undeclared row')
    p.need(cp['binding'] == release['binding'], 'binding changed')
    receipt = p.checked(cp['receipt']); summary = receipt['summary']
    directory = Path(cp['receipt']['path']).parents[2] / 'cells' / row['cell_id']
    p.need(p.read(directory / 'summary.json') == summary, 'summary differs from receipt')
    p.need(receipt['measurement_valid'] is True and summary['measurement_valid'] is True
           and summary['fixed_window_valid'] is True and summary['gpu_count'] == 8
           and summary['power_source_verified'] is True and receipt['clock_restore_complete'] is True
           and receipt['child_stopped'] is True and receipt.get('child_exitcode') == 0
           and not receipt['outer_cleanup_errors'] and not summary.get('runtime_error')
           and not summary.get('incomplete_drain'), 'invalid measurement/cleanup')
    p.need(len(receipt['restoration']) == len(binding['instances']) and
           all(v.get('complete') is True for v in receipt['restoration'].values()), 'native cleanup incomplete')
    p.need(summary['trace_sha256'] == row['trace_sha256'] and
           summary['comparison_system'] == row['system'], 'trace/system mismatch')
    # Recheck the SLO actually supplied to the controller, rather than only rescoring.
    expected_config = p.read(binding['configs'][row['dataset']])
    expected_config.update(journal=str(directory / 'control.jsonl'), slo_scale=row['slo_scale'],
        slo_protocol='per-dataset-slo-v1', slo_attainment_target=.9,
        slo_ttft_s=row['slo_ttft_s'], slo_tpot_s=row['slo_tpot_s'], comparison_system=row['system'])
    p.need(p.read(directory / 'runtime_config.json') == expected_config, 'actual runtime SLO/config differs')
    for side in ('before', 'after'):
        identities = p.read(Path(cp['receipt']['path']).parent / ('identity.' + side + '.json'))
        p.need(len(identities) == len(binding['instances']), 'engine count differs')
        for actual, expected in zip(identities, binding['instances']):
            c = actual['container']; e = expected['container']
            p.need(c['Id'] == e['id'] and c['Image'] == e['image'] and c['State']['StartedAt'] == e['StartedAt']
                   and c['State']['Running'], 'engine identity differs')
            p.need(all(actual['provenance'].get(k) == v for k, v in expected['provenance'].items()), 'engine source differs')
            native = actual['runtime']
            p.need(not native.get('error') and not native.get('runtime_error') and
                   all(native.get(k) == 0 for k in ('active', 'running', 'waiting')) and
                   native.get('kv_allocations') == {} and native.get('transfer_allocations') == {}, 'native residue/fault')
            p.need(native.get('transfer_send_healthy') is not False and not native.get('transfer_send_failed')
                   and not native.get('transfer_inflight_sends') and not native.get('transfer_inflight_receives')
                   and not native.get('transfer_buffered_tensors') and native.get('scheduler_budget_pending') is None,
                   'unsettled native transfer/budget')
    with (directory / 'bench.csv').open() as stream:
        requests = list(csv.DictReader(stream))
    trace = p.read(row['trace'])
    window = p.read(directory / 'arrival_window.json')
    p.need(window['arrival_window_s'] == row['arrival_window_s'] == 100 and
           window['request_hard_timeout_s'] == row['request_hard_timeout_s'] == 120 and
           window['effective_slo_s'] == {'ttft': row['slo_ttft_s'], 'tpot': row['slo_tpot_s']},
           'actual arrival window or effective SLO differs')
    epoch = window['arrival_epoch_s']
    p.need(close(window['arrival_window_end_s'], epoch + 100), 'actual arrival window end differs')
    prescribed = {str(i): r for i, r in enumerate(trace['requests'])}
    for request in requests:
        expected_request = prescribed[str(request['request_id'])]
        p.need(close(request['planned_arrival_s'], epoch + expected_request['arrival_s']),
               'planned arrival differs from frozen trace')
        p.need(close(request['request_deadline_s'], float(request['planned_arrival_s']) + 120),
               'request deadline differs from frozen protocol')
    events = [json.loads(line) for line in (directory / 'control.jsonl').read_text().splitlines() if line]
    classification = classify_requests(requests, trace, row, events)
    report = p.load(release['raw_auditor'], 'slo_legacy_raw_audit')
    raw_proof = report.audit_raw(summary, directory, dict(
        trace=dict(path=row['trace'], sha256=row['trace_sha256']), original_point=row))
    metrics = p.load(release['measurement_auditor'], 'slo_original_metric_auditor')
    # Only raw-journal-proven zero-output failures may supply absent token usage.
    metrics.diagnosed_zero_output = lambda _: (set(classification['zero_output_ids']), classification)
    verified = metrics.audit_checkpoint(checkpoint['path'])
    p.need(len(classification['failures']) == row['n_requests'] - verified['completed_work_requests'],
           'failure denominator differs')
    result = dict(row)
    result.update(verified)
    result.update(measurement_valid=True, service_terminal_valid=True, independently_recomputed=True,
        strict_slo_recomputed=True, energy_measured_gpu_count=8,
        unknown_error_count=0, capacity_failures_independently_audited=True,
        completed_work_throughput_rps=verified['request_throughput_rps'],
        generated_token_throughput_tps=verified['token_throughput_tps'],
        checkpoint=checkpoint, binding=cp['binding'], qualification=cp['qualification'],
        verification=dict(raw=raw_proof, service_terminal=classification),
        service_failures=classification, work_complete=verified['work_complete'],
        frozen_arrival_offsets_verified=True,
        admission_rejections=sum(bool(r.get('admission_rejection')) for r in requests),
        actual_slo_config_verified=True, measurement_purpose='normal')
    return result
