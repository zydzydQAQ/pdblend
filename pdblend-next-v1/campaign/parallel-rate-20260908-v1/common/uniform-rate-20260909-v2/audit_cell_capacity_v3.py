"""Read-only, independent request/SLO/energy/arrival audit of uniform cells."""
import csv
from pathlib import Path

import support as p


def baseline_timeout_classification(rows, summary, row, zero_output_ids=()):
    """Accept only explicit client deadline exhaustion after full raw audit.

    This does not infer capacity from an arbitrary HTTP failure, missing row,
    cancellation, native fault, or incomplete sampler. Callers must also
    independently verify before/after native health and all cleanup.
    """
    p.need(row['system'] != 'pdblend' and summary['measurement_valid'] is True,
           'only a valid baseline may receive this classification')
    trace = p.checked(dict(path=row['trace'], sha256=row['trace_sha256']))
    p.need(len(rows) == len(trace['requests']) == summary['n_expected'], 'request count mismatch')
    failures = []
    for actual, expected in zip(rows, trace['requests']):
        successful = (actual.get('success', '').lower() in ('1', 'true') and not actual.get('error')
            and actual.get('token_count_source') == 'server_usage'
            and actual.get('token_ids_verified', '').lower() in ('1', 'true')
            and int(actual['input_tokens']) == expected['prompt_len']
            and int(actual['generated_tokens']) == expected['output_len'])
        if successful:
            continue
        if actual['request_id'] in zero_output_ids:
            failures.append(actual['request_id'])
            continue
        p.need(actual.get('success', '').lower() in ('0', 'false')
            and actual.get('request_timeout', '').lower() in ('1', 'true')
            and actual.get('error') in ('request_hard_timeout', 'request_hard_timeout_before_dispatch')
            and actual.get('http_status', '') in ('', '200')
            and actual.get('admission_rejection', '').lower() in ('', 'false', '0'),
            'baseline incomplete work is not an explicit isolated request deadline')
        failures.append(actual['request_id'])
    p.need(failures and len(failures) == summary['n_expected'] - summary['completed_work_requests'],
           'failure classification denominator mismatch')
    return dict(classification='baseline_explicit_request_hard_timeout', independently_diagnosed=True,
        failed_request_ids=failures, timeout_bound_s=row['request_hard_timeout_s'],
        no_PDB_capacity_boundary_claim=True, full_request_denominator_preserved=True,
        missing_terminal_usage_is_not_zero_generation=True)


def audit(checkpoint):
    cp = p.checked(checkpoint)
    for path, digest in cp['artifacts'].items():
        p.need(p.sha(path) == digest, 'raw artifact changed: ' + path)
    row = cp['row']
    release = p.checked(cp['release'])
    p.need(row in release['rows'] and cp['declaration'] == release['declaration'], 'row outside frozen release')
    binding = p.checked(cp['binding'])
    p.need(cp['binding'] == release['binding'], 'execution binding changed')
    receipt = p.checked(cp['receipt'])
    summary = receipt['summary']
    directory = Path(cp['receipt']['path']).parents[2] / 'cells' / row['cell_id']
    p.need(p.read(directory / 'summary.json') == summary, 'summary differs from receipt')
    p.need(receipt['measurement_valid'] is True and summary['measurement_valid'] is True
           and summary['fixed_window_valid'] is True and summary['gpu_count'] == 8
           and summary['power_source_verified'] is True and receipt['clock_restore_complete'] is True
           and receipt['child_stopped'] is True and not receipt['outer_cleanup_errors'], 'invalid measurement/cleanup')
    p.need(all(v.get('complete') is True for v in receipt['restoration'].values()), 'native cleanup incomplete')
    p.need(summary['trace_sha256'] == row['trace_sha256']
           and summary['comparison_system'] == row['system'], 'trace/system mismatch')
    report = p.load(release['raw_auditor'], 'uniform_independent_raw_report')
    raw = p.load(release['additional_metrics'], 'uniform_additional_raw_metrics')
    proof = report.audit_raw(summary, directory, dict(
        trace=dict(path=row['trace'], sha256=row['trace_sha256']), original_point=row))
    additional = raw.audit_additional_metrics(summary, directory)
    arrivals = p.load(release['arrival_auditor'], 'uniform_arrivals')
    with (directory / 'bench.csv').open() as stream:
        bench_rows = list(csv.DictReader(stream))
        timing = arrivals.recompute(bench_rows, row['n_requests'])
    identity_files = [Path(cp['receipt']['path']).parent / ('identity.' + side + '.json')
                      for side in ('before', 'after')]
    for path in identity_files:
        identity = p.read(path)
        p.need(len(identity) == len(binding['instances']), 'engine identity count differs')
        for actual, expected in zip(identity, binding['instances']):
            c = actual['container']
            p.need(c['Id'] == expected['container']['id'] and c['Image'] == expected['container']['image']
                   and c['State']['StartedAt'] == expected['container']['StartedAt']
                   and c['State']['Running'], 'actual engine identity differs')
            p.need(all(actual['provenance'].get(k) == v for k, v in expected['provenance'].items()),
                   'actual engine provenance differs')
            native = actual['runtime']
            p.need(not native.get('error') and not native.get('runtime_error'), 'native runtime fault')
            p.need(all(native.get(k) == 0 for k in ('active', 'running', 'waiting'))
                   and native.get('kv_allocations') == {} and native.get('transfer_allocations') == {},
                   'native pre/post measurement residue')
            p.need(native.get('transfer_send_healthy') is not False
                   and not native.get('transfer_send_failed')
                   and not native.get('transfer_inflight_sends') and not native.get('transfer_inflight_receives')
                   and not native.get('transfer_buffered_tensors')
                   and native.get('scheduler_budget_pending') is None, 'native transfer/budget fault')
    result = dict(row, **{k: summary.get(k) for k in report.METRICS})
    result.update(additional['normalized_metrics'])
    result.update(measurement_valid=True, work_complete=summary['work_complete'],
        completed_work_requests=summary['completed_work_requests'], n_expected=summary['n_expected'],
        expected_generated_tokens=summary['expected_generated_tokens'], generated_tokens=summary['generated_tokens'],
        measurement_duration_s=summary['measurement_duration_s'], good_requests=summary['good_requests'],
        completed_work_throughput_rps=summary['completed_work_requests'] / summary['measurement_duration_s'],
        generated_token_throughput_tps=summary['generated_tokens'] / summary['measurement_duration_s'],
        gpu_util_per_gpu=proof['gpu_util_per_gpu'], energy_measured_gpu_count=8,
        checkpoint=checkpoint, qualification=cp['qualification'], binding=cp['binding'],
        verification=dict(raw=proof, additional=additional, arrivals=timing), independently_recomputed=True)
    metrics = p.load(release['measurement_auditor'], 'uniform_v2_strict_metrics')
    verified = metrics.audit_checkpoint(checkpoint['path'])
    result.update(verified)
    result['measurement_purpose'] = row.get('measurement_purpose', 'normal')
    if not summary['work_complete'] and row['system'] != 'pdblend':
        zero_ids = [str(x['request_id']) for x in verified.get('zero_output_diagnosis', {}).get('refusal_details', [])]
        if any(r.get('admission_rejection') == 'admission_queue_full' for r in bench_rows):
            helper_path = Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1/common/token-evidence-v2/audit_controller_queue_429_v1.py')
            helper_ref = dict(path=str(helper_path), sha256='f3a89aac070472b5eecfd47e3841c5c912dc13ca0f28f5fa764c116783b02890')
            queue_auditor = p.load(helper_ref, 'uniform_independent_controller_queue_429')
            queue_proof = queue_auditor.audit(checkpoint)
            zero_ids.extend(str(x['request_id']) for x in queue_proof['refusal_details'])
            classification = baseline_timeout_classification(bench_rows, summary, row, zero_ids)
            classification.update(classification='baseline_explicit_controller_admission_queue_full',
                passed=True, independently_recomputed=True, controller_queue_evidence=queue_proof,
                controller_rejections=queue_proof['controller_rejections'],
                actual_request_timeouts=queue_proof['request_timeouts'],
                queue_rejections_are_not_timeouts=True)
            result['verification']['controller_queue_full'] = queue_proof
            result['baseline_service_failure'] = classification
        else:
            result['baseline_service_failure'] = baseline_timeout_classification(bench_rows, summary, row, zero_ids)
    return result
