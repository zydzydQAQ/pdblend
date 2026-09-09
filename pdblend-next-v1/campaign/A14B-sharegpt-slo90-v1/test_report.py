"""CPU fixtures only; these synthetic observations are never experiment results."""
import copy
import csv
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('slo90_report_under_test', Path(__file__).with_name('report.py'))
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def fixture(tmp_path, *, failures=(), n=10, good_count=None):
    trace = dict(model='14b', dataset='sharegpt', seed=701,
                 requests=[dict(arrival_s=i, prompt_len=2, output_len=3) for i in range(n)])
    trace_path = tmp_path / 'trace.json'
    trace_path.write_text(json.dumps(trace))
    requests = []
    for i in range(n):
        request = dict(request_id=str(i), success='1', error='', input_tokens='2', generated_tokens='3',
                       token_count_source='server_usage', token_ids_verified='1', n_text_chunks='3',
                       ttft_s='1.0', tpot_s='0.01', planned_arrival_s=str(1000 + i),
                       actual_dispatch_s=str(1000.125 + i), dispatch_delay_s='.125',
                       first_token_s=str(1001 + i), request_deadline_s=str(1120 + i),
                       open_loop_independent='True', request_timeout='False', admission_rejection='', http_status='')
        if i in failures:
            request.update(success='0', error='hard request deadline', request_timeout='True',
                           token_count_source='missing', token_ids_verified='', generated_tokens='0',
                           n_text_chunks='0', ttft_s='', tpot_s='', first_token_s='')
        elif good_count is not None and i >= good_count:
            request.update(ttft_s='3.0', first_token_s=str(1003 + i))
        requests.append(request)
    write_csv(tmp_path / 'bench.csv', requests)
    power = []
    for timestamp in (999, 1000, 1050, 1100, 1101):
        power.append(dict(t_s=timestamp, **{f'gpu{i}_w': 100 for i in range(8)},
                          **{f'gpu{i}_util_pct': 50 for i in range(8)}))
    write_csv(tmp_path / 'power.csv', power)
    complete = n - len(failures)
    good = complete if good_count is None else good_count
    summary = dict(measurement_valid=True, fixed_window_valid=True, gpu_count=8,
                   power_source_verified=True, power_source_errors=[], drain_complete=True,
                   incomplete_drain=False, post_measurement_cleanup=dict(cleanup_complete=True),
                   open_loop_verified=True, arrival_fidelity_valid=True,
                   fixed_window=dict(arrival_window_s=100, request_hard_timeout_s=120,
                       drain_after_arrival_window_s=120, slo_scale=.5, arrival_epoch_s=1000,
                       effective_slo_s=dict(ttft=2.5, tpot=.075)),
                   trace_sha256=report.sha(trace_path), n_expected=n, offered_requests=n,
                   completed_work_requests=complete, good_requests=good, generated_tokens=complete * 3,
                   expected_generated_tokens=n * 3, failed_requests=len(failures),
                   slo_attainment=good / n, request_timeouts=len(failures), admission_rejections=0,
                   ttft_avg_s=(good + 3 * (complete - good)) / complete if complete else float('nan'),
                   tpot_avg_s=.01 if complete else float('nan'), goodput_measurement_rps=good / 100,
                   measurement_start_s=1000, measurement_end_s=1100, measurement_duration_s=100,
                   work_complete=complete == n, energy_j=80000, gpu_util_per_gpu=[.5] * 8,
                   gpu_util=.5, energy_per_good_request_j=80000 / good if good else None,
                   dispatch_delay_max_s=.125, dispatch_lateness_limit_s=None)
    receipt = dict(cell_id='test-only-cell', system='pdblend', summary=summary,
                   trace_sha256=summary['trace_sha256'], measurement_valid=True, child_stopped=True,
                   child_exitcode=0, clock_restore_complete=True, outer_cleanup_errors=[],
                   restoration={'engine': {'complete': True}}, full_operation_energy_j=90000,
                   started_s=999)
    row = dict(cell_id='test-only-cell', attempt_id='cpu-only-1', system='pdblend', slo_scale=.5,
               rate_rps=.4, source_version='cpu-fixture-version', executing_host='cpu-fixture-host',
               trace=dict(path=str(trace_path), sha256=report.sha(trace_path)), raw_dir=str(tmp_path),
               summary_path=str(tmp_path / 'summary.json'), receipt_path=str(tmp_path / 'receipt.json'))
    return summary, receipt, row, requests


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_observation(summary, receipt, row):
    Path(row['summary_path']).write_text(json.dumps(summary))
    Path(row['receipt_path']).write_text(json.dumps(receipt))
    row['receipt_sha256'] = report.sha(row['receipt_path'])
    row['summary_sha256'] = report.sha(row['summary_path'])


def test_exactly_ninety_continues_with_valid_timeout(tmp_path):
    summary, receipt, row, _ = fixture(tmp_path, failures=(9,))
    result = report.validate_result(summary, receipt, row)
    assert result['measurement_valid'], result
    assert result['slo_attainment'] == .9 and result['slo_target_met']
    assert result['stop_eligible'] and not result['below_slo90']
    assert not result['work_complete'] and result['request_timeouts'] == 1


def test_below_target_stops_and_failed_denominator_is_retained(tmp_path):
    summary, receipt, row, _ = fixture(tmp_path, failures=(8, 9))
    result = report.validate_result(summary, receipt, row)
    assert result['measurement_valid'] and result['below_slo90']
    assert result['good_requests'] == 8 and result['offered_requests'] == 10
    assert result['energy_j'] == 80000 and result['full_operation_energy_j'] == 90000
    assert result['energy_per_good_request_j'] == 10000
    assert result['completed_throughput_measurement_rps'] == .08


def test_completion_throughput_uses_observed_tail_and_optional_producer_field(tmp_path):
    summary, receipt, row, _ = fixture(tmp_path, failures=(8, 9))
    summary.update(measurement_end_s=1101, measurement_duration_s=101,
                   goodput_measurement_rps=8 / 101, energy_j=80800,
                   energy_per_good_request_j=10100)
    result = report.validate_result(summary, receipt, row)
    assert result['measurement_valid'], result
    assert result['completed_throughput_measurement_rps'] == 8 / 101
    assert result['completed_throughput_arrival_window_rps'] == .08
    assert result['measurement_duration_s'] == 101
    summary['completed_work_throughput_rps'] = .08
    result = report.validate_result(summary, receipt, row)
    assert not result['measurement_valid'] and 'completed work throughput' in result['technical_error']
    summary['completed_work_throughput_rps'] = 8 / 101
    assert report.validate_result(summary, receipt, row)['measurement_valid']


@pytest.mark.parametrize('admission_mark', ['', 'True'])
def test_clock_503_is_technical_even_if_mislabelled_as_admission(tmp_path, admission_mark):
    summary, receipt, row, requests = fixture(tmp_path, failures=(9,))
    requests[9].update(error='HTTP 503: frequency coverage-limited: physical recovery unconfirmed',
                       request_timeout='False', admission_rejection=admission_mark, http_status='503')
    write_csv(tmp_path / 'bench.csv', requests)
    result = report.validate_result(summary, receipt, row)
    assert not result['measurement_valid'] and not result['below_slo90']
    assert 'technical request failure' in result['technical_error']


def test_legal_capacity_rejection_is_valid_workload_outcome(tmp_path):
    summary, receipt, row, requests = fixture(tmp_path, failures=(8, 9))
    for index in (8, 9):
        requests[index].update(error='capacity queue full', http_status='503', request_timeout='False',
                               admission_rejection='True')
    write_csv(tmp_path / 'bench.csv', requests)
    summary.update(request_timeouts=0, admission_rejections=2)
    result = report.validate_result(summary, receipt, row)
    assert result['measurement_valid'] and result['below_slo90'], result


def test_unknown_503_cannot_establish_capacity_endpoint(tmp_path):
    summary, receipt, row, requests = fixture(tmp_path, failures=(8, 9))
    requests[8].update(error='HTTP 503 unknown engine failure', http_status='503', request_timeout='False')
    write_csv(tmp_path / 'bench.csv', requests)
    assert not report.validate_result(summary, receipt, row)['stop_eligible']


def test_corrupted_summary_or_eight_board_energy_rejected(tmp_path):
    summary, receipt, row, _ = fixture(tmp_path)
    summary['good_requests'] = 9
    assert not report.validate_result(summary, receipt, row)['measurement_valid']
    summary['good_requests'] = 10
    summary['energy_j'] = 70000
    result = report.validate_result(summary, receipt, row)
    assert 'eight-GPU primary energy' in result['technical_error']


def test_epoch_comparison_does_not_allow_second_of_timestamp_drift(tmp_path):
    summary, receipt, row, requests = fixture(tmp_path)
    for req in requests:
        req['planned_arrival_s'] = str(float(req['planned_arrival_s']) + 1)
        req['actual_dispatch_s'] = str(float(req['actual_dispatch_s']) + 1)
        req['request_deadline_s'] = str(float(req['request_deadline_s']) + 1)
    write_csv(tmp_path / 'bench.csv', requests)
    result = report.validate_result(summary, receipt, row)
    assert 'planned trace arrival' in result['technical_error']
    with pytest.raises(ValueError):
        report.time_close(1788814547.0, 1788814546.0, 'epoch')


def test_declared_lateness_bound_and_ttft_include_delay(tmp_path):
    summary, receipt, row, requests = fixture(tmp_path)
    summary.update(dispatch_lateness_limit_s=.1, dispatch_lateness_within_declared_limit=False)
    result = report.validate_result(summary, receipt, row)
    assert 'lateness bound' in result['technical_error']
    summary.update(dispatch_lateness_limit_s=None)
    requests[0]['ttft_s'] = '.875'  # dispatch-to-token would hide the .125 delay
    write_csv(tmp_path / 'bench.csv', requests)
    result = report.validate_result(summary, receipt, row)
    assert 'TTFT includes dispatch delay' in result['technical_error']


def test_predeclared_linear_p99_boundary_is_shared_with_worker(tmp_path):
    summary, receipt, row, _ = fixture(tmp_path)
    row.update(dispatch_delay_max_limit_s=1., dispatch_delay_p99_limit_s=.1)
    result = report.validate_result(summary, receipt, row)
    assert not result['measurement_valid'] and 'arrival-lag bound' in result['technical_error']
    assert report.linear_percentile([0.] * 99 + [.5], 99) == pytest.approx(.005)
    row['dispatch_delay_p99_limit_s'] = .125
    result = report.validate_result(summary, receipt, row)
    assert result['measurement_valid'] and result['dispatch_delay_p99_s'] == .125


def test_report_retains_worker_technical_verdict_even_when_raw_window_is_valid(tmp_path):
    summary, receipt, row, _ = fixture(tmp_path)
    save_observation(summary, receipt, row)
    row['technical_error'] = 'inherited node lease changed after execution'
    result = report.inspect_record(row, {})
    assert result['raw_requests_recomputed'] and not result['measurement_valid']
    assert not result['stop_eligible'] and result['technical_error'] == row['technical_error']


def test_zero_good_keeps_undefined_metrics_and_partial_output_unknown(tmp_path):
    summary, receipt, row, requests = fixture(tmp_path, failures=tuple(range(10)))
    requests[0]['n_text_chunks'] = '2'
    write_csv(tmp_path / 'bench.csv', requests)
    result = report.validate_result(summary, receipt, row)
    assert result['measurement_valid'] and result['below_slo90'], result
    assert result['energy_per_good_request_j'] is None and result['ttft_avg_s'] is None
    assert result['partial_stream_usage_unknown'] == 1


def test_source_version_duplicates_and_cross_host_are_not_selected(tmp_path):
    summary, receipt, row, _ = fixture(tmp_path)
    point = report.validate_result(summary, receipt, row)
    with pytest.raises(ValueError, match='duplicate valid'):
        report.summarize([point, copy.deepcopy(point)])
    other = dict(point, rate_rps=.6, source_version='different')
    with pytest.raises(ValueError, match='multiple valid source'):
        report.summarize([point, other])
    other.update(source_version=point['source_version'], executing_host='another-host')
    with pytest.raises(ValueError, match='one selected host'):
        report.summarize([point, other])


def test_old_measurement_and_tampered_receipt_are_rejected(tmp_path):
    summary, receipt, row, _ = fixture(tmp_path)
    save_observation(summary, receipt, row)
    result = report.inspect_record(row, {'created_s': 1000})
    assert 'predates fresh' in result['technical_error']
    Path(row['receipt_path']).write_text('{}')
    assert 'receipt digest' in report.inspect_record(row, {})['technical_error']


def test_report_outputs_all_eight_metrics_both_scales_and_missing_baselines(tmp_path):
    summary, receipt, row, _ = fixture(tmp_path, good_count=8)
    save_observation(summary, receipt, row)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'protocol_id': 'a14b-sharegpt-slo90-v1', 'created_s': 998}))
    state = tmp_path / 'state.json'
    state.write_text(json.dumps({'records': [row]}))
    out = tmp_path / 'cpu-only-report'
    result = report.build_report(manifest, state, out)
    totals = result['totals']
    assert totals['valid_measurements'] == 1 and totals['work_complete'] == 1 and totals['slo90_met'] == 0
    assert totals['baseline_pairs_required'] == 4 and totals['baseline_pairs_valid'] == 0
    assert totals['endpoints']['0.5']['rate_rps'] == .4 and totals['endpoints']['2.0'] is None
    assert len(list(out.glob('*.png'))) == len(list(out.glob('*.pdf'))) == 2
    assert (out / 'points.csv').is_file() and (out / 'baseline-coverage.csv').is_file()
    assert (out / 'operations.csv').is_file()
    assert 'completed_throughput_measurement_rps' in {m[0] for m in report.METRICS}
    assert 'gpu_util' not in {m[0] for m in report.METRICS}
    assert len(result['inputs']) >= 7  # declaration/state/receipt/summary/trace/bench/power
    with pytest.raises(ValueError, match='new output'):
        report.build_report(manifest, state, out)


def test_operations_keep_failure_energy_and_nested_restore_receipt_separate(tmp_path):
    def reference(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return dict(path=str(path), sha256=report.sha(path))

    deployment = reference('deploy.json', dict(complete=False, measurement_valid=False,
        full_operation_energy_j=1234, all8_operation_energy_j=1234, errors=['readiness failed']))
    gate = reference('gate.json', dict(complete=True, measurement_valid=True, full_operation_energy_j=456))
    restore = reference('restore-receipt.json', dict(complete=False, measurement_valid=False,
        full_operation_energy_j=789, errors=['native cleanup failed']))
    wrapper = reference('restore-result.json', dict(hostname='cpu-only-host', restoration_receipt=restore))
    state = dict(stages={'pdblend': dict(hostname='cpu-only-host', deployment_receipt=deployment, gate=gate)},
                 restoration=wrapper)
    rows, inputs = report.operation_records(state)
    assert [r['energy_j'] for r in rows] == [1234, 456, 789]
    assert [r['measurement_valid'] for r in rows] == [False, True, False]
    assert all(r['evidence_valid'] and r['energy_windows_must_not_be_added'] for r in rows)
    assert rows[2]['receipt_path'] == restore['path'] and rows[2]['executing_host'] == 'cpu-only-host'
    assert len(inputs) == 4
    Path(gate['path']).write_text('{}')
    rows, _ = report.operation_records(state)
    assert not rows[1]['evidence_valid'] and rows[1]['energy_j'] is None


def test_restore_wrapper_failure_cannot_promote_valid_inner_meter(tmp_path):
    raw = tmp_path / 'restore-receipt.json'
    raw.write_text(json.dumps(dict(complete=True, measurement_valid=True, full_operation_energy_j=123)))
    wrapper = dict(complete=True, measurement_valid=False, error='node lease changed after restoration',
                   restoration_receipt=dict(path=str(raw), sha256=report.sha(raw)))
    rows, _ = report.operation_records(dict(restoration=wrapper))
    assert rows[0]['energy_j'] == 123 and rows[0]['observed_measurement_valid']
    assert not rows[0]['measurement_valid']
    assert rows[0]['wrapper_error'] == wrapper['error']
