"""Read-only report and pre-stop validation for the fresh ShareGPT SLO scan.

Runner API: validate_result(summary, receipt, row) -> verdict. ``row`` supplies
trace {path, sha256}, raw_dir (or summary_path), cell_id, system, slo_scale,
rate_rps and source_version. No GPU or experiment actions are performed here.
"""
import argparse
import bisect
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

SYSTEMS = ('pdblend', 'mixed', 'distserve', 'dynamollm', 'ecoserve')
METRICS = (
    ('energy_j', 'Total GPU energy (kJ)', .001),
    ('energy_per_good_request_j', 'Energy / SLO-good request (J)', 1),
    ('slo_attainment', 'Joint SLO attainment (%)', 100),
    ('completed_throughput_measurement_rps', 'Completed work, full measurement (req/s)', 1),
    ('ttft_avg_s', 'Mean TTFT (s)', 1),
    ('tpot_avg_s', 'Mean TPOT (s)', 1),
    ('goodput_measurement_rps', 'Goodput, full measurement (req/s)', 1),
    ('completion_fraction', 'Prescribed work completion (%)', 100),
)


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def truth(value):
    return str(value).lower() in ('1', 'true')


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(actual, expected, field):
    if expected is None:
        need(actual is None or isinstance(actual, float) and math.isnan(actual),
             'undefined metric must not be zero: ' + field)
    else:
        need(number(actual) and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8),
             'raw metric mismatch: ' + field)


def time_close(actual, expected, field):
    # Epoch-valued timestamps must use an absolute tolerance. Relative
    # tolerance at today's epoch would otherwise permit seconds of lateness.
    need(number(actual) and number(expected) and math.isclose(actual, expected, rel_tol=0, abs_tol=5e-6),
         'raw timestamp mismatch: ' + field)


def linear_percentile(values, percentile):
    """The frozen producer uses NumPy's linear percentile convention."""
    ordered = sorted(values)
    need(ordered and 0 <= percentile <= 100, 'invalid percentile input')
    position = (len(ordered) - 1) * percentile / 100
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def integrate(rows, start, end, columns):
    times = [float(r['t_s']) for r in rows]
    need(len(times) > 1 and times[0] <= start < end <= times[-1]
         and all(b > a for a, b in zip(times, times[1:])), 'power coverage/time order invalid')
    values = [[float(r[c]) for c in columns] for r in rows]
    need(all(math.isfinite(v) and v >= 0 for row in values for v in row), 'invalid power sample')

    def at(t):
        i = min(max(bisect.bisect_right(times, t) - 1, 0), len(times) - 2)
        weight = (t - times[i]) / (times[i + 1] - times[i])
        return [a + weight * (b - a) for a, b in zip(values[i], values[i + 1])]

    samples = [(start, at(start))] + [(t, v) for t, v in zip(times, values) if start < t < end] + [(end, at(end))]
    return [sum((t1 - t0) * (v0[g] + v1[g]) / 2
                for (t0, v0), (t1, v1) in zip(samples, samples[1:])) for g in range(len(columns))]


def _technical_request(row):
    """Capacity rejection/timeout is a workload outcome; broken clock/HTTP is not."""
    if truth(row.get('success')) and not row.get('error'):
        return None
    error = str(row.get('error', ''))
    lowered = error.lower()
    clock_faults = ('frequency coverage-limited', 'frequency_coverage_limited',
                    'physical recovery unconfirmed', 'first-token phase frequency promise',
                    'clock uncertainty', 'physical_command_uncertainty', 'clock restore')
    if any(text in lowered for text in clock_faults):
        return error or 'technical frequency failure'
    status = str(row.get('http_status', ''))
    if truth(row.get('request_timeout')) and status in ('', '200'):
        return None
    if truth(row.get('admission_rejection')) and status in ('', '429', '503'):
        return None
    return error or 'unclassified failed request (not a declared timeout/rejection)'


def validate_result(summary, receipt, row):
    """Validate a closed physical attempt; only valid joint < .9 may stop a scan.

    Raw work and eight-board energy are independently recomputed. Missing usage
    on a partial stream stays unknown; its confirmed output counter is retained.
    The return value deliberately separates validity, complete work and SLO.
    """
    result = {k: row.get(k) for k in ('cell_id', 'attempt_id', 'system', 'slo_scale',
                                      'rate_rps', 'source_version', 'executing_host', 'receipt_path', 'summary_path')}
    result.update(attempted=True, measurement_observed=bool(summary), measurement_valid=False,
                  technical_error=None, status='technical_failure', stop_eligible=False,
                  below_slo90=False, slo_target_met=False, work_complete=None,
                  energy_j=clean(summary.get('energy_j')),
                  full_operation_energy_j=clean(receipt.get('full_operation_energy_j')),
                  energy_windows_must_not_be_added=True, primary_energy_verified=False,
                  observed_summary_metrics=clean({key: summary.get(key) for key in
                      ('n_expected', 'offered_requests', 'good_requests', 'completed_work_requests',
                       'request_timeouts', 'admission_rejections', 'generated_tokens',
                       'expected_generated_tokens', 'slo_attainment')}))
    try:
        need(receipt.get('cell_id') == row['cell_id'], 'receipt is for another cell')
        need(receipt.get('system') == row['system'], 'receipt system differs')
        need(clean(receipt.get('summary')) == clean(summary), 'receipt and summary disagree')
        need(receipt.get('measurement_valid') is True and summary.get('measurement_valid') is True,
             'producer declares measurement invalid')
        need(summary.get('fixed_window_valid') is True and summary.get('gpu_count') == 8,
             'fixed window or eight-board inventory invalid')
        need(receipt.get('child_stopped') is True and receipt.get('child_exitcode') == 0,
             'producer did not finish cleanly')
        need(receipt.get('clock_restore_complete') is True and not receipt.get('outer_cleanup_errors'),
             'clock or outer cleanup incomplete')
        need(not any(receipt.get(k) for k in ('error', 'sampling_error', 'integration_error')),
             'operation reports a technical error')
        need(not any(summary.get(k) for k in ('runtime_error', 'sampling_error', 'measurement_window_error', 'timestamp_errors')),
             'summary reports a technical error')
        need(summary.get('power_source_verified') is True and not summary.get('power_source_errors'),
             'power source not verified')
        need(summary.get('drain_complete') is True and not summary.get('incomplete_drain'), 'native drain incomplete')
        need(summary.get('post_measurement_cleanup', {}).get('cleanup_complete') is True,
             'post-measurement cleanup incomplete')
        restoration = receipt.get('restoration')
        need(isinstance(restoration, dict) and restoration and
             all(v.get('complete') is True for v in restoration.values()), 'native restoration incomplete')
        need(summary.get('open_loop_verified') is True and summary.get('arrival_fidelity_valid') is True,
             'open-loop arrival evidence invalid')
        scale = float(row['slo_scale'])
        need(scale in (.5, 2), 'unexpected SLO scale')
        window = summary['fixed_window']
        for field, expected in (('arrival_window_s', 100), ('request_hard_timeout_s', 120),
                                ('drain_after_arrival_window_s', 120), ('slo_scale', scale)):
            close(window.get(field), expected, field)
        close(window['effective_slo_s']['ttft'], 5 * scale, 'ShareGPT TTFT limit')
        close(window['effective_slo_s']['tpot'], .15 * scale, 'ShareGPT TPOT limit')
        trace_ref = row['trace']
        need(sha(trace_ref['path']) == trace_ref['sha256'], 'trace digest differs')
        trace = read(trace_ref['path'])
        need(trace.get('model') == '14b' and trace.get('dataset') == 'sharegpt'
             and trace.get('seed') == 701, 'trace is outside the declared model/dataset/seed')
        need(summary.get('trace_sha256') == trace_ref['sha256'] == receipt.get('trace_sha256'), 'executed trace differs')
        result['trace_sha256'] = trace_ref['sha256']
        raw_dir = Path(row.get('raw_dir') or Path(row['summary_path']).parent)
        inputs = {str(Path(path).resolve()): sha(path)
                  for path in (trace_ref['path'], raw_dir / 'bench.csv', raw_dir / 'power.csv')}
        with (raw_dir / 'bench.csv').open() as stream:
            requests = list(csv.DictReader(stream))
        n = len(trace['requests'])
        need(n > 0 and len(requests) == n, 'offered request denominator differs')
        need([r['request_id'] for r in requests] == [str(i) for i in range(n)], 'request identities differ')
        complete = good = generated = 0
        delays, ttfts, tpots = [], [], []
        partial_usage_unknown = 0
        for req, prescribed in zip(requests, trace['requests']):
            technical = _technical_request(req)
            need(technical is None, 'technical request failure: ' + str(technical))
            emitted = int(req['generated_tokens'])
            need(emitted >= 0, 'negative output count')
            generated += emitted
            work = (truth(req.get('success')) and not req.get('error')
                    and req.get('token_count_source') == 'server_usage'
                    and truth(req.get('token_ids_verified'))
                    and int(req['input_tokens']) == prescribed['prompt_len']
                    and emitted == prescribed['output_len'])
            complete += work
            if not work and req.get('token_count_source') != 'server_usage' and int(req.get('n_text_chunks') or 0) > 0:
                partial_usage_unknown += 1
            ttft = float(req['ttft_s']) if req.get('ttft_s') else None
            tpot = float(req['tpot_s']) if req.get('tpot_s') else None
            for value, values in ((ttft, ttfts), (tpot, tpots)):
                if value is not None:
                    need(math.isfinite(value) and value >= 0, 'invalid raw latency')
                    values.append(value)
            good += bool(work and ttft is not None and tpot is not None
                         and ttft < 5 * scale and tpot < .15 * scale)
            planned = float(req['planned_arrival_s'])
            dispatched = float(req['actual_dispatch_s'])
            delay = dispatched - planned
            need(math.isfinite(delay) and delay >= -1e-6 and truth(req.get('open_loop_independent')),
                 'invalid independent dispatch evidence')
            close(float(req['dispatch_delay_s']), delay, 'per-request dispatch delay')
            time_close(planned, window['arrival_epoch_s'] + float(prescribed['arrival_s']), 'planned trace arrival')
            time_close(float(req['request_deadline_s']), planned + 120, 'deadline must include dispatch delay')
            if ttft is not None:
                time_close(float(req['first_token_s']) - planned, ttft, 'TTFT includes dispatch delay')
            delays.append(delay)
        expected_tokens = sum(r['output_len'] for r in trace['requests'])
        start, end = summary['measurement_start_s'], summary['measurement_end_s']
        need(number(start) and number(end) and end - start >= 100, 'measurement window too short')
        close(summary['measurement_duration_s'], end - start, 'measurement duration')
        normalized = dict(n_expected=n, offered_requests=n, completed_work_requests=complete,
                          good_requests=good, generated_tokens=generated, expected_generated_tokens=expected_tokens,
                          failed_requests=n - complete, slo_attainment=good / n,
                          request_timeouts=sum(truth(r.get('request_timeout')) for r in requests),
                          admission_rejections=sum(truth(r.get('admission_rejection')) for r in requests),
                          ttft_avg_s=statistics.fmean(ttfts) if ttfts else None,
                          tpot_avg_s=statistics.fmean(tpots) if tpots else None,
                          goodput_measurement_rps=good / (end - start))
        for field, expected in normalized.items():
            close(summary.get(field), expected, field)
        completed_throughput = complete / (end - start)
        if 'completed_work_throughput_rps' in summary:
            close(summary['completed_work_throughput_rps'], completed_throughput,
                  'completed work throughput, full measurement')
        complete_work = complete == n and generated == expected_tokens
        need(summary.get('work_complete') is complete_work, 'work completion flag differs')
        with (raw_dir / 'power.csv').open() as stream:
            powers = list(csv.DictReader(stream))
        energy = sum(integrate(powers, start, end, [f'gpu{i}_w' for i in range(8)]))
        close(summary['energy_j'], energy, 'eight-GPU primary energy')
        util = [v / (end - start) / 100 for v in integrate(powers, start, end, [f'gpu{i}_util_pct' for i in range(8)])]
        need(len(summary['gpu_util_per_gpu']) == 8 and all(0 <= v <= 1 for v in util), 'utilization inventory invalid')
        for actual, expected in zip(summary['gpu_util_per_gpu'], util):
            close(actual, expected, 'per-GPU utilization')
        close(summary['gpu_util'], statistics.fmean(util), 'eight-GPU mean utilization')
        energy_good = energy / good if good else None
        close(summary.get('energy_per_good_request_j'), energy_good, 'energy per good request')
        close(summary.get('dispatch_delay_max_s'), max(delays), 'maximum dispatch lateness')
        p99 = linear_percentile(delays, 99)
        if 'dispatch_delay_p99_s' in summary:
            close(summary['dispatch_delay_p99_s'], p99, 'p99 dispatch lateness')
        declared_max = row.get('dispatch_delay_max_limit_s')
        declared_p99 = row.get('dispatch_delay_p99_limit_s')
        if declared_max is not None or declared_p99 is not None:
            need(number(declared_max) and declared_max >= 0 and number(declared_p99) and declared_p99 >= 0,
                 'both predeclared arrival-lag bounds are required')
            need(max(delays) <= declared_max and p99 <= declared_p99,
                 'predeclared arrival-lag bound violated')
        limit = summary.get('dispatch_lateness_limit_s')
        if limit is not None:
            need(number(limit) and limit >= 0 and max(delays) <= limit
                 and summary.get('dispatch_lateness_within_declared_limit') is True,
                 'declared dispatch lateness bound violated')
        need(all(sha(path) == digest for path, digest in inputs.items()), 'raw files changed while being audited')
        result.update(normalized, energy_j=energy, energy_per_good_request_j=energy_good,
                      completed_throughput_measurement_rps=completed_throughput,
                      completed_throughput_arrival_window_rps=complete / 100,
                      measurement_duration_s=end - start, arrival_window_s=100,
                      gpu_util=statistics.fmean(util), completion_fraction=complete / n,
                      work_complete=complete_work, measurement_valid=True, stop_eligible=True,
                      below_slo90=good * 10 < n * 9, slo_target_met=good * 10 >= n * 9,
                      status='valid_complete' if complete_work else 'valid_incomplete',
                      dispatch_delay_max_s=max(delays), dispatch_delay_mean_s=statistics.fmean(delays),
                      dispatch_delay_p99_s=p99, dispatch_delay_percentile_method='linear',
                      dispatch_delay_max_limit_s=declared_max, dispatch_delay_p99_limit_s=declared_p99,
                      dispatch_lateness_limit_s=limit, partial_stream_usage_unknown=partial_usage_unknown,
                      all_eight_gpu_energy_reintegrated=True, primary_energy_verified=True,
                      raw_requests_recomputed=True, raw_input_sha256=inputs)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        result['technical_error'] = str(exc)
    return clean(result)


def inspect_record(record, manifest):
    row = dict(record)
    path = row.get('receipt_path')
    if not path or not Path(path).exists():
        return dict(row, measurement_valid=False, status='awaiting_receipt', attempted=True,
                    measurement_observed=False, stop_eligible=False, below_slo90=False,
                    technical_error=row.get('technical_error'), work_complete=None)
    try:
        for key in ('dispatch_delay_max_limit_s', 'dispatch_delay_p99_limit_s'):
            if key in manifest:
                need(row.get(key) == manifest[key], 'arrival-lag declaration differs: ' + key)
        if row.get('receipt_sha256'):
            need(sha(path) == row['receipt_sha256'], 'receipt digest differs')
        receipt = read(path)
        if not row.get('summary_path'):
            row['summary_path'] = str(Path(path).parents[2] / 'cells' / row['cell_id'] / 'summary.json')
        if row.get('summary_sha256'):
            need(sha(row['summary_path']) == row['summary_sha256'], 'summary digest differs')
        summary = read(row['summary_path']) if Path(row['summary_path']).exists() else receipt.get('summary', {})
        need(isinstance(row.get('source_version'), str) and row['source_version'], 'missing frozen source version')
        expected_version = manifest.get('versions', {}).get(row['system'])
        if expected_version is not None:
            need(row['source_version'] == expected_version, 'source version differs from the declaration')
        fresh_after = manifest.get('created_s')
        if fresh_after is not None:
            need(receipt.get('started_s', receipt.get('operation_start_s', 0)) >= fresh_after,
                 'observation predates fresh campaign declaration')
        verdict = validate_result(summary, receipt, row)
        if row.get('technical_error'):
            verdict.update(measurement_valid=False, stop_eligible=False, below_slo90=False,
                           slo_target_met=False, status='technical_failure',
                           technical_error=row['technical_error'])
        return verdict
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(row, measurement_valid=False, status='evidence_error', attempted=True,
                    measurement_observed=True, stop_eligible=False, below_slo90=False,
                    technical_error=str(exc), work_complete=None)


def _key(row):
    return float(row['slo_scale']), float(row['rate_rps']), row['system']


def summarize(points):
    versions = {}
    for point in points:
        if point['measurement_valid']:
            versions.setdefault(point['system'], set()).add(point['source_version'])
    need(all(len(v) == 1 for v in versions.values()), 'multiple valid source versions for one system; separate reports required')
    hosts = {p.get('executing_host') for p in points if p['measurement_valid']}
    need(len(hosts) <= 1, 'one sweep must execute on one selected host')
    valid = {}
    for point in points:
        if point['measurement_valid']:
            need(_key(point) not in valid, 'duplicate valid attempt; no best-attempt selection')
            valid[_key(point)] = point
    endpoints, coverage = {}, []
    for scale in (.5, 2.):
        pdb = sorted((p for p in valid.values() if p['system'] == 'pdblend' and float(p['slo_scale']) == scale),
                     key=lambda p: float(p['rate_rps']))
        below = [p for p in pdb if p['below_slo90']]
        endpoint = below[0] if below else None
        if endpoint:
            need(not any(float(p['rate_rps']) > float(endpoint['rate_rps']) for p in pdb),
                 'PDB has observations beyond first valid below-90 endpoint')
        endpoints[str(scale)] = dict(rate_rps=endpoint['rate_rps'], slo_attainment=endpoint['slo_attainment'],
                                     cell_id=endpoint['cell_id']) if endpoint else None
        for p in pdb:
            for system in SYSTEMS[1:]:
                b = valid.get((scale, float(p['rate_rps']), system))
                paired = bool(b and p['trace_sha256'] == b['trace_sha256'])
                coverage.append(dict(slo_scale=scale, rate_rps=p['rate_rps'], system=system,
                                     pdb_cell_id=p['cell_id'], baseline_cell_id=b['cell_id'] if b else None,
                                     same_trace_valid_pair=paired, baseline_work_complete=b.get('work_complete') if b else None))
    complete = all(value is not None for value in endpoints.values()) and bool(coverage) and all(p['same_trace_valid_pair'] for p in coverage)
    return dict(campaign_complete=complete, attempted=len(points), measurement_observed=sum(p.get('measurement_observed', False) for p in points),
                valid_measurements=sum(p['measurement_valid'] for p in points),
                work_complete=sum(p['measurement_valid'] and p.get('work_complete') is True for p in points),
                slo90_met=sum(p['measurement_valid'] and p.get('slo_target_met', False) for p in points),
                status_counts=dict(Counter(p['status'] for p in points)), endpoints=endpoints,
                baseline_pairs_required=len(coverage), baseline_pairs_valid=sum(p['same_trace_valid_pair'] for p in coverage),
                source_versions={k: next(iter(v)) for k, v in versions.items()},
                executing_host=next(iter(hosts)) if hosts else None), coverage


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def operation_records(state):
    """Expose setup/gate/restore energy without summing overlapping windows."""
    rows, inputs = [], {}

    def observation(kind, source, context=None):
        context = context or {}
        row = dict(operation=kind, executing_host=context.get('hostname'),
                   energy_windows_must_not_be_added=True, evidence_valid=False,
                   energy_j=None, measurement_valid=False)
        try:
            if isinstance(source, dict) and source.get('path'):
                path = Path(source['path']).resolve()
                row.update(receipt_path=str(path), declared_sha256=source.get('sha256'))
                digest = sha(path)
                need(source.get('sha256') == digest, 'operation receipt digest differs')
                row['receipt_sha256'] = digest
                inputs[str(path)] = digest
                data = read(path)
            else:
                need(isinstance(source, dict), 'operation receipt/reference missing')
                data = source
            # Restore wrappers retain the actual measured operation separately.
            if data.get('restoration_receipt'):
                measured = observation(kind, data['restoration_receipt'], data)
                measured.update(wrapper_path=row.get('receipt_path'),
                                wrapper_measurement_valid=data.get('measurement_valid'),
                                wrapper_error=data.get('error'),
                                observed_measurement_valid=measured['measurement_valid'])
                measured['measurement_valid'] = bool(measured['measurement_valid'] and
                                                     data.get('measurement_valid') is True)
                return measured
            row.update(evidence_valid=True, executing_host=data.get('hostname', row['executing_host']),
                       complete=data.get('complete'), measurement_valid=data.get('measurement_valid') is True,
                       operation_start_s=data.get('operation_start_s', data.get('measurement_start_s')),
                       operation_end_s=data.get('operation_end_s', data.get('measurement_end_s')),
                       measured_gpu_count=data.get('measured_gpu_count'),
                       energy_j=clean(data.get('full_operation_energy_j', data.get('all8_operation_energy_j'))),
                       error=clean(data.get('errors', data.get('error', data.get('cleanup_errors')))),
                       energy_scope=data.get('energy_scope', 'separate all-eight qualification operation'))
            need(row['energy_j'] is None or number(row['energy_j']) and row['energy_j'] >= 0,
                 'invalid observed operation energy')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(evidence_valid=False, measurement_valid=False, error=str(exc))
        return row

    for stage, value in state.get('stages', {}).items():
        for field, kind in (('deployment_receipt', 'deployment'), ('gate', 'qualification')):
            if value.get(field):
                rows.append(observation(stage + ':' + kind, value[field], value))
    if state.get('restoration'):
        rows.append(observation('original_host_restoration', state['restoration']))
    return rows, inputs


def make_figures(points, totals, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(SYSTEMS, ('#1b7652', '#707b8c', '#bf6a2a', '#7860aa', '#247ea4')))
    for scale in (.5, 2.):
        scale_points = [p for p in points if float(p['slo_scale']) == scale]
        rates = sorted({float(p['rate_rps']) for p in scale_points})
        fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
        for axis, (metric, label, factor) in zip(axes.flat, METRICS):
            for system in SYSTEMS:
                selected = {float(p['rate_rps']): p for p in scale_points
                            if p['system'] == system and p['measurement_valid']}
                values = [selected[r].get(metric) if r in selected else None for r in rates]
                values = [float(v) * factor if v is not None else float('nan') for v in values]
                axis.plot(rates, values, '-o', label=system, color=colors[system], linewidth=1.6, markersize=4)
            if metric == 'slo_attainment':
                axis.axhline(90, color='#a32b37', linestyle='--', linewidth=1, label='90% target')
                axis.set_ylim(-2, 103)
            endpoint = totals['endpoints'][str(scale)]
            if endpoint:
                axis.axvline(float(endpoint['rate_rps']), color=colors['pdblend'], linestyle=':', linewidth=1)
            axis.set(xlabel='Offered rate (req/s)', ylabel=label)
            axis.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        endpoint = totals['endpoints'][str(scale)]
        suffix = f"first valid PDB <90%: {endpoint['rate_rps']} req/s" if endpoint else 'PDB endpoint not yet observed'
        fig.suptitle(f'14B ShareGPT | SLO scale {scale:g} | seed 701 | 100 s arrivals\n{suffix}; gaps are unmeasured or invalid')
        for extension in ('png', 'pdf'):
            fig.savefig(out / f'14b-sharegpt-slo{scale:g}.{extension}', dpi=180)
        plt.close(fig)


def build_report(manifest_path, state_path, out_dir, *, figures=True):
    manifest_path, state_path, out = map(Path, (manifest_path, state_path, out_dir))
    need(not out.exists(), 'report requires a new output directory')
    manifest, state = read(manifest_path), read(state_path)
    need(manifest.get('protocol_id') == 'a14b-sharegpt-slo90-v1', 'report declaration is for another protocol')
    need(number(manifest.get('created_s')), 'fresh campaign created_s is required to reject historical measurements')
    records = state.get('records', [])
    need(isinstance(records, list), 'state.records must be a list')
    input_hashes = {str(manifest_path.resolve()): sha(manifest_path), str(state_path.resolve()): sha(state_path)}
    for record in records:
        for field in ('receipt_path', 'summary_path', 'checkpoint_path'):
            path = record.get(field)
            if path and Path(path).is_file():
                input_hashes[str(Path(path).resolve())] = sha(path)
    points = [inspect_record(r, manifest) for r in records]
    for point in points:
        input_hashes.update(point.get('raw_input_sha256', {}))
    totals, coverage = summarize(points)
    operations, operation_inputs = operation_records(state)
    input_hashes.update(operation_inputs)
    report = dict(schema='a14b-sharegpt-slo90-report-v1', created_s=time.time(),
                  protocol_id=manifest.get('protocol_id'), totals=totals, points=points,
                  baseline_coverage=coverage, operations=operations, scan_state=state.get('scan'), inputs=input_hashes,
                  single_seed_screen=True, energy_windows_must_not_be_added=True,
                  old_measurements_reused=False)
    need(all(sha(path) == digest for path, digest in input_hashes.items()), 'input changed during report snapshot')
    out.mkdir(parents=True)
    (out / 'results.json').write_text(json.dumps(clean(report), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    csv_write(out / 'points.csv', points)
    csv_write(out / 'baseline-coverage.csv', coverage)
    csv_write(out / 'operations.csv', operations)
    lines = ['# 14B ShareGPT SLO 90% 新实验', '',
             f"执行节点：{totals['executing_host'] or '尚未观察到'}。已尝试 {totals['attempted']} 次；已有测量观测 {totals['measurement_observed']} 次；有效测量 {totals['valid_measurements']} 次；完整工作 {totals['work_complete']} 次；joint SLO ≥90% {totals['slo90_met']} 次。", '',
             '两档分别在 PDB 首个有效 joint SLO <90% 的 rate 停止；恰好90%继续。技术故障不作为阈值终点。有效超时和合法拒绝留在全部 offered 请求分母。', '',
             f"同 trace 基线配对有效 {totals['baseline_pairs_valid']}/{totals['baseline_pairs_required']}。基线在 PDB 所有有效已测 rate 上补齐；未测或无效点留空。", '',
             '| SLO scale | 首个有效 PDB <90% rate | joint SLO |', '|---|---:|---:|']
    for scale, endpoint in totals['endpoints'].items():
        lines.append(f"| {scale} | {endpoint['rate_rps'] if endpoint else '尚未观察到'} | {format(endpoint['slo_attainment'], '.2%') if endpoint else '—'} |")
    lines += ['', '全八卡主测能量包含100秒到达窗口及实际尾部；外层操作能量单列，与主测窗口重叠，不相加。缺失最终 usage 的部分流式输出量未知，不按零输出解释。', '',
              '完成吞吐和 goodput 均用实际完整测量时长作分母；逐点 CSV 另列100秒到达窗口归一化的完成吞吐及八卡利用率。部署、资格检查和最终恢复的已观测能量列于 operations.csv，保留来源和失败状态，不并入逐点主测能量。', '',
              '单个 seed 701 的短测及自适应停止结果，不给独立种子置信区间；未达到终点不代表无限容量。各系统固定版本列于 results.json，不逐点选择最低能耗版本。', '',
              '到达延迟同时列出实测最大值、均值和线性插值p99，按行内预声明上限核验；迟发超限是技术无效，不是SLO终点。TTFT及120秒硬超时均从原计划到达时刻计算。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    if figures:
        make_figures(points, totals, out)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--out-dir', '--out', required=True, type=Path)
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    result = build_report(args.manifest, args.state, args.out_dir, figures=not args.no_figures)
    print(json.dumps(result['totals'], ensure_ascii=False))


if __name__ == '__main__':
    main()
