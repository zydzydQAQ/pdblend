"""Independent scalar evidence extraction; never edits CSV, sources, or queue.

The CLI waits for the explicitly authorized next job's early loading phase.
It reads the failed native completion once, hashes those same bytes, then
retains scalar request receipts for subsequent reviews. Native latency fields
are diagnostics, not a fabricated canonical drain or energy measurement.
"""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import math
import re
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/2026-09-24/profile-saturation-repaired-v1-independent-metrics-review'
QUEUE = ROOT/'results/2026-09-22/three-model/queue.json'
WINDOW = ROOT/'results/2026-09-22/three-model/queue-attempts/comparison-7b-b71abb0f68b33c93/attempt-0001-6dd7a6d545664f46848b85b332dc4d93/session/windows/7b-distserve-sharegpt-x1.953125-seed701'
RECEIPT_SHA = '25b87e6eadc6851f124c1e0ef57e75c6ae3c484ea39a128719c0591314a70341'
TARGET_JOB = 'comparison-7b-dd9742f537d25cc3'
FIELDS = ('request_id', 'replica', 'arrival_s', 'scheduled_s', 'submitted_s',
          'finished_s', 'input_tokens', 'output_tokens', 'completion_tokens',
          'ok', 'terminal', 'native_receipts_complete', 'ttft_s', 'tpot_s',
          'error', 'event_count', 'token_ids_sha256', 'journal_path')


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read_bound(path, expected=None, limit=None):
    path = Path(path)
    if limit is not None and path.stat().st_size > limit:
        raise ValueError('size exceeds predeclared limit: '+str(path))
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError('bound artifact SHA differs: '+str(path))
    return json.loads(data), dict(path=str(path.resolve()), sha256=actual), len(data)


def small_ref(ref):
    return read_bound(ref['path'], ref['sha256'], 1024*1024)[0]


def stats(values):
    values = sorted(values)
    n = len(values)
    return dict(samples=n, mean_s=sum(values)/n if n else None,
                max_s=values[-1] if n else None,
                p99_low_sample=n < 100,
                **{f'p{p}_s': values[math.ceil(p*n/100)-1] if n else None
                   for p in (50, 90, 95, 99)})


def summarize(native, *, point, expected_count):
    if native.get('trace_sha256') != point['trace']['sha256']:
        raise ValueError('native/frozen trace mismatch')
    origin = native.get('service_started_s')
    if not finite(origin):
        raise ValueError('actual native service start unavailable')
    duration = point['duration_s']
    reduced, seen, delays, events = [], set(), [], []
    good = successful = finished_window_good = delivered = 0
    ttfts, tpots, errors = [], [], Counter()
    timing_invalid, client_timing_missing, good_tokens = [], [], 0
    for full in native.get('outcomes', []):
        row = {key: full[key] for key in FIELDS if key in full}
        match = re.fullmatch(r'distserve-701-(\d+)', str(row.get('request_id', '')))
        if match is None:
            raise ValueError('outcome index cannot be bound')
        idx = int(match[1])
        if idx in seen or not 0 <= idx < expected_count:
            raise ValueError('duplicate or out-of-cohort native outcome')
        seen.add(idx)
        row['idx'] = idx
        scheduled, submitted, finished = (row.get(k) for k in
                                         ('scheduled_s', 'submitted_s', 'finished_s'))
        arrival = row.get('arrival_s')
        if (not finite(arrival) or not 0 <= arrival < duration or
                not finite(scheduled) or not math.isclose(scheduled, origin+arrival,
                                                         rel_tol=0, abs_tol=1e-6)):
            raise ValueError('native scheduled time differs from service anchor')
        if all(finite(v) for v in (submitted, finished)) and finished >= submitted >= scheduled:
            delays.append(submitted-scheduled)
            if finished > submitted:
                events.extend(((submitted, 1), (finished, -1)))
        else:
            client_timing_missing.append(idx)
        n, maximum = row.get('completion_tokens'), row.get('output_tokens')
        if type(n) is int and n >= 0:
            delivered += n
        native_success = (row.get('ok') is True and row.get('terminal') is True
                          and row.get('native_receipts_complete') is True
                          and not row.get('error') and type(n) is int
                          and type(maximum) is int and n == maximum and maximum >= 2)
        timing_ok = (type(n) is int and n >= 2 and
                     finite(row.get('ttft_s')) and row['ttft_s'] >= 0 and
                     finite(row.get('tpot_s')) and row['tpot_s'] >= 0 and
                     finite(finished) and finished >= scheduled and
                     scheduled + row['ttft_s'] + row['tpot_s']*(n-1) <= finished + 1e-5)
        if native_success:
            successful += 1
            if timing_ok:
                ttfts.append(row['ttft_s']); tpots.append(row['tpot_s'])
                is_good = (row['ttft_s'] <= point['slo']['ttft_s'] and
                           row['tpot_s'] <= point['slo']['tpot_s'])
                good += is_good
                if is_good:
                    good_tokens += n
                    finished_window_good += finished <= origin+duration
            else:
                timing_invalid.append(idx)
        if row.get('error'):
            errors[str(row['error'])] += 1
        reduced.append(row)
    outstanding = peak = 0
    for _, delta in sorted(events):
        outstanding += delta
        peak = max(peak, outstanding)
    all_accounted = seen == set(range(expected_count))
    counts = dict(expected_requests=expected_count, recorded_outcomes=len(reduced),
                  all_outcomes_accounted=all_accounted, native_successful_requests=successful,
                  native_failed_or_missing_requests=expected_count-successful,
                  native_joint_slo_requests=good, native_joint_slo_rate=good/expected_count,
                  native_success_rate=successful/expected_count,
                  native_delivered_completion_tokens=delivered,
                  native_joint_slo_completion_tokens=good_tokens,
                  native_good_requests_finished_by_window_end=finished_window_good)
    st_ttft, st_tpot = stats(ttfts), stats(tpots)
    finite_ends = [r['finished_s'] for r in reduced if finite(r.get('finished_s'))]
    horizon = max([origin+duration, *finite_ends])
    native_slo = bool(all_accounted and successful == expected_count and
                      not timing_invalid and good/expected_count >= .9 and
                      st_ttft['p99_s'] is not None and st_ttft['p99_s'] <= point['slo']['ttft_s'] and
                      st_tpot['p99_s'] is not None and st_tpot['p99_s'] <= point['slo']['tpot_s'])
    return dict(schema='distserve-native-failed-window-scalar-review/v1',
                counts=counts, native_request_slo_pass=native_slo,
                native_ttft=st_ttft, native_tpot=st_tpot,
                native_goodput_finished_window_request_s_lower_bound=finished_window_good/duration,
                native_cohort_goodput_request_s=good/(horizon-origin),
                native_cohort_goodput_token_s=good_tokens/(horizon-origin),
                request_observation_horizon_s=horizon,
                service_started_s=origin, service_ended_s=origin+duration,
                native_service_finished_s=native.get('service_finished_s'),
                native_status=native.get('status'), native_complete=native.get('complete'),
                native_cleanup_errors=native.get('cleanup_errors'),
                native_hardware_executed=native.get('hardware_executed'),
                native_error=native.get('error'), native_error_counts=dict(errors),
                missing_outcome_indices=sorted(set(range(expected_count))-seen),
                invalid_success_timing_indices=timing_invalid,
                client_pre_dispatch_delay=stats(delays),
                client_peak_pre_dispatch_outstanding=peak if not client_timing_missing else None,
                client_timing_missing_indices=client_timing_missing,
                client_send_queue_status='unknown_actual_send_and_connector_queue',
                client_timestamp_semantics='submitted_s before runtime.handle; scheduled_s from native service anchor plus frozen arrival',
                latency_method='frozen DistServe deployment computes TTFT from planned arrival and TPOT from client first/last token arrivals',
                limitation='Scalar native evidence only; full client journal not replayed. finished_s is after stream handling, so in-window goodput is a conservative lower bound. Native completion does not prove outer drain success.',
                canonical_modified=False, used_for_ranking=False,
                canonical_energy_service_j=None, canonical_energy_tail_j=None,
                requests=sorted(reduced, key=lambda r: r['idx']))


def phase(queue_path, job_id):
    queue = json.loads(Path(queue_path).read_text())
    active = [v for v in queue['leases'].values() if v.get('status') == 'active']
    wanted = [v for v in active if v.get('job_id') == job_id]
    observed = dict(observed_s=time.time(), active_job_ids=[v['job_id'] for v in active])
    if len(wanted) != 1 or len(active) != 1:
        return dict(observed, ready=False, reason='target_not_sole_active_lease')
    lease = wanted[0]
    session = Path(lease['attempt_dir'])/'session'
    windows = session/'windows'
    points = list(windows.glob('*/point.json')) if windows.exists() else []
    resets = list(windows.glob('*/reset.json')) if windows.exists() else []
    qualifies = (session/'qualification.json').exists()
    ready = not points and not resets and not qualifies
    return dict(observed, ready=ready, reason='early_loading' if ready else 'loading_slot_missed',
                claimed_at=lease['claimed_at'], attempt_dir=lease['attempt_dir'],
                point_count=len(points), reset_count=len(resets), qualification_present=qualifies)


def context():
    receipt, receipt_ref, _ = read_bound(WINDOW/'receipt.json', RECEIPT_SHA, 1024*1024)
    point, point_ref, _ = read_bound(WINDOW/'point.json', receipt['artifacts']['point.json'], 1024*1024)
    if digest(point) != receipt['point_sha256']:
        raise ValueError('point digest differs')
    source = small_ref(point['source_manifest'])
    if source['source_sha256'] != point['revision']:
        raise ValueError('execution source differs')
    pd_point = small_ref(point['pd_boundary_point'])
    if pd_point['trace'] != point['trace']:
        raise ValueError('same-cohort reference differs')
    pd_receipt, pd_ref, _ = read_bound(Path(point['pd_boundary_point']['path']).parent/'receipt.json', limit=1024*1024)
    if pd_receipt['point_sha256'] != digest(pd_point):
        raise ValueError('same-cohort point binding differs')
    expected = pd_receipt['result']['metrics']['offered_requests']
    return receipt, point, dict(receipt=receipt_ref, point_artifact=point_ref,
                               point_sha256=receipt['point_sha256'], source_manifest=point['source_manifest'],
                               trace=point['trace'], offered_count_evidence=pd_ref,
                               expected_requests=expected, trace_raw_rehashed=False)


def extract(job_id):
    destination = OUT/'distserve-sharegpt-lower-failed-native-review.json'
    if destination.exists():
        value = json.loads(destination.read_text())
        if value['bindings']['receipt']['sha256'] != RECEIPT_SHA:
            raise ValueError('existing review belongs to another receipt')
        return dict(status='already_cached', path=str(destination))
    before = phase(QUEUE, job_id)
    if not before['ready']:
        return dict(status='deferred', phase=before)
    receipt, point, bindings = context()
    before = phase(QUEUE, job_id)
    if not before['ready']:
        return dict(status='deferred', phase=before)
    began = time.time(); clock = time.perf_counter()
    native, native_ref, size = read_bound(WINDOW/'run/completion.json',
                                       receipt['artifacts']['run/completion.json'], 40*1024*1024)
    ended = time.time()
    after = phase(QUEUE, job_id)
    summary = summarize(native, point=point, expected_count=bindings['expected_requests'])
    requests = summary.pop('requests')
    reduced_path = OUT/'distserve-sharegpt-lower-native-scalar-requests.json'
    reduced_path.write_text(json.dumps(requests, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n')
    summary.update(bindings=bindings, native_completion=native_ref,
                   scalar_requests=dict(path=str(reduced_path), sha256=hashlib.sha256(reduced_path.read_bytes()).hexdigest()),
                   extraction=dict(raw_bytes_read=size, started_s=began, read_and_parse_finished_s=ended,
                                   wall_elapsed_to_summary_s=time.perf_counter()-clock,
                                   phase_before=before, phase_after_read=after,
                                   same_early_loading_phase_before_after=after['ready'] and before.get('attempt_dir')==after.get('attempt_dir')),
                   method=dict(path=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return dict(status='extracted', path=str(destination), sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                counts=summary['counts'], native_request_slo_pass=summary['native_request_slo_pass'],
                client_delay=summary['client_pre_dispatch_delay'], peak=summary['client_peak_pre_dispatch_outstanding'],
                phase=after, raw_read_elapsed_s=ended-began)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-job', default=TARGET_JOB)
    parser.add_argument('--wait-seconds', type=float, default=0)
    args = parser.parse_args()
    deadline = time.monotonic()+args.wait_seconds
    while True:
        result = extract(args.wait_for_job)
        if result['status'] != 'deferred' or time.monotonic() >= deadline:
            print(json.dumps(result, sort_keys=True)); return
        time.sleep(min(20, max(0, deadline-time.monotonic())))


if __name__ == '__main__':
    main()
