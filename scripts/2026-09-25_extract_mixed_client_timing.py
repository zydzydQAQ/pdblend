"""One-pass native scalar outcomes review; no prompt or token-event replay."""
from pathlib import Path
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import math
import time

from importlib import import_module

common = import_module('2026-09-25_extract_failed_distserve_native')

OUT = common.OUT
WINDOWS = common.ROOT/'results/2026-09-22/three-model/queue-attempts/comparison-7b-0d6782f15e6c0f27/attempt-0001-6e0ff084085940f7af0fb911837b1e89/session/windows'


def reduce_timing(rows, *, origin, duration, offered):
    seen, delays, spans, missing, early = set(), [], [], [], []
    buckets = {i: [] for i in range(math.ceil(duration/30))}
    errors, success = Counter(), 0
    for row in rows:
        idx = row.get('idx')
        if type(idx) is not int or not 0 <= idx < offered or idx in seen:
            raise ValueError('duplicate or unknown request index')
        seen.add(idx)
        scheduled, submitted, finished = (row.get(k) for k in ('scheduled_s', 'submitted_s', 'finished_s'))
        arrival = row.get('arrival_s')
        if (not common.finite(arrival) or not 0 <= arrival < duration or
                not common.finite(scheduled) or
                not math.isclose(scheduled, origin+arrival, rel_tol=0, abs_tol=1e-6)):
            raise ValueError('native scheduled timestamp differs from bound service start')
        if common.finite(submitted) and common.finite(finished) and finished >= submitted:
            delta = submitted-scheduled
            if delta < 0: early.append(idx)
            delays.append(delta); buckets[int(arrival//30)].append(delta)
            if finished > submitted: spans.extend(((submitted, 1), (finished, -1)))
        else: missing.append(idx)
        success += row.get('correct') is True
        if row.get('error'): errors[str(row['error'])] += 1
    outstanding = peak = 0
    for _, delta in sorted(spans):
        outstanding += delta; peak = max(peak, outstanding)
    if seen != set(range(offered)):
        raise ValueError('native outcome cohort incomplete')
    return dict(offered_requests=offered, observed_outcomes=len(seen),
                native_correct_requests=success, native_errors=dict(errors),
                scheduled_to_pre_dispatch_delay=common.stats(delays),
                delay_by_30s_arrival_bucket={str(i*30): common.stats(v) for i, v in buckets.items()},
                peak_pre_dispatch_outstanding=peak if not missing else None,
                missing_client_timing_indices=missing,
                early_pre_dispatch_indices=early,
                actual_send_queue_status='unknown_actual_send_and_connector_queue',
                client_concurrency_limit=None,
                concurrency_limit_evidence='no explicit numerical connector limit bound by this review',
                timestamp_semantics='submitted_s is recorded before policy.route and generate; outstanding counts submitted-to-finished intervals, not actual HTTP sends',
                phase_interval_semantics='submitted inclusive, finished exclusive')


def rows_from_bytes(raw):
    with gzip.open(io.BytesIO(raw), 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if row.pop('_journal_schema', None) != 'pdblend-journal-v1':
                raise ValueError('unexpected native outcome schema')
            if any(k in row for k in ('payload_ref', 'token_payload_ref')):
                raise ValueError('scalar outcomes unexpectedly require payload replay')
            yield row


def run(job_id):
    target = OUT/'mixed-four-client-timing-review.json'
    if target.exists(): return dict(status='already_cached', path=str(target))
    if not (OUT/'distserve-sharegpt-lower-recovered-service-power.json').exists():
        return dict(status='deferred', reason='priority_native_and_service_review_not_ready')
    phase = common.phase(common.QUEUE, job_id)
    if not phase['ready']: return dict(status='deferred', phase=phase)
    records = []
    for window in sorted(WINDOWS.iterdir()):
        path = OUT/('mixed-client-timing-'+window.name+'.json')
        if path.exists():
            records.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            continue
        receipt, receipt_ref, _ = common.read_bound(window/'receipt.json', limit=1024*1024)
        point, point_ref, _ = common.read_bound(window/'point.json', receipt['artifacts']['point.json'], 1024*1024)
        if common.digest(point) != receipt['point_sha256'] or point['system'] != 'mixed':
            raise ValueError('point binding differs')
        native, native_ref, _ = common.read_bound(window/'run/native-result.json',
                                                receipt['artifacts']['run/native-result.json'], 1024*1024)
        source = common.small_ref(point['source_manifest'])
        if source['source_sha256'] != point['revision']:
            raise ValueError('source manifest revision differs')
        raw_path = window/'run'/native['outcomes_path']
        if raw_path.resolve().parent != (window/'run').resolve():
            raise ValueError('native outcomes path escapes window')
        expected_sha = receipt['artifacts']['run/'+native['outcomes_path']]
        if expected_sha != native['outcomes_sha256'] or raw_path.stat().st_size > 1024*1024:
            raise ValueError('raw size or binding differs')
        before = common.phase(common.QUEUE, job_id)
        if not before['ready']: return dict(status='deferred', phase=before)
        began = time.time(); clock = time.perf_counter()
        raw = raw_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError('native outcomes checksum differs')
        summary = reduce_timing(rows_from_bytes(raw), origin=native['started_s'],
                                duration=point['duration_s'],
                                offered=receipt['result']['metrics']['offered_requests'])
        ended = time.time(); after = common.phase(common.QUEUE, job_id)
        summary.update(schema='mixed-native-client-timing-review/v1',
            receipt=receipt_ref, point_artifact=point_ref, point_sha256=receipt['point_sha256'],
            point_id=point['name'], run_id=point.get('run_id'), revision=point['revision'],
            source_manifest=point['source_manifest'], trace=point['trace'],
            native_result=native_ref, outcomes=dict(path=str(raw_path), sha256=expected_sha),
            method=dict(path=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
            canonical_modified=False, used_for_ranking=False,
            extraction=dict(started_s=began, finished_s=ended, wall_elapsed_s=time.perf_counter()-clock,
                            raw_bytes_read=len(raw), phase_before=before, phase_after=after,
                            same_early_loading_phase_before_after=after['ready'] and before.get('attempt_dir')==after.get('attempt_dir')))
        path.write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False)+'\n')
        records.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    target.write_text(json.dumps(dict(schema='mixed-four-client-timing-review/v1', records=records,
                                     canonical_modified=False, used_for_ranking=False), indent=2, sort_keys=True)+'\n')
    return dict(status='extracted', path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-job', default=common.TARGET_JOB)
    parser.add_argument('--wait-seconds', type=float, default=0)
    args = parser.parse_args(); deadline = time.monotonic()+args.wait_seconds
    while True:
        value = run(args.wait_for_job)
        if value['status'] != 'deferred' or time.monotonic() >= deadline:
            print(json.dumps(value, sort_keys=True)); return
        time.sleep(min(20, max(0, deadline-time.monotonic())))


if __name__ == '__main__': main()
