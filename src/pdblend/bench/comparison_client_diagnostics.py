"""Cached client timing diagnostics from already hash-verified window artifacts.

PD's submitted_s is captured before its concurrency semaphore. These timestamps
describe observable scheduling and outstanding work, never actual HTTP send
queue length or a proven client/GPU capacity limit. Original SLO metrics are
not changed by this module.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
import json
import math
from pathlib import Path

VERSION = 'submitted-before-semaphore-client-diagnostics/v1'
_CACHE = {}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _stats(values):
    values = sorted(values)
    n = len(values)
    return {f'client_schedule_delay_p{p}_s': values[math.ceil(n*p/100)-1] if n else None
            for p in (50, 90, 95, 99)} | dict(
        client_schedule_delay_mean_s=sum(values)/n if n else None,
        client_schedule_delay_max_s=values[-1] if n else None,
        client_schedule_delay_samples=n)


def _peak(events):
    # Half-open intervals: a request finishing at t is removed before another
    # starting at t. Zero-duration intervals do not add outstanding work.
    current = peak = 0
    for _, delta in sorted(events):
        current += delta
        peak = max(peak, current)
    return peak


def summarize(records, *, expected_requests=None):
    delays, events, seen = [], [], set()
    count = missing = invalid = duplicates = 0
    for row in records:
        count += 1
        if not isinstance(row, dict):
            invalid += 1; continue
        idx = row.get('idx')
        if idx is not None:
            if idx in seen:
                duplicates += 1; continue
            seen.add(idx)
        scheduled, submitted, finished = (row.get(k) for k in ('scheduled_s', 'submitted_s', 'finished_s'))
        if any(value is None for value in (scheduled, submitted, finished)):
            missing += 1; continue
        if (not all(_finite(v) for v in (scheduled, submitted, finished))
                or submitted < scheduled or finished < submitted):
            invalid += 1; continue
        delays.append(submitted-scheduled)
        if finished > submitted:
            events.extend(((submitted, 1), (finished, -1)))
    return dict(_stats(delays), client_timing_records=count, client_timing_missing_records=missing,
        client_timing_invalid_records=invalid, client_timing_duplicate_records=duplicates,
        client_timing_coverage_fraction=len(delays)/expected_requests if expected_requests else None,
        client_peak_outstanding=_peak(events) if delays else None,
        client_timing_complete=bool(count and count == expected_requests and len(delays) == count))


def _records(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        if '.jsonl' in path.name:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        else:
            rows = json.load(stream)
            if not isinstance(rows, list):
                raise ValueError('client timing artifact must contain request rows')
            yield from rows


def _cached(path, checksum, expected_requests):
    key = (VERSION, checksum, '.jsonl' in path.name, expected_requests)
    if key not in _CACHE:
        try:
            value = summarize(_records(path), expected_requests=expected_requests)
            value['client_timing_parse_error'] = ''
        except (ValueError, TypeError) as exc:
            # Unsupported diagnostic shapes must not suppress an otherwise
            # recorded experiment. Byte-integrity checks remain the exporter's
            # responsibility and occur before this optional reader.
            value = dict(summarize([], expected_requests=expected_requests),
                         client_timing_parse_error=type(exc).__name__+': '+str(exc))
        _CACHE[key] = value
    return deepcopy(_CACHE[key])


def annotate(path, receipt, point, metrics, *, active_run_id=None):
    """Read only paths in the receipt's already-validated artifact mapping."""
    directory = Path(path).parent
    artifacts = receipt.get('artifacts', {})
    result = dict(client_diagnostics_version=VERSION, client_bottleneck_status='unsupported_missing_artifact',
        client_timing_source='', client_timing_source_sha256='', client_send_queue_status='unknown_send_queue',
        client_submitted_timestamp_semantics=('before_client_concurrency_semaphore' if point['system'] == 'pdblend'
                                             else 'producer_submitted_timestamp_not_verified_as_network_send'),
        client_outstanding_semantics='submitted_to_finished_including_client_wait_and_server_time',
        client_configured_concurrency_limit=None, potential_client_concurrency_limit=None,
        client_timing_records=0, client_timing_missing_records=0, client_timing_invalid_records=0,
        client_timing_duplicate_records=0, client_timing_coverage_fraction=None,
        client_peak_outstanding=None, client_timing_complete=False, client_timing_parse_error='', **_stats([]))
    if not point.get('run_id') or (active_run_id is not None and point['run_id'] != active_run_id):
        result['client_bottleneck_status'] = 'not_recomputed_historical'
        return result
    # Only PD's actual LoadClient outcomes are an allowed fallback. Other
    # systems keep unsupported unless their canonical rows expose all fields.
    candidates = ['run/comparison-requests.json']
    if point['system'] == 'pdblend':
        candidates += ['run/outcomes.jsonl', 'run/outcomes.jsonl.gz']
    found = False
    for name in candidates:
        checksum = artifacts.get(name)
        if checksum is None:
            continue
        found = True
        summary = _cached(directory/name, checksum, metrics.get('offered_requests'))
        result.update(summary, client_timing_source=name, client_timing_source_sha256=checksum)
        if summary['client_schedule_delay_samples']:
            break
    if not result['client_schedule_delay_samples']:
        result['client_bottleneck_status'] = 'unsupported_missing_timestamps' if found else 'unsupported_missing_artifact'
        return result
    # The shared PD client has a frozen default limit of 2048 for these native
    # comparison calls. submitted_s precedes its semaphore, so exceeding it is
    # only evidence of a potential limit, not observed send concurrency.
    limit = 2048 if point['system'] == 'pdblend' else None
    potential = result['client_peak_outstanding'] >= limit if limit is not None else None
    result.update(client_configured_concurrency_limit=limit,
        potential_client_concurrency_limit=potential,
        client_bottleneck_status=('potential_client_concurrency_limit' if potential else
            'observable_timing_no_limit_evidence' if result['client_timing_complete'] else 'partial_timing_evidence'))
    return result
