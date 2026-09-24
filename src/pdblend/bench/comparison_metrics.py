"""Policy-independent client-visible metrics for a frozen arrival cohort.

Window goodput counts only complete, correct, SLO-satisfying requests whose
terminal event arrives by the service boundary. Token throughput counts actual
delivered tokens, including partial/failed requests. Cohort goodput includes
drain and uses the actual terminal-outcome horizon, never the shorter window.
No engine token times or HTTP-close times substitute for client token times.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass
import math
from pathlib import Path
import re
from typing import Mapping

PROTOCOL = 'client-window-and-drained-cohort-v1'
CLIENT_EVENTS = {'mixed': 'mixed_client_sse', 'distserve': 'distserve_client_sse',
                 'ecoserve': 'eco_client_sse', 'dynamollm': 'dynamo_sse',
                 'pdblend': 'pdblend_client_sse'}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _dict(value):
    return asdict(value) if is_dataclass(value) else dict(value)


def event_token_count(event, cumulative=0):
    """Return (delivered delta, exact, new cumulative) without retokenizing text.

    Legacy text frames are recorded as approximate events so missing token
    identity stays explicit. Final usage alone cannot recover delivery times.
    """
    if isinstance(event.get('token_ids'), list):
        count, exact = len(event['token_ids']), True
    elif type(event.get('pdblend_generated_tokens')) is int and event['pdblend_generated_tokens'] >= 0:
        count, exact = event['pdblend_generated_tokens'], True
    elif type(event.get('token_index')) is int:
        count, exact = event['token_index'] - cumulative, True
        if count < 0:
            raise ValueError('client token index regressed')
    elif any(isinstance(c.get('logprobs'), dict) and isinstance(c['logprobs'].get('tokens'), list)
             for c in event.get('choices', [])):
        count = sum(len(c.get('logprobs', {}).get('tokens', [])) for c in event['choices']
                    if isinstance(c.get('logprobs'), dict))
        exact = True
    else:
        count = sum(bool(c.get('text')) for c in event.get('choices', []))
        exact = count == 0
    return count, exact, cumulative + count


def _request_index(row, trace_indices):
    if row.get('idx') in trace_indices:
        return row['idx']
    rid = str(row.get('request_id', ''))
    match = re.fullmatch(r'r(\d+)', rid) or re.search(r'-701-(\d+)$', rid)
    if match and int(match[1]) in trace_indices:
        return int(match[1])
    raise ValueError('outcome cannot be bound to frozen request: ' + rid)


def canonical_outcomes(system, trace, outcomes, *, service_started_s, journal=None):
    """Adapt five native journals or public ``Outcome`` objects to scalar rows.

    ``trace`` is Request objects, request dicts, or a {requests: [...]} manifest.
    ``outcomes`` may be the native completion dict containing ``outcomes``.
    ``journal`` is an iter_journal-compatible path or iterable of decoded rows.
    Only the system's *client* event stream is selected, avoiding EcoServe and
    DistServe native/client double counting. Missing outcomes remain failures.
    """
    if system not in CLIENT_EVENTS or not _finite(service_started_s):
        raise ValueError('supported system and actual service start required')
    requests = [_dict(r) for r in (trace['requests'] if isinstance(trace, Mapping) else trace)]
    indices = {r.get('idx', i) for i, r in enumerate(requests)}
    if len(indices) != len(requests):
        raise ValueError('duplicate frozen request idx')
    outcomes = outcomes.get('outcomes', []) if isinstance(outcomes, Mapping) else outcomes
    by_idx = {}
    for value in outcomes:
        row = _dict(value)
        idx = _request_index(row, indices)
        if idx in by_idx:
            raise ValueError('duplicate terminal outcome for request ' + str(idx))
        by_idx[idx] = row
    streams = defaultdict(list)
    journal_available = journal is not None
    if isinstance(journal, (str, Path)):
        from pdblend.results.journal import iter_journal
        journal = iter_journal(journal)
    for event in journal or ():
        if event.get('kind', event.get('event')) != CLIENT_EVENTS[system]:
            continue
        idx = _request_index(event, indices)
        streams[idx].append(event)
    result = []
    for position, req in enumerate(requests):
        idx = req.get('idx', position)
        original = by_idx.get(idx)
        if original is None:
            result.append(dict(idx=idx, error='missing_outcome', terminal=False,
                               completion_tokens=None, token_events=[], token_events_complete=False))
            continue
        row = dict(original, idx=idx)
        row['scheduled_s'] = service_started_s + req['arrival_s']
        native_failed = bool(row.get('error') or row.get('ok') is False or row.get('correct') is False)
        events, cumulative, terminal_s = [], 0, None
        selected = streams.get(idx, [])
        for envelope in selected:
            payload = envelope.get('payload', {})
            stamp = payload.get('received_s', envelope.get('at_s'))
            if not _finite(stamp):
                raise ValueError('client event lacks receipt timestamp')
            count, exact, cumulative = event_token_count(payload, cumulative)
            if count:
                events.append(dict(received_s=stamp, count=count, exact=exact))
            if (payload.get('finished') is True or any(c.get('finish_reason') is not None
                                                     for c in payload.get('choices', []))):
                terminal_s = stamp
        if selected:
            row['token_events'] = events
            row['token_events_complete'] = all(e['exact'] for e in events)
            if events:
                row['first_token_s'], row['last_token_s'] = events[0]['received_s'], events[-1]['received_s']
            if terminal_s is not None:
                row.update(terminal=True, terminal_s=terminal_s)
            row.setdefault('completion_tokens', cumulative if all(e['exact'] for e in events) else None)
            if all(e['exact'] for e in events) and row.get('completion_tokens') != cumulative:
                row['native_reported_completion_tokens'] = row.get('completion_tokens')
                if native_failed:
                    # Several native collectors initialize count=0 and only
                    # update it on successful return. Client journal delivery
                    # remains authoritative after timeout/stream failure.
                    row['completion_tokens'] = cumulative
                else:
                    # A claimed successful stream with contradictory exact
                    # counts is invalid evidence; do not repair it into success.
                    row['error'] = 'native_client_token_count_mismatch'
        elif (journal_available and native_failed and row.get('completion_tokens') == 0
              and _finite(row.get('finished_s'))):
            # A complete client journal with no frames plus an explicit final
            # failure outcome proves zero delivered tokens (e.g. rejection or
            # timeout before first token). Missing outcomes still stay unknown.
            row.update(token_events=[], token_events_complete=True, observed_zero_tokens=True)
        if row.get('ok') is False or row.get('correct') is False:
            row['error'] = row.get('error') or 'native_output_validation_failed'
        # finished_s is a terminal outcome timestamp, not the last-token time.
        # Its use for successful terminal timing requires explicit proof of a
        # complete stream; journal timestamps take precedence when available.
        if row.get('terminal') is True and row.get('terminal_s') is None:
            row['terminal_s'] = row.get('finished_s')
        result.append(row)
    return result


def _stats(values, prefix):
    ordered = sorted(values)
    n = len(ordered)
    return {f'{prefix}_p{p}_s': ordered[math.ceil(p / 100 * n) - 1] if n else None
            for p in (50, 90, 95, 99)} | {
                f'{prefix}_mean_s': sum(ordered) / n if n else None,
                f'{prefix}_max_s': ordered[-1] if n else None,
                f'{prefix}_samples': n, f'{prefix}_p99_low_sample': n < 100}


def reduce_comparison(trace, outcomes, *, service_started_s, duration_s=150.,
                      slo=(1., .1), token_events=None, observed_until_s=None):
    """Reduce canonical outcomes. Missing requests stay in every denominator.

    ``observed_until_s`` is the actual end of observation, required to report
    cohort throughput when an offered request has no terminal outcome. It
    never fabricates a terminal event or makes that request successful.
    Optional ``token_events`` maps frozen idx to client delivery event lists.
    Zero-count energy ratios should use :func:`safe_ratio` in the caller.
    """
    if (not _finite(service_started_s) or not _finite(duration_s) or duration_s <= 0
            or len(slo) != 2 or any(not _finite(v) or v <= 0 for v in slo)):
        raise ValueError('finite service clock, positive duration and SLO required')
    if observed_until_s is not None and (not _finite(observed_until_s) or observed_until_s < service_started_s):
        raise ValueError('observation end precedes service start')
    trace = [_dict(r) for r in (trace['requests'] if isinstance(trace, Mapping) else trace)]
    indices = {r.get('idx', i) for i, r in enumerate(trace)}
    if len(indices) != len(trace):
        raise ValueError('duplicate frozen request idx')
    by_idx = {}
    for value in outcomes:
        row = _dict(value)
        idx = _request_index(row, indices)
        if idx in by_idx:
            raise ValueError('duplicate outcome')
        by_idx[idx] = row
    end = service_started_s + duration_s
    details, ttfts, tpots, terminal_times = [], [], [], []
    window_tokens = tail_tokens = 0
    token_evidence_complete = True
    unresolved = 0
    for position, req in enumerate(trace):
        idx = req.get('idx', position)
        if not _finite(req['arrival_s']) or not 0 <= req['arrival_s'] < duration_s:
            raise ValueError('request arrival outside frozen service window')
        maximum = req['max_tokens']
        if type(maximum) is not int or maximum < 2:
            raise ValueError('comparison requires at least two requested tokens')
        row = by_idx.get(idx, dict(error='missing_outcome'))
        scheduled = service_started_s + req['arrival_s']
        if row.get('scheduled_s') is not None and (not _finite(row['scheduled_s'])
                or not math.isclose(row['scheduled_s'], scheduled, abs_tol=1e-6, rel_tol=0)):
            raise ValueError('scheduled clock differs from frozen arrival')
        n = row.get('completion_tokens')
        first, last, terminal = (row.get(k) for k in ('first_token_s', 'last_token_s', 'terminal_s'))
        outcome_end = terminal
        if not _finite(outcome_end):
            outcome_end = row.get('finished_s')
        if _finite(outcome_end) and outcome_end >= scheduled:
            terminal_times.append(outcome_end)
        else:
            unresolved += 1
        timing = (all(_finite(v) for v in (first, last, terminal))
                  and scheduled <= first <= last <= terminal)
        full = (not row.get('error') and row.get('terminal') is True
                and row.get('ok') is not False and row.get('correct') is not False
                and type(n) is int and n == maximum)
        success = bool(full and timing)
        ttft = first - scheduled if success else None
        tpot = (last - first) / (n - 1) if success else None
        good = success and ttft <= slo[0] and tpot <= slo[1]
        in_window = success and terminal <= end
        if success:
            ttfts.append(ttft); tpots.append(tpot)
        delivered = (token_events or {}).get(idx, row.get('token_events'))
        complete = row.get('token_events_complete') is True
        if token_events is not None and idx in token_events:
            complete = True
        total = 0
        previous = scheduled
        row_window_tokens = row_tail_tokens = 0
        for event in delivered or []:
            stamp, count = event.get('received_s'), event.get('count')
            if (not _finite(stamp) or stamp < previous or type(count) is not int or count <= 0):
                raise ValueError('invalid client token delivery event')
            previous = stamp
            total += count
            complete = complete and event.get('exact', True) is True
            if service_started_s <= stamp <= end:
                row_window_tokens += count
            elif stamp > end:
                row_tail_tokens += count
        complete = complete and type(n) is int and total == n
        token_evidence_complete &= complete
        if complete:
            window_tokens += row_window_tokens; tail_tokens += row_tail_tokens
        details.append(dict(idx=idx, successful=success, joint_slo=bool(good),
            completed_in_window=bool(in_window), window_good=bool(good and in_window),
            ttft_s=ttft, tpot_s=tpot, completion_tokens=n,
            native_reported_completion_tokens=row.get('native_reported_completion_tokens'),
            token_events_complete=bool(complete),
            timing_complete=bool(timing),
            timeout=bool(row.get('timeout') or 'timeout' in str(row.get('error', '')).lower()
                         or 'deadline exceeded' in str(row.get('error', '')).lower()),
            terminal_after_window=bool(_finite(outcome_end) and outcome_end > end),
            pending_at_window_end=not _finite(outcome_end) or outcome_end > end,
            error=row.get('error') or (None if success else 'incomplete_output_or_timing')))
    offered = len(trace)
    successful = sum(r['successful'] for r in details)
    good = sum(r['joint_slo'] for r in details)
    window_success = sum(r['completed_in_window'] for r in details)
    window_good = sum(r['window_good'] for r in details)
    output_tokens = sum(r['completion_tokens'] for r in details if r['successful'])
    good_tokens = sum(r['completion_tokens'] for r in details if r['joint_slo'])
    window_good_tokens = sum(r['completion_tokens'] for r in details if r['window_good'])
    horizon = max([end, *terminal_times,
                   *([observed_until_s] if unresolved and observed_until_s is not None else [])])
    elapsed = horizon - service_started_s if not unresolved or observed_until_s is not None else None
    result = dict(measurement_protocol_version=PROTOCOL, service_started_s=service_started_s,
        duration_s=duration_s, service_finished_s=end, offered_requests=offered,
        successful_requests=successful, failed_requests=offered-successful,
        timeout_requests=sum(r['timeout'] for r in details),
        invalid_timing_requests=sum(not r['timing_complete'] for r in details),
        unresolved_requests=unresolved, output_tokens=output_tokens,
        joint_slo_requests=good, good_output_tokens=good_tokens,
        success_rate=safe_ratio(successful, offered), joint_slo_rate=safe_ratio(good, offered),
        window_successful_requests=window_success, window_good_requests=window_good,
        window_good_output_tokens=window_good_tokens,
        window_delivered_tokens=window_tokens if token_evidence_complete else None,
        tail_delivered_tokens=tail_tokens if token_evidence_complete else None,
        throughput_request_s=window_success/duration_s,
        throughput_token_s=window_tokens/duration_s if token_evidence_complete else None,
        goodput_request_s=window_good/duration_s, goodput_token_s=window_good_tokens/duration_s,
        cohort_good_requests=good, cohort_good_output_tokens=good_tokens,
        cohort_goodput_request_s=safe_ratio(good, elapsed),
        cohort_goodput_token_s=safe_ratio(good_tokens, elapsed),
        cohort_throughput_request_s=safe_ratio(successful, elapsed),
        cohort_throughput_token_s=safe_ratio(output_tokens, elapsed),
        request_elapsed_s=elapsed, request_tail_s=max(0., elapsed-duration_s) if elapsed is not None else None,
        tail_completed_requests=sum(r['terminal_after_window'] for r in details),
        pending_at_window_end_requests=sum(r['pending_at_window_end'] for r in details),
        token_timing_complete=bool(token_evidence_complete),
        slo_ttft_s=slo[0], slo_tpot_s=slo[1], request_metrics=details,
        **_stats(ttfts, 'ttft'), **_stats(tpots, 'tpot'))
    result['slo_pass'] = bool(offered and successful == offered and good/offered >= .9
        and result['ttft_p99_s'] <= slo[0] and result['tpot_p99_s'] <= slo[1])
    return result


def safe_ratio(numerator, denominator):
    """Missing or zero denominator stays missing, never an invented zero."""
    if not _finite(numerator) or not _finite(denominator) or denominator <= 0:
        return None
    return numerator / denominator
