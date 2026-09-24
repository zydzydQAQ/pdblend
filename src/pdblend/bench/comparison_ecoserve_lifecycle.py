"""Independent replay of explicitly selected EcoServe wrapper cancellation."""
from __future__ import annotations

from .comparison_acceptance import _need, _finite, _equal

MODE = 'cohort_cancel_and_serial_close/v1'
CONFIG_KEY = 'eco_comparison_lifecycle'


def audit_lifecycle_events(events, native, config):
    """Return exact permitted cancelled-read IDs, never an interval exemption."""
    _need(config.get(CONFIG_KEY) == MODE, 'comparison lifecycle mode is absent')
    summary = native.get('comparison_lifecycle', {})
    requests = {r['request_id']: r for r in native['outcomes']}
    _need(summary.get('mode') == MODE and summary.get('request_tasks') == len(requests)
          and summary.get('policy_changed') is False and summary.get('hardware_qualification') is False
          and summary.get('serial_close_started') is True, 'native lifecycle summary differs')

    def one(kind):
        rows = [r for r in events if r.get('kind') == kind]
        _need(len(rows) == 1 and _finite(rows[0].get('at_s')), 'lifecycle boundary missing/duplicate: ' + kind)
        return rows[0]

    wait = one('eco_comparison_cohort_wait')
    wait_close = one('eco_comparison_close_wait')
    begin_close = one('eco_comparison_close_begin')
    end_close = one('eco_comparison_close_end')
    closed = one('eco_closed')
    service_end = native['service_started_s'] + native['duration_s']
    _need(wait['at_s'] >= service_end and wait.get('timeout_s') == config['request_timeout_s']
          and _finite(wait.get('monotonic_s')) and wait.get('requests') == sorted(requests),
          'lifecycle cohort wait differs from complete offered request set')
    _need(wait_close['at_s'] <= begin_close['at_s'] <= closed['at_s'] <= end_close['at_s']
          and max(r['finished_s'] for r in requests.values()) <= wait_close['at_s']
          and begin_close.get('resize_lock_held') is True and begin_close.get('cohort_tasks_done') is True
          and end_close.get('resize_lock_held') is True and end_close.get('controller_closed') is True
          and end_close.get('runtime_started') is False
          and begin_close.get('cancel_id') == end_close.get('cancel_id') == 'controller-close',
          'lifecycle serial close lacks complete cohort/lock/close evidence')
    # A membership transaction cannot straddle the lock-held close boundary.
    _need(not any(r.get('kind') in ('eco_membership_prepare', 'eco_membership_commit',
                                  'eco_membership_rollback', 'eco_controller_failure')
                  and begin_close['at_s'] <= r.get('at_s', -1) <= end_close['at_s'] for r in events),
          'membership/controller event occurred during locked close')
    causes = {'controller-close': (begin_close, end_close['at_s'], None)}
    cancel = [r for r in events if r.get('kind') == 'eco_comparison_cohort_cancel_begin']
    cancel_end = [r for r in events if r.get('kind') == 'eco_comparison_cohort_cancel_end']
    if cancel:
        _need(len(cancel) == len(cancel_end) == 1, 'cohort cancellation boundary duplicated or incomplete')
        begin, end = cancel[0], cancel_end[0]
        pending = begin.get('pending_requests')
        _need(summary.get('cohort_cancelled') is True and native.get('error') == 'TimeoutError()'
              and begin.get('reason') == 'cohort_timeout' and begin.get('error') == 'TimeoutError()'
              and begin.get('cancel_id') == end.get('cancel_id') == 'cohort-cancel'
              and isinstance(pending, list) and pending == sorted(set(pending)) and pending
              and set(pending) <= set(requests) and end.get('requests') == pending
              and all(_finite(r.get('at_s')) for r in (begin, end))
              and _finite(begin.get('monotonic_s'))
              and begin['monotonic_s'] - wait['monotonic_s'] >= wait['timeout_s']
              and wait['at_s'] <= begin['at_s'] <= end['at_s'] <= wait_close['at_s']
              and all(begin['at_s'] <= requests[r]['finished_s'] <= end['at_s'] for r in pending),
              'cohort cancellation is not a proven wrapper timeout')
        causes['cohort-cancel'] = (begin, end['at_s'], set(pending))
    else:
        _need(not cancel_end and summary.get('cohort_cancelled') is False,
              'unbound cohort cancellation summary/end')

    receipts = {}
    for row in events:
        if row.get('kind') != 'eco_http_receipt':
            continue
        identifier = row.get('lifecycle_http_id')
        _need(isinstance(identifier, str) and identifier.startswith('http-')
              and identifier not in receipts, 'HTTP lifecycle operation ID missing/duplicate')
        _need(row.get('lifecycle_request_id') in requests or row.get('lifecycle_request_id') is None,
              'HTTP operation belongs to an unknown request')
        receipts[identifier] = row
    allowed = set()
    bound = set()
    for cause_id, (begin, end_s, pending_requests) in causes.items():
        operations = begin.get('pending_http')
        _need(isinstance(operations, list), 'cancellation lacks exact pending HTTP operations')
        for op in operations:
            identifier = op.get('http_id')
            _need(identifier in receipts and identifier not in bound, 'pending HTTP operation missing/duplicate')
            bound.add(identifier)
            row = receipts[identifier]
            _need(row.get('lifecycle_cancel_id') == cause_id
                  and row.get('lifecycle_request_id') == op.get('request_id')
                  and all(_equal(row.get(k), op.get(k)) for k in ('instance_id', 'method', 'path', 'body'))
                  and _finite(op.get('entered_s')) and _finite(row.get('started_s'))
                  and _finite(row.get('at_s'))
                  and op['entered_s'] <= row['started_s'] <= begin['at_s'] <= row['at_s'] <= end_s,
                  'HTTP cancellation causal binding differs')
            _need((pending_requests is None and op.get('request_id') is None)
                  or (pending_requests is not None and op.get('request_id') in pending_requests),
                  'HTTP cancellation has no matching wrapper-owned request/task')
            _need(row.get('error') == 'CancelledError()' and row.get('method') == 'GET'
                  and row.get('body') is None
                  and row.get('path', '').split('?', 1)[0] in ('/baseline/state', '/baseline/events'),
                  'wrapper cancellation affected a mutation or non-read error')
            allowed.add(identifier)
    for identifier, row in receipts.items():
        if row.get('lifecycle_cancel_id') is not None:
            _need(identifier in bound, 'HTTP cancellation cites an absent causal operation')
    return dict(mode=MODE, allowed_cancelled_read_ids=sorted(allowed),
                policy_changed=False, independently_replayed=True)


def bound_read_cancel(row, proof):
    return bool(proof and proof.get('independently_replayed') is True
                and row.get('lifecycle_http_id') in proof['allowed_cancelled_read_ids'])
