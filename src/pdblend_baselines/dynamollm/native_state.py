"""Independent validation of native scheduler and KV ownership evidence.

Frozen from pdblend.online.native_control on 2026-09-25. This module has no
PDblend controller, planner, policy, or engine imports.
"""
from __future__ import annotations


class NativeControlError(RuntimeError):
    pass


def validate_state(value: dict, *, generation: int, tp: int, pp: int = 1,
                   drained: bool = False, request_id: str | None = None,
                   observed_after_s: float | None = None) -> dict:
    """Never substitute proxy counts or HTTP success for native evidence."""
    if (not isinstance(value, dict) or value.get('generation') != generation
            or (value.get('tp'), value.get('pp')) != (tp, pp)
            or value.get('native_evidence_complete') is not True
            or value.get('transport_healthy') is not True):
        raise NativeControlError('native identity, generation or transport evidence missing')
    if observed_after_s is not None and value.get('native_at_s', 0) < observed_after_s:
        raise NativeControlError('stale native scheduler evidence')
    ranks = value.get('ranks')
    if (not isinstance(ranks, list) or len(ranks) != tp * pp
            or {r.get('rank') for r in ranks} != set(range(tp * pp))
            or any(r.get('generation') != generation or r.get('native_evidence_complete') is not True
                   or r.get('healthy') is not True for r in ranks)):
        raise NativeControlError('missing or stale native rank evidence')
    for field in ('all_queue', 'running', 'waiting', 'retained_kv_requests'):
        if not isinstance(value.get(field), list):
            raise NativeControlError('native inventory missing: ' + field)
        if drained and value[field] or request_id is not None and request_id in value[field]:
            raise NativeControlError('native request still owns ' + field)
    if (value.get('pending_transfers') != 0 or value.get('transfer_allocations') != {}
            or any(r.get('pending_transfers') != 0 or r.get('transfer_allocations') != {} for r in ranks)):
        raise NativeControlError('native transfers have not been released')
    allocations = value.get('kv_allocations')
    if not isinstance(allocations, dict):
        raise NativeControlError('native KV inventory missing')
    if drained:
        total, free, reserved = (value.get(k) for k in ('total_blocks', 'free_blocks', 'reserved_blocks'))
        if (allocations or any(type(x) is not int for x in (total, free, reserved))
                or total <= 0 or reserved < 0 or free < total - reserved):
            raise NativeControlError('native KV blocks not fully returned')
    elif request_id is not None and request_id in allocations:
        raise NativeControlError('cancelled request retains KV blocks')
    return value

