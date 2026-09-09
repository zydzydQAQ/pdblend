"""Evaluation-v3 request clocks and completion recovery, never SLO relabeling."""
import ipaddress
import math
from dataclasses import replace

from aiohttp import web


PROTOCOL = 'evaluation-v3'


def benchmark_timing(request, config, received_s):
    """Only the explicitly opted-in loopback benchmark can supply its clock."""
    planned = received_s
    dispatch = None
    supplied = request.headers.get('X-PDBlend-Planned-Arrival-S')
    if supplied is not None:
        peer = request.transport.get_extra_info('peername') if request.transport else None
        try:
            address = ipaddress.ip_address(peer[0]) if peer else None
            loopback = bool(address and (address.is_loopback or
                (getattr(address, 'ipv4_mapped', None) and address.ipv4_mapped.is_loopback)))
        except (ValueError, TypeError):
            loopback = False
        if (config.get('evaluation_protocol') != PROTOCOL or not loopback or
                request.headers.get('X-PDBlend-Evaluation-Protocol') != PROTOCOL):
            raise web.HTTPBadRequest(text='planned arrival requires the opted-in loopback benchmark')
        try:
            planned = float(supplied)
            raw_dispatch = request.headers.get('X-PDBlend-Actual-Dispatch-S')
            dispatch = float(raw_dispatch) if raw_dispatch is not None else None
            if not math.isfinite(planned) or planned <= 0 or planned > received_s + .005:
                raise ValueError('invalid planned arrival')
            if dispatch is not None and (not math.isfinite(dispatch) or
                    dispatch < planned - .005 or dispatch > received_s + .005):
                raise ValueError('invalid dispatch time')
        except (TypeError, ValueError, OverflowError) as exc:
            raise web.HTTPBadRequest(text='invalid benchmark request clock') from exc
    timeout = float(config.get('request_timeout_s', 120.))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('positive finite hard request timeout required')
    return dict(planned_arrival_s=planned, actual_dispatch_s=dispatch,
                handler_arrival_s=received_s, hard_deadline_s=planned + timeout)


def recovery_budget(request, now, *, only_if_late=False):
    """A temporary planner input; measured and live SLO budgets stay unchanged.

    Expired SLOs do not grant unlimited execution: recovery has the original
    hard deadline. Unexpired existing requests retain their actual SLO guards.
    """
    if only_if_late and request.next_token_remaining(now) >= 0:
        return request
    deadline = request.hard_deadline_s
    if deadline is None or deadline <= now:
        return request
    return replace(request, ttft_s=deadline-request.arrival_s,
                   tpot_s=max(request.tpot_s, deadline-now))


def recovery_snapshot(snapshot, now):
    return replace(snapshot, instances=tuple(replace(instance,
        requests=tuple(recovery_budget(r, now, only_if_late=True) for r in instance.requests))
        for instance in snapshot.instances))


def engine_residual(raw, now, ttl_s=1.):
    """Require explicit, recent empty-engine evidence, including KV ownership."""
    fields = ('active', 'running', 'waiting', 'kv_allocations', 'transfer_allocations',
              'transfer_buffered_tensors', 'transfer_inflight_receives')
    residual = {name:raw.get(name) for name in fields if raw.get(name)}
    missing = [name for name in fields if name not in raw]
    if missing:
        residual['missing_fields'] = missing
    stamp = raw.get('timestamp')
    if not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or not 0 <= now-stamp <= ttl_s:
        residual['stale_timestamp'] = stamp
    transfer_stamp = raw.get('transfer_observed_s')
    if (not isinstance(transfer_stamp, (int, float)) or not math.isfinite(transfer_stamp)
            or not 0 <= now-transfer_stamp <= ttl_s):
        residual['stale_transfer_timestamp'] = transfer_stamp
    if raw.get('transfer_inflight_sends'):
        residual['transfer_inflight_sends'] = raw['transfer_inflight_sends']
    if raw.get('error') or raw.get('runtime_error') or raw.get('transport_healthy') is not True:
        residual['error'] = raw.get('error') or raw.get('runtime_error') or 'transport health unconfirmed'
    if raw.get('generation') != raw.get('acknowledged_generation'):
        residual['generation_unconfirmed'] = True
    return residual
