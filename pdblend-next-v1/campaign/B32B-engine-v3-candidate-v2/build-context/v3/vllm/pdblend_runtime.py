"""Versioned scheduler controls and explicit request-scoped transfer identities.

Importable without torch, for protocol tests. Peers are configured at startup;
request IDs never contain network addresses or filesystem paths.
"""
import json
import os
import re
from collections import deque

from .pdblend_budget import apply_budget

_ID = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def transfer_id(nonce, phase, source, target):
    if phase not in ("p", "d") or not all(_ID.fullmatch(x) for x in (nonce, source, target)):
        raise ValueError("invalid transfer identity")
    return ":".join(("pdb", nonce, phase, source, target))


def parse_transfer(request_id):
    fields = str(request_id).split(":")
    if len(fields) != 5 or fields[0] != "pdb":
        return None
    _, nonce, phase, source, target = fields
    if transfer_id(nonce, phase, source, target) != request_id:
        raise ValueError("invalid transfer identity")
    return dict(nonce=nonce, phase=phase, source=source, target=target)


def transfer_phase(model_input, phase):
    return any((parse_transfer(rid) or {}).get("phase") == phase
               for rid in (model_input.request_ids_to_seq_ids or {}))


def read_runtime(scheduler, *, force_refresh=False):
    """Observe controls and apply budgets only on the scheduler owner thread."""
    path = os.environ.get("PDBLEND_RUNTIME_PATH")
    if not path:
        return None
    try:
        if os.environ.get("PDBLEND_ASYNC_IO") == "1":
            from .pdblend_io import initialize_engine_io
            io = initialize_engine_io(scheduler)
            cache = io.caches.get("runtime")
            if cache is None or cache.path != path:
                raise ValueError("runtime control path changed after startup")
            payload, error, checked_s = (cache.force_refresh() if force_refresh else cache.read())
            scheduler._pdblend_control_checked_s = checked_s
            if error or payload is None:
                raise ValueError(error or "runtime control not observed")
        else:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            from .pdblend_io import _validate
            _validate(payload, "runtime")
        apply_budget(scheduler, payload)
        scheduler._pdblend_runtime_error = None
        return payload
    except (OSError, ValueError, KeyError, TypeError) as exc:
        scheduler._pdblend_runtime_error = str(exc)
        # Keep the last applied chunking mode while draining. A reduced token
        # budget must never turn a long accepted prompt into an ignored prompt.
        applied = getattr(scheduler, "_pdblend_applied_runtime", {})
        if "scheduler_budget" in applied:
            return dict(applied, generation=-1, admit_prefill=False)
        return dict(generation=-1, role="mixed", mode="temporal", admit_prefill=False)


def schedule(scheduler, runtime):
    chunked = runtime["role"] == "mixed" and runtime["mode"] == "continuous"
    scheduler.scheduler_config.chunked_prefill_enabled = chunked
    hidden={}
    pending = getattr(scheduler, "_pdblend_budget_pending", None)
    if not runtime["admit_prefill"] or pending:
        hidden['waiting']=scheduler.waiting
        scheduler.waiting=deque()
    if pending:
        hidden['swapped'] = scheduler.swapped
        scheduler.swapped = deque()
    if not runtime.get('admit_decode',True):
        # Profiling can retain existing KV while admitting additional prefills,
        # then release an actual decode batch. No request or KV is fabricated.
        for attr in ('running','swapped'):
            hidden.setdefault(attr, getattr(scheduler,attr))
            setattr(scheduler,attr,deque())
    try:
        return (scheduler._schedule_chunked_prefill() if chunked and (runtime["admit_prefill"] or pending
                    or "scheduler_budget" in runtime)
                else scheduler._schedule_default())
    finally:
        for attr,original in hidden.items():
            original.extend(getattr(scheduler,attr))
            setattr(scheduler,attr,original)


def transfer_state(worker, cancel_nonce=None):
    """Executor RPC: observe or cancel request-scoped transport on every rank."""
    from vllm.distributed.kv_transfer import has_kv_transfer_group, get_kv_transfer_group
    if not has_kv_transfer_group():
        return dict(buffered_tensors=0, inflight_receives=0, listener_alive=True)
    return get_kv_transfer_group().transport.request_state(cancel_nonce)


def prepare_peers(worker, peer_ids):
    """Warm resident communication before the measured serving window."""
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    connector = get_kv_transfer_group()
    for peer_id in peer_ids:
        peer = connector.peers[peer_id]
        for rank in range(int(peer['tp'])):
            address = '%s:%d' % (peer['host'],int(peer['kv_port'])+rank)
            connector.transport._create_connect(address)
    return dict(peers=list(peer_ids),rank=connector.rank)


def register_peer(worker,peer_id,peer):
    """New versioned peer IDs prevent reuse of a dead process's NCCL handle."""
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    connector=get_kv_transfer_group()
    if peer_id in connector.peers and connector.peers[peer_id]!=peer:
        raise ValueError('peer identity collision')
    connector.peers[peer_id]=dict(peer)
    return dict(id=peer_id,rank=connector.rank)
