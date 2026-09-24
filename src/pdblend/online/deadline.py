"""Conservative proxy queue timing; this is not native scheduler telemetry."""
from __future__ import annotations

import heapq
import math


def mixed_queue_prediction(records, *, model, frequency, max_num_seqs,
                           input_tokens, max_tokens, now):
    """Model running slots separately from FIFO requests awaiting a first token.

    A first token is only a proxy for native execution: preemption is not
    observable here. Keep the provenance explicit, and refuse inconsistent
    proxy states instead of dropping excess requests by clamping the batch.
    """
    records = list(records)
    running, waiting = [], []
    for record in records:
        started = (record.first_token_s if record.path == 'M'
                   else getattr(record, 'first_decode_token_s', None))
        (running if started is not None else waiting).append(record)
    if len(running) > max_num_seqs:
        raise ValueError('observed_proxy_queue: running ownership exceeds scheduler slots')
    batch = min(max_num_seqs, max(1, len(records) + int(input_tokens is not None)))
    context = max([1] + [r.input_tokens + r.max_tokens for r in records]
                  + ([] if input_tokens is None else [input_tokens + max_tokens]))
    supported = getattr(model, 'decode_supported', None)
    if not callable(supported) or not supported(batch, context, frequency):
        raise ValueError('observed_proxy_queue: decode query outside measured coverage')
    step = model.step_seconds(batch, context, frequency)
    if not math.isfinite(step) or step < 0:
        raise ValueError('observed_proxy_queue: invalid step prediction')
    slots = [max(0, r.max_tokens - r.tokens_so_far) * step for r in running]
    slots.extend([0.] * (max_num_seqs - len(slots)))
    heapq.heapify(slots)
    prefill_cursor = 0.
    first_tokens = {}

    def enqueue(prompt, output):
        nonlocal prefill_cursor, slots
        slot_ready = heapq.heappop(slots)
        start = max(slot_ready, prefill_cursor)
        prefill = model.prefill_seconds(prompt, frequency)
        if not math.isfinite(prefill) or prefill < 0:
            raise ValueError('observed_proxy_queue: invalid prefill prediction')
        # Mixed prefill also pauses existing decodes. Do not count its work
        # as parallel execution in the other virtual scheduler slots.
        slots = [release + prefill if release > start else release for release in slots]
        heapq.heapify(slots)
        prefill_cursor = start + prefill
        first = prefill_cursor + step
        heapq.heappush(slots, first + max(0, output - 1) * step)
        return first, start, prefill

    for r in sorted(waiting, key=lambda r: (r.submitted_s, r.request_id)):
        first_tokens[r.request_id] = enqueue(r.input_tokens, r.max_tokens)[0]
    first, wait, prefill = (None, None, None) if input_tokens is None else enqueue(input_tokens, max_tokens)
    return dict(source='observed_proxy_queue', native_scheduler_observed=False,
                running=len(running), waiting=len(waiting), owned=len(records),
                running_batch=batch, context_tokens=context, step_s=step,
                scheduler_wait_s=wait, prefill_s=prefill, ttft_s=first,
                waiting_first_token_s=first_tokens,
                oldest_waiting_age_s=max([max(0., now-r.submitted_s) for r in waiting] or [0.]))
