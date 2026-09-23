"""Validation contract shared with the neutral V1 serving bridge.

No V0 scheduler or transfer_state imports are retained. The bridge obtains the
snapshot on the EngineCore owner thread and aggregates every worker rank ACK.
"""
from __future__ import annotations

import math
import time
from .transport import validate_rank_ack


def validate_empty_state(state, *, generation, now=None):
    stamp = state.get('timestamp')
    now = time.time() if now is None else now
    if (type(stamp) not in (int, float) or not math.isfinite(stamp) or not 0 <= now-stamp <= .5
            or state.get('generation') != generation
            or state.get('acknowledged_generation') != generation
            or state.get('evidence_complete') is not True
            or state.get('transport_healthy') is not True
            or state.get('accepting') is not False
            or any(state.get(key) != 0 for key in ('active', 'running', 'waiting'))
            or state.get('kv_allocations') != {} or state.get('transfer_allocations') != {}
            or state.get('free_kv_tokens') != state.get('total_kv_tokens')
            or state.get('total_kv_tokens', 0) <= 0):
        raise RuntimeError('native scheduler/generation/KV drain proof incomplete')
    # total_kv_tokens is allocatable capacity, excluding the V1 null block.
    if 'num_gpu_blocks' in state:
        if (state.get('reserved_blocks') != 1
                or state.get('free_blocks') != state['num_gpu_blocks'] - 1):
            raise RuntimeError('V1 native free block count differs from allocatable capacity')
    return state


def aggregate_drain(state, ranks, *, tp, generation):
    validate_empty_state(state, generation=generation)
    validate_rank_ack({'ranks': ranks}, {'tp': tp, 'generation': generation})
    if any(row.get('drained') is not True or row.get('cuda_synchronized') is not True
           or row.get('active_weight_sessions') != 0 for row in ranks):
        raise RuntimeError('all-rank weight communication drain proof required')
    return dict(drained=True, owner_ack=True, generation=generation, ranks=ranks,
                state=state, timestamp=time.time())
