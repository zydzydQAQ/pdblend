"""Work still owned by engines, independent of rolling request history."""
from pdblend.planner.forecast import InFlightWork


def backlog_snapshot(router):
    # Uncertain/cancel-pending executions remain reserved until native ACK.
    active = {r.request_id: r for rows in router.active.values() for r in rows}
    return tuple(InFlightWork(
        request_id=r.request_id, input_tokens=r.input_tokens,
        remaining_output_tokens=max(0, r.max_tokens-r.tokens_so_far),
        waiting_prefill_tokens=r.input_tokens if r.first_token_s is None else 0,
        kv_tokens=r.input_tokens+max(0, r.tokens_so_far),
        branch=r.path, pool_id=r.pool_id) for r in active.values())
