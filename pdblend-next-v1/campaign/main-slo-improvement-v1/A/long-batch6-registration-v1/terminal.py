"""Strict terminal empty-event reader, adapted only to actual A budget2048."""
import copy,hashlib,json,math
def require(ok,why):
    if not ok:raise ValueError(why)
def finite(value):return type(value) in (int,float) and math.isfinite(value)

def trailing_empty(raw, events):
    """Reject every alteration except one complete, terminal, zero-work event.

    All timestamps and identities are checked before removing the last row.
    Empty internal rows are rejected: separated decode runs cannot be joined.
    The original complete event file/hash and empty row remain in the report.
    """
    require(len(events) >= 66, 'complete nonempty sequence and terminal empty event required')
    ids = {r['request_id'] for r in raw['requests']}
    generation = raw['runtime_before']['generation']
    previous = None
    empty = []
    for index, event in enumerate(events):
        owned, pf, dc, tokens = (event.get(k) for k in ('request_ids', 'prefill', 'decode', 'tokens'))
        start, end = event.get('started_s'), event.get('finished_s')
        require(isinstance(owned, list) and len(owned) == len(set(owned)) and set(owned) <= ids,
                'foreign or duplicate actual owner IDs')
        require(all(type(v) is int and v >= 0 for v in (pf, dc, tokens))
                and pf + dc == len(owned) and tokens <= 2048, 'invalid owner phase/token counts')
        require(event.get('generation') == generation and event.get('role') == 'mixed'
                and event.get('mode') == 'continuous', 'owner generation/role/mode changed')
        require(finite(start) and finite(end) and raw['measurement_start_s'] <= start < end <= raw['measurement_end_s']
                and (previous is None or start >= previous - 1e-6), 'owner event chronology differs')
        previous = end
        if not owned:
            require(pf == dc == tokens == 0, 'empty owner step contains work')
            empty.append(index)
        else:
            require(tokens > 0 and ((pf > 0 and tokens > dc) or (pf == 0 and tokens == dc)),
                    'nonempty owner phase/token work differs')
    require(empty == [len(events) - 1], 'exactly one trailing empty event required; internal empty step is unsupported')
    last = events[-1]
    require(all(r.get('success') is True and r.get('done_marker') is True for r in raw['requests']),
            'empty tail cannot qualify incomplete requests')
    return events[:-1], dict(index=len(events) - 1, event=copy.deepcopy(last),
        canonical_sha256=hashlib.sha256(json.dumps(last, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        semantics='zero model work; retained in original whole-batch and whole-operation energy',
        internal_empty_steps_accepted=False, original_event_indices_preserved=True)
