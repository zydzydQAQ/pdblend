"""No CLI/GPU launcher. Synchronous reference driver for a future reviewed child.

Dependency injection permits actual LLMEngine interfaces to be CPU tested. No
stored token oracle is loaded here: every output comes from engine.step().
"""
import time
from schedule_selection import select_default


def require(ok, message):
    if not ok: raise RuntimeError(message)


def run_reference(engine, requests, params_factory, emit, guard):
    """Two sequential solo preambles, then one native-default matched pair.

    The future operation owns process lifetime and abort/native/clock cleanup;
    this function always restores its record-only schedule wrapper. guard must
    check the real absolute wall AND monotonic child deadline before each call.
    """
    require(len(requests) == 4, 'four predeclared requests')
    require([len(r['body']['prompt']) for r in requests] == [96, 192, 96, 192], 'original inputs')
    require(len({r['request_uuid'] for r in requests}) == 4, 'unique explicit UUIDs')
    for r in requests:
        b = r['body']
        require(b == dict(prompt=[9707, 1879, 13] * (len(b['prompt']) // 3),
            max_tokens=64, temperature=0, top_p=1, seed=0, ignore_eos=True, stream=False),
            'original full64 input/sampling unchanged')
    require(len(engine.scheduler) == 1, 'one actual scheduler owner, TP workers are separate')
    scheduler = engine.scheduler[0]
    require(scheduler.scheduler_config.chunked_prefill_enabled is True, 'original chunked configuration required')
    require(scheduler.scheduler_config.max_num_batched_tokens == 8192 and
            scheduler.scheduler_config.max_num_seqs == 32, 'original budget')
    require(not scheduler.running and not scheduler.waiting and not scheduler.swapped,
            'fresh independent owner idle')
    selection = select_default(engine, emit)
    selection.__enter__()
    original_had_schedule = "schedule" in scheduler.__dict__
    original_instance_schedule = scheduler.__dict__.get("schedule")
    original = scheduler.schedule
    scheduled = []
    outputs = {r['request_uuid']: [] for r in requests}
    finished = set()
    owned = set()

    def record():
        result = original()
        metadata, decision, _ = result
        scheduled.append(dict(prefill=decision.num_prefill_groups,
            decode=len(decision.scheduled_seq_groups) - decision.num_prefill_groups,
            tokens=decision.num_batched_tokens, request_ids=[m.request_id for m in metadata],
            preempted=decision.preempted, blocks_to_swap_in=decision.blocks_to_swap_in,
            blocks_to_swap_out=decision.blocks_to_swap_out, blocks_to_copy=decision.blocks_to_copy))
        return result

    def add(r):
        guard()
        rid, b = r['request_uuid'], r['body']
        owned.add(rid)
        emit(dict(kind='add_intent', request_id=rid, body=b, wall_s=time.time()))
        params = params_factory(temperature=b['temperature'], top_p=b['top_p'],
            max_tokens=b['max_tokens'], ignore_eos=b['ignore_eos'], seed=b['seed'])
        engine.add_request(rid, {'prompt_token_ids': list(b['prompt'])}, params)

    def step(expected):
        guard()
        scheduled.clear()
        started = time.time()
        responses = engine.step()
        require(len(scheduled) == 1, 'exact one actually executed scheduler step')
        event = dict(scheduled[0], started_s=started, finished_s=time.time())
        emit(dict(kind='executed_step', **event))
        got = (event['prefill'], event['decode'], event['tokens'], event['request_ids'])
        require(got == expected, 'actual native prefill/decode trajectory differs')
        require(event['preempted'] == 0 and not any(event[k] for k in
            ('blocks_to_swap_in', 'blocks_to_swap_out', 'blocks_to_copy')), 'unexpected cache movement')
        seen = set()
        for response in responses:
            rid = response.request_id
            require(rid in owned and rid not in seen and rid in event['request_ids'], 'unexpected output UUID')
            seen.add(rid)
            require(len(response.outputs) == 1, 'one greedy sequence only')
            tokens = list(response.outputs[0].token_ids)
            require(tokens[:-1] == outputs[rid] and len(tokens) == len(outputs[rid]) + 1,
                    'one genuine new output per eager native step')
            require(all(type(t) is int and t >= 0 for t in tokens), 'actual token IDs')
            outputs[rid] = tokens
            require(response.finished is (len(tokens) == 64), 'exact full64 finish')
            if response.finished: finished.add(rid)
            emit(dict(kind='output', request_id=rid, token_ids=tokens, finished=response.finished))
        require(seen == set(event['request_ids']), 'no missing scheduled outputs')

    scheduler.schedule = record
    try:
        for r in requests[:2]:
            rid = r['request_uuid']
            add(r)
            step((1, 0, len(r['body']['prompt']), [rid]))
            for _ in range(63): step((0, 1, 1, [rid]))
            require(rid in finished and not engine.has_unfinished_requests(), 'solo preamble must finish')
        first, second = requests[2:]
        a, b = first['request_uuid'], second['request_uuid']
        add(first)
        step((1, 0, 96, [a]))
        for _ in range(4): step((0, 1, 1, [a]))
        require(len(outputs[a]) == 5, 'second arrives after fifth actual output; no sleep substitute')
        add(second)
        step((1, 0, 192, [b]))
        for _ in range(59): step((0, 2, 2, [a, b]))
        require(a in finished and len(outputs[b]) == 60, 'native pair end alignment')
        for _ in range(4): step((0, 1, 1, [b]))
        require(len(finished) == 4 and not engine.has_unfinished_requests(), 'four full64 native terminals')
        return dict(complete=True, token_ids_by_request_uuid=outputs,
                    original_baseline_gate_changed=False, performance_evidence=False)
    finally:
        if original_had_schedule:
            scheduler.schedule = original_instance_schedule
        else:
            del scheduler.schedule
        selection.__exit__(None, None, None)
