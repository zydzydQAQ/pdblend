"""Independent Mixed policy over the shared native engine transport.

This runner owns neither GPU leases nor model lifecycle. It can therefore be
used for capacity calibration and for a resident comparison without importing
any of the other systems' planners or profiles.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import aiohttp

from pdblend_baselines.mixed_policy import MixedLeastLoadPolicy, MixedReplica
from .client import Request, nearest_rank


async def generate(session, url, payload):
    events = []
    async with session.post(url+'/baseline/generate', json=payload) as response:
        if response.status != 200:
            raise RuntimeError(f'native generate {response.status}: {(await response.text())[:1000]}')
        async for line in response.content:
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                break
            event = json.loads(data)
            event['received_s'] = time.time()
            events.append(event)
    if not events or events[-1].get('finished') is not True:
        raise RuntimeError('native generation has no terminal event')
    return dict(events=events, token_ids=[t for e in events for t in e['token_ids']])


def metrics(outcomes: list[dict], offered: int, slo: tuple[float, float]) -> dict:
    if offered <= 0 or len(outcomes) != offered:
        raise ValueError('one terminal outcome is required per offered request')
    ttfts, tpots, good, tokens, good_tokens = [], [], 0, 0, 0
    for row in outcomes:
        ok = row.get('correct') is True
        n = int(row.get('completion_tokens', 0))
        tokens += n
        if ok:
            a, b = row.get('ttft_s'), row.get('tpot_s')
            if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in (a, b)):
                raise ValueError('successful request has invalid timing')
            ttfts.append(a); tpots.append(b)
            if a <= slo[0] and b <= slo[1]:
                good += 1; good_tokens += n
    return dict(offered=offered, correct=len(ttfts), success_rate=len(ttfts)/offered,
                joint_slo_rate=good/offered, output_tokens=tokens, good_output_tokens=good_tokens,
                ttft_p99_s=nearest_rank(ttfts, .99), tpot_p99_s=nearest_rank(tpots, .99),
                slo_ttft_s=slo[0], slo_tpot_s=slo[1],
                passed=len(ttfts) == offered and good/offered >= .9
                    and bool(ttfts) and nearest_rank(ttfts, .99) <= slo[0]
                    and nearest_rank(tpots, .99) <= slo[1])


async def execute(specs, trace: list[Request], out: Path, *, duration_s: float,
                  slo: tuple[float, float], seed: int, request_timeout_s: float = 180.) -> dict:
    if (not specs or len({s.tp for s in specs}) != 1 or not trace
            or duration_s <= 0 or any(not 0 <= r.arrival_s < duration_s for r in trace)):
        raise ValueError('Mixed requires fixed TP and a nonempty bounded trace')
    out.mkdir(parents=True, exist_ok=False)
    (out/'requests.json').write_text(json.dumps(dict(seed=seed, duration_s=duration_s,
        requests=[asdict(r) for r in trace]), sort_keys=True)+'\n')
    replicas = [MixedReplica(s.instance_id, s.tp, max_num_seqs=s.max_num_seqs) for s in specs]
    urls = {s.instance_id:s.base_url for s in specs}
    policy = MixedLeastLoadPolicy(specs[0].tp)
    start = time.monotonic(); started_s = time.time()
    events = (out/'events.jsonl').open('x')
    outcomes = (out/'outcomes.jsonl').open('x')
    def emit(handle, row):
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n'); handle.flush()
    rows = []
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=request_timeout_s)) as session:
            async def submit(r):
                await asyncio.sleep(max(0., start+r.arrival_s-time.monotonic()))
                rid = f'mixed-{seed}-{r.idx}'
                # Latency includes client/event-loop delay from the scheduled
                # arrival, so overload cannot disappear through coordinated omission.
                row = dict(idx=r.idx, request_id=rid, arrival_s=r.arrival_s,
                    scheduled_s=started_s+r.arrival_s, submitted_s=time.time(),
                    input_tokens=len(r.prompt), max_tokens=r.max_tokens, completion_tokens=0,
                    sampling_seed=seed, correct=False)
                route = policy.route(rid, replicas)
                try:
                    if route is None:
                        raise RuntimeError('no accepting Mixed replica')
                    row['instance_id'] = route.instance_id
                    emit(events, dict(event='route', at_s=time.time(), **asdict(route)))
                    result = await generate(session, urls[route.instance_id], dict(
                        request_id=rid, prompt=r.prompt, max_tokens=r.max_tokens,
                        seed=seed, temperature=0, ignore_eos=True))
                    stamp_tokens = [(event['received_s'], token) for event in result['events']
                                    for token in event['token_ids']]
                    # Native events retain client receipt timestamps; timestamps
                    # are not reconstructed from the total request duration.
                    row.update(result)
                    row['completion_tokens'] = len(stamp_tokens)
                    if stamp_tokens:
                        row['ttft_s'] = stamp_tokens[0][0]-row['scheduled_s']
                        row['tpot_s'] = ((stamp_tokens[-1][0]-stamp_tokens[0][0])/(len(stamp_tokens)-1)
                                         if len(stamp_tokens)>1 else 0.)
                    row['correct'] = len(stamp_tokens) == r.max_tokens and result['events'][-1].get('finished') is True
                except Exception as exc:
                    row['error'] = f'{type(exc).__name__}: {exc}'
                    if route is not None:
                        try:
                            async with session.post(urls[route.instance_id]+'/baseline/cancel',
                                                    json=dict(request_id=rid)) as response:
                                row['cancel'] = dict(status=response.status, body=await response.json())
                        except Exception as cancel_exc:
                            row['cancel_error'] = repr(cancel_exc)
                finally:
                    if route is not None:
                        policy.complete(route.instance_id, replicas)
                        emit(events, dict(event='release', request_id=rid, at_s=time.time(),
                            active={r.instance_id:r.active_requests for r in replicas}))
                row['finished_s'] = time.time()
                emit(outcomes, row)
                return row
            rows = await asyncio.gather(*(submit(r) for r in trace))
        await asyncio.sleep(max(0., start+duration_s-time.monotonic()))
    finally:
        events.close(); outcomes.close()
    return dict(system='mixed', native_runner=True, started_s=started_s, finished_s=time.time(),
                duration_s=duration_s, seed=seed, metrics=metrics(rows, len(trace), slo),
                counts_reclaimed=not any(r.active_requests for r in replicas),
                routing_policy='independent_least_load_fixed_tp')
