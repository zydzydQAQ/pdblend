"""Run an immutable seed-701 trace through the independent native P/D runtime.

Services must already be deployed. This command never loads models or leases
GPUs, and does not qualify a complete profile, PP deployment or energy ranking.
"""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import time

from .request_runtime import DistServeRuntime
from .runtime import MappedDistServeTransport
from pdblend.results.journal import CompactJournal, payload_receipt


def result_receipt(value):
    """Keep scheduling/KV receipts without copying the accumulated SSE again."""
    return {item.name:getattr(value,item.name) for item in fields(value)
            if item.name not in ('events','token_ids')}


def load_trace(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or value.get('seed') != 701 or not value.get('requests'):
        raise ValueError('nonempty shared trace with explicit seed 701 required')
    previous = -1.
    for index, row in enumerate(value['requests']):
        arrival, prompt, count = row.get('arrival_s'), row.get('prompt'), row.get('max_tokens')
        if (type(arrival) not in (int, float) or not math.isfinite(arrival) or arrival < max(0., previous)
                or not isinstance(prompt, list) or not prompt or any(type(token) is not int or token < 0 for token in prompt)
                or type(count) is not int or not 1 <= count <= 512 or len(prompt)+count > 8192):
            raise ValueError('invalid shared trace request: '+str(index))
        previous = arrival
    return value


async def execute(args):
    trace = load_trace(args.trace)
    duration = getattr(args, 'duration', None)
    if duration is not None and (type(duration) not in (int, float)
            or not math.isfinite(duration) or duration <= 0
            or trace['requests'][-1]['arrival_s'] >= duration):
        raise ValueError('positive observation duration must contain all trace arrivals')
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out/name).exists() for name in ('events.jsonl', 'events.jsonl.gz', 'completion.json')):
        raise FileExistsError('refusing to overwrite DistServe execution evidence')
    raw = CompactJournal(args.out/'events.jsonl.gz')
    def journal(event, **fields):
        raw.write(dict(event=event, **fields))
    transport = MappedDistServeTransport(args.prefill_url, args.decode_url,
        prefill_address=args.prefill_address, decode_address=args.decode_address)
    runtime = DistServeRuntime(transport, tp=args.tp, pp=args.pp, max_batch_size=args.max_batch_size,
        request_timeout_s=args.request_timeout, journal=journal)
    result = dict(system='distserve', seed=701, tp=args.tp, pp=args.pp, status='running', complete=False,
        scope='independent_native_request_pipeline', formal_eligible=False, energy_comparable=False,
        complete_reproduction=False, gpu_batch_equivalence_qualified=False,
        pp_pipeline_qualified=False, offline_topology_search_qualified=False,
        request_golden_comparison_performed=False,
        requested_duration_s=duration,
        trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(), started_s=time.time(), outcomes=[])
    tasks, cleanup = [], []
    try:
        caps = await runtime.start(); result['capabilities'] = caps
        for key in ('model_id', 'tokenizer_hash'):
            if key in trace and trace[key] != caps['P'][key]:
                raise ValueError('trace/deployment '+key+' differs')
        result['trace_model_identity_complete'] = all(key in trace for key in ('model_id', 'tokenizer_hash'))
        start = time.monotonic()
        result['service_window_started_s'] = time.time()
        journal('distserve_service_window_start', at_s=result['service_window_started_s'],
                duration_s=duration, seed=701)
        async def request(index, row):
            await asyncio.sleep(max(0., start+row['arrival_s']-time.monotonic()))
            rid = 'distserve-701-'+str(index)
            outcome = dict(request_id=rid, arrival_s=row['arrival_s'], input_tokens=len(row['prompt']),
                           output_tokens=row['max_tokens'], submitted_s=time.time(), events=[])
            try:
                async for event in runtime.handle(dict(row, seed=701, ignore_eos=True, temperature=0), rid):
                    outcome['events'].append(event)
                value = runtime.results[rid]
                steps = {item['step'] for item in value.receipts}
                expected = ({'prefill', 'release'} if row['max_tokens'] == 1 else
                            {'prefill', 'expect_load', 'transfer', 'load_ack', 'release'})
                terminal = bool(outcome['events']) and outcome['events'][-1].get('finished') is True
                outcome.update(ok=value.status == 'completed' and value.tokens == row['max_tokens']
                               and terminal and steps == expected,
                               terminal_observed=terminal, native_receipts_complete=steps == expected,
                               result=result_receipt(value))
            except Exception as exc:
                outcome.update(ok=False, error=repr(exc))
                if rid in runtime.results: outcome['result'] = result_receipt(runtime.results[rid])
            outcome['finished_s'] = time.time()
            arrivals = [event['received_s'] for event in outcome['events'] for token in event['token_ids']]
            outcome['ttft_s'] = arrivals[0]-outcome['submitted_s'] if arrivals else None
            outcome['tpot_s'] = ((arrivals[-1]-arrivals[0])/(len(arrivals)-1) if len(arrivals)>1 else None)
            outcome.update(payload_receipt(outcome.pop('events'),journal_path='events.jsonl.gz',request_id=rid))
            if 'result' in outcome:
                outcome['result'].pop('events',None);outcome['result'].pop('token_ids',None)
            result['outcomes'].append(outcome)
        tasks = [asyncio.create_task(request(index, row)) for index, row in enumerate(trace['requests'])]
        await asyncio.gather(*tasks)
        if duration is not None:
            await asyncio.sleep(max(0., start+duration-time.monotonic()))
        result.update(service_window_finished_s=time.time(), observed_duration_s=time.monotonic()-start)
        journal('distserve_service_window_end', at_s=result['service_window_finished_s'],
                elapsed_s=result['observed_duration_s'])
        result.update(status='passed' if all(row['ok'] for row in result['outcomes']) else 'failed',
                      complete=len(result['outcomes']) == len(trace['requests']) and all(row['ok'] for row in result['outcomes']))
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
    finally:
        for task in tasks:
            if not task.done(): task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if runtime.ready:
            try: await runtime.close()
            except Exception as exc: cleanup.append(repr(exc))
        if cleanup: result.update(status='failed', complete=False)
        result.update(cleanup_errors=cleanup, quarantined_roles=sorted(runtime.quarantined), finished_s=time.time())
        raw.close()
        result['journal_path']='events.jsonl.gz'
        result['events_sha256'] = hashlib.sha256((args.out/'events.jsonl.gz').read_bytes()).hexdigest()
        (args.out/'trace.json').write_text(json.dumps(trace, allow_nan=False)+'\n')
        (args.out/'completion.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('prefill-url', 'decode-url', 'prefill-address', 'decode-address'):
        parser.add_argument('--'+field, required=True)
    parser.add_argument('--tp', type=int, required=True); parser.add_argument('--pp', type=int, default=1)
    parser.add_argument('--trace', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-batch-size', type=int, default=8)
    parser.add_argument('--request-timeout', type=float, default=180.)
    parser.add_argument('--duration', type=float,
                        help='keep the real observation window open after the last request')
    args = parser.parse_args(argv)
    result = asyncio.run(execute(args))
    print(json.dumps({key:result.get(key) for key in ('status', 'complete', 'error', 'cleanup_errors')}))
    return 0 if result['complete'] else 2


if __name__ == '__main__': raise SystemExit(main())
