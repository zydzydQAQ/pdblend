"""Independent DistServe request-pipeline probe on an owned or resident P/D pair.

This exercises the real author admission queues and native KV protocol. It is
not a profiler, an offline topology search, or full GPU-batch reproduction.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from pdblend.bench.gates import random_prompt
from pdblend.engine.launcher import Fleet
from pdblend.bench.metering import Gpus
from pdblend_runtime.probe import NativeSpec, call, generate
from pdblend_runtime.cleanup import cleanup_owned
from .request_runtime import DistServeRuntime
from .runtime import MappedDistServeTransport


def specifications(model, gpus, tp, base_port):
    if tp not in (1, 2, 4) or len(gpus) != 2*tp or len(set(gpus)) != len(gpus):
        raise ValueError('two disjoint symmetric TP groups required')
    return [NativeSpec('distserve-'+role, tuple(gpus[i*tp:(i+1)*tp]), base_port+16*i,
            model, tp=tp, max_num_seqs=32, extra_args=('--enforce-eager',))
            for i, role in enumerate(('P', 'D'))]


def trace_rows():
    return [dict(request_id='distserve-701-'+name, phase=phase, prompt=random_prompt(length, 701+length),
                 max_tokens=count, seed=701, temperature=0., ignore_eos=True)
            for name, phase, length, count in (
                ('512', 'concurrent', 512, 16), ('2048', 'concurrent', 2048, 16),
                ('7168', 'long', 7168, 16), ('one', 'single_token', 512, 1),
                ('cancel', 'cancel', 512, 512), ('recover', 'recovery', 512, 16))]


async def run_resident(prefill_url, decode_url, prefill_address, decode_address, tp, out):
    """Reuse existing native services without loading, stopping or profiling them."""
    import aiohttp
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if any((out/name).exists() for name in ('runtime.json', 'events.jsonl', 'trace.json')):
        raise FileExistsError('refusing to overwrite DistServe GPU probe evidence')
    raw = (out/'events.jsonl').open('x')
    journal_rows = []
    def journal(event, **fields):
        row = dict(event=event, **fields); journal_rows.append(row)
        raw.write(json.dumps(row, allow_nan=False)+'\n'); raw.flush()
    transport = MappedDistServeTransport(prefill_url, decode_url,
        prefill_address=prefill_address, decode_address=decode_address)
    runtime = DistServeRuntime(transport, tp=tp, pp=1, max_batch_size=2,
                              request_timeout_s=90., poll_s=.01, journal=journal)
    result = dict(system='distserve', seed=701, tp=tp, pp=1, status='running', complete=False,
        scope='independent_native_request_pipeline', formal_eligible=False, energy_comparable=False,
        complete_reproduction=False, gpu_batch_equivalence_qualified=False,
        pp_pipeline_qualified=False, offline_topology_search_qualified=False,
        recovery_scope='real_cancellation_only', profiles_collected=False, outcomes={}, references={})
    active = []; cleanup = []
    rows = trace_rows()
    try:
        caps = await runtime.start(); result['capabilities'] = caps
        trace = dict(seed=701, model_id=caps['P']['model_id'], tokenizer_hash=caps['P']['tokenizer_hash'], requests=rows)
        (out/'trace.json').write_text(json.dumps(trace, allow_nan=False)+'\n')
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
            # Exact greedy fixed-work goldens on the same decoder and model.
            for row in rows[:3]:
                rid = row['request_id']
                result['references'][rid] = await generate(session, decode_url,
                    dict(row, request_id='distserve-reference-'+rid))
            repeated = await generate(session, decode_url,
                dict(rows[0], request_id='distserve-reference-repeat'))
            if repeated['token_ids'] != result['references'][rows[0]['request_id']]['token_ids']:
                raise RuntimeError('same-topology deterministic golden differs')
            result['reference_repeat'] = repeated

            async def consume(row):
                rid = row['request_id']; outcome = dict(events=[], submitted_s=time.time())
                result['outcomes'][rid] = outcome
                try:
                    async for event in runtime.handle(row, rid):
                        outcome['events'].append(event)
                    value = runtime.results[rid]
                    outcome.update(result=asdict(value), finished_s=time.time())
                    expected = result['references'].get(rid, result['references'][rows[0]['request_id']])['token_ids'][:row['max_tokens']]
                    if value.status != 'completed' or value.token_ids != expected or not outcome['events'][-1]['finished']:
                        raise RuntimeError('full request output does not match independent golden: '+rid)
                    expected_steps = {'prefill', 'release'} if row['max_tokens'] == 1 else {'prefill', 'expect_load', 'transfer', 'load_ack', 'release'}
                    if {item['step'] for item in value.receipts} != expected_steps:
                        raise RuntimeError('native pipeline receipts incomplete: '+rid)
                    outcome.update(ok=True, golden_match=True)
                except BaseException as exc:
                    outcome.update(error=repr(exc), finished_s=time.time())
                    if rid in runtime.results: outcome['result'] = asdict(runtime.results[rid])
                    raise

            active = [asyncio.create_task(consume(row)) for row in rows[:2]]
            await asyncio.gather(*active)
            first, second = (result['outcomes'][row['request_id']] for row in rows[:2])
            result['concurrent_requests_observed'] = (max(first['submitted_s'], second['submitted_s']) <
                                                      min(first['finished_s'], second['finished_s']))
            if not result['concurrent_requests_observed']:
                raise RuntimeError('submitted requests never overlapped')
            for row in rows[2:4]: await consume(row)

            row = rows[4]; rid = row['request_id']; cancel_events = []; ready = asyncio.Event()
            async def cancel_consumer():
                try:
                    async for event in runtime.handle(row, rid):
                        cancel_events.append(event)
                        if event['token_index'] >= 2: ready.set()
                except Exception as exc:
                    result['cancel_stream_error'] = repr(exc)
            task = asyncio.create_task(cancel_consumer()); active.append(task)
            await asyncio.wait_for(ready.wait(), 30)
            before = runtime._state('D', await transport.state('D'))
            target_id = runtime.contexts[rid]['target_id']
            blocks = before['kv_allocations'].get(target_id)
            if (rid in runtime.results or target_id not in before['all_queue'] or not blocks
                    or any(not group for group in blocks) or cancel_events[-1].get('finished')):
                raise RuntimeError('cancellation lacks actual in-flight decoder KV evidence')
            cancelled = await runtime.cancel(rid)
            await asyncio.wait_for(task, 30)
            result['cancel'] = dict(before=before, target_request_id=target_id,
                acknowledgement=cancelled, events=cancel_events, result=asdict(runtime.results[rid]))
            if (cancelled.get('acknowledged') is not True or cancelled['status'] != 'cancelled'
                    or not {'cancel_D', 'cancel_P'} <= {item['step'] for item in cancelled['receipts']}
                    or not 2 <= runtime.results[rid].tokens < row['max_tokens']):
                raise RuntimeError('real in-flight cancellation acknowledgement incomplete')
            result['recovery'] = await runtime.recover()
            await consume(rows[5])
            result['drain'] = [await call(session, url, '/baseline/drain', dict(timeout_s=15))
                               for url in (prefill_url, decode_url)]
            result['native_events'] = [await call(session, url, '/baseline/events?after_seq=0')
                                      for url in (prefill_url, decode_url)]
        result.update(status='passed', complete=True)
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
    finally:
        for task in active:
            if not task.done(): task.cancel()
        await asyncio.gather(*active, return_exceptions=True)
        if runtime.ready:
            try: await runtime.close()
            except Exception as exc: cleanup.append(repr(exc))
        if cleanup: result.update(status='failed', complete=False)
        result.update(cleanup_errors=cleanup, quarantined_roles=sorted(runtime.quarantined), finished_s=time.time())
        raw.close()
        result['events_sha256'] = hashlib.sha256((out/'events.jsonl').read_bytes()).hexdigest()
        if (out/'trace.json').is_file(): result['trace_sha256'] = hashlib.sha256((out/'trace.json').read_bytes()).hexdigest()
        (out/'runtime.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True); parser.add_argument('--tp', type=int, required=True)
    parser.add_argument('--gpus', required=True); parser.add_argument('--base-port', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--trace', type=Path,
                        help='run the supplied shared seed-701 trace through the native runtime')
    parser.add_argument('--duration', type=float, default=100.)
    args = parser.parse_args(argv)
    gpus = [int(g) for g in args.gpus.split(',')]
    specs = specifications(args.model, gpus, args.tp, args.base_port)
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    if (out/'completion.json').exists(): raise FileExistsError('probe completion already exists')
    result = dict(system='distserve', model=args.model, tp=args.tp, pp=1, seed=701,
        status='running', complete=False, formal_eligible=False, energy_comparable=False,
        scope='independent_native_request_pipeline', profiles_collected=False,
        specs=[asdict(spec) for spec in specs], started_s=time.time(),
        source_sha256=os.environ.get('PDBLEND_SOURCE_SHA256'), image_digest=os.environ.get('PDBLEND_IMAGE_ID'))
    meter = Gpus(gpus); sampler = meter.sampler(interval_s=.1); fleet = Fleet(specs, out/'logs')
    try:
        sampler.start(); result['startup'] = fleet.start_all(timeout_s=600)
        p, d = specs
        if args.trace is None:
            result['runtime'] = asyncio.run(run_resident(p.base_url, d.base_url, p.zmq_address, d.zmq_address,
                                                       args.tp, out/'request-pipeline'))
        else:
            from .run_native import execute
            options = SimpleNamespace(trace=args.trace, out=out/'request-pipeline', tp=args.tp, pp=1,
                max_batch_size=8, request_timeout=180., prefill_url=p.base_url, decode_url=d.base_url,
                prefill_address=p.zmq_address, decode_address=d.zmq_address, duration=args.duration)
            result['runtime'] = asyncio.run(execute(options))
            result.update(shared_trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
                          requested_duration_s=args.duration, request_golden_comparison_performed=False)
        result.update(status=result['runtime']['status'], complete=result['runtime']['complete'])
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
    finally:
        result['cleanup_errors'] = cleanup_owned(fleet, meter, sampler)
        if result['cleanup_errors']: result.update(status='failed', complete=False)
        result.update(finished_s=time.time(), group_energy_j=sampler.total_energy_j(), sampler_error=sampler.error)
        (out/'power.json').write_text(json.dumps(dict(samples=sampler.samples, frequency_samples=sampler.frequency_samples)))
        (out/'completion.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result.get(key) for key in ('status', 'complete', 'error')}, indent=2), flush=True)
    return 0 if result['complete'] else 1


if __name__ == '__main__': raise SystemExit(main())
