"""Pinned V1 server plus policy-neutral native execution and measurement RPCs."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

router = APIRouter(prefix='/baseline')


def engine(request):
    return request.app.state.engine_client


async def scheduler(request, operation, payload=None):
    return await engine(request).engine_core.call_utility_async('native_operation', operation, payload or {})


async def workers(request, method, **kwargs):
    rows = await engine(request).collective_rpc(method, kwargs=kwargs)
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise HTTPException(503, 'missing native rank acknowledgements')
    ranks = [row.get('rank') for row in rows]
    expected = request.app.state.native_tp * request.app.state.native_pp
    if len(ranks) != expected or set(ranks) != set(range(expected)):
        raise HTTPException(503, 'incomplete native rank coverage')
    return rows


@router.get('/state')
async def state(request: Request):
    rank_observation_started_s = time.time()
    ranks = await workers(request, 'native_worker_state')
    rank_observation_finished_s = time.time()
    # Read the owner's inventory last. Its timestamp describes all_queue and
    # KV block IDs; a later RPC/HTTP timestamp must never relabel that snapshot.
    value = await scheduler(request, 'state')
    transfers = {str(row['rank']): row['transfer_allocations'] for row in ranks if row.get('transfer_allocations')}
    complete = value['native_evidence_complete'] and all(row.get('native_evidence_complete') and
                    row.get('generation') == value['generation'] for row in ranks)
    value.update(ranks=ranks, transfer_allocations=transfers,
                 dynamo_weights_ready=all(row.get('dynamo_weights_ready') is True for row in ranks),
                 pending_transfers=sum(row.get('pending_transfers', 0) for row in ranks),
                 native_evidence_complete=complete, evidence_complete=complete,
                 transport_healthy=all(row.get('healthy') for row in ranks), timestamp=time.time(),
                 active=len(value['running']), scheduler_at_s=value['native_at_s'], response_at_s=time.time(),
                 rank_observation_started_s=rank_observation_started_s,
                 rank_observation_finished_s=rank_observation_finished_s,
                 rank_observed_at_s=[dict(rank=row['rank'], started_s=rank_observation_started_s,
                                         finished_s=rank_observation_finished_s) for row in ranks],
                 atomic_rank_scheduler_snapshot=False, atomic_snapshot=False)
    return value


@router.get('/capability')
async def capability(request: Request):
    native = await state(request)
    return dict(request.app.state.native_identity, tp=native['tp'], pp=native['pp'],
                supported=native['native_evidence_complete'], native_evidence_complete=native['native_evidence_complete'],
                mechanisms=dict(scheduler_state=True, generation_control=True, cancellation=True,
                                cuda_measurement=True, retained_kv=all(r.get('retained_kv_supported') for r in native['ranks']), pp_transfer=False),
                state=native)


@router.get('/events')
async def events(request: Request, after_seq: int = 0):
    return await scheduler(request, 'events', dict(after_seq=after_seq))


@router.post('/control')
async def control(request: Request):
    payload = await request.json()
    if payload.get('operation') in ('prefill', 'decode'):
        raise HTTPException(501, 'stage execution uses explicit DistServe retained-KV endpoints')
    current = await state(request)
    generation = payload.get('generation', current['generation'])
    if generation != current['generation']:
        if type(generation) is not int or generation < current['generation']:
            raise HTTPException(409, 'stale generation')
        if current['all_queue'] or current['pending_transfers'] or current['transfer_allocations']:
            raise HTTPException(409, 'generation change requires drained native execution')
        # Quiesce admission before installing a new generation on actual ranks.
        await scheduler(request, 'control', dict(accepting=False))
        ranks = await workers(request, 'native_generation_set', generation=generation)
        if not all(r.get('acknowledged') is True and r.get('generation') == generation for r in ranks):
            raise HTTPException(503, 'native generation installation incomplete; engine quiesced')
        payload.setdefault('accepting', current['accepting'])
        receipt = await scheduler(request, 'control', payload)
        return dict(receipt, ranks=ranks)
    return await scheduler(request, 'control', payload)


@router.post('/cancel')
async def cancel(request: Request):
    payload = await request.json(); rid = payload['request_id']
    await engine(request).abort(rid)
    deadline = time.monotonic()+float(payload.get('timeout_s', 10))
    worker_cancelled, retained_release = False, None
    while True:
        receipt = await state(request)
        if rid in receipt['retained_kv_requests']:
            # AsyncLLM.abort filters requests already removed from its output
            # processor. A completed prefill can still own scheduler/worker KV
            # after that removal, so it needs the explicit retained release.
            # Never release uncertain transport buffers or incomplete ranks.
            safe = (receipt['native_evidence_complete'] and receipt['transport_healthy']
                    and not receipt['pending_transfers'] and not receipt['transfer_allocations'])
            if safe:
                ranks = await workers(request, 'native_kv_operation', operation='release',
                                      payload=dict(held_request_id=rid))
                if not all(row.get('acknowledged') is True and row.get('released') is True
                           and row.get('generation') == receipt['generation'] for row in ranks):
                    raise HTTPException(503, 'retained cancellation lacks every rank release ACK')
                released = await scheduler(request, 'release', dict(request_id=rid))
                retained_release = dict(released, ranks=ranks)
                worker_cancelled = True
                continue
        elif not worker_cancelled:
            # Completed receive metadata can be retired; uncertain transfers
            # remain quarantined and cannot be reported as a clean cancel.
            await workers(request, 'native_kv_operation', operation='cancel', payload=dict(request_id=rid))
            worker_cancelled = True
            continue
        pending = (rid in receipt['all_queue'] or receipt['pending_transfers'] or
                   any(rid in name for slots in receipt['transfer_allocations'].values() for name in slots))
        if not pending:
            return dict(acknowledged=True, cancelled=True, request_id=rid,
                        generation=receipt['generation'], native_state=receipt,
                        retained_release=retained_release)
        if time.monotonic() >= deadline:
            # The pinned vLLM HTTP exception handler requires message: str.
            raise HTTPException(409, json.dumps(dict(reason='cancellation resources not released', state=receipt)))
        await asyncio.sleep(.05)


async def drain_engine(request, payload):
    await scheduler(request, 'control', dict(accepting=False, admit_prefill=True, admit_decode=True))
    deadline = time.monotonic()+float(payload.get('timeout_s', 30))
    while True:
        value = await state(request)
        if not value['all_queue'] and not value['transfer_allocations'] and not value['pending_transfers']:
            return dict(value, acknowledged=True, drained=True, inflight=0, live_kv_blocks=0,
                        live_kv_tokens=0, pending_transfers=0)
        if time.monotonic() >= deadline:
            raise HTTPException(409, dict(reason='drain not complete', state=value))
        await asyncio.sleep(.05)


@router.post('/drain')
async def drain(request: Request):
    return await drain_engine(request, await request.json())


async def generate_events(request, payload, *, private=False):
    from vllm import SamplingParams
    prompt = payload.get('prompt', payload.get('prompt_token_ids', payload.get('prompt_tokens')))
    if not isinstance(prompt, list) or not prompt or any(type(v) is not int or v < 0 for v in prompt):
        raise HTTPException(400, 'native generation requires tokenized prompt')
    rid = payload['request_id']
    if not private and not (await scheduler(request, 'state'))['accepting']:
        raise HTTPException(409, 'engine admission is quiesced')
    maximum = int(payload.get('max_tokens', 16))
    params = SamplingParams(temperature=0, max_tokens=maximum, ignore_eos=bool(payload.get('ignore_eos', True)),
                            seed=int(payload.get('seed', 701)))
    count, finished = 0, False
    try:
        async for output in engine(request).generate(dict(prompt_token_ids=prompt), params, rid):
            for choice in output.outputs:
                ids = list(choice.token_ids)
                delta = ids[count:]
                count = len(ids)
                finished = bool(output.finished)
                yield dict(request_id=rid, token_ids=delta, token_index=count,
                           text=choice.text, finished=finished, finish_reason=choice.finish_reason,
                           choices=[dict(index=choice.index, text=choice.text, finish_reason=choice.finish_reason)],
                           usage=dict(prompt_tokens=len(prompt), completion_tokens=count), at_s=time.time())
    finally:
        if not finished:
            await engine(request).abort(rid)


@router.post('/generate')
async def generate(request: Request):
    payload = await request.json()
    # Check admission before HTTP headers are committed.
    if not (await scheduler(request, 'state'))['accepting']:
        raise HTTPException(409, 'engine admission is quiesced')
    async def stream():
        async for row in generate_events(request, payload):
            yield 'data: '+json.dumps(row)+'\n\n'
        yield 'data: [DONE]\n\n'
    return StreamingResponse(stream(), media_type='text/event-stream')


@router.post('/measurement/start')
async def measurement_start(request: Request):
    p = await request.json()
    native = await scheduler(request, 'state')
    if native['all_queue']:
        raise HTTPException(409, 'drain requests before changing measurement scope')
    rows = await workers(request, 'native_measurement_start', scope=p['scope'], system=p['system'])
    return dict(acknowledged=all(row.get('acknowledged') for row in rows), ranks=rows)


@router.get('/measurement/samples')
async def measurement_samples(request: Request):
    return dict(ranks=await workers(request, 'native_measurement_samples'))


@router.post('/measurement/stop')
async def measurement_stop(request: Request):
    return dict(ranks=await workers(request, 'native_measurement_stop'))


def instance_uuids():
    import pynvml
    pynvml.nvmlInit()
    parent = os.environ.get('PDBLEND_GPU_UUIDS', '').split(',')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    result = []
    for value in visible:
        if value.startswith('GPU-'):
            result.append(value)
        elif value.isdigit():
            index = int(value)
            if parent and parent[0].startswith('GPU-'):
                result.append(parent[index])
            else:
                result.append(pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(index)))
    if not result or len(set(result)) != len(result):
        raise ValueError('explicit unique GPU lease mapping is required')
    return result


def clock_operation(frequency):
    import pynvml
    uuids = instance_uuids(); rows=[]
    if frequency is not None and frequency not in (900,1200,1500,1800,2100,2520):
        raise ValueError('frequency outside this campaign')
    for uuid in uuids:
        handle=pynvml.nvmlDeviceGetHandleByUUID(uuid)
        if frequency is None:
            pynvml.nvmlDeviceResetGpuLockedClocks(handle)
        else:
            pynvml.nvmlDeviceSetGpuLockedClocks(handle, frequency, frequency)
        rows.append(dict(gpu_uuid=uuid, frequency_mhz=pynvml.nvmlDeviceGetClockInfo(handle,pynvml.NVML_CLOCK_SM),
                         power_w=pynvml.nvmlDeviceGetPowerUsage(handle)/1000))
    return dict(acknowledged=True, success=True, requested_frequency_mhz=frequency, gpus=rows, at_s=time.time())


@router.post('/clock')
async def clock(request: Request):
    p=await request.json()
    return await asyncio.to_thread(clock_operation, int(p.get('frequency_mhz', p.get('frequency'))))


@router.post('/park')
async def park(request: Request):
    value=await state(request)
    if value['all_queue'] or value['transfer_allocations'] or value['pending_transfers']:
        raise HTTPException(409,'parking requires empty native queues and KV transfer state')
    return await asyncio.to_thread(clock_operation, None)


@router.post('/distserve/prefill')
async def distserve_prefill(request: Request):
    payload = await request.json()
    cap = await capability(request)
    if not cap['mechanisms']['retained_kv'] or cap['pp'] != 1:
        raise HTTPException(501, 'native retained P/D requires P2P and PP1')
    address = request.app.state.native_kv_address
    if not address:
        raise HTTPException(501, 'no source KV address')
    original_id = payload['request_id']
    rid = f'distserve-hold-{original_id}___prefill_addr_{address}___decode_addr_127.0.0.1:0_{original_id}'
    outputs=[]
    async for row in generate_events(request, dict(payload, request_id=rid, max_tokens=1)):
        outputs.append(row)
    ranks=await workers(request, 'native_kv_operation', operation='state', payload={})
    if not all(rid in r['held_requests'] for r in ranks):
        raise HTTPException(503, 'prefill completed without retained KV on every rank')
    return dict(acknowledged=True, request_id=original_id, retained_handle=rid, kv_handle=rid,
                source_address=address, tp=cap['tp'], outputs=outputs, ranks=ranks)


@router.post('/distserve/transfer')
async def distserve_transfer(request: Request):
    payload=await request.json()
    value=await scheduler(request,'state')
    if payload.get('target_tp') != value['tp'] or value['pp'] != 1:
        raise HTTPException(501,'asymmetric P/D is not qualified')
    if payload.get('generation') != value['generation']:
        raise HTTPException(409,'stale KV transfer generation')
    ranks=await workers(request,'native_kv_operation',operation='transfer',payload=payload)
    if not all(row.get('acknowledged') is True for row in ranks):
        raise HTTPException(503,'missing native KV send ACK')
    return dict(acknowledged=True,ranks=ranks,transaction_id=payload['transaction_id'],
                held_request_id=payload['held_request_id'],target_request_id=payload['target_request_id'])


@router.post('/distserve/load_ack')
async def distserve_load_ack(request: Request):
    payload=await request.json()
    ranks=await workers(request,'native_kv_operation',operation='query_load',payload=payload)
    generation=(await scheduler(request,'state'))['generation']
    if not all(r.get('acknowledged') is True and r.get('generation') == generation and
               r.get('transaction_id') == payload.get('transaction_id') and
               r.get('target_request_id') == payload.get('target_request_id') and
               r.get('expected_layers', 0) > 0 and r.get('loaded_layers') == r.get('expected_layers') for r in ranks):
        raise HTTPException(503, 'incomplete request/generation/rank/layer KV load evidence')
    return dict(acknowledged=True,ranks=ranks,request_id=payload.get('target_request_id'),
                generation=generation,transaction_id=payload.get('transaction_id'))


@router.post('/distserve/expect_load')
async def distserve_expect_load(request: Request):
    payload=await request.json()
    value=await scheduler(request,'state')
    if payload.get('generation') != value['generation']:
        raise HTTPException(409,'stale target KV generation')
    ranks=await workers(request,'native_kv_operation',operation='expect_load',payload=payload)
    if not all(r.get('acknowledged') is True for r in ranks):
        raise HTTPException(503,'target ranks did not register KV receive')
    return dict(acknowledged=True,ranks=ranks)


@router.post('/distserve/release')
async def distserve_release(request: Request):
    payload=await request.json(); rid=payload['held_request_id']
    ranks=await workers(request,'native_kv_operation',operation='release',payload=payload)
    if not all(row.get('released') is True for row in ranks):
        raise HTTPException(503,'not all source ranks released retained references')
    receipt=await scheduler(request,'release',dict(request_id=rid))
    return dict(receipt,ranks=ranks)


def _dynamo_context(request):
    app = request.app.state
    if not hasattr(app, 'dynamo_context'):
        app.dynamo_context = dict(lock=asyncio.Lock(), sessions={}, transferred={}, verified={},
            weights_ready=os.environ.get('DYNAMO_DUMMY', '0') != '1', failed=False)
    return app.dynamo_context


async def _dynamo_rank_operation(request, operation, payload, generation):
    from pdblend_baselines.dynamollm.transport import validate_rank_ack
    rows = await workers(request, 'dynamo_operation', operation=operation,
                         payload=dict(payload, expected_generation=generation))
    validate_rank_ack({'ranks': rows}, {'tp': request.app.state.native_tp, 'generation': generation},
                      transaction_id=payload.get('transaction_id'))
    return rows


async def _dynamo_native_drain(request, payload, generation):
    from pdblend_baselines.dynamollm.native_hooks import aggregate_drain
    await drain_engine(request, payload)
    ranks = await _dynamo_rank_operation(request, 'drain', payload, generation)
    current = await state(request)
    proof = dict(current, running=len(current['running']), waiting=len(current['waiting']))
    proof.setdefault('active', len(current['all_queue']))
    proof.setdefault('timestamp', current.get('native_at_s'))
    proof.setdefault('evidence_complete', current.get('native_evidence_complete'))
    proof.setdefault('transport_healthy', current.get('healthy'))
    return aggregate_drain(proof, ranks, tp=request.app.state.native_tp, generation=generation)


async def _dynamo_verify(request, payload, generation, context):
    operation_id = payload.get('operation_id')
    if operation_id not in context['transferred'] or context['sessions']:
        raise HTTPException(409, 'complete transfer and closed communicators required before golden')
    if operation_id in context['verified']:
        return context['verified'][operation_id]
    golden = json.loads(os.environ.get('DYNAMO_GOLDEN_JSON', '{}'))
    identity = request.app.state.native_identity
    if (golden.get('model_id') != identity['model_id'] or golden.get('tp') != request.app.state.native_tp
            or golden.get('engine_revision') != identity['engine_revision']
            or golden.get('seed') != 701 or not golden.get('prompt') or not golden.get('token_ids')
            or len(golden['token_ids']) > 64 or len(golden.get('source_sha256', '')) != 64):
        raise HTTPException(409, 'model-bound new-stack same-TP golden required')
    for key in ('model_hash', 'tokenizer_hash', 'image_digest'):
        if key in golden and golden[key] != identity.get(key):
            raise HTTPException(409, 'target golden provenance differs: ' + key)
    native = await scheduler(request, 'state')
    if native['accepting'] or native['all_queue']:
        raise HTTPException(409, 'private golden requires closed idle target')
    await _dynamo_native_drain(request, payload, generation)
    tokens = []
    finished = False
    async def probe():
        nonlocal finished
        work = dict(prompt=golden['prompt'], request_id='dynamo-private-' + operation_id,
                    max_tokens=len(golden['token_ids']), seed=golden['seed'])
        async for event in generate_events(request, work, private=True):
            tokens.extend(event['token_ids'])
            finished = bool(event.get('finished'))
    await asyncio.wait_for(probe(), float(payload.get('probe_timeout_s', 60)))
    if not finished or tokens != golden['token_ids']:
        raise HTTPException(409, 'target output differs from frozen same-TP golden')
    drained = await _dynamo_native_drain(request, payload, generation)
    result = dict(ok=True, verified=True, generation=generation, operation_id=operation_id,
                  transaction_id=payload['transaction_id'], token_ids=tokens,
                  golden_source_sha256=golden['source_sha256'], drain=drained,
                  hardware_qualified=False)
    context['verified'][operation_id] = result
    return result


@router.post('/dynamollm/{operation}')
async def dynamollm(request: Request, operation: str):
    allowed = {'quiesce', 'resume', 'drain', 'describe', 'open', 'transfer', 'close', 'verify', 'activate'}
    if operation not in allowed:
        raise HTTPException(404, 'unknown Dynamo native operation')
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(400, 'Dynamo operation requires a JSON object')
    context = _dynamo_context(request)
    async with context['lock']:
        current = await scheduler(request, 'state')
        generation = current['generation']
        if payload.get('expected_generation', generation) != generation:
            raise HTTPException(409, 'Dynamo expected generation differs')
        if context['failed']:
            raise HTTPException(503, 'Dynamo communication state uncertain; isolate and rebuild')
        try:
            if operation in ('quiesce', 'resume'):
                if operation == 'resume' and not context['weights_ready']:
                    raise HTTPException(409, 'unverified target weights cannot resume')
                result = await scheduler(request, 'control', dict(accepting=operation == 'resume',
                    admit_prefill=True, admit_decode=True, generation=generation))
                return dict(result, accepting=operation == 'resume', generation=generation)
            if operation == 'drain':
                return await _dynamo_native_drain(request, payload, generation)
            if operation in ('open', 'transfer', 'close', 'verify', 'activate'):
                tx = payload.get('transaction_id')
                if not isinstance(tx, str) or not tx or len(tx) > 128:
                    raise HTTPException(400, 'bounded transaction_id required')
            if operation == 'verify':
                return await _dynamo_verify(request, payload, generation, context)
            if operation == 'activate':
                verified = context['verified'].get(payload.get('operation_id'))
                if not verified or verified['transaction_id'] != payload['transaction_id']:
                    raise HTTPException(409, 'verified transaction required before activation')
                ranks = await _dynamo_rank_operation(request, 'mark_ready', dict(payload, verified=True), generation)
                if not all(row.get('weights_ready') is True for row in ranks):
                    raise HTTPException(503, 'complete ready ACK missing')
                await scheduler(request, 'control', dict(accepting=True, admit_prefill=True,
                                                        admit_decode=True, generation=generation))
                context['weights_ready'] = True
                return dict(activated=True, active=True, generation=generation, ranks=ranks,
                            transaction_id=payload['transaction_id'], operation_id=payload['operation_id'])
            if operation in ('transfer', 'close'):
                session = context['sessions'].get(payload.get('session_id'))
                if not session or session['transaction_id'] != payload['transaction_id']:
                    raise HTTPException(409, 'matching open Dynamo session required')
                if operation == 'transfer':
                    sending = all(rank in payload.get('source_ranks', []) for rank in
                        range(session['rank_offset'], session['rank_offset'] + request.app.state.native_tp))
                    if not sending and (context['weights_ready'] or current['accepting'] or current['all_queue']):
                        raise HTTPException(409, 'receive requires a never-activated idle dummy target')
            rows = await _dynamo_rank_operation(request, operation, payload, generation)
            if operation == 'open':
                context['sessions'][payload['session_id']] = dict(payload)
            elif operation == 'close':
                context['sessions'].pop(payload['session_id'])
            elif operation == 'transfer' and all(row.get('source') is False for row in rows):
                operation_id = payload.get('operation_id')
                if (not operation_id or operation_id != payload['transaction_id'] or any(
                        row.get('operation_id') != operation_id or row.get('target_complete') is not True
                        or row.get('received_bytes', 0) <= 0 or row.get('parameter_count', 0) <= 0 for row in rows)):
                    raise HTTPException(503, 'incomplete target weight coverage')
                context['transferred'][operation_id] = rows
            return dict(ranks=rows, acknowledged=True, transaction_id=payload.get('transaction_id'),
                        generation=generation)
        except BaseException:
            if operation in ('open', 'transfer', 'close', 'verify', 'activate', 'drain'):
                context['failed'] = True
                try:
                    await scheduler(request, 'control', dict(accepting=False, admit_prefill=True, admit_decode=True))
                except BaseException:
                    pass
            raise


def serving_identity(args):
    receipt_path=os.environ.get('PDBLEND_MODEL_VERIFICATION_RECEIPT')
    if not receipt_path:
        raise ValueError('native serving requires verified model receipt')
    receipt=json.loads(Path(receipt_path).read_text())
    item=next((r for r in receipt['models'].values() if Path(r['model_path']).resolve()==Path(args.model).resolve()),None)
    if not receipt.get('all_pass') or not item or item.get('verified') is not True:
        raise ValueError('model is not verified in serving receipt')
    hashes={}
    model_root=Path(args.model).resolve()
    for kind,key in [('weight','model_hash'),('tokenizer','tokenizer_hash')]:
        files=[r for r in item['files'] if r['kind']==kind]
        paths=[(model_root/record['path']).resolve() for record in files]
        if not files or any(not path.is_relative_to(model_root) or not path.is_file() or
                            path.stat().st_size!=record['bytes'] for path,record in zip(paths,files)):
            raise ValueError('model files differ from verification receipt')
        hashes[key]=hashlib.sha256(json.dumps([(r['path'],r['bytes'],r['sha256']) for r in files],separators=(',',':'),sort_keys=True).encode()).hexdigest()
    return dict(model_id=item['model_id'],model=item['model_id'],**hashes,
                engine_revision='vllm-0.10.1.1',engine_version='vllm-0.10.1.1',
                image_digest=os.environ.get('PDBLEND_IMAGE_ID'),source_revision=os.environ.get('PDBLEND_SOURCE_SHA256'),
                gpu_uuids=instance_uuids(),verification_receipt_sha256=hashlib.sha256(Path(receipt_path).read_bytes()).hexdigest())


def main():
    from vllm.entrypoints.openai import api_server
    from vllm.utils import FlexibleArgumentParser
    import uvloop
    argv=list(sys.argv[1:])
    if argv and not argv[0].startswith('-'):
        argv=['--model',argv[0],*argv[1:]]
    for flag,value in [('--scheduler-cls','pdblend_runtime.native_v1.NativeScheduler'),
                       ('--worker-cls','pdblend_runtime.native_v1.NativeWorker')]:
        if flag not in argv:
            argv += [flag,value]
    parser=api_server.make_arg_parser(FlexibleArgumentParser(description=__doc__))
    args=parser.parse_args(argv)
    api_server.validate_parsed_serve_args(args)
    # Fail before loading weights if the mounted model or receipt differs.
    identity=serving_identity(args)
    original=api_server.build_app
    def build_app(values):
        app=original(values)
        app.include_router(router)
        app.state.native_tp=values.tensor_parallel_size
        app.state.native_pp=values.pipeline_parallel_size
        app.state.native_identity=identity
        kv_config=values.kv_transfer_config
        kv_port=(kv_config.get('kv_port') if isinstance(kv_config,dict) else getattr(kv_config,'kv_port',None)) if kv_config else None
        app.state.native_kv_address=f'127.0.0.1:{kv_port}' if kv_port else None
        return app
    api_server.build_app=build_app
    uvloop.run(api_server.run_server(args))


if __name__=='__main__':main()
