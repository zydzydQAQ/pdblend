"""Execute pinned vLLM request/KV lifetime code on CPU, without importing vLLM.

Only GPU allocation, connector I/O and the engine RPC wire are substitutes.
Request termination, KV allocation bounds, native events and HTTP drain are
the shipped source bodies, not a test-defined copy of their behavior.
"""
import ast
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
import enum
import heapq
import hashlib
import importlib.util
import logging
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS

import pytest


PINNED = {
    'v1/request.py': '93c25fba1385c2d04fcb553b37228beb045ea46830061921a377d3b5c6653ec7',
    'v1/core/sched/utils.py': 'fe0446f2051afe429de56291d78a7ea5f598f86ce205ef78df581a41e31e082d',
    'v1/core/sched/scheduler.py': 'f42c72e0848d669b07a39ececd624431a89aa55a806b568efd86b4af80dea7a1',
    'v1/core/kv_cache_manager.py': 'b317f99f1b9692d0c8bdfe184c9155aab8685bba15f8a94e59171c2ebbf4b50a',
    'v1/core/sched/request_queue.py': '73f7e18902fea1e38f540a11fe8289a691364e3311a2c69b5030506607db6c8e',
}


def execute(nodes, namespace):
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])),
                 '<pinned-vllm-ecoserve-contract>', 'exec'), namespace)


@pytest.fixture(scope='module')
def pinned():
    spec = importlib.util.find_spec('vllm')
    if spec is None:
        pytest.skip('requires the pinned image; no GPU or vLLM import is used')
    root = Path(next(iter(spec.submodule_search_locations)))
    sources = {}
    for name, digest in PINNED.items():
        raw = (root/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        sources[name] = ast.parse(raw)
    namespace = dict(__name__=__name__, enum=enum, Enum=enum.Enum, time=time, os=os, deque=deque,
        ABC=ABC, abstractmethod=abstractmethod, heapq=heapq,
        defaultdict=defaultdict, dataclass=dataclass, ConstantList=lambda values: values,
        FinishReason=NS(STOP='stop', LENGTH='length', ABORT='abort'),
        SchedulerInterface=object, MULTIMODAL_REGISTRY=None,
        EngineCoreOutput=NS, EngineCoreOutputs=NS, logger=logging.getLogger(__name__))
    for name in ('v1/request.py', 'v1/core/sched/request_queue.py', 'v1/core/sched/utils.py', 'v1/core/kv_cache_manager.py',
                 'v1/core/sched/scheduler.py'):
        nodes = [node for node in sources[name].body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                    and target.id == '_FINISHED_REASON_MAP' for target in node.targets)]
        execute(nodes, namespace)
    native = ast.parse((Path(__file__).resolve().parents[2]/'src/pdblend_runtime/native_v1.py').read_text())
    execute([node for node in native.body if isinstance(node, ast.ClassDef)
             and node.name == 'NativeScheduler'], namespace)
    assert 'vllm' not in sys.modules
    return namespace


def scheduler(pinned, *, delayed=False):
    """Initialize only state needed by the actual output/abort/free methods."""
    core = object.__new__(pinned['NativeScheduler'])
    allocated = {}
    core.requests, core.running, core.waiting = {}, [], []
    core.max_model_len, core.block_size = 8192, 16
    core.cache_config = NS(num_gpu_blocks=2049)
    core.parallel_config = NS(tensor_parallel_size=1, pipeline_parallel_size=1)
    core.kv_cache_manager = NS(
        block_pool=NS(get_num_free_blocks=lambda: 2048-sum(len(v) for v in allocated.values())),
        get_block_ids=lambda rid: (allocated.get(rid, []),), free=lambda req: allocated.pop(req.request_id))
    core.encoder_cache_manager = NS(free=lambda req: None)
    core.structured_output_manager = NS(should_advance=lambda req: False)
    core.connector = (NS(request_finished=lambda req, blocks: (True, None),
                         update_connector_output=lambda output: None) if delayed else None)
    core.finished_req_ids, core.finished_recving_kv_req_ids = set(), set()
    core.finished_req_ids_dict, core.log_stats = None, False
    core.native_generation, core.native_seq = 2, 0
    core.native_role, core.native_mode = 'mixed', 'temporal'
    core.native_accepting = core.native_admit_prefill = core.native_admit_decode = True
    core.native_max_batch, core.native_max_tokens = 32, 8192
    core.max_num_running_reqs, core.max_num_scheduled_tokens = 32, 8192
    core.native_events, core.native_held = deque(maxlen=100000), set()
    core.native_last_schedule, core.native_prefill_mode = [], False
    return core, allocated


def request(pinned, identifier='pressure', *, inputs=7168, outputs=512):
    params = NS(max_tokens=outputs, guided_decoding=None, extra_args=None,
                ignore_eos=True, stop_token_ids=[], logprobs=None)
    return pinned['Request'](identifier, [100]*inputs, None, None, None, params, None, 151645)


@pytest.mark.parametrize('inputs,outputs', [(128, 1), (7168, 512)])
def test_actual_finished_output_frees_native_all_queue_and_kv(pinned, inputs, outputs):
    core, allocated = scheduler(pinned)
    req = request(pinned, inputs=inputs, outputs=outputs)
    req.status = pinned['RequestStatus'].RUNNING
    core.requests[req.request_id] = req
    core.running.append(req)
    allocated[req.request_id] = list(range(1, (inputs+outputs+15)//16+1))
    for index in range(outputs):
        core.native_last_schedule = [req.request_id]
        core.native_prefill_mode = index == 0
        core._native_event('schedule')
        output = core.update_from_output(NS(num_scheduled_tokens={req.request_id: inputs if index == 0 else 1},
            scheduled_spec_decode_tokens={}), NS(sampled_token_ids=[[151645]], spec_token_ids=None,
            logprobs=None, prompt_logprobs_dict={}, pooler_output=None, num_nans_in_logits=None,
            req_id_to_index={req.request_id: 0}, kv_connector_output=None))[0].outputs[0]
        assert req.num_output_tokens == index+1
        assert (output.finish_reason == 'length') == (index+1 == outputs)
        assert (req.request_id in core.native_state()['all_queue']) == (index+1 < outputs)
    assert req.num_tokens == inputs+outputs
    assert core.native_state()['kv_allocations'] == {} and not core.running
    assert core.native_state()['free_kv_tokens'] == core.native_state()['total_kv_tokens']
    assert core.native_events[-1]['kind'] == 'step_completed'
    assert core.native_events[-1]['all_queue'] == []


def test_actual_kv_allocator_refuses_shortage_without_allocating_max_num_seqs(pinned):
    manager = object.__new__(pinned['KVCacheManager'])
    manager.max_model_len, manager.enable_caching = 8192, False
    manager.kv_cache_config = NS(kv_cache_groups=[object()])
    calls, available = [], [447]
    manager.block_pool = NS(get_num_free_blocks=lambda: available[0])
    manager.coordinator = NS(remove_skipped_blocks=lambda *args: None,
        get_num_blocks_to_allocate=lambda **kw: (kw['num_tokens']+15)//16,
        save_new_computed_blocks=lambda *args: calls.append(('save', args)),
        allocate_new_blocks=lambda rid, count: calls.append(('allocate', count)) or ([NS(block_id=1)],))
    req = request(pinned)
    assert manager.allocate_slots(req, 7168) is None
    assert calls == []  # No over-capacity GPU write or future-32-sequence reservation.
    available[0] = 448
    assert manager.allocate_slots(req, 7168).get_block_ids() == ([1],)
    assert calls[-1] == ('allocate', 7168)
    req.num_computed_tokens = 8190
    available[0] = 512
    manager.allocate_slots(req, 512)
    assert calls[-1] == ('allocate', 8192)  # Hard model-length ceiling remains real.


@pytest.mark.asyncio
@pytest.mark.parametrize('pending,received', [(1, {}), (0, {'request-layer': 'received'})])
async def test_actual_native_drain_rejects_worker_pending_or_received_kv(pinned, pending, received):
    pytest.importorskip('fastapi')
    from fastapi import HTTPException
    from pdblend_runtime.serve import drain_engine
    core, _ = scheduler(pinned)
    async def utility(method, operation, payload):
        return core.native_operation(operation, payload)
    async def worker(method, kwargs):
        return [dict(rank=0, generation=2, native_evidence_complete=True, healthy=True,
                     transfer_allocations=received, pending_transfers=pending)]
    client = NS(engine_core=NS(call_utility_async=utility), collective_rpc=worker)
    http = NS(app=NS(state=NS(engine_client=client, native_tp=1, native_pp=1)))
    with pytest.raises(HTTPException) as caught:
        await drain_engine(http, dict(timeout_s=0))
    assert caught.value.status_code == 409
    assert not core.native_accepting


def test_actual_abort_keeps_delayed_transfer_visible_until_real_connector_ack(pinned):
    core, allocated = scheduler(pinned, delayed=True)
    req = request(pinned)
    req.status = pinned['RequestStatus'].RUNNING
    core.requests[req.request_id], allocated[req.request_id] = req, [1, 2]
    core.running.append(req)
    core.finish_requests(req.request_id, pinned['RequestStatus'].FINISHED_ABORTED)
    assert not core.running and req.is_finished()
    assert core.native_state()['all_queue'] == [req.request_id]
    assert core.native_state()['kv_allocations'] == {req.request_id: [[1, 2]]}
    assert core.native_events[-1]['kind'] == 'request_terminated'
    core._update_from_kv_xfer_finished(NS(finished_recving=None, finished_sending=[req.request_id]))
    assert core.native_state()['all_queue'] == [] and allocated == {}


@pytest.mark.parametrize('stage', ['before_prefill', 'during_prefill', 'after_retention'])
def test_distserve_retained_prefix_cancellation_releases_actual_pinned_request(pinned, stage):
    core, allocated = scheduler(pinned)
    core.waiting = pinned['FCFSRequestQueue']()
    core.connector = NS(request_finished=lambda request, blocks: (False, None))
    req = request(pinned, identifier='distserve-hold-cancel-contract', inputs=128, outputs=1)
    core.requests[req.request_id] = req
    allocated[req.request_id] = [] if stage == 'before_prefill' else [1, 2]
    if stage == 'before_prefill':
        core.waiting.append(req)
    elif stage == 'during_prefill':
        req.status = pinned['RequestStatus'].RUNNING; req.num_computed_tokens = 64
        core.running.append(req)
    else:
        req.status = pinned['RequestStatus'].FINISHED_LENGTH_CAPPED; req.num_computed_tokens = 128
        assert core._free_request(req)['native_retained_request'] == req.request_id
        assert req.request_id in core.native_held and req.request_id in core.requests
    core.finish_requests(req.request_id, pinned['RequestStatus'].FINISHED_ABORTED)
    assert not core.native_held and not core.requests and not core.running and not core.waiting
    assert not allocated and core.native_state()['all_queue'] == []
