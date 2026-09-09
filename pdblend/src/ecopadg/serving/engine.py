"""Bounded asynchronous HTTP adapter around the shared vLLM V0 engine.

The dedicated engine thread owns CUDA, scheduling and request mutations. The
HTTP loop only transports data; phase admission is enforced inside the engine.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import partial
import json
import hashlib
import os
from pathlib import Path
import time
import uuid

from aiohttp import web


SOURCE_FILES_AT_IMPORT={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in Path(__file__).parent.glob('*.py')}


class EngineService:
    def __init__(self, config):
        self.config = config
        self.worker = ThreadPoolExecutor(1, thread_name_prefix="engine-owner")
        self.streams = {}
        self.finished = set()
        self.accepting = True
        self.error = None
        self.wake = asyncio.Event()
        self.control_lock = asyncio.Lock()
        self.state = dict(generation=config.get("initial_generation",0), role=config.get("role", "mixed"),
                          mode=config.get("mode", "continuous"), admit_prefill=True,admit_decode=True)
        self.runtime_path = Path(config["runtime_dir"]) / (config["id"] + ".control.json")
        self.timeline = self.runtime_path.with_suffix(".events.jsonl")
        self.snapshot = {}

    def fail(self,message):
        self.error=message
        self.accepting=False
        for queue in self.streams.values():
            while not queue.empty(): queue.get_nowait()
            queue.put_nowait(RuntimeError(message))

    async def call(self, func, *args, timeout_s=None):
        if self.error:
            raise RuntimeError(self.error)
        future=asyncio.get_running_loop().run_in_executor(self.worker,partial(func,*args))
        try:
            return await asyncio.wait_for(future,timeout_s or self.config.get('operation_timeout_s',45))
        except asyncio.TimeoutError as exc:
            # Native CUDA/NCCL work cannot be cancelled by cancelling a Python
            # future. Quarantine this engine and fail its streams immediately;
            # a lifecycle replacement is required before it can serve again.
            message='engine operation timed out; instance quarantined, replacement required'
            self.fail(message)
            raise RuntimeError(message) from exc

    def write_state(self, state):
        self.runtime_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.runtime_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(self.runtime_path)

    def initialize(self):
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ["PDBLEND_RUNTIME_PATH"] = str(self.runtime_path)
        # NCCL caches channel settings at its first initialization. Set these
        # before TP collectives, or later P2P peers disagree on bootstrap size.
        os.environ["NCCL_MAX_NCHANNELS"] = "8"
        os.environ["NCCL_MIN_NCHANNELS"] = "8"
        from vllm.engine.arg_utils import EngineArgs
        from vllm.engine.llm_engine import LLMEngine
        from vllm.config import KVTransferConfig
        self.write_state(self.state)
        c = self.config
        args = dict(model=c["model"], tensor_parallel_size=c["tp"],
                    dtype="bfloat16", max_model_len=c.get("max_model_len", 8192),
                    max_num_seqs=c.get("max_num_seqs", 32),
                    max_num_batched_tokens=c.get("max_num_batched_tokens", 8192),
                    gpu_memory_utilization=c.get("gpu_memory_utilization", .85),
                    enforce_eager=True, enable_chunked_prefill=True,
                    enable_prefix_caching=False, disable_log_stats=True,
                    distributed_executor_backend="mp" if c["tp"] > 1 else "uni")
        if c.get('retained_weights'):
            from vllm.pdblend_weights import RetainedWeightLoader
            args.update(load_format=RetainedWeightLoader,
                model_loader_extra_config=dict(retained_weights=c['retained_weights']))
        if c.get("peers"):
            args["kv_transfer_config"] = KVTransferConfig(
                kv_connector="PDBlendConnector", kv_role="kv_both",
                engine_id=c["id"], kv_rank=0, kv_parallel_size=1,
                kv_port=c["kv_port"], kv_buffer_size=c.get('transfer_buffer_bytes',2*1024**3),
                kv_connector_extra_config=dict(peers=c["peers"], http_port=c["port"],
                    mem_pool_size_gb=1, send_type="PUT", recv_timeout_s=30,
                    request_id_safety=True,verify_recompute=c.get("verify_recompute",False)))
            args['kv_transfer_config'].kv_connector_extra_config['verify_transport']=c.get('verify_transport',False)
        self.engine = LLMEngine.from_engine_args(EngineArgs(**args))
        hf=self.engine.model_config.hf_config
        self.transfer_bytes_per_token=(2*hf.num_hidden_layers*(hf.num_key_value_heads//c['tp'])*
            (hf.hidden_size//hf.num_attention_heads)*2 + hf.hidden_size*2)
        self.transfer_snapshot={}
        self.transfer_snapshot_s=0
        # Capture scheduling, then attach execution boundaries only after step
        # succeeds. A scheduling intent alone is never treated as execution.
        self.scheduled = []
        for scheduler in self.engine.scheduler:
            original = scheduler.schedule
            def record(original=original):
                output = original()
                metadata, scheduled, _ = output
                p = scheduled.num_prefill_groups
                d = len(scheduled.scheduled_seq_groups) - p
                self.scheduled.append(dict(prefill=p, decode=d,
                    tokens=scheduled.num_batched_tokens,
                    request_ids=[m.request_id for m in metadata]))
                return output
            scheduler.schedule = record
        self.event_file = self.timeline.open("a", buffering=1)
        self.update_snapshot()

    def control_rpc(self, method):
        """Run only on the owner thread, between model steps.

        V0 TP workers stay inside their execution loop after a step. They
        cannot consume RPC messages until the driver broadcasts loop exit.
        The next execute_model call restarts them without reloading weights.
        """
        executor = self.engine.model_executor
        executor.stop_remote_worker_execution_loop()
        return executor.collective_rpc(method)

    def update_snapshot(self):
        if self.config.get('peers') and time.monotonic()-self.transfer_snapshot_s>.1:
            from vllm.pdblend_runtime import transfer_state
            transfers=self.control_rpc(transfer_state)
            allocations={rid:max(s.get('allocations',{}).get(rid,0) for s in transfers)
                         for s in transfers for rid in s.get('allocations',{})}
            self.transfer_snapshot=dict(
                free_transfer_bytes=max(0,min(s['buffer_capacity_bytes'] for s in transfers)-sum(allocations.values())),
                transfer_bytes_per_token=self.transfer_bytes_per_token,
                transfer_allocations=allocations,
                transport_healthy=all(s['listener_alive'] for s in transfers))
            self.transfer_snapshot_s=time.monotonic()
        sched = self.engine.scheduler[0]
        blocks = sched.block_manager
        block_size = self.engine.cache_config.block_size
        allocations = {}
        for group in sched.running:
            unfinished=[seq for seq in group.get_seqs() if not seq.is_finished()]
            if unfinished:
                allocations[group.request_id] = sum(len(blocks.get_block_table(seq)) * block_size
                                                    for seq in unfinished)
        self.snapshot = dict(id=self.config["id"], timestamp=time.time(),
            **self.transfer_snapshot,
            **self.state, running=len(allocations), waiting=len(sched.waiting),
            free_kv_tokens=blocks.get_num_free_gpu_blocks() * block_size,
            total_kv_tokens=self.engine.cache_config.num_gpu_blocks * block_size,
            kv_allocations=allocations,
            diagnostic_recompute=self.config.get("verify_recompute", False),
            diagnostic_transport=self.config.get('verify_transport',False),
            runtime_error=getattr(sched, "_pdblend_runtime_error", None),
            acknowledged_generation=getattr(sched, "_pdblend_runtime", {}).get("generation", -1))

    def add(self, rid, body):
        from vllm import SamplingParams
        from vllm.pdblend_runtime import parse_transfer
        meta = parse_transfer(rid)
        phase = meta["phase"] if meta else "m"
        if meta:
            pair=[self.config["peers"][meta["source"]]["tp"],self.config["peers"][meta["target"]]["tp"]]
            validated=self.config.get("validated_tp_pairs",[[1,1]])
            if pair not in validated and not (self.config.get("verify_recompute",False) or
                                             self.config.get('verify_transport',False)):
                raise ValueError("TP transfer pair has not passed numerical validation")
        if phase != {"mixed": "m", "prefill": "p", "decode": "d"}[self.state["role"]]:
            raise ValueError("request phase does not match resident role")
        prompt = body["prompt"]
        if isinstance(prompt, list):
            prompt = {"prompt_token_ids": prompt}
        params = SamplingParams(temperature=body.get("temperature", 0),
            top_p=body.get("top_p", 1), max_tokens=int(body["max_tokens"]),
            ignore_eos=body.get("ignore_eos", True), seed=body.get("seed", 0))
        self.engine.add_request(rid, prompt, params)
        self.update_snapshot()

    def step(self):
        self.scheduled.clear()
        started = time.time()
        outputs = self.engine.step()
        ended = time.time()
        for event in self.scheduled:
            self.event_file.write(json.dumps(dict(event, started_s=started,
                finished_s=ended, generation=self.state["generation"],
                role=self.state["role"], mode=self.state["mode"])) + "\n")
        self.update_snapshot()
        return outputs

    async def loop(self):
        try:
            while True:
                if not self.streams:
                    self.wake.clear()
                    try:
                        await asyncio.wait_for(self.wake.wait(),timeout=.1)
                    except asyncio.TimeoutError:
                        # Incoming KV may change while this engine has no
                        # scheduled request. Observe it without faking freshness.
                        await self.call(self.update_snapshot)
                        continue
                outputs = await self.call(self.step)
                for output in outputs:
                    queue = self.streams.get(output.request_id)
                    if queue is None:
                        continue
                    try:
                        queue.put_nowait(output)
                    except asyncio.QueueFull:
                        # Slow/disconnected consumers cannot block every request.
                        await self.call(self.engine.abort_request, output.request_id)
                        while not queue.empty():
                            queue.get_nowait()
                        queue.put_nowait(RuntimeError("stream backpressure limit"))
                await asyncio.sleep(.001 if not outputs else 0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.fail(str(exc))

    async def completions(self, request):
        if not self.accepting or self.error:
            raise web.HTTPServiceUnavailable(text=self.error or "instance draining")
        if len(self.streams) >= self.config.get("max_pending", 128):
            raise web.HTTPTooManyRequests(text="bounded admission queue full")
        body = await request.json()
        rid = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        if rid in self.streams or rid in self.finished:
            raise web.HTTPConflict(text="duplicate request id")
        queue = asyncio.Queue(maxsize=64)
        self.streams[rid] = queue
        response = None
        try:
            try:
                await self.call(self.add, rid, body)
            except (ValueError, KeyError, TypeError) as exc:
                raise web.HTTPBadRequest(text=str(exc)) from exc
            self.wake.set()
            streaming = body.get("stream", False)
            if streaming:
                response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await response.prepare(request)
            text_so_far, tokens_so_far = "", 0
            token_times, token_ids = [], []
            while True:
                output = await asyncio.wait_for(queue.get(), timeout=120)
                if isinstance(output, Exception):
                    raise output
                completion = output.outputs[0]
                ids = list(completion.token_ids)
                new_ids = ids[tokens_so_far:]
                now = time.time()
                token_times.extend([now] * len(new_ids))
                token_ids.extend(new_ids)
                delta = completion.text[len(text_so_far):]
                text_so_far, tokens_so_far = completion.text, len(ids)
                usage = dict(prompt_tokens=len(output.prompt_token_ids),
                             completion_tokens=len(ids), total_tokens=len(output.prompt_token_ids)+len(ids))
                if streaming:
                    event = dict(id=rid, choices=[dict(text=delta, index=0,
                        finish_reason=completion.finish_reason)],
                        token_ids=new_ids, token_index=len(ids),
                        usage=usage if output.finished else None)
                    await response.write(("data: " + json.dumps(event) + "\n\n").encode())
                if output.finished:
                    break
            if streaming:
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
                return response
            return web.json_response(dict(id=rid, choices=[dict(text=text_so_far,
                index=0, finish_reason=completion.finish_reason)], usage=usage,
                token_ids=token_ids, token_received_s=token_times))
        except web.HTTPException:
            raise
        except (ConnectionError, asyncio.CancelledError):
            raise
        except Exception as exc:
            if response is not None:
                with suppress(ConnectionError):
                    await response.write(("data: " + json.dumps({"error": str(exc)}) + "\n\n").encode())
                return response
            raise web.HTTPInternalServerError(text=str(exc)) from exc
        finally:
            try:
                if not self.error:
                    await self.call(self.engine.abort_request,rid)
            finally:
                self.streams.pop(rid,None)
                self.finished.add(rid)
                # Tombstones bound memory; nonce generation belongs to the router.
                if len(self.finished)>100000: self.finished.pop()

    async def control(self, request):
        payload = await request.json()
        payload.setdefault("admit_decode",True)
        async with self.control_lock:
            generation = payload.get("generation")
            if generation == self.state["generation"] and payload == self.state:
                return web.json_response(self.state)
            if type(generation) is not int or generation != self.state["generation"] + 1:
                raise web.HTTPConflict(text="expected next generation")
            if (payload.get("role") not in ("mixed", "prefill", "decode")
                    or payload.get("mode") not in ("continuous", "temporal")
                    or type(payload.get("admit_prefill")) is not bool
                    or type(payload.get("admit_decode")) is not bool):
                raise web.HTTPBadRequest(text="invalid role or scheduling mode")
            old = self.state.copy()
            self.accepting = False
            try:
                # Same-role temporal admission can change between engine steps;
                # role/mode changes drain all requests first.
                if (payload["role"], payload["mode"]) != (old["role"], old["mode"]):
                    deadline = time.monotonic() + 30
                    while self.streams:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("role change did not drain")
                        await asyncio.sleep(.01)
                await self.call(self.commit_state, payload)
                if self.snapshot["acknowledged_generation"] != generation:
                    raise RuntimeError("engine did not acknowledge generation")
            except Exception as exc:
                # Rollback itself has a newer version; stale acknowledgements
                # cannot reapply a rejected role change.
                rollback = dict(old, generation=generation + 1)
                await self.call(self.commit_state, rollback)
                raise web.HTTPConflict(text="control rolled back: " + str(exc)) from exc
            finally:
                self.accepting = not bool(self.error)
            return web.json_response(self.state)

    def commit_state(self, payload):
        from vllm.pdblend_runtime import read_runtime, transfer_state
        if payload["role"] != self.state["role"]:
            states = self.control_rpc(transfer_state)
            if any(s["buffered_tensors"] or s["inflight_receives"]
                   or not s["listener_alive"] for s in states):
                raise RuntimeError("transport has not drained")
        self.write_state(payload)
        self.state = payload.copy()
        for scheduler in self.engine.scheduler:
            read_runtime(scheduler)
        self.update_snapshot()

    async def status(self, request):
        stamp = self.snapshot.get("timestamp", 0)
        return web.json_response(dict(self.snapshot, timestamp=stamp, active=len(self.streams),
                                      accepting=self.accepting and self.snapshot.get('transport_healthy',True), error=self.error))

    async def provenance(self,request):
        import vllm
        return web.json_response(dict(instance_id=self.config['id'],pid=os.getpid(),
            engine_version=vllm.__version__,source_files_at_import=SOURCE_FILES_AT_IMPORT,
            model=self.config['model'],tp=self.config['tp'],dtype='bfloat16',
            max_model_len=self.config.get('max_model_len',8192),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))

    async def cancel(self, request):
        from vllm.pdblend_runtime import transfer_state, parse_transfer
        payload = await request.json()
        rid = payload["request_id"]
        meta = parse_transfer(rid)
        targets = [key for key in self.streams if key == rid or
                   (meta and (parse_transfer(key) or {}).get("nonce") == meta["nonce"])]
        for key in targets:
            await self.call(self.engine.abort_request, key)
            queue = self.streams.get(key)
            if queue is not None:
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(RuntimeError("request cancelled"))
        states = await self.call(self.control_rpc,
                                  partial(transfer_state, cancel_nonce=meta["nonce"] if meta else rid))
        return web.json_response(dict(cancelled=rid, transfers=states))

    async def prepare_peers(self, request):
        from vllm.pdblend_runtime import prepare_peers
        payload=await request.json()
        async with self.control_lock:
            if self.streams:
                raise web.HTTPConflict(text="peer preparation requires drained instance")
            peers=payload["peers"]
            if any(p not in self.config.get("peers",{}) or p==self.config["id"] for p in peers):
                raise web.HTTPBadRequest(text="unknown or local peer")
            self.accepting=False
            try:
                result=await self.call(self.control_rpc,
                                      partial(prepare_peers,peer_ids=peers))
            finally:
                self.accepting=not bool(self.error)
        return web.json_response(dict(ready=result))

    async def drain(self,request):
        from vllm.pdblend_runtime import transfer_state
        payload=await request.json()
        async with self.control_lock:
            if payload.get('expected_generation')!=self.state['generation']:
                raise web.HTTPConflict(text='stale drain generation')
            self.accepting=False
            try:
                deadline=time.monotonic()+30
                while self.streams:
                    if time.monotonic()>deadline:
                        raise TimeoutError('drain deadline expired')
                    await asyncio.sleep(.01)
                transfers=await self.call(self.control_rpc,transfer_state)
                if any(s['buffered_tensors'] or s['inflight_receives'] or not s['listener_alive'] for s in transfers):
                    raise RuntimeError('transport did not drain')
                await self.call(self.commit_state,dict(self.state,generation=self.state['generation']+1,
                    admit_prefill=False))
                return web.json_response(dict(self.state,accepting=False,drained=True))
            except Exception:
                self.accepting=not bool(self.error)
                raise

    async def retain_weights(self,request):
        from vllm.pdblend_weights import export_weights
        from vllm.pdblend_runtime import transfer_state
        payload=await request.json()
        # Directory names are generated by the local topology manager. Do not
        # turn this endpoint into an arbitrary filesystem writer.
        transaction=payload.get('transaction','')
        if not transaction or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in transaction):
            raise web.HTTPBadRequest(text='invalid transaction identifier')
        root=Path(self.config.get('weight_cache_root',self.runtime_path.parent.parent/'weights'))/transaction
        async with self.control_lock:
            if payload.get('expected_generation')!=self.state['generation']:
                raise web.HTTPConflict(text='stale weight retention transaction')
            self.accepting=False
            try:
                deadline=time.monotonic()+30
                while self.streams:
                    if time.monotonic()>deadline:
                        raise TimeoutError('weight retention did not drain')
                    await asyncio.sleep(.01)
                transfers=await self.call(self.control_rpc,transfer_state)
                if any(t['buffered_tensors'] or t['inflight_receives'] or not t['listener_alive'] for t in transfers):
                    raise RuntimeError('KV transport must drain before retaining weights')
                ranks=await self.call(self.control_rpc,partial(export_weights,directory=str(root)),timeout_s=600)
                identities={r['model_config_sha256'] for r in ranks}
                if len(identities)!=1 or len(ranks)!=self.config['tp']:
                    raise RuntimeError('source weight ranks disagree')
                manifest=dict(schema=1,complete=True,tp=self.config['tp'],
                    model_config_sha256=identities.pop(),ranks=sorted(ranks,key=lambda r:r['rank']))
                await asyncio.to_thread((root/'manifest.json').write_text,json.dumps(manifest))
                return web.json_response(dict(retained_weights=str(root),manifest=manifest,
                    generation=self.state['generation'],accepting=False))
            except Exception:
                self.accepting=not bool(self.error)
                raise

    async def register_peer(self,request):
        from vllm.pdblend_runtime import register_peer
        payload=await request.json()
        peer_id=payload.get('id','')
        peer=payload.get('peer',{})
        if (not peer_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in peer_id)
                or peer.get('host')!='127.0.0.1' or peer.get('tp') not in (1,2,4,8)
                or type(peer.get('kv_port')) is not int or not 1024<=peer['kv_port']<=65000):
            raise web.HTTPBadRequest(text='invalid local peer')
        async with self.control_lock:
            if peer_id in self.config.get('peers',{}) and self.config['peers'][peer_id]!=peer:
                raise web.HTTPConflict(text='peer identities are immutable; allocate a new versioned id')
            ranks=await self.call(self.control_rpc,partial(register_peer,peer_id=peer_id,peer=peer))
            self.config.setdefault('peers',{})[peer_id]=peer
            return web.json_response(dict(id=peer_id,registered=ranks))

    async def start(self, app):
        await self.call(self.initialize,timeout_s=900)
        self.task = asyncio.create_task(self.loop())

    async def stop(self, app):
        self.accepting = False
        self.task.cancel()
        with suppress(asyncio.CancelledError):
            await self.task
        try:
            if not self.error: await self.call(self.event_file.close)
        finally:
            self.worker.shutdown(wait=False,cancel_futures=True)

    def application(self):
        app = web.Application(client_max_size=16 * 1024**2)
        app.router.add_post("/v1/completions", self.completions)
        app.router.add_get("/health", self.status)
        app.router.add_get("/runtime", self.status)
        app.router.add_get('/provenance',self.provenance)
        app.router.add_post("/control", self.control)
        app.router.add_post("/cancel", self.cancel)
        app.router.add_post("/prepare-peers", self.prepare_peers)
        app.router.add_post("/retain-weights", self.retain_weights)
        app.router.add_post("/drain", self.drain)
        app.router.add_post("/register-peer", self.register_peer)
        app.on_startup.append(self.start)
        app.on_cleanup.append(self.stop)
        return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    service = EngineService(config)
    web.run_app(service.application(), host="127.0.0.1", port=config["port"], access_log=None)


if __name__ == "__main__":
    main()
