"""End-to-end CPU test: fake vLLM engines (aiohttp) behind the real proxy, controller and load client."""
import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web

from pdblend.bench.client import LoadClient, Request, slo_attainment
from pdblend.bench.run import _point, offline_forecast, window_energy
from pdblend.control.planner import SLO
from pdblend.control.policies import get_policy
from synthetic import synthetic_model


class FakeEngineServer:
    """Streams `max_tokens` tokens at a fixed cadence; supports remote prefill/decode handoff and sleep."""

    def __init__(self, gpu: int, tpot_s: float = 0.005):
        self.gpu, self.tpot_s = gpu, tpot_s
        self.state = "ready"
        self.seen_kv_params = []
        self.seen_request_ids = []
        self.seen_bodies = []
        self.native_active = set()
        self.native_cancelled = set()
        self.native_accepting = True
        self.native_tp, self.native_pp, self.native_generation = 1, 1, 0
        self.app = web.Application(client_max_size=64 * 1024 * 1024)
        self.app.router.add_post("/v1/completions", self.completions)
        self.app.router.add_get("/health", self.health)
        self.app.router.add_post("/sleep", self.sleep)
        self.app.router.add_post("/wake_up", self.wake)
        # CPU contract fixtures only; none of these receipts qualify hardware.
        self.app.router.add_get('/baseline/state', self.native_state)
        self.app.router.add_post('/baseline/drain', self.native_drain)
        self.app.router.add_post('/baseline/control', self.native_control)
        self.app.router.add_post('/baseline/cancel', self.native_cancel)

    def native_snapshot(self):
        return dict(generation=self.native_generation, tp=self.native_tp, pp=self.native_pp,
                    native_at_s=time.time(), native_evidence_complete=True, transport_healthy=True,
                    healthy=True, accepting=self.native_accepting, all_queue=sorted(self.native_active),
                    running=sorted(self.native_active), waiting=[], retained_kv_requests=[],
                    kv_allocations={rid: [1] for rid in self.native_active},
                    pending_transfers=0, transfer_allocations={}, total_blocks=4096,
                    free_blocks=4096-len(self.native_active), reserved_blocks=0,
                    ranks=[dict(rank=rank, generation=self.native_generation, native_evidence_complete=True,
                                healthy=True, at_s=time.time(), pending_transfers=0, transfer_allocations={})
                           for rank in range(self.native_tp * self.native_pp)])

    async def native_state(self, request):
        return web.json_response(self.native_snapshot())

    async def native_drain(self, request):
        payload = await request.json()
        self.native_accepting = False
        deadline = time.monotonic() + payload.get('timeout_s', 10)
        while self.native_active and time.monotonic() < deadline:
            await asyncio.sleep(.005)
        return web.json_response(dict(self.native_snapshot(), acknowledged=not self.native_active,
                                      drained=not self.native_active))

    async def native_control(self, request):
        payload = await request.json()
        self.native_accepting = payload.get('accepting', self.native_accepting)
        self.native_generation = payload.get('generation', self.native_generation)
        return web.json_response(dict(self.native_snapshot(), acknowledged=True))

    async def native_cancel(self, request):
        payload = await request.json()
        rid = payload['request_id']
        self.native_cancelled.add(rid)
        self.native_active.discard(rid)
        return web.json_response(dict(acknowledged=True, cancelled=True, request_id=rid,
                                      generation=self.native_generation, native_state=self.native_snapshot()))

    async def health(self, request):
        return web.json_response({})

    async def sleep(self, request):
        self.state = f"sleep{request.query.get('level', '1')}"
        return web.json_response({})

    async def wake(self, request):
        self.state = "ready"
        return web.json_response({})

    async def completions(self, request):
        rid = request.headers['X-Request-Id']
        self.native_active.add(rid)
        try:
            return await self._completion(request)
        finally:
            self.native_active.discard(rid)

    async def _completion(self, request):
        body = await request.json()
        self.seen_bodies.append(body)
        assert "request_id" not in body
        self.seen_request_ids.append(request.headers["X-Request-Id"])
        kv = body.get("kv_transfer_params")
        if kv:
            self.seen_kv_params.append(kv)
        await asyncio.sleep(0.002 + len(body["prompt"]) * 2e-6)
        if not body.get("stream", True):
            payload = dict(choices=[dict(text="t0", index=0, finish_reason="length",
                           logprobs=dict(tokens=["token_id:9000"]) if body.get("logprobs") is not None else None)],
                           usage=dict(prompt_tokens=len(body["prompt"]), completion_tokens=1))
            if kv and kv.get("do_remote_decode"):
                payload["kv_transfer_params"] = dict(remote_engine_id=f"e{self.gpu}", remote_block_ids=[1, 2],
                                                     remote_host="127.0.0.1", remote_port=1234)
            return web.json_response(payload)
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        n = int(body.get("max_tokens", 1))
        first = 1 if body['prompt'][-1] == 9000 and (self.seen_request_ids[-1].startswith('___prefill_addr_') or
                     (kv and kv.get('do_remote_prefill'))) else 0
        for i in range(n):
            if request.headers['X-Request-Id'] in self.native_cancelled:
                return resp
            choice=dict(text=f"t{first+i}",index=0)
            if body.get('logprobs') is not None:choice['logprobs']=dict(tokens=[f'token_id:{9000+first+i}'])
            await resp.write(b"data: " + json.dumps(dict(choices=[choice])).encode() + b"\n\n")
            await asyncio.sleep(self.tpot_s)
        await resp.write(b"data: " + json.dumps(dict(choices=[], usage=dict(prompt_tokens=len(body["prompt"]), completion_tokens=n))).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp


class FakeInstance:
    def __init__(self, iid, gpu, port, kv_connector="NixlConnector"):
        self.spec = SimpleNamespace(instance_id=iid, max_num_seqs=256, max_model_len=8192,
                                    gpus=(gpu,), base_url=f"http://127.0.0.1:{port}",
                                    kv_connector=kv_connector, zmq_address=f"127.0.0.1:{port + 20000}",
                                    tp=1, pp=1, generation=0)
        self.server = FakeEngineServer(gpu)
        self.runner = None

    async def serve(self):
        self.server.native_tp, self.server.native_pp = self.spec.tp, self.spec.pp
        self.server.native_generation = self.spec.generation
        self.runner = web.AppRunner(self.server.app)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", int(self.spec.base_url.rsplit(":", 1)[1])).start()

    def start(self): self.server.state = "ready"
    def wait_ready(self): return 0.0
    def stop(self, timeout_s=30.0): self.server.state = "off"
    def sleep(self, level=1): self.server.state = f"sleep{level}"; return 0.0
    def wake_up(self): self.server.state = "ready"; return 0.0


class FakeFleet:
    def __init__(self, n, base_port=18300):
        self.instances = {f"i{k}": FakeInstance(f"i{k}", k, base_port + k) for k in range(n)}

    def __getitem__(self, iid): return self.instances[iid]
    def events(self): return []


class FakeSampler:
    def __init__(self, gpus):
        self.gpus, self.samples, self._task = gpus, [], None
        self.utilization_samples, self.frequency_samples = [], []
        self.frequency_readings, self.frequency_errors = [], []
        self.frequency_requested = None
        self._frequency_lock = threading.Lock()

    def capture_frequency(self, requested=None, reason='explicit'):
        # A deterministic CPU sensor, retaining the fixture's fixed 2520 MHz.
        # Transition hooks call this from a thread, alongside periodic samples.
        with self._frequency_lock:
            if requested is None and self.frequency_requested is not None:
                requested = self.frequency_requested()
            requested = requested or {}
            stamp = time.time()
            rows = [dict(gpu=gpu, requested_mhz=requested.get(gpu), observed_mhz=2520.,
                         read_started_s=stamp, read_finished_s=stamp, t_s=stamp,
                         error=None, reason=reason, source_id='cpu_fixture:constant_clock')
                    for gpu in self.gpus]
            self.frequency_readings.extend(rows)
            self.frequency_samples.append((stamp, [2520.] * len(self.gpus)))
            return rows
    def start(self):
        self._task = asyncio.get_event_loop().create_task(self._loop())
    async def _loop(self):
        while True:
            self.samples.append((time.time(), [100.0] * len(self.gpus)))
            self.utilization_samples.append((time.time(), [50.0] * len(self.gpus)))
            self.capture_frequency(reason='periodic')
            await asyncio.sleep(0.02)
    def stop(self):
        if self._task: self._task.cancel()


class FakeGpus:
    def __init__(self, gpus): self.gpus, self.clocks = list(gpus), {}
    def set_clock(self, g, mhz): self.clocks[g] = mhz
    def reset_clock(self, g): self.clocks[g] = None
    def park(self, g): self.clocks[g] = "parked"
    def unpark(self, g): self.clocks[g] = None
    def reset_all(self): pass
    def sampler(self, gpus=None, interval_s=0.1): return FakeSampler(self.gpus)


def make_trace(n=40, rate=40.0, in_tokens=300, out_tokens=8):
    return [Request(i, i / rate, [7] * in_tokens, out_tokens, "t") for i in range(n)]


@pytest.mark.parametrize("policy", ["mixed", "pdblend", "distserve_static", "dynamollm", "ecoserve"])
def test_point_end_to_end(tmp_path, policy):
    fleet = FakeFleet(4)
    model = synthetic_model()
    slo = SLO(5.0, 0.15)

    async def go():
        for inst in fleet.instances.values():
            await inst.serve()
        try:
            return await _point(fleet, FakeGpus([0, 1, 2, 3]), model, get_policy(policy), slo, make_trace(),
                                make_trace(4, rate=10.0), tmp_path, 18299, period_s=0.5, tail_timeout_s=10,
                                min_warm_s=0.0)
        finally:
            for inst in fleet.instances.values():
                await inst.runner.cleanup()
    result = asyncio.run(go())
    att = result["slo"]
    assert att["offered"] == 40 and att["succeeded"] == 40, att
    assert att["joint_slo_rate"] == 1.0
    assert result["energy_j"] > 0 and result["window_energy_j"] <= result["energy_j"] + 1e-6
    assert (tmp_path / "outcomes.jsonl").exists() and (tmp_path / "controller.jsonl").exists()
    roles = set(result["final_roles"].values())
    if policy == "mixed":
        assert roles == {"M"}
    elif policy == "pdblend":
        # The four-slot fake fleet is below the PDblend M>=4 safety floor;
        # all-M is therefore the expected safe layout in this unit test.
        assert roles == {"M"} or roles & {"L1", "off"}
    elif policy == "distserve_static":
        assert roles == {"P", "D"}
    elif policy == "dynamollm":
        # This direct _point smoke lasts <60 s and supplies no history replay.
        # ScaleInst must fail open until a complete observed bin is available.
        assert roles == {"M"}
        assert result["controller"]["events"].get("park", 0) == 0
    else:
        assert roles <= {"M", "idle"}


def test_pd_path_through_proxy(tmp_path):
    """Force a P/D split and check the fake engines saw remote_decode then remote_prefill params."""
    from pdblend.proxy.router import Router
    from pdblend.proxy.server import Proxy
    fleet = FakeFleet(2, base_port=18400)

    async def go():
        for inst in fleet.instances.values():
            await inst.serve()
        urls = {i: inst.spec.base_url for i, inst in fleet.instances.items()}
        router = Router(list(urls))
        router.set_roles({"i0": "P", "i1": "D"})
        proxy = Proxy(urls, router)
        runner = web.AppRunner(proxy.app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 18399).start()
        try:
            outs = await LoadClient("http://127.0.0.1:18399").replay(make_trace(5, rate=50.0))
        finally:
            await runner.cleanup()
            for inst in fleet.instances.values():
                await inst.runner.cleanup()
        return outs
    outs = asyncio.run(go())
    assert all(o.error is None and o.path == "PD" and o.completion_tokens == 8 for o in outs)
    p, d = fleet["i0"].server, fleet["i1"].server
    assert all(k["do_remote_decode"] for k in p.seen_kv_params) and len(p.seen_kv_params) == 5
    assert all(k["do_remote_prefill"] and k["remote_block_ids"] == [1, 2] for k in d.seen_kv_params)
    assert p.seen_request_ids == d.seen_request_ids == [f"r{i}" for i in range(5)]


def test_pd_path_through_proxy_p2p_nccl(tmp_path):
    """P2pNccl: both legs share a routing id naming the two ZMQ addresses and carry no kv_transfer_params."""
    from pdblend.engine.client import PDTransfer
    from pdblend.proxy.router import Router
    from pdblend.proxy.server import Proxy
    fleet = FakeFleet(2, base_port=18500)

    async def go():
        for inst in fleet.instances.values():
            await inst.serve()
        urls = {i: inst.spec.base_url for i, inst in fleet.instances.items()}
        router = Router(list(urls))
        router.set_roles({"i0": "P", "i1": "D"})
        transfer = PDTransfer("P2pNcclConnector", {i: inst.spec.zmq_address for i, inst in fleet.instances.items()})
        proxy = Proxy(urls, router, transfer=transfer)
        runner = web.AppRunner(proxy.app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 18499).start()
        try:
            outs = await LoadClient("http://127.0.0.1:18499").replay(make_trace(3, rate=50.0))
        finally:
            await runner.cleanup()
            for inst in fleet.instances.values():
                await inst.runner.cleanup()
        return outs
    outs = asyncio.run(go())
    assert all(o.error is None and o.path == "PD" and o.completion_tokens == 8 for o in outs)
    p, d = fleet["i0"].server, fleet["i1"].server
    assert not p.seen_kv_params and not d.seen_kv_params
    expected = [f"___prefill_addr_127.0.0.1:38500___decode_addr_127.0.0.1:38501_r{i}" for i in range(3)]
    assert p.seen_request_ids == d.seen_request_ids == expected


@pytest.mark.parametrize("path", ["M", "PD", "ecoserve"])
@pytest.mark.parametrize("chunking", ["separate", "coalesced", "fragmented"])
def test_stream_updates_live_token_observations(monkeypatch, path, chunking):
    from unittest.mock import AsyncMock

    from pdblend.control.policies.baselines import EcoRouter
    from pdblend.control.shield import Shield
    from pdblend.proxy.router import Router
    from pdblend.proxy.server import Proxy

    now = [1000.0]
    monkeypatch.setattr("pdblend.proxy.router.time.time", lambda: now[0])
    slo = SLO(5.0, 0.1)
    roles = {"i0": "P", "i1": "D"} if path == "PD" else {"i0": "M", "i1": "M"}
    router = EcoRouter(list(roles), synthetic_model(), slo) if path == "ecoserve" else Router(list(roles))
    router.set_roles(roles)
    record = router.dispatch("r", 100, 7)
    proxy = Proxy({i: f"http://127.0.0.1:{8100 + k}" for k, i in enumerate(roles)}, router)
    event = b'data: {"choices":[{"text":"%s"}]}\n\n'
    empty, a, b, c = [event % text for text in (b"", b"a", b"b", b"c")]
    finish = b'data: {"choices":[{"text":"","finish_reason":"length"}]}\n\n'
    usage = b'data: {"choices":[],"usage":{"completion_tokens":7}}\n\n'
    if chunking == "separate":
        chunks = [(empty, 0), (a, 1), (b, 2), (c, 3), (finish, 3), (usage, 3)]
    elif chunking == "coalesced":
        chunks = [(empty + a + b, 2), (c + finish + usage, 3)]
    else:
        cut = a.index(b'"text":"') + len(b'"text":"')
        chunks = [(empty + a[:cut], 0), (a[cut:] + b + c[:cut], 2), (c[cut:] + finish + usage, 3)]

    async def stream_chunks():
        first_at = None
        for chunk, expected in chunks:
            now[0] += 0.1
            yield chunk
            assert record.tokens_so_far == expected
            assert record.finished_s is None and record.completion_tokens == 0
            assert record in router.active[record.decode_instance]
            assert router.loads[record.decode_instance].inflight_seqs == 1
            if expected:
                if first_at is None:
                    first_at = now[0]
                assert record.first_token_s == first_at
                assert router.loads[record.prefill_instance].inflight_prefill_tokens == 0
            else:
                assert record.first_token_s is None
                assert router.loads[record.prefill_instance].inflight_prefill_tokens == 100
        yield b"data: [DONE]\n\n"

    class Upstream:
        status = 200
        content = SimpleNamespace(iter_any=stream_chunks)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    proxy.session = SimpleNamespace(post=lambda *args, **kwargs: Upstream())
    response = SimpleNamespace(write=AsyncMock())
    assert asyncio.run(proxy._stream_leg(record, {}, "r", response)) == 7
    assert record.tokens_so_far == 3
    assert b"".join(call.args[0] for call in response.write.await_args_list) == empty + a + b + c + finish + usage
    probe_at = record.first_token_s + 1.0
    pressure = Shield(slo).observe([record], probe_at)
    assert pressure.decode and pressure.tpot_p90 == pytest.approx(0.5)
    if path == "ecoserve":
        assert router._slack_ms(record, probe_at) == pytest.approx(-700.0)
    router.finish(record, 7)
    assert record.completion_tokens == 7
    assert router.loads[record.decode_instance].inflight_seqs == 0
    assert router.loads[record.prefill_instance].inflight_prefill_tokens == 0


def test_window_energy_and_offline_forecast():
    samples = [(0.0, [50.0, 50.0]), (1.0, [50.0, 50.0]), (2.0, [150.0, 50.0])]
    e, w = window_energy(samples, 0.0, 2.0)
    assert abs(e - (100 + 150)) < 1e-9 and abs(w - 125) < 1e-9
    fc = offline_forecast(make_trace(20, rate=10.0))
    assert abs(fc.rate_rps - 20 / 1.9) < 1e-6 and fc.input_mean == 300 and fc.output_mean == 8


def test_resident_tp_uses_real_proxy_and_separate_pool_controllers(tmp_path):
    """CPU HTTP integration proves wiring, never GPU rank/KV correctness."""
    from pdblend.bench.tp_runtime import mechanism_pd_plan
    from pdblend.control.planner import Plan
    fleet = FakeFleet(3, base_port=19500)
    small, large = synthetic_model(), synthetic_model()
    large.tp = 2
    for index, instance in enumerate(fleet.instances.values()):
        instance.spec.tp = 1 if index == 0 else 2
        instance.spec.pp = 1
        instance.spec.pool_id = 'small' if index == 0 else 'large'
        instance.spec.generation = 2
        instance.spec.model = 'synthetic'
        instance.spec.profile_key = f'test-pool-{instance.spec.pool_id}'
        instance.spec.gpus = (0,) if index == 0 else ((1, 2) if index == 1 else (3, 4))
    trace = [Request(i, i / 10, [7] * (512 if i % 2 == 0 else 2048), 8, 'resident-test') for i in range(8)]
    fixed = {'small': Plan({'M': 1}, 2520, 2520, 2520, 1024, 0, 0, 0),
             'large': mechanism_pd_plan(2)}

    async def go():
        for instance in fleet.instances.values():
            await instance.serve()
        try:
            return await _point(fleet, FakeGpus(range(5)), small, get_policy('pdblend'), SLO(5, .15),
                                trace, [], tmp_path, 19499, .5, 10, min_warm_s=0,
                                sampling_seed=701, pool_models={'small': small, 'large': large},
                                pool_fixed_plans=fixed)
        finally:
            for instance in fleet.instances.values():
                await instance.runner.cleanup()
    result = asyncio.run(go())
    assert result['slo']['succeeded'] == 8
    assert result['controller']['mode'] == 'resident_hetero_tp'
    routes = [json.loads(line) for line in (tmp_path / 'routes.jsonl').read_text().splitlines()]
    assert {row['tp'] for row in routes} == {1, 2}
    assert {row['path'] for row in routes} == {'M', 'PD'}
    assert all(row['generation'] == 2 and row['profile_key'] for row in routes)
    assert not result['quarantined_instances']
    assert (tmp_path / 'pools/small/controller.jsonl').is_file()
    assert (tmp_path / 'pools/large/controller.jsonl').is_file()
