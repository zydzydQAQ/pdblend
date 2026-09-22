"""End-to-end CPU test: fake vLLM engines (aiohttp) behind the real proxy, controller and load client."""
import asyncio
import json
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
        self.app = web.Application(client_max_size=64 * 1024 * 1024)
        self.app.router.add_post("/v1/completions", self.completions)
        self.app.router.add_get("/health", self.health)
        self.app.router.add_post("/sleep", self.sleep)
        self.app.router.add_post("/wake_up", self.wake)

    async def health(self, request):
        return web.json_response({})

    async def sleep(self, request):
        self.state = f"sleep{request.query.get('level', '1')}"
        return web.json_response({})

    async def wake(self, request):
        self.state = "ready"
        return web.json_response({})

    async def completions(self, request):
        body = await request.json()
        assert "request_id" not in body
        self.seen_request_ids.append(request.headers["X-Request-Id"])
        kv = body.get("kv_transfer_params")
        if kv:
            self.seen_kv_params.append(kv)
        await asyncio.sleep(0.002 + len(body["prompt"]) * 2e-6)
        if not body.get("stream", True):
            payload = dict(choices=[dict(text="x", index=0)],
                           usage=dict(prompt_tokens=len(body["prompt"]), completion_tokens=1))
            if kv and kv.get("do_remote_decode"):
                payload["kv_transfer_params"] = dict(remote_engine_id=f"e{self.gpu}", remote_block_ids=[1, 2],
                                                     remote_host="127.0.0.1", remote_port=1234)
            return web.json_response(payload)
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        n = int(body.get("max_tokens", 1))
        for i in range(n):
            await resp.write(b"data: " + json.dumps(dict(choices=[dict(text=f"t{i}", index=0)])).encode() + b"\n\n")
            await asyncio.sleep(self.tpot_s)
        await resp.write(b"data: " + json.dumps(dict(choices=[], usage=dict(prompt_tokens=len(body["prompt"]), completion_tokens=n))).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp


class FakeInstance:
    def __init__(self, iid, gpu, port, kv_connector="NixlConnector"):
        self.spec = SimpleNamespace(instance_id=iid, gpus=(gpu,), base_url=f"http://127.0.0.1:{port}",
                                    kv_connector=kv_connector, zmq_address=f"127.0.0.1:{port + 20000}")
        self.server = FakeEngineServer(gpu)
        self.runner = None

    async def serve(self):
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
    def start(self):
        self._task = asyncio.get_event_loop().create_task(self._loop())
    async def _loop(self):
        while True:
            self.samples.append((time.time(), [100.0] * len(self.gpus)))
            self.utilization_samples.append((time.time(), [50.0] * len(self.gpus)))
            self.frequency_samples.append((time.time(), [2520.0] * len(self.gpus)))
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
