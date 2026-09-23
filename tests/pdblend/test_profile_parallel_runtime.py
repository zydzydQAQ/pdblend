import asyncio
from types import SimpleNamespace

import pytest

import pdblend.profile.profiler as profiler_module
from pdblend.profile.profiler import Profiler


class _Client:
    instances = []

    def __init__(self, instance_id, base_url):
        self.instance_id = instance_id
        self.base_url = base_url
        self.__class__.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def complete(self, *args, **kwargs):
        return SimpleNamespace(error=None)


class _Fleet:
    def __init__(self, specs):
        self.instances = {s.instance_id: SimpleNamespace(spec=s) for s in specs}

    def __getitem__(self, key):
        return self.instances[key]


def _profiler(tmp_path, *, parallel_step=1.0):
    p = Profiler.__new__(Profiler)
    p.freqs = (2100, 2520)
    p.specs = [SimpleNamespace(instance_id="i0", base_url="i0", gpus=(0,)),
               SimpleNamespace(instance_id="i1", base_url="i1", gpus=(1,))]
    p.out_dir = tmp_path
    p.raw = {}
    p.decode_repeats = 3
    p._lock = lambda freq, gpus: None
    p._checkpoints = 0
    p._checkpoint = lambda: setattr(p, "_checkpoints", p._checkpoints + 1)
    calls = []

    async def decode(client, gpus, batch, ctx, steps, tag):
        calls.append(("decode", client.instance_id, tuple(gpus), tag))
        return {"step_seconds": parallel_step if "concurrent" in tag else 1.0,
                "power_w": 100.0, "batch": batch, "context_tokens": ctx,
                "repeats": [{"step_seconds": 1.0, "power_w": 100.0}] * 3}

    async def sections(client, gpus, sections, freqs, *, checkpoint=False):
        calls.append(("sections", client.instance_id, tuple(freqs), checkpoint))

    async def prefill(client, gpus, freqs=None, checkpoint=True):
        calls.append(("serial", "prefill", tuple(freqs), checkpoint))

    async def decode_section(client, gpus, freqs=None, checkpoint=True):
        calls.append(("serial", "decode", tuple(freqs), checkpoint))

    async def mixed(client, gpus, freqs=None, checkpoint=True):
        calls.append(("serial", "mixed", tuple(freqs), checkpoint))

    p._decode_batch = decode
    p.run_sections_for_client = sections
    p._prefill, p._decode, p._mixed = prefill, decode_section, mixed
    return p, _Fleet(p.specs), calls


def test_parallel_online_measures_isolated_then_concurrent_and_shards(tmp_path, monkeypatch):
    _Client.instances.clear()
    monkeypatch.setattr(profiler_module, "EngineClient", _Client)
    p, fleet, calls = _profiler(tmp_path)
    asyncio.run(p._parallel_online(fleet, ("decode",)))
    measurements = [row for row in calls if row[0] == "decode"]
    assert len(measurements) == 4
    assert [row[1] for row in measurements[:2]] == ["i0", "i1"]
    assert all("isolated" in row[3] for row in measurements[:2])
    assert all("concurrent" in row[3] for row in measurements[2:])
    shards = [row for row in calls if row[0] == "sections"]
    assert {row[2] for row in shards} == {(2100,), (2520,)}
    assert p.raw["measured_mode"] == "parallel"
    assert p._checkpoints >= 1


def test_parallel_online_failed_interference_falls_back_to_serial(tmp_path, monkeypatch):
    monkeypatch.setattr(profiler_module, "EngineClient", _Client)
    p, fleet, calls = _profiler(tmp_path, parallel_step=1.2)
    asyncio.run(p._parallel_online(fleet, ("prefill", "decode", "mixed")))
    assert p.raw["measured_mode"] == "serial_fallback"
    assert any(row[0] == "serial" and row[3] is True for row in calls)
    assert "error" in p.raw["parallel_interference"]
