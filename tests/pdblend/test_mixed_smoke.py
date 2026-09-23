"""Independent Mixed smoke contracts; fake HTTP clients, no GPU operations."""
import asyncio
from dataclasses import asdict
import json
from types import SimpleNamespace
import time

import pytest

from pdblend.engine.client import Completion
from pdblend.engine.launcher import make_specs
from pdblend.bench.client import Request
from pdblend.bench.mixed_smoke import build_smoke_trace, replay, run, smoke_manifest, trace_digest, valid_completion
from pdblend.bench.smoke_trace import build_smoke_trace as shared_smoke_trace
from pdblend_baselines.mixed_policy import MixedReplica


class FakeClient:
    def __init__(self, instance_id, *, fail=False):
        self.instance_id, self.fail = instance_id, fail

    async def complete(self, prompt, max_tokens, request_id, *, on_token=None, seed=None):
        assert seed == 701
        await asyncio.sleep(.001)
        if self.fail:
            raise RuntimeError("failed HTTP request")
        row = Completion(request_id, self.instance_id, time.time(), prompt_tokens=len(prompt))
        for _ in range(max_tokens):
            stamp = time.time()
            if row.first_token_s is None:
                row.first_token_s = stamp
            row.token_times_s.append(stamp)
            row.text += "x"
            if on_token:
                on_token(row, stamp)
        row.completion_tokens = max_tokens
        row.finished_s = time.time()
        return row


def test_trace_is_reproducible_and_keeps_seed_and_shape_scope():
    left, right = build_smoke_trace(), build_smoke_trace()
    assert trace_digest(left) == trace_digest(right)
    assert [asdict(row) for row in left] == [asdict(row) for row in shared_smoke_trace(100, 701, .2)]
    assert {request.input_tokens for request in left} == {128, 512, 2048}
    assert all(request.max_tokens == 16 and 0 <= request.arrival_s < 100 for request in left)
    with pytest.raises(ValueError, match="701"):
        build_smoke_trace(seed=1701)


@pytest.mark.asyncio
async def test_live_policy_routes_concurrent_streams_and_releases_counters(tmp_path):
    replicas = [MixedReplica("i0", 2), MixedReplica("i1", 2)]
    clients = {row.instance_id: FakeClient(row.instance_id) for row in replicas}
    trace = [Request(i, 0, [1000] * 128, 16) for i in range(4)]
    result = await replay(clients, replicas, trace, tmp_path, duration_s=0)
    assert result["passed"] and result["route_count"] == result["correct"] == 4
    assert result["peak_active_requests"] == 4 and result["counts_reclaimed"]
    routes = [json.loads(line) for line in (tmp_path / "routes.jsonl").read_text().splitlines()]
    assert [row["instance_id"] for row in routes] == ["i0", "i1", "i0", "i1"]
    assert {row["policy"] for row in routes} == {"fixed_tp_least_load"}
    assert all(row.active_requests == 0 for row in replicas)
    assert len((tmp_path / "sse-events.jsonl").read_text().splitlines()) == 64


@pytest.mark.asyncio
async def test_transport_error_cannot_leak_counts_or_become_passing_smoke(tmp_path):
    replicas = [MixedReplica("i0", 1), MixedReplica("i1", 1)]
    clients = {row.instance_id: FakeClient(row.instance_id, fail=row.instance_id == "i1") for row in replicas}
    result = await replay(clients, replicas, [Request(i, 0, [1000], 16) for i in range(2)],
                          tmp_path, duration_s=0)
    assert not result["passed"] and result["correct"] == 1 and result["counts_reclaimed"]
    outcomes = [json.loads(line) for line in (tmp_path / "outcomes.jsonl").read_text().splitlines()]
    assert any("failed HTTP" in row.get("error", "") for row in outcomes if row.get("error"))


def test_metadata_never_claims_formal_or_cancellation_qualification(tmp_path, monkeypatch):
    monkeypatch.setenv("PDBLEND_SOURCE_SHA256", "source-hash")
    monkeypatch.setenv("PDBLEND_IMAGE_ID", "image-hash")
    model = SimpleNamespace(model_id="Qwen2.5-32B-Instruct", model_hash="weights", tokenizer_hash="tokenizer",
                            manifest_sha256="config", verification_receipt="verified.json")
    specs = make_specs(model.model_id, [0, 1, 2, 3], tp=2, kv_connector=None)
    metadata = smoke_manifest(model, specs, [{"uuid": "GPU-a"}], build_smoke_trace(), duration_s=100, seed=701)
    assert metadata["system"] == "mixed" and metadata["profile_required"] is False
    assert not metadata["formal_eligible"] and not metadata["energy_comparable"] and not metadata["hardware_qualified"]
    assert metadata["tp"] == 2 and metadata["pp"] == 1
    assert metadata["seed_policy"] == "single_seed_701" and metadata["source_sha256"] == "source-hash"
    assert metadata["warmup_seed"] == 9701
    assert len(metadata["independent_policy_sha256"]) == 64
    assert "native_cancellation" in metadata["missing_gates"]
    assert all(row["kv_connector"] is None for row in metadata["specs"])


def test_bad_config_still_writes_failed_development_receipt_without_starting_gpu(tmp_path):
    result = run("7b", [0], 1, tmp_path)
    assert not result["complete"] and not result["formal_eligible"] and result["status"] == "failed"
    assert "two disjoint" in result["error"]
    assert json.loads((tmp_path / "completion.json").read_text())["energy_comparable"] is False


@pytest.mark.asyncio
async def test_truncated_stream_is_not_valid_even_with_nonempty_text():
    result = asdict(await FakeClient("i0").complete([1000], 15, "r", seed=701))
    assert not valid_completion(result, max_tokens=16)
