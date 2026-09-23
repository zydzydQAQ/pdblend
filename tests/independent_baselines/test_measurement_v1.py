import json
import sys
import time
import types
from pathlib import Path

import pytest

from pdblend_baselines.measurement_v1 import (
    CapabilityReceipt,
    DynamoEndToEndMeasurement,
    DynamoLoadMeasurement,
    DynamoLoadProfileBuilder,
    DynamoShapeMeasurement,
    DynamoShapeProfileBuilder,
    EcoServeForwardMeter,
    MixedEndToEndMeasurement,
    NativeStageCollector,
    NativeStageSample,
    UnsupportedMeasurement,
    detect_vllm_v1_capabilities,
    reduce_native_stage_samples,
)
from pdblend_baselines.mixed_policy import MixedLeastLoadPolicy, MixedReplica


def test_capability_detector_returns_unsupported_receipt_without_vllm():
    receipt = detect_vllm_v1_capabilities(module=types.SimpleNamespace(__version__="0.10.1.1"),
                                           cuda_available=True)
    assert receipt.supported is False
    assert receipt.reason in {"distributed_group_api_missing", "v1_execute_model_api_missing"}
    assert receipt.hardware_qualified is False
    assert isinstance(receipt.as_dict(), dict)


def test_stage_samples_require_real_native_metadata_and_full_physical_coverage():
    samples = [NativeStageSample("7b", "prefill", 2, 2, rank, rank % 2, rank // 2,
                                 2, 128, 128, 1.0 + rank)
               for rank in range(4)]
    reduced = reduce_native_stage_samples(samples, tp=2, pp=2, role="prefill")
    assert reduced["hardware_qualified"] is True
    assert len(reduced["ranks"]) == 4
    with pytest.raises(ValueError, match="coverage"):
        reduce_native_stage_samples(samples[:-1], tp=2, pp=2, role="prefill")
    with pytest.raises(ValueError, match="role"):
        NativeStageSample("7b", "mixed", 1, 1, 0, 0, 0, 1, 1, 1, 1.)


def test_native_collector_never_converts_http_or_wall_time_to_gpu_time():
    class Event:
        events = []
        def __init__(self): self.actions = []; Event.events.append(self)
        def record(self): self.actions.append("record")
        def synchronize(self): self.actions.append("synchronize")
        def elapsed_time(self, other): return 2.5

    class Tokens:
        def numel(self): return 8

    class Input:
        seq_lens = [8]
        query_lens = [8]
        is_prompt = True
        input_tokens = Tokens()
        virtual_engine = 0

    class Runner:
        def execute_model(self, model_input): return "output"

    worker = types.SimpleNamespace(model_runner=Runner())
    collector = NativeStageCollector(model_id="7b", tp=1, pp=1, rank=0, tp_rank=0,
                                     pp_rank=0, event_factory=Event)
    collector.install(worker)
    assert worker.model_runner.execute_model(Input()) == "output"
    sample = collector.collect()[0]
    assert sample.gpu_elapsed_ms == 2.5
    assert sample.measurement_scope.startswith("cuda_event")
    assert Event.events[0].actions == ["record"]
    assert Event.events[1].actions == ["record", "synchronize"]


def test_ecoserve_requires_five_samples_and_cuda_receipt(monkeypatch):
    meter = EcoServeForwardMeter(model_id="7b", tp=1)
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(UnsupportedMeasurement) as error:
        meter.measure(16, lambda: None, samples=5)
    assert error.value.receipt.adapter == "ecoserve_gpu_forward"
    assert error.value.receipt.hardware_qualified is False
    with pytest.raises(ValueError, match="at least five"):
        # Validation happens before CUDA access for malformed sample counts.
        meter.measure(16, lambda: None, samples=4)


def test_dynamo_shape_profile_is_rectangular_and_load_history_is_separate(tmp_path):
    builder = DynamoShapeProfileBuilder(model_id="14b", tps=(1,), frequencies_mhz=(2520,),
                                        input_tokens=(16, 32), context_tokens=(64,), batches=(1,))
    row = dict(model_id="14b", tp=1, frequency_mhz=2520, context_tokens=64, batch=1,
               prefill_s=.01, iteration_s=.02, prefill_power_w=100., decode_power_w=101.,
               samples=5, source_sha256="a" * 64)
    builder.record(DynamoShapeMeasurement(input_tokens=16, **row))
    with pytest.raises(ValueError, match="incomplete"):
        builder.document()
    builder.record(DynamoShapeMeasurement(input_tokens=32, **row))
    doc = builder.document()
    assert doc["rectangular"] is True and len(doc["points"]) == 2
    out = builder.write(tmp_path / "shape.json")
    assert json.loads(out.read_text())["model_id"] == "14b"
    load = DynamoLoadProfileBuilder(model_id="14b", source_sha256="b" * 64)
    load.record(DynamoLoadMeasurement(1., 16, 32, "SS"))
    load_doc = load.document()
    assert "records" in load_doc and "points" not in load_doc


def test_dynamo_endpoint_adapter_requires_native_token_timestamps(tmp_path):
    adapter = DynamoEndToEndMeasurement(model_id="32b", system="dynamollm", tp=2)
    now = time.time()
    row = adapter.measure("r1", 16, lambda: {"first_token_s": now + .01,
                                               "last_token_s": now + .02, "output_tokens": 10})
    assert row["measurement_scope"] == "native_endpoint_token_timestamps"
    assert row["hardware_qualified"] is False
    path = adapter.write(tmp_path / "events.jsonl")
    assert json.loads(path.read_text())["request_id"] == "r1"
    with pytest.raises(ValueError, match="timestamps"):
        adapter.measure("r2", 16, lambda: {"output_tokens": 3})


def test_mixed_endpoint_adapter_keeps_fixed_system_identity():
    import time
    now = time.time()
    adapter = MixedEndToEndMeasurement(model_id="7b", tp=1)
    row = adapter.measure("m1", 8, lambda: {"first_token_s": now + .01,
                                             "last_token_s": now + .02, "output_tokens": 3})
    assert row["system"] == "mixed" and row["tp"] == 1


def test_mixed_fixed_tp_least_load_policy_routes_only_compatible_replicas():
    replicas = [MixedReplica("a", tp=1, active_requests=1, max_num_seqs=4),
                MixedReplica("b", tp=1, active_requests=0, max_num_seqs=4),
                MixedReplica("wrong", tp=2, active_requests=0)]
    policy = MixedLeastLoadPolicy(tp=1)
    assert policy.route("r1", replicas).instance_id == "b"
    assert policy.route("r2", replicas).instance_id == "a"
    assert policy.routes[0].policy == "fixed_tp_least_load"
    policy.complete("a", replicas)
    replicas[0].accepting = False
    replicas[1].active_requests = replicas[1].max_num_seqs
    assert policy.route("r3", replicas) is None
