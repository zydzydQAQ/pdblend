import hashlib
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from pdblend.profile.window_sampling import (
    measure_shared_prefill_windows, plan_shared_prefill_windows, summarize_window,
)


class Model:
    system = "pdblend"
    model = "/models/Qwen2.5-7B-Instruct"
    tp = 2
    pp = 1
    bounded_coverage = {"declared": True}
    max_context = 8191
    bias = 1.0

    def decode_supported(self, batch, ctx, freq):
        return freq == 1500 and 1 <= ctx <= self.max_context

    def step_seconds(self, batch, ctx, freq):
        return .1 * self.bias

    def decode_power_w(self, batch, freq, *, ctx=None):
        return 200.0

    def to_json(self):
        return json.dumps(dict(bias=self.bias, max_context=self.max_context))


def raw(batch=4, ctx=4096, capacity=100000):
    return dict(system="pdblend", model_id="Qwen2.5-7B-Instruct", tp=2, pp=1,
                kv_capacity_tokens=capacity,
                decode=[dict(freq_mhz=1500, batch=batch, context_tokens=ctx,
                             step_seconds=.1, step_repeats=[.1, .1, .1])])


def test_plan_full_tail_reservation_and_unsupported_domain_fallback():
    model = Model()
    plan = plan_shared_prefill_windows(raw(), model, batch=4, context=4096, freq_mhz=1500)
    assert plan["status"] == "ready"
    assert plan["max_tokens"] > 21 / .1 + 17
    assert plan["reserved_kv_tokens"] == 4 * (4096 + plan["max_tokens"])
    assert plan["batch_prefills_saved"] == 2
    model.max_context = 4200
    fallback = plan_shared_prefill_windows(raw(), model, batch=4, context=4096, freq_mhz=1500)
    assert fallback["reason"] == "continuous_windows_exceed_frozen_context_coverage"


def test_reservation_checks_entire_generation_not_64_token_head():
    result = plan_shared_prefill_windows(raw(capacity=18800), Model(), batch=4, context=4096, freq_mhz=1500)
    assert result["status"] == "fallback_required"
    assert result["reason"] == "continuous_windows_exceed_measured_kv_reservation"
    missing = plan_shared_prefill_windows(raw(), Model(), batch=2, context=2048, freq_mhz=1500)
    assert missing["reason"] == "missing_same_shape_measured_latency_for_tail_reservation"


def test_no_baseline_sharing_or_shortened_gates():
    data = raw(); data["system"] = "distserve"
    with pytest.raises(ValueError, match="PDBlend"):
        plan_shared_prefill_windows(data, Model(), batch=4, context=4096, freq_mhz=1500)
    with pytest.raises(ValueError, match="repeats"):
        plan_shared_prefill_windows(raw(), Model(), batch=4, context=4096, freq_mhz=1500, repeats=1)


def test_effective_context_uses_actual_generated_token_indices():
    times = [[float(x) for x in range(20)]]
    row = summarize_window(token_times=times, context=100, start_s=2, end_s=12,
                           power=[(2, [100, 90]), (12, [100, 90])], frequency=[(5, [1500, 1500])],
                           gpu_count=2, settle_s=2)
    # Output j at timestamp j has context L+j. Window excludes timestamp 2.
    assert row["effective_context_tokens"] == 107.5
    assert row["observed_context_min"] == 103
    assert row["observed_context_max"] == 112
    assert row["min_steps"] == 10
    assert row["power_w"] == 190
    with pytest.raises(ValueError, match="power or frequency"):
        summarize_window(token_times=times, context=100, start_s=2, end_s=12,
                         power=[(2, [100])], frequency=[], gpu_count=1, settle_s=2)


class Clock:
    def __init__(self, batch):
        self.now = 1000.
        self.live = [SimpleNamespace(token_times_s=[998.4 + i * .1 for i in range(17)], error=None)
                     for _ in range(batch)]

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.now += seconds
        for request in self.live:
            while request.token_times_s[-1] + .1 <= self.now + 1e-8:
                request.token_times_s.append(request.token_times_s[-1] + .1)


class Sampler:
    error = None

    def __init__(self, clock):
        self.clock = clock

    def start(self):
        self.start_s = self.clock()

    def stop(self):
        self.samples = [(self.start_s + .5, [100, 100]), (self.clock(), [100, 100])]
        self.frequency_samples = [(self.start_s + 1, [1500, 1500])]


def profiler(tmp_path, clock):
    return SimpleNamespace(raw=raw(), out_dir=tmp_path, decode_repeats=3,
        decode_settle_s=2, decode_measure_s=5, parallel_layout={"instances": ["tp2"]},
        meter=SimpleNamespace(sampler=lambda gpus: Sampler(clock)))


@pytest.mark.asyncio
async def test_three_windows_share_prefill_and_keep_context_specific_predictions(tmp_path):
    clock = Clock(4)
    enters = []

    @asynccontextmanager
    async def background(profiler, client, plan, tag):
        enters.append(plan)
        yield clock.live, []

    result = await measure_shared_prefill_windows(profiler(tmp_path, clock), None, [0, 1],
        model=Model(), freq_mhz=1500, batch=4, context=4096, tag="test",
        _clock=clock, _sleep=clock.sleep, _background_factory=background)
    assert result["status"] == "sampled"
    assert len(enters) == 1
    assert len(result["row"]["repeats"]) == 3
    contexts = [r["effective_context_tokens"] for r in result["row"]["repeats"]]
    assert contexts[0] < contexts[1] < contexts[2]
    for row in result["row"]["repeats"]:
        payload = (tmp_path / row["samples_file"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == row["samples_sha256"]
        assert len(json.loads(payload)["token_times_s"]) == 4
        assert row["prediction"]["context_tokens"] == row["effective_context_tokens"]
        assert row["min_steps"] >= 8


@pytest.mark.asyncio
async def test_prediction_failure_is_preserved_and_not_fallback(tmp_path):
    clock = Clock(4)

    @asynccontextmanager
    async def background(*args):
        yield clock.live, []

    model = Model(); model.bias = 1.25
    result = await measure_shared_prefill_windows(profiler(tmp_path, clock), None, [0, 1],
        model=model, freq_mhz=1500, batch=4, context=4096, tag="bad-prediction",
        _clock=clock, _sleep=clock.sleep, _background_factory=background)
    assert result["status"] == "prediction_failed"
    assert len(result["row"]["repeats"]) == 3
    assert len(result["row"]["prediction_failures"]) == 3


@pytest.mark.asyncio
async def test_early_stream_end_explicit_fallback_without_partial_profile(tmp_path):
    clock = Clock(4)

    class Task:
        def done(self): return clock.now >= 1010
        def cancelled(self): return False
        def result(self): return SimpleNamespace(error=None)

    @asynccontextmanager
    async def background(*args):
        yield clock.live, [Task()]

    result = await measure_shared_prefill_windows(profiler(tmp_path, clock), None, [0, 1],
        model=Model(), freq_mhz=1500, batch=4, context=4096, tag="early",
        _clock=clock, _sleep=clock.sleep, _background_factory=background)
    assert result["status"] == "fallback_required"
    assert result["row"] is None
    attempt = json.loads((tmp_path / result["attempt"]["path"]).read_text())
    assert len(attempt["completed_windows"]) == 1


@pytest.mark.asyncio
async def test_prediction_failure_cannot_be_hidden_by_later_early_end(tmp_path):
    clock = Clock(4)

    class Task:
        def done(self): return clock.now >= 1010
        def cancelled(self): return False
        def result(self): return SimpleNamespace(error=None)

    @asynccontextmanager
    async def background(*args):
        yield clock.live, [Task()]

    model = Model(); model.bias = 1.25
    result = await measure_shared_prefill_windows(profiler(tmp_path, clock), None, [0, 1],
        model=model, freq_mhz=1500, batch=4, context=4096, tag="fail-then-short",
        _clock=clock, _sleep=clock.sleep, _background_factory=background)
    assert result["status"] == "prediction_failed"
    assert result["row"] is None
    assert result["prediction_failures"]
