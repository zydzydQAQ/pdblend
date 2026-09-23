"""Optional PDBlend-only consecutive decode windows after one batch prefill.

Nothing in this module changes active profiling jobs.  Callers must explicitly
select this path and handle ``fallback_required`` by invoking the existing
measurement.  Prediction failures are preserved; they are never a fallback
reason.  Consecutive windows are repeated measurements of one decode run, not
independent cold starts, workloads, or seeds.
"""
from __future__ import annotations

import asyncio
import bisect
import hashlib
import json
import math
import statistics
import time
from contextlib import asynccontextmanager
from pathlib import Path


def plan_shared_prefill_windows(raw, model, *, batch, context, freq_mhz,
                                repeats=3, settle_s=2.0, measure_s=5.0,
                                max_model_len=8192, max_num_batched_tokens=8192):
    """Conservatively reserve a full tail using this topology's training data.

    Prior latency estimates choose a reservation; live timestamps remain the
    authority.  Exceeding the reservation or frozen coverage returns an explicit
    fallback, not nominal-context substitution or extrapolation.
    """
    if (raw.get("system") != "pdblend" or model.system != "pdblend" or
            raw.get("model_id") != Path(model.model).name or raw.get("tp") != model.tp or
            raw.get("pp") != 1 or model.pp != 1):
        raise ValueError("shared windows require the same PDBlend model and PP1 topology")
    if type(repeats) is not int or repeats < 3 or settle_s < 2 or measure_s < 5:
        raise ValueError("decode requires >=3 repeats, >=2 seconds settle and >=5 seconds measurement")
    if (type(batch) is not int or batch < 1 or type(context) is not int or context < 1 or
            not all(math.isfinite(x) for x in (settle_s, measure_s)) or
            type(max_model_len) is not int or not 1 <= max_model_len <= 8192 or
            type(max_num_batched_tokens) is not int or max_num_batched_tokens < 1):
        raise ValueError("invalid decode shape or window bounds")
    plan = dict(schema=1, system="pdblend", model_id=raw["model_id"], tp=model.tp, pp=1,
                batch=batch, context_tokens=context, freq_mhz=freq_mhz, repeats=int(repeats),
                settle_s=float(settle_s), measure_s=float(measure_s), max_model_len=max_model_len,
                same_decode_run=True, independent_prefill_repeats=False,
                fallback_method="Profiler._decode_batch", formal_eligible=False)

    def fallback(reason):
        return dict(plan, status="fallback_required", reason=reason)

    if not model.bounded_coverage or not model.decode_supported(batch, context, freq_mhz):
        return fallback("missing_or_unsupported_frozen_coverage")
    capacity = raw.get("kv_capacity_tokens", 0)
    if not isinstance(capacity, (int, float)) or not math.isfinite(capacity) or capacity <= 0:
        return fallback("missing_measured_kv_capacity")
    points = [r for r in raw.get("decode", []) if r.get("freq_mhz") == freq_mhz and
              r.get("batch") == batch and r.get("context_tokens") == context]
    timings = [float(t) for r in points for t in r.get("step_repeats", [r.get("step_seconds", 0)])
               if t is not None and math.isfinite(t) and t > 0]
    if not timings:
        return fallback("missing_same_shape_measured_latency_for_tail_reservation")
    fastest = min(timings)
    # Reserve 15% faster generation, staggered prefill progress, the existing
    # 16-token all-stream barrier, and 16 terminal guard tokens.  This estimate
    # is explicitly checked again against actual stream progress while running.
    lead = math.ceil(batch * context / max_num_batched_tokens)
    tail = math.ceil(repeats * (settle_s + measure_s) / (fastest * .85)) + lead + 17 + 16
    plan.update(max_tokens=tail, fastest_observed_step_s=fastest,
                estimated_prefill_lead_tokens=lead, kv_capacity_tokens=capacity,
                reserved_kv_tokens=batch * (context + tail),
                decode_window_seconds=repeats * (settle_s + measure_s),
                batch_prefills_saved=repeats - 1)
    if context + tail > max_model_len:
        return fallback("continuous_windows_exceed_model_length")
    if batch * (context + tail) > .9 * capacity:
        return fallback("continuous_windows_exceed_measured_kv_reservation")
    if not model.decode_supported(batch, context + tail - 1, freq_mhz):
        return fallback("continuous_windows_exceed_frozen_context_coverage")
    if getattr(model, 'decode_power_overrides', {}) and not all(
            model.decode_power_supported(batch, c, freq_mhz) for c in (context, context + tail - 1)):
        return fallback("continuous_windows_exceed_frozen_power_coverage")
    return dict(plan, status="ready")


def summarize_window(*, token_times, context, start_s, end_s, power, frequency,
                     gpu_count, settle_s, measurement_s=5.0):
    """Derive actual decode contexts from stream token indices, not the prompt.

    Output token with zero-based index j was generated with context L+j.
    Only arrivals in (start, end] enter a window; the first output (prefill) must
    precede it.  Timestamp arrays are preserved in the corresponding artifact.
    """
    if settle_s < 2 or measurement_s < 5 or end_s - start_s < measurement_s:
        raise ValueError("insufficient settle or measurement window")
    if not token_times or gpu_count < 1:
        raise ValueError("empty decode or GPU group")
    starts, ends, counts, contexts = [], [], [], []
    lower, upper = [], []
    for times in token_times:
        if (not times or any(not math.isfinite(x) for x in times) or
                any(b < a for a, b in zip(times, times[1:]))):
            raise ValueError("invalid or non-monotonic token timestamps")
        a, b = bisect.bisect_right(times, start_s), bisect.bisect_right(times, end_s)
        if a < 1 or b - a < 8:
            raise ValueError("every request needs prefill and eight decode steps before window completion")
        starts.append(a); ends.append(b); counts.append(b - a)
        contexts.extend(range(context + a, context + b))
        lower.append(context + a); upper.append(context + b - 1)
    powers = [row for row in power if start_s <= row[0] <= end_s]
    clocks = [row for row in frequency if start_s <= row[0] <= end_s]
    if len(powers) < 2 or len(clocks) < 1:
        raise ValueError("missing per-window power or frequency samples")
    for rows in (powers, clocks):
        if (any(not math.isfinite(t) or len(values) != gpu_count or
                any(not math.isfinite(v) or v < 0 for v in values) for t, values in rows) or
                any(b[0] <= a[0] for a, b in zip(rows, rows[1:]))):
            raise ValueError("invalid GPU sample values or timestamps")
    return dict(batch=len(token_times), context_tokens=context,
                effective_context_tokens=statistics.fmean(contexts),
                observed_context_min=min(lower), observed_context_max=max(upper),
                effective_context_method="mean_context_of_observed_decode_token_events",
                start_token_counts=starts, end_token_counts=ends,
                start_s=start_s, end_s=end_s, settle_s=settle_s,
                steady_window_s=end_s - start_s,
                step_seconds=(end_s - start_s) / statistics.median(counts),
                steps=int(statistics.median(counts)), min_steps=min(counts),
                power_w=statistics.fmean(sum(values) for _, values in powers),
                power_samples=len(powers), frequency_samples=len(clocks),
                mean_freq_mhz=statistics.fmean(v for _, values in clocks for v in values))


class _EarlyEnd(RuntimeError):
    pass


def _running(tasks):
    for task in tasks:
        if task.done():
            if task.cancelled():
                raise _EarlyEnd("decode cancelled before all consecutive windows completed")
            result = task.result()
            if result.error:
                raise RuntimeError(f"decode failed: {result.error}")
            raise _EarlyEnd("decode exhausted reserved output tokens")


async def _wait_for_tokens(live, tasks, targets, timeout_s=600):
    deadline = time.monotonic() + timeout_s
    while True:
        _running(tasks)
        if all(r is not None and len(r.token_times_s) >= n for r, n in zip(live, targets)):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("background decode did not reach stable-token barrier")
        await asyncio.sleep(.01)


@asynccontextmanager
async def _background(profiler, client, plan, tag):
    from ..bench.gates import random_prompt
    batch, context = plan["batch"], plan["context_tokens"]
    live = [None] * batch

    def callback(i):
        def update(result, now):
            live[i] = result
        return update

    tasks = [asyncio.create_task(client.complete(
        random_prompt(context, 1000 * batch + i), plan["max_tokens"], f"{tag}-shared-{i}",
        on_token=callback(i))) for i in range(batch)]
    try:
        # Use the same all-stream barriers as the existing profiler, without
        # borrowing another system's profiles or baseline admission logic.
        await _wait_for_tokens(live, tasks, [1] * batch)
        targets = [len(r.token_times_s) + 16 for r in live]
        await _wait_for_tokens(live, tasks, targets)
        yield live, tasks
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def measure_shared_prefill_windows(profiler, client, gpus, *, model, freq_mhz,
                                         batch, context, tag, max_model_len=8192,
                                         training_raw=None,
                                         _clock=time.time, _sleep=asyncio.sleep,
                                         _background_factory=None):
    """Measure an explicitly selected shared-prefill point; never auto-fallback.

    Return ``sampled`` or ``prediction_failed`` with a full ``row`` and raw
    per-window evidence.  ``fallback_required`` has no acceptable row and must
    not enter a profile.  Unexpected transport/sampling failures propagate.
    The frozen model is queried separately at each window's observed context.
    """
    planning_raw = profiler.raw
    if training_raw is not None:
        for key in ("system", "model_id", "tp", "pp"):
            if training_raw.get(key) != profiler.raw.get(key):
                raise ValueError("training and live measurement identities differ")
        live_capacity = profiler.raw.get("kv_capacity_tokens", 0)
        planning_raw = dict(training_raw, kv_capacity_tokens=min(
            live_capacity, training_raw.get("kv_capacity_tokens", 0)))
    plan = plan_shared_prefill_windows(planning_raw, model, batch=batch, context=context,
        freq_mhz=freq_mhz, repeats=profiler.decode_repeats, settle_s=profiler.decode_settle_s,
        measure_s=profiler.decode_measure_s, max_model_len=max_model_len)
    if plan["status"] != "ready":
        return dict(status="fallback_required", plan=plan, row=None)
    if len(gpus) != model.tp or len(set(gpus)) != len(gpus):
        raise ValueError("shared windows need exactly one nonoverlapping TP group")
    if Path(tag).name != tag or not tag:
        raise ValueError("measurement tag must be a single filename component")
    frozen_json = model.to_json()
    frozen_sha = hashlib.sha256(frozen_json.encode()).hexdigest()
    factory = _background_factory or _background
    samples_dir = profiler.out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    repeats = []
    immediate_failures = []

    def save_attempt(status, reason=None):
        path = samples_dir / f"{tag}-shared-attempt.json"
        path.write_text(json.dumps(dict(status=status, reason=reason, plan=plan,
            frozen_model_serialization_sha256=frozen_sha, completed_windows=repeats,
            formal_eligible=False), indent=2) + "\n")
        return dict(path=str(path.relative_to(profiler.out_dir)),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    try:
        async with factory(profiler, client, plan, tag) as (live, tasks):
            if len(live) != batch:
                raise ValueError("background batch differs from planned batch")
            for index in range(plan["repeats"]):
                settle_started = _clock()
                await _sleep(plan["settle_s"])
                _running(tasks)
                sampler = profiler.meter.sampler(gpus)
                sampler.start()
                start = _clock()
                try:
                    await _sleep(plan["measure_s"])
                    end = _clock()
                    _running(tasks)
                finally:
                    sampler.stop()
                if sampler.error:
                    raise RuntimeError(f"decode sampler: {sampler.error}")
                timestamps = [list(r.token_times_s) for r in live]
                evidence = dict(start_s=start, end_s=end, settle_start_s=settle_started,
                    token_times_s=timestamps, prompt_context_tokens=context,
                    power=[x for x in sampler.samples if start <= x[0] <= end],
                    frequency=[x for x in sampler.frequency_samples if start <= x[0] <= end],
                    shared_decode_run=tag, window_index=index, reservation=plan,
                    frozen_model_serialization_sha256=frozen_sha)
                path = samples_dir / f"{tag}-shared-r{index}.json"
                path.write_text(json.dumps(evidence) + "\n")
                row = summarize_window(token_times=timestamps, context=context,
                    start_s=start, end_s=end, power=evidence["power"], frequency=evidence["frequency"],
                    gpu_count=len(gpus), settle_s=start - settle_started,
                    measurement_s=plan["measure_s"])
                evidence.update(start_token_counts=row["start_token_counts"], end_token_counts=row["end_token_counts"])
                path.write_text(json.dumps(evidence) + "\n")
                row.update(samples_file=str(path.relative_to(profiler.out_dir)),
                    samples_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    concurrency=batch, parallel_layout=profiler.parallel_layout,
                    measured_gpu_ids=list(gpus), shared_decode_run=tag, window_index=index)
                repeats.append(row)
                if (row["observed_context_max"] >= max_model_len or
                        batch * (row["observed_context_max"] + 1) > .9 * plan["kv_capacity_tokens"]):
                    raise _EarlyEnd("observed context exceeded model or measured KV reservation")
                if not all(model.decode_supported(batch, c, freq_mhz) for c in
                           (row["observed_context_min"], row["effective_context_tokens"], row["observed_context_max"])):
                    raise _EarlyEnd("observed consecutive window exceeded frozen context coverage")
                if (getattr(model, 'decode_power_overrides', {}) and not
                        model.decode_power_supported(batch, row['effective_context_tokens'], freq_mhz)):
                    raise _EarlyEnd("observed consecutive window exceeded frozen power coverage")
                predicted = model.step_seconds(batch, row["effective_context_tokens"], freq_mhz)
                error = abs(predicted / row["step_seconds"] - 1)
                power_error = abs(model.decode_power_w(batch, freq_mhz, ctx=row["effective_context_tokens"]) / row["power_w"] - 1)
                row["prediction"] = dict(seconds=predicted, relative_error=error,
                                          power_relative_error=power_error,
                                          context_tokens=row["effective_context_tokens"])
                if not math.isfinite(error) or error > .10:
                    immediate_failures.append(dict(window_index=index, metric="decode_timing", relative_error=error))
                if not math.isfinite(power_error) or power_error > .15:
                    immediate_failures.append(dict(window_index=index, metric="decode_power_max", relative_error=power_error))
        if model.to_json() != frozen_json:
            raise RuntimeError("frozen model changed during consecutive windows")
    except _EarlyEnd as exc:
        status = "prediction_failed" if immediate_failures else "fallback_required"
        artifact = save_attempt(status, str(exc))
        return dict(status=status, plan=plan, reason=str(exc), row=None, attempt=artifact,
                    prediction_failures=immediate_failures)
    except BaseException as exc:
        save_attempt("failed_measurement", f"{type(exc).__name__}: {exc}")
        raise
    failures = list(immediate_failures)
    power_errors = [r["prediction"]["power_relative_error"] for r in repeats]
    powers = [r["power_w"] for r in repeats]
    if (not all(math.isfinite(x) for x in power_errors) or statistics.fmean(power_errors) > .10 or
            max(power_errors) > .15 or min(powers) <= 0 or
            statistics.stdev(powers) / statistics.fmean(powers) > .10):
        failures.append(dict(metric="decode_power", relative_errors=power_errors))
    status = "prediction_failed" if failures else "sampled"
    row = dict(batch=batch, context_tokens=context, freq_mhz=freq_mhz, concurrency=batch,
        effective_context_tokens=statistics.median(r["effective_context_tokens"] for r in repeats),
        step_seconds=statistics.median(r["step_seconds"] for r in repeats),
        power_w=statistics.median(powers), power_repeats=powers,
        step_repeats=[r["step_seconds"] for r in repeats], repeats=repeats,
        steady_window_s=min(r["steady_window_s"] for r in repeats),
        steps=min(r["steps"] for r in repeats),
        power_samples=sum(r["power_samples"] for r in repeats),
        frequency_samples=sum(r["frequency_samples"] for r in repeats),
        parallel_layout=profiler.parallel_layout, measured_gpu_ids=list(gpus),
        sampling_method="consecutive_windows_shared_prefill", independent_prefill_repeats=False,
        requires_per_window_evaluation=True, prediction_failures=failures,
        frozen_model_serialization_sha256=frozen_sha, formal_eligible=False)
    return dict(status=status, plan=plan, row=row, attempt=save_attempt(status))
