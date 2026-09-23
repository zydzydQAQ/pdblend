"""Measured profiling grid feeding profile.model.fit; runtime depends on topology.

prefill:  input x freq, back-to-back single requests for >= window seconds
decode:   batch x context x freq, concurrent requests; step time from token arrivals,
          power averaged over the interval where all requests are decoding
mixed:    decode batch + chunk prefill probes to validate the additive interference model
static:   active idle per frequency, sleep L1/L2, off; wake latency
transfer: P->D handoff overhead vs mixed TTFT (same method as gate G0)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import subprocess
import statistics
import time
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from contextlib import asynccontextmanager, AsyncExitStack
from pathlib import Path
from typing import Sequence

from pdblend.bench.gates import kv_bytes_per_token, random_prompt
from pdblend.bench.metering import FREQUENCY_TIERS, Gpus
from pdblend.engine.client import EngineClient, PDTransfer, pd_complete
from pdblend.engine.handoff_timing import PROTOCOL as HANDOFF_PROTOCOL, measure_handoff
from pdblend.engine.launcher import Fleet, make_specs
from pdblend.model_registry import ModelRegistry
from pdblend.profile.identity import ProfileKey, provenance
from pdblend.profile.query.model import DecodePoint, PrefillPoint, StaticState, fit
from pdblend.profile.collection.parallel import partition_frequencies
from pdblend.profile.collection.parallel import evaluate_interference

PREFILL_INPUTS = (128, 512, 1024, 2048, 4096, 7168)
DECODE_BATCHES = (1, 4, 8, 16, 32, 64, 128, 256)
DECODE_CONTEXTS = (256, 1024, 4096)
DECODE_STEPS = 64
RAW_SCHEMA = 2
MIXED_PROBES = ((8, 512), (8, 2048), (32, 512), (32, 2048))
MIXED_FREQS = (1500, 2100, 2520)
TRANSFER_INPUTS = (512, 2048, 7168)


def parallel_layout_metadata(specs) -> dict:
    """Return the physical layout used by a profile measurement.

    Keeping this manifest in raw evidence makes it possible to distinguish
    interference measured across independent instances from interference
    between requests sharing one engine.  ``InstanceSpec`` is intentionally
    duck-typed here so this helper remains usable by lightweight tests.
    """
    instances = []
    all_gpus = []
    for spec in specs:
        gpus = [int(g) for g in spec.gpus]
        all_gpus.extend(gpus)
        stage_map = getattr(spec, "stage_map", None)
        if stage_map is None:
            tp, pp = int(spec.tp), int(getattr(spec, "pp", 1))
            stage_map = {stage: tuple(gpus[stage * tp:(stage + 1) * tp]) for stage in range(pp)}
        instances.append(dict(instance_id=spec.instance_id, gpus=gpus,
                              tp=int(spec.tp), pp=int(getattr(spec, "pp", 1)),
                              stage_map={str(k): list(v) for k, v in stage_map.items()}))
    return dict(instances=instances, gpus=all_gpus,
                gpu_count=len(all_gpus), unique_gpu_count=len(set(all_gpus)),
                instance_count=len(instances))


def window_mean_power(samples, start_s: float, end_s: float, gpu_index: int = 0) -> float | None:
    inside = [row[1][gpu_index] for row in samples if start_s <= row[0] <= end_s]
    return statistics.fmean(inside) if inside else None


@contextmanager
def _load_flock():
    """Serialize model loading across profiling jobs when a coordination mount exists."""
    try:
        import fcntl
    except (OSError, ImportError):
        # Local development and containers without /coord retain historical behavior.
        yield
        return
    coord = Path(os.environ.get("PDBLEND_COORD_DIR", "/coord"))
    try:
        coord.mkdir(parents=True, exist_ok=True)
        lock_path = coord / "load.lock"
        lock = lock_path.open("a+")
    except OSError:
        yield
        return
    with lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class Profiler:
    def __init__(self, model: str, gpus: Sequence[int], tp: int = 1, pp: int = 1,
                 system: str = "pdblend", role: str = "mixed", workload_shape: str = "default",
                 hardware_id: str = "", engine_revision: str = "vllm-0.10.1.1",
                 freqs: Sequence[int] = FREQUENCY_TIERS,
                 window_s: float = 2.0, out_dir: Path = Path("results/v2/profile"), kv_connector: str | None = "P2pNcclConnector",
                 mixed_freqs: Sequence[int] = (1500, 2100, 2520), decode_repeats: int = 3,
                 decode_settle_s: float = 2.0, decode_measure_s: float = 5.0, base_port: int = 8100,
                 parallel_instances: bool = False):
        self.model, self.gpus, self.tp, self.pp, self.system = model, list(gpus), tp, pp, system
        self.role, self.workload_shape = role, workload_shape
        self.hardware_id, self.engine_revision = hardware_id or "unknown", engine_revision
        model_key = Path(model).name
        model_root = os.environ.get("PDBLEND_MODELS_DIR")
        if not model_root:
            model_root = str(Path(model).parent) if Path(model).is_absolute() else "/models"
        receipt = os.environ.get("PDBLEND_MODEL_VERIFICATION_RECEIPT")
        self.model_spec = ModelRegistry(model_root, verification_receipt=receipt).get(model_key) if receipt else ModelRegistry(model_root).get(model_key)
        self.model_spec.validate_topology(tp, pp, require_memory=False)
        self.freqs, self.window_s = tuple(freqs), window_s
        self.mixed_freqs = tuple(mixed_freqs)
        if decode_repeats < 3 or decode_settle_s < 2 or decode_measure_s < 5:
            raise ValueError("profile-v2 requires >=3 repeats, >=2 s settle and >=5 s measurement")
        self.decode_repeats = int(decode_repeats)
        self.decode_settle_s = float(decode_settle_s)
        self.decode_measure_s = float(decode_measure_s)
        self.parallel_instances = bool(parallel_instances)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.meter = Gpus(self.gpus)
        self.specs = make_specs(model, self.gpus, tp=tp, pp=pp, base_port=base_port, kv_connector=kv_connector)
        self.parallel_layout = parallel_layout_metadata(self.specs)
        self.concurrency = dict(decode_max_batch=max(DECODE_BATCHES),
                                mixed_background_max_batch=max(b for b, _ in MIXED_PROBES),
                                prefill_inflight=1, transfer_inflight=1)
        self.transfer = PDTransfer(kv_connector, {s.instance_id: s.zmq_address for s in self.specs})
        self.kv_bpt = kv_bytes_per_token(self.specs[0].model_path)
        self.profile_key = ProfileKey(system, self.model_spec.model_id, engine_revision, self.hardware_id,
                                      tp, pp, role, workload_shape)
        self.raw: dict = dict(schema=RAW_SCHEMA, model=model, model_id=self.model_spec.model_id,
                              gpus=self.gpus, tp=tp, pp=pp, system=system, role=role,
                              workload_shape=workload_shape, profile_key=self.profile_key.as_dict(),
                              profile_namespace=self.profile_key.namespace(),
                              parallel_layout=self.parallel_layout,
                              concurrency=self.concurrency,
                              freqs=list(self.freqs),
                              prefill=[], decode=[], mixed=[], static={}, transfer=[], freq_switch_s=[],
                              kv_bytes_per_token=self.kv_bpt,
                              model_hash=self.model_spec.model_hash,
                              tokenizer_hash=self.model_spec.tokenizer_hash,
                              verification_receipt=self.model_spec.verification_receipt,
                              config=dict(decode_repeats=self.decode_repeats,
                                           decode_settle_s=self.decode_settle_s,
                                           decode_measure_s=self.decode_measure_s,
                                           mixed_freqs=list(self.mixed_freqs), base_port=base_port,
                                           decode_batches=list(DECODE_BATCHES),
                                           parallel_instances=self.parallel_instances),
                              environment=self._environment_metadata(self.gpus),
                              provenance=provenance(model=self.model_spec, engine_revision=engine_revision,
                                                    hardware_id=self.hardware_id))
        self._snapshot_concurrency_environment()

    def _snapshot_concurrency_environment(self) -> None:
        # The worker refreshes its live receipt during sampling. Bind an
        # immutable snapshot, so a later heartbeat cannot invalidate raw.json.
        allocation = self.out_dir / "concurrency-environment.json"
        if not allocation.is_file():
            return
        payload = allocation.read_bytes()
        document = json.loads(payload)
        digest = hashlib.sha256(payload).hexdigest()
        path = self.out_dir / "samples" / f"concurrency-{digest}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)
        self.raw["concurrency_environment"] = {
            **document, "samples_file": str(path.relative_to(self.out_dir)),
            "samples_sha256": digest,
        }

    def _environment_metadata(self, gpus: Sequence[int]) -> dict:
        def pkg(name):
            try:
                return version(name)
            except PackageNotFoundError:
                return None
        uuids = []
        try:
            nvml = self.meter.backend._nvml
            if nvml is not None:
                for g in gpus:
                    value = nvml.nvmlDeviceGetUUID(self.meter.backend._handle(g))
                    uuids.append(value.decode() if isinstance(value, bytes) else value)
        except Exception:
            uuids = []
        if not uuids:
            try:
                rows = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                    text=True, timeout=5).splitlines()
                wanted = set(int(g) for g in gpus)
                uuids = [line.split(",", 1)[1].strip() for line in rows
                         if int(line.split(",", 1)[0].strip()) in wanted]
            except Exception:
                uuids = []
        return dict(timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    hostname=platform.node(), python=platform.python_version(),
                    vllm=pkg("vllm"), torch=pkg("torch"), cuda=os.environ.get("CUDA_VERSION"),
                    image_digest=os.environ.get("PDBLEND_IMAGE_DIGEST") or os.environ.get("PDBLEND_IMAGE_ID"),
                    source_hash=os.environ.get("PDBLEND_SOURCE_HASH") or os.environ.get("PDBLEND_SOURCE_SHA256"),
                    hardware_id=os.environ.get("PDBLEND_HARDWARE_ID") or self.hardware_id,
                    gpu_uuids=uuids or os.environ.get("NVIDIA_VISIBLE_DEVICES"))

    def _lock(self, freq: int, gpus: Sequence[int]) -> float:
        started = time.time()
        for g in gpus:
            self.meter.set_clock(g, freq)
        return time.time() - started

    # ---- prefill -------------------------------------------------------------------------------
    async def _prefill(self, client: EngineClient, gpus: Sequence[int],
                       freqs: Sequence[int] | None = None, checkpoint: bool = True) -> None:
        for f in self.freqs if freqs is None else tuple(freqs):
            self._lock(f, gpus)
            for n in PREFILL_INPUTS:
                if any(r["freq_mhz"] == f and r["input_tokens"] == n for r in self.raw["prefill"]):
                    continue
                prompt = random_prompt(n, n)
                await client.complete(prompt, 1, f"pw-{f}-{n}")
                times = []
                with self.meter.measure(gpus) as m:
                    t_end = time.time() + self.window_s
                    i = 0
                    while time.time() < t_end or len(times) < 3:
                        r = await client.complete(prompt, 1, f"p-{f}-{n}-{i}")
                        i += 1
                        if r.error:
                            raise RuntimeError(f"prefill {f}/{n}: {r.error}")
                        times.append(r.first_token_s - r.submitted_s)
                busy = sum(times) / m["duration_s"]
                self.raw["prefill"].append(dict(freq_mhz=f, input_tokens=n, seconds=statistics.median(times),
                                                concurrency=1, parallel_layout=self.parallel_layout,
                                                measured_gpu_ids=list(gpus),
                                                power_w=m["mean_power_w"] / len(gpus) if self.tp == 1 else m["mean_power_w"],
                                                runs=len(times), busy_fraction=busy))
                print(f"prefill f={f} n={n}: {statistics.median(times)*1e3:.1f} ms {m['mean_power_w']:.0f} W", flush=True)
            if checkpoint:
                self._checkpoint()

    # ---- decode ---------------------------------------------------------------------------------
    @staticmethod
    def _require_running(tasks):
        for task in tasks:
            if task.done():
                if task.cancelled():
                    raise RuntimeError("background decode cancelled before measurement ended")
                result = task.result()
                raise RuntimeError(f"background decode ended early: {result.error or 'exhausted tokens'}")

    async def _wait_progress(self, live, tasks, targets, timeout_s=600.0):
        deadline = time.monotonic() + timeout_s
        while True:
            self._require_running(tasks)
            if all(r is not None and len(r.token_times_s) >= n for r, n in zip(live, targets)):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("background decode did not reach stable-token barrier")
            await asyncio.sleep(0.01)

    @asynccontextmanager
    async def _background(self, client, batch, ctx, tag):
        """Keep all streams alive; measure only after every prompt has finished prefill.

        Completion callbacks are profiler-only. Cancellation closes the HTTP response
        and aborts generation on the engine; all tasks are joined before returning.
        """
        live = [None] * batch
        def callback(i):
            def update(result, now):
                live[i] = result
            return update
        # A short generation can finish before the five-second steady window
        # on TP2/TP4. Keep enough tail tokens for the measurement while
        # bounding KV reservation for very large batches.
        max_tokens = max(512, DECODE_STEPS + 32, min(2048, 65536 // max(1, batch)))
        tasks = [asyncio.create_task(client.complete(
            random_prompt(ctx, 1000 * batch + i), max_tokens, f"{tag}-{i}", on_token=callback(i)))
            for i in range(batch)]
        try:
            await self._wait_progress(live, tasks, [1] * batch)
            # These 16 tokens must occur AFTER the last background prefill, not during it.
            targets = [len(r.token_times_s) + 16 for r in live]
            await self._wait_progress(live, tasks, targets)
            yield live, tasks
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(0.1)

    async def _decode_batch_once(self, client: EngineClient, gpus: Sequence[int], batch: int, ctx: int,
                                 steps: int, tag: str, *, before_measure=None) -> dict:
        async with self._background(client, batch, ctx, tag) as (live, tasks):
            # Settle under continuous decode, not idle. NVML is a trailing ~1 s average.
            await asyncio.sleep(self.decode_settle_s)
            self._require_running(tasks)
            if before_measure is not None:
                await before_measure()
                self._require_running(tasks)
            sampler = self.meter.sampler(gpus)
            sampler.start()
            start = time.time()
            start_counts = [len(r.token_times_s) for r in live]
            try:
                await asyncio.sleep(self.decode_measure_s)
                self._require_running(tasks)
                end = time.time()
                end_counts = [len(r.token_times_s) for r in live]
            finally:
                sampler.stop()
            if sampler.error:
                raise RuntimeError(f"decode sampler: {sampler.error}")
            counts = [b - a for a, b in zip(start_counts, end_counts)]
            if min(counts) < 8 or end - start < self.decode_measure_s:
                raise RuntimeError("insufficient full-batch steady measurement")
            power_samples = [x for x in sampler.samples if start <= x[0] <= end]
            clocks = [x for x in sampler.frequency_samples if start <= x[0] <= end]
            if len(power_samples) < 2 or not clocks:
                raise RuntimeError("missing power/frequency samples")
            power = [window_mean_power(power_samples, start, end, i) for i in range(len(gpus))]
            effective_context = statistics.fmean(ctx + (a + b) / 2 for a, b in zip(start_counts, end_counts))
            evidence = self.out_dir / "samples" / f"{tag}.json"
            evidence.parent.mkdir(exist_ok=True)
            evidence.write_text(json.dumps(dict(start_s=start, end_s=end, power=power_samples,
                frequency=clocks, start_token_counts=start_counts, end_token_counts=end_counts)))
            evidence_sha256 = hashlib.sha256(evidence.read_bytes()).hexdigest()
            return dict(batch=batch, context_tokens=ctx, effective_context_tokens=effective_context,
                concurrency=batch, parallel_layout=self.parallel_layout,
                measured_gpu_ids=list(gpus),
                step_seconds=(end - start) / statistics.median(counts),
                power_w=sum(power) if self.tp > 1 else statistics.fmean(power),
                steady_window_s=end - start, start_s=start, end_s=end,
                steps=int(statistics.median(counts)), min_steps=min(counts),
                power_samples=len(power_samples), frequency_samples=len(clocks),
                mean_freq_mhz=statistics.fmean(v for _, values in clocks for v in values),
                samples_file=str(evidence.relative_to(self.out_dir)), samples_sha256=evidence_sha256)

    async def _decode_batch(self, client: EngineClient, gpus: Sequence[int], batch: int, ctx: int,
                            steps: int, tag: str, *, before_measure=None) -> dict:
        warm = await client.complete(random_prompt(ctx, 700000 + batch * 1000 + ctx), 4, f"{tag}-warmup")
        if warm.error:
            raise RuntimeError(f"decode warmup {tag}: {warm.error}")
        repeats = [await self._decode_batch_once(client, gpus, batch, ctx, steps, f"{tag}-r{r}",
                   **({'before_measure': lambda r=r: before_measure(r)} if before_measure else {}))
                   for r in range(self.decode_repeats)]
        return dict(batch=batch, context_tokens=ctx, concurrency=batch,
            parallel_layout=self.parallel_layout,
            measured_gpu_ids=list(gpus),
            effective_context_tokens=statistics.median(r["effective_context_tokens"] for r in repeats),
            step_seconds=statistics.median(r["step_seconds"] for r in repeats),
            power_w=statistics.median(r["power_w"] for r in repeats),
            power_repeats=[r["power_w"] for r in repeats],
            step_repeats=[r["step_seconds"] for r in repeats],
            steady_window_s=min(r["steady_window_s"] for r in repeats),
            steps=min(r["steps"] for r in repeats), repeats=repeats,
            power_samples=sum(r["power_samples"] for r in repeats),
            frequency_samples=sum(r["frequency_samples"] for r in repeats))

    def _decode_grid(self, freqs: Sequence[int] | None = None) -> list[tuple[int, int, int]]:
        """(freq, ctx, batch) combos still missing from raw["decode"], within 90% of the KV capacity."""
        cap = self.raw.get("kv_capacity_tokens") or 0
        have = {(d["freq_mhz"], d["context_tokens"], d["batch"]) for d in self.raw["decode"] if len(d.get("repeats", [])) >= self.decode_repeats}
        active_freqs = self.freqs if freqs is None else tuple(freqs)
        return [(f, ctx, b) for f in active_freqs for ctx in DECODE_CONTEXTS for b in DECODE_BATCHES
                if not (cap and b * (ctx + DECODE_STEPS) > 0.9 * cap) and (f, ctx, b) not in have]

    async def _decode(self, client: EngineClient, gpus: Sequence[int],
                      freqs: Sequence[int] | None = None, checkpoint: bool = True) -> None:
        locked = None
        for f, ctx, b in self._decode_grid(freqs):
            if f != locked:
                self._lock(f, gpus)
                locked = f
                # NVML power lags by ~1 s; the first point (B=1) is short enough for the previous
                # section's draw to leak into its window
                await asyncio.sleep(max(1.5, self.decode_settle_s))
            row = await self._decode_batch(client, gpus, b, ctx, DECODE_STEPS, f"d-{f}-{ctx}-{b}")
            row["freq_mhz"] = f
            self.raw["decode"].append(row)
            print(f"decode f={f} ctx={ctx} B={b}: {row['step_seconds']*1e3:.2f} ms/step {row['power_w']:.0f} W", flush=True)
            if checkpoint:
                self._checkpoint()

    # ---- mixed interference ----------------------------------------------------------------------
    async def _mixed(self, client: EngineClient, gpus: Sequence[int],
                     freqs: Sequence[int] | None = None, checkpoint: bool = True,
                     *, reference_raw: dict | None = None) -> None:
        reference = self.raw if reference_raw is None else reference_raw
        if reference_raw is not None:
            for key in ('system', 'model_id', 'model_hash', 'tokenizer_hash', 'tp', 'pp'):
                if not self.raw.get(key) or reference.get(key) != self.raw[key]:
                    raise ValueError('mixed reference identity differs: '+key)
        have = {(r["freq_mhz"], r["batch"], r["chunk_tokens"]) for r in self.raw["mixed"] if r.get("valid")}
        active_freqs = self.mixed_freqs if freqs is None else tuple(freqs)
        for f in active_freqs:
            if f not in self.freqs:
                continue
            self._lock(f, gpus)
            for batch, chunk in MIXED_PROBES:
                if (f, batch, chunk) in have:
                    continue
                ctx = 1024
                base = next((r for r in reference["decode"] if r["freq_mhz"] == f
                             and r["batch"] == batch and r["context_tokens"] == ctx), None)
                prefill = next((r for r in reference["prefill"]
                              if r["freq_mhz"] == f and r["input_tokens"] == chunk), None)
                alone = prefill['seconds'] if prefill else None
                row = dict(freq_mhz=f, batch=batch, chunk_tokens=chunk, context_tokens=ctx,
                           concurrency=batch, parallel_layout=self.parallel_layout,
                           measured_gpu_ids=list(gpus),
                           interference="decode_with_chunked_prefill",
                           base_step_s=base["step_seconds"] if base else None, alone_prefill_s=alone)
                if reference_raw is not None:
                    row['reference_binding'] = dict(
                        decode_evidence_source=base.get('evidence_source') if base else None,
                        prefill_evidence_source=prefill.get('evidence_source') if prefill else None,
                        decode_reference_sha256=hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest(),
                        prefill_reference_sha256=hashlib.sha256(json.dumps(prefill, sort_keys=True).encode()).hexdigest())
                if base is None or alone is None:
                    row.update(valid=False, invalid_reason="missing_base_step_or_prefill")
                    self._record_mixed(row)
                    if checkpoint:
                        self._checkpoint()
                    continue
                try:
                    async with self._background(client, batch, ctx, f"m-{f}-{batch}-{chunk}") as (live, tasks):
                        await asyncio.sleep(self.decode_settle_s)
                        self._require_running(tasks)
                        stable_start = time.time()
                        probes = []
                        for repeat in range(3):
                            self._require_running(tasks)
                            r = await client.complete(random_prompt(chunk, 77), 1,
                                                      f"mp-{f}-{batch}-{chunk}-r{repeat}")
                            self._require_running(tasks)
                            if r.error or r.first_token_s is None:
                                raise RuntimeError(r.error or "probe missing first token")
                            probes.append(r)
                        # Include gaps that straddle a probe submission/end boundary.
                        await asyncio.sleep(max(0.1, 2 * base["step_seconds"]))
                        self._require_running(tasks)
                        gaps = [b - a for r in live for a, b in zip(r.token_times_s, r.token_times_s[1:])
                                if any(a <= p.finished_s and b >= p.submitted_s for p in probes)]
                        if not gaps:
                            raise RuntimeError("no background decode intervals overlap probes")
                        ttfts = [r.ttft_s for r in probes]
                        evidence = self.out_dir / "samples" / f"mixed-{f}-{batch}-{chunk}.json"
                        evidence.parent.mkdir(exist_ok=True)
                        evidence.write_text(json.dumps(dict(background=[r.token_times_s for r in live],
                            probes=[dict(start_s=r.submitted_s, first_s=r.first_token_s,
                                         end_s=r.finished_s) for r in probes])))
                        row.update(valid=True, stable_tokens=16, stable_start_s=stable_start,
                            stable_end_s=time.time(), probe_ttft_s=statistics.median(ttfts),
                            probe_ttft_samples=ttfts, probe_ttft_p95=sorted(ttfts)[-1],
                            decode_max_stall_s=max(gaps), additive_pred_s=base["step_seconds"] + alone,
                            samples_file=str(evidence.relative_to(self.out_dir)))
                except (RuntimeError, TimeoutError) as exc:
                    row.update(valid=False, invalid_reason=str(exc))
                if row.get("samples_file"):
                    evidence_path = self.out_dir / row["samples_file"]
                    if evidence_path.is_file():
                        row["samples_sha256"] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
                self._record_mixed(row)
                print(f"mixed f={f} B={batch} chunk={chunk}: {row}", flush=True)
                if checkpoint:
                    self._checkpoint()

    def _record_mixed(self, row: dict) -> None:
        fields = ("freq_mhz", "batch", "chunk_tokens")
        key = tuple(row.get(k) for k in fields)
        previous = [r for r in self.raw["mixed"] if tuple(r.get(k) for k in fields) == key]
        if previous:
            self.raw.setdefault("mixed_attempt_history", []).extend(previous)
        self.raw["mixed"] = [r for r in self.raw["mixed"] if tuple(r.get(k) for k in fields) != key]
        self.raw["mixed"].append(row)

    async def run_sections_for_client(self, client: EngineClient, gpus: Sequence[int],
                                      sections: Sequence[str], freqs: Sequence[int],
                                      *, checkpoint: bool = False) -> None:
        """Run online sections for one resident instance's frequency shard.

        Callers gather this method for resident instances. Checkpoints execute
        synchronously on the shared event loop, so each completed point is
        saved before the next coroutine can update raw data.
        """
        await client.complete(random_prompt(256, 1), 8, "warmup")
        for name, section in (("prefill", self._prefill), ("decode", self._decode), ("mixed", self._mixed)):
            if name in sections:
                active = tuple(f for f in freqs if f in self.mixed_freqs) if name == "mixed" else freqs
                await section(client, gpus, freqs=active, checkpoint=checkpoint)

    async def _parallel_online(self, fleet: Fleet, sections: Sequence[str]) -> None:
        """Run online sections concurrently, with disjoint frequency shards."""
        shards = partition_frequencies(self.freqs, len(self.specs))
        qualification_id = str(time.time_ns())
        async with AsyncExitStack() as stack:
            clients = []
            for spec in self.specs:
                clients.append(await stack.enter_async_context(
                    EngineClient(spec.instance_id, spec.base_url)))
            # First establish a representative isolated-vs-parallel receipt.
            # This is deliberately a small point so failure can fall back to a
            # serial profile without invalidating the whole run.
            check_freq = 2100 if 2100 in self.freqs else self.freqs[0]
            isolated = []
            try:
                for index, (client, spec) in enumerate(zip(clients, self.specs)):
                    self._lock(check_freq, spec.gpus)
                    isolated.append(await self._decode_batch(
                        client, spec.gpus, 8, 1024, DECODE_STEPS,
                        f"parallel-isolated-{qualification_id}-{index}-{check_freq}"))
                for spec in self.specs:
                    self._lock(check_freq, spec.gpus)
                parallel = await asyncio.gather(*(
                    self._decode_batch(client, spec.gpus, 8, 1024, DECODE_STEPS,
                                      f"parallel-concurrent-{qualification_id}-{index}-{check_freq}")
                    for index, (client, spec) in enumerate(zip(clients, self.specs))))
                checks = [evaluate_interference(base, together, limit=.05)
                          for base, together in zip(isolated, parallel)]
                validation = dict(passed=all(item["passed"] for item in checks), checks=checks)
                receipt = dict(complete=True, measured_mode="parallel", point=dict(
                    freq_mhz=check_freq, batch=8, context_tokens=1024), isolated=isolated,
                    parallel=parallel, validation=validation)
                receipt_path = self.out_dir / "samples" / f"parallel-interference-{qualification_id}.json"
                receipt_path.parent.mkdir(parents=True, exist_ok=True)
                receipt_path.write_text(json.dumps(receipt, indent=1, sort_keys=True, default=str))
                receipt["samples_file"] = str(receipt_path.relative_to(self.out_dir))
                receipt["samples_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                self.raw["parallel_interference"] = receipt
                self.raw["measured_mode"] = "parallel" if validation["passed"] else "serial_fallback"
                if not validation["passed"]:
                    raise RuntimeError("parallel interference exceeded 5%")
            except Exception as exc:
                self.raw["parallel_interference"] = dict(complete=False, measured_mode="serial_fallback",
                                                           error=str(exc), point=dict(freq_mhz=check_freq, batch=8,
                                                           context_tokens=1024))
                self.raw["measured_mode"] = "serial_fallback"
                # Keep the rest of the profile valid and attributable.
                for name, section in (("prefill", self._prefill), ("decode", self._decode), ("mixed", self._mixed)):
                    if name in sections:
                        await section(clients[0], self.specs[0].gpus, freqs=self.freqs, checkpoint=True)
                self._checkpoint()
                return
            await asyncio.gather(*(
                self.run_sections_for_client(client, fleet[spec.instance_id].spec.gpus,
                                              sections, shard, checkpoint=True)
                for client, spec, shard in zip(clients, self.specs, shards) if shard))
            self._checkpoint()
        # A single writer checkpoint is the barrier that makes all shards
        # visible atomically to resume and acceptance tooling.
        self._checkpoint()

    # ---- static states ---------------------------------------------------------------------------
    def _static(self, inst, gpus: Sequence[int]) -> None:
        st = self.raw["static"]
        for f in self.freqs:
            self._lock(f, gpus)
            m = self.meter.settle_and_measure(3.0, gpus, settle_s=1.5)
            st[f"active_idle@{f}"] = dict(power_w=m["mean_power_w"], wake_s=0.0)
        self.meter.reset_all()
        m = self.meter.settle_and_measure(3.0, gpus, settle_s=1.5)
        st["active_idle_reset"] = dict(power_w=m["mean_power_w"], wake_s=0.0)
        for g in gpus:
            self.meter.park(g)
        m = self.meter.settle_and_measure(4.0, gpus, settle_s=2.0)
        t = time.time()
        for g in gpus:
            self.meter.unpark(g)
        self._lock(max(self.freqs), gpus)
        wake = time.time() - t
        asyncio.run(self._probe(inst))
        st["parked"] = dict(power_w=m["mean_power_w"], wake_s=wake)
        self.meter.reset_all()
        inst.stop()
        release_s = self.meter.wait_released(gpus)
        m = self.meter.settle_and_measure(4.0, gpus, settle_s=2.0)
        with _load_flock():
            inst.start()
            ready = inst.wait_ready()
        asyncio.run(self._probe(inst))
        st["off"] = dict(power_w=m["mean_power_w"], wake_s=ready, release_s=release_s)
        switches = []
        for a, b in ((900, 2520), (2520, 900), (1500, 2100)):
            self._lock(a, gpus)
            time.sleep(0.2)
            switches.append(self._lock(b, gpus))
        self.raw["freq_switch_s"] = switches

    async def _probe(self, inst):
        async with EngineClient(inst.spec.instance_id, inst.spec.base_url) as c:
            r = await c.complete(random_prompt(64, 3), 4, f"probe-{time.time():.0f}")
            if r.error:
                raise RuntimeError(r.error)

    # ---- KV transfer -----------------------------------------------------------------------------
    async def _transfer(self, p_inst, d_inst) -> None:
        if any(row.get('protocol') != HANDOFF_PROTOCOL for row in self.raw['transfer']):
            raise ValueError('legacy transfer timing cannot be resumed under carry-first-token; retain it as history and collect a separate transfer artifact')
        async with EngineClient(p_inst.spec.instance_id, p_inst.spec.base_url) as pc, \
                EngineClient(d_inst.spec.instance_id, d_inst.spec.base_url) as dc:
            completed = {row["input_tokens"] for row in self.raw["transfer"]}
            for n in TRANSFER_INPUTS:
                if n in completed:
                    continue
                rows = []
                for r in range(3):
                    prompt = random_prompt(n, 900 + r)
                    mixed = await dc.complete(prompt, 4, f"tm-{n}-{r}")
                    pre, dec = await pd_complete(self.transfer, pc, dc, prompt, 4, f"tp-{n}-{r}")
                    if mixed.error or mixed.ttft_s is None or pre.error or dec is None or dec.error:
                        raise RuntimeError(mixed.error or pre.error or (dec.error if dec else "missing decode"))
                    rows.append(measure_handoff(mixed, pre, dec))
                overhead=statistics.median(row['overhead_s'] for row in rows)
                self.raw["transfer"].append(dict(input_tokens=n, overhead_s=overhead, runs=len(rows),
                                                  protocol=HANDOFF_PROTOCOL, timing_repeats=rows,
                                                  concurrency=1, parallel_layout=self.parallel_layout,
                                                  measured_gpu_ids=list(p_inst.spec.gpus)+list(d_inst.spec.gpus)))
                self._checkpoint()
                print(f"transfer n={n}: second-output overhead {overhead*1e3:.1f} ms", flush=True)

    # ---- driver ----------------------------------------------------------------------------------
    def run(self, sections=("prefill", "decode", "mixed", "static", "transfer")) -> Path:
        started = time.time()
        if not sections:
            return self._finish_profile(started)
        with Fleet(self.specs, self.out_dir / "logs") as fleet:
            with _load_flock():
                fleet.start_all()
            inst = fleet[self.specs[0].instance_id]
            gpus = list(inst.spec.gpus)
            self.raw["kv_capacity_tokens"] = self._kv_capacity(inst)

            async def online():
                if self.parallel_instances and len(self.specs) > 1:
                    await self._parallel_online(fleet, sections)
                    return
                async with EngineClient(inst.spec.instance_id, inst.spec.base_url) as client:
                    await client.complete(random_prompt(256, 1), 8, "warmup")
                    for name, section in (("prefill", self._prefill), ("decode", self._decode), ("mixed", self._mixed)):
                        if name in sections:
                            await section(client, gpus)
                            self._checkpoint()
            async def measure_all():
                await online()
                self.meter.reset_all()
                if "transfer" in sections and len(self.specs) >= 2:
                    await self._transfer(inst, fleet[self.specs[1].instance_id])
                    self._checkpoint()
                if "static" in sections:
                    await asyncio.to_thread(self._static, inst, gpus)

            async def coordinated():
                from pdblend.profile.collection.wave import ProfileWave
                wave = ProfileWave.from_environment()
                if wave is None:
                    await measure_all()
                    return
                try:
                    await wave.qualify_external(self, fleet)
                    async with wave.measurement():
                        await measure_all()
                except BaseException as exc:
                    wave.write("error", {"error": repr(exc)})
                    raise
            try:
                asyncio.run(coordinated())
            finally:
                # Keep every completed row even if one frequency shard fails.
                self._checkpoint()
                self.meter.reset_all()
        self.meter.reset_all()
        return self._finish_profile(started)

    def _finish_profile(self, started: float) -> Path:
        self.raw["elapsed_s"] = self.raw.get("elapsed_s", 0.0) + time.time() - started
        self._checkpoint()
        model = self.to_model()
        # Keep measurement completion separate from calibration eligibility: a
        # complete raw profile can be useful for diagnosis, but only a profile
        # that passes the independent provenance, coverage and error audit may
        # feed a planner or formal matrix.
        from pdblend.profile.calibration.acceptance import quality_audit
        audit = quality_audit(self.raw, model, self.out_dir)
        (self.out_dir / "quality-audit.json").write_text(
            json.dumps(audit, indent=1, sort_keys=True, default=str) + "\n")
        model.save(self.out_dir / "profile.json")
        (self.out_dir / "completion.json").write_text(json.dumps({
            "status": "passed", "complete": True, "hardware_qualified": not audit["failures"],
            "formal_eligible": False, "quality_passed": not audit["failures"],
            "missing_gates": ["independent_holdout", "correctness_and_kv", "campaign_acceptance"],
            "quality_failures": audit["failures"],
            "profile": str((self.out_dir / "profile.json").resolve()),
            "profile_key": self.profile_key.as_dict(),
            "raw_sha256": __import__("hashlib").sha256((self.out_dir / "raw.json").read_bytes()).hexdigest(),
        }, indent=2, sort_keys=True) + "\n")
        print(f"profile written to {self.out_dir} in {self.raw['elapsed_s']/60:.1f} min; residuals: "
              + json.dumps({k: round(v, 3) for k, v in model.residuals.items()}), flush=True)
        return self.out_dir / "profile.json"

    def _checkpoint(self) -> None:
        self._snapshot_concurrency_environment()
        # Publish raw only after its immutable evidence snapshots exist. A
        # crash while writing the next checkpoint leaves the previous raw and
        # all of its bindings readable. The row split is an archive partition,
        # not an independent holdout: to_model fits all measured rows.
        sample, holdout = {}, {}
        for section in ("prefill", "decode", "mixed", "transfer"):
            rows = list(self.raw.get(section, ()))
            sample[section] = rows[::2]
            holdout[section] = rows[1::2]
        sample["static"] = self.raw.get("static", {})
        holdout["static"] = self.raw.get("static", {})
        def snapshot(name, data):
            payload = json.dumps(data, indent=1, sort_keys=True, default=str)
            digest = hashlib.sha256(payload.encode()).hexdigest()
            path = self.out_dir / "samples" / f"profile-{name}-{digest}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(payload)
            return path
        sample_path = snapshot("sample", sample)
        holdout_path = snapshot("holdout", holdout)
        from pdblend.profile.identity import evidence_binding, sha256_value
        bindings = {"sample": evidence_binding([sample_path], kind="sample"),
                    "holdout": evidence_binding([holdout_path], kind="holdout")}
        self.raw["evidence_bindings"] = bindings
        self.raw["holdout_independent"] = False
        identity_payload = {key: value for key, value in self.raw.items() if key != "identity_sha256"}
        self.raw["identity_sha256"] = sha256_value(identity_payload)
        path = self.out_dir / "raw.json"
        temp = path.with_suffix(".tmp")
        with temp.open("w") as file:
            file.write(json.dumps(self.raw, indent=1, sort_keys=True, default=str))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)

    def resume(self) -> tuple[str, ...]:
        """Load an earlier raw.json and return the sections it already holds, so a rerun can skip them."""
        path = self.out_dir / "raw.json"
        if not path.exists():
            return ()
        old = json.loads(path.read_text())
        from pdblend.profile.identity import sha256_value
        if old.get("identity_sha256") != sha256_value({k: v for k, v in old.items() if k != "identity_sha256"}):
            raise ValueError("cannot resume profile with invalid checkpoint digest")
        if old.get("schema") != RAW_SCHEMA or old.get("config") != self.raw.get("config"):
            raise ValueError("incompatible raw schema/settings: use a new profile output directory")
        for key in ("source_hash", "image_digest", "gpu_uuids", "hardware_id"):
            if old.get("environment", {}).get(key) != self.raw.get("environment", {}).get(key):
                raise ValueError(f"cannot resume profile with changed {key}")
        for key in ("model_hash", "tokenizer_hash", "profile_key", "parallel_layout"):
            if old.get(key) != self.raw.get(key):
                raise ValueError(f"cannot resume profile with changed {key}")
        for key in ("parallel_interference", "external_interference", "concurrency_environment"):
            evidence = old.get(key) or {}
            name = evidence.get("samples_file")
            if name:
                sample = (self.out_dir / name).resolve()
                if (not sample.is_relative_to(self.out_dir.resolve()) or not sample.is_file()
                        or hashlib.sha256(sample.read_bytes()).hexdigest() != evidence.get("samples_sha256")):
                    raise ValueError(f"cannot resume profile with invalid receipt: {key}")
        for section in ("decode", "mixed"):
            for row in old.get(section, ()):
                for evidence in (row, *row.get("repeats", ())):
                    name = evidence.get("samples_file")
                    if not name:
                        continue
                    sample = (self.out_dir / name).resolve()
                    if (not sample.is_relative_to(self.out_dir.resolve()) or not sample.is_file()
                            or hashlib.sha256(sample.read_bytes()).hexdigest() != evidence.get("samples_sha256")):
                        raise ValueError(f"cannot resume profile with invalid sample: {name}")
        if (old.get("model"), old.get("model_id"), old.get("tp"), old.get("pp"), old.get("system"), old.get("role"), old.get("freqs")) != (self.model, self.model_spec.model_id, self.tp, self.pp, self.system, self.role, list(self.freqs)):
            return ()
        self.raw.update(old)
        self._snapshot_concurrency_environment()
        done = []
        if {(r["freq_mhz"], r["input_tokens"]) for r in old.get("prefill", [])} >= {
                (f, n) for f in self.freqs for n in PREFILL_INPUTS}:
            done.append("prefill")
        if {(r["freq_mhz"], r["batch"], r["chunk_tokens"]) for r in old.get("mixed", []) if r.get("valid")} >= {
                (f, b, c) for f in self.mixed_freqs if f in self.freqs for b, c in MIXED_PROBES}:
            done.append("mixed")
        if {r["input_tokens"] for r in old.get("transfer", [])} >= set(TRANSFER_INPUTS):
            done.append("transfer")
        if old.get("decode") and not self._decode_grid():
            done.append("decode")
        static_keys = {f"active_idle@{f}" for f in self.freqs} | {"active_idle_reset", "parked", "off"}
        return tuple(done) + (("static",) if static_keys <= set(old.get("static", {})) else ())

    def _kv_capacity(self, inst) -> int:
        log = self.out_dir / "logs" / f"{inst.spec.instance_id}.log"
        try:
            for line in log.read_text(errors="ignore").splitlines():
                if "GPU KV cache size:" in line:
                    digits = "".join(ch for ch in line.split("GPU KV cache size:")[-1].split("tokens")[0] if ch.isdigit())
                    if digits:
                        return int(digits)
        except OSError:
            pass
        return 0

    def to_model(self):
        r = self.raw
        static = {k: StaticState(v["power_w"], v.get("wake_s", 0.0)) for k, v in r["static"].items()}
        for f in self.freqs:
            static.setdefault(f"active_idle@{f}", StaticState(0.0))
        model = fit([PrefillPoint(p["freq_mhz"], p["input_tokens"], p["seconds"], p["power_w"]) for p in r["prefill"]],
                   [DecodePoint(d["freq_mhz"], d["batch"], d.get("effective_context_tokens", d["context_tokens"]), d["step_seconds"], d["power_w"],
                                tuple(d.get("power_repeats", ())), float(d.get("steady_window_s", 0.0)))
                    for d in r["decode"] if d["power_w"] is not None],
                   static, [(t["input_tokens"], t["overhead_s"]) for t in r["transfer"]],
                   kv_bytes_per_token=r.get("kv_bytes_per_token", self.kv_bpt), kv_capacity_tokens=r.get("kv_capacity_tokens", 0),
                   freq_switch_s=statistics.median(r["freq_switch_s"]) if r["freq_switch_s"] else 0.15,
                   model=self.model, tp=self.tp, pipeline_parallel=self.pp,
                   system=self.system, profile_key=self.profile_key.as_dict())
        mixed = r.get("mixed", [])
        model.quality["mixed"] = dict(valid=sum(bool(x.get("valid")) for x in mixed),
            invalid=sum(not x.get("valid", False) for x in mixed),
            missing_base=sum(x.get("base_step_s") is None for x in mixed),
            rows=mixed)
        return model


def load_raw(path: Path, model: str, tp: int, kv_bpt: int):
    """Rebuild a PerfModel from raw.json without hardware (used by tests and re-fits)."""
    p = Profiler.__new__(Profiler)
    p.raw = json.loads(Path(path).read_text())
    p.freqs = tuple(p.raw["freqs"])
    p.model, p.tp, p.kv_bpt = model, tp, kv_bpt
    p.pp = int(p.raw.get("pp", 1))
    p.system = p.raw.get("system", "pdblend")
    p.profile_key = ProfileKey(**p.raw["profile_key"]) if p.raw.get("profile_key") else ProfileKey(
        p.system, p.raw.get("model_id", model), "unknown", "unknown", tp, p.pp)
    return p.to_model()
