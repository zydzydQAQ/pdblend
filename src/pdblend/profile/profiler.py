"""Small profiling grid (~150 points, <1 h per model) feeding profile.model.fit.

prefill:  input x freq, back-to-back single requests for >= window seconds
decode:   batch x context x freq, concurrent requests; step time from token arrivals,
          power averaged over the interval where all requests are decoding
mixed:    decode batch + chunk prefill probes to validate the additive interference model
static:   active idle per frequency, sleep L1/L2, off; wake latency
transfer: P->D handoff overhead vs mixed TTFT (same method as gate G0)
"""
from __future__ import annotations

import asyncio
import json
import os
import platform
import subprocess
import statistics
import time
from importlib.metadata import PackageNotFoundError, version
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Sequence

from ..bench.gates import kv_bytes_per_token, random_prompt
from ..bench.metering import FREQUENCY_TIERS, Gpus
from ..engine.client import EngineClient, PDTransfer, pd_complete
from ..engine.launcher import Fleet, make_specs
from .model import DecodePoint, PrefillPoint, StaticState, fit

PREFILL_INPUTS = (128, 512, 1024, 2048, 4096, 7168)
DECODE_BATCHES = (1, 4, 8, 16, 32, 64, 128, 256)
DECODE_CONTEXTS = (256, 1024, 4096)
DECODE_STEPS = 64
RAW_SCHEMA = 2
MIXED_PROBES = ((8, 512), (8, 2048), (32, 512), (32, 2048))
MIXED_FREQS = (1500, 2100, 2520)
TRANSFER_INPUTS = (512, 2048, 7168)


def window_mean_power(samples, start_s: float, end_s: float, gpu_index: int = 0) -> float | None:
    inside = [row[1][gpu_index] for row in samples if start_s <= row[0] <= end_s]
    return statistics.fmean(inside) if inside else None


class Profiler:
    def __init__(self, model: str, gpus: Sequence[int], tp: int = 1, freqs: Sequence[int] = FREQUENCY_TIERS,
                 window_s: float = 2.0, out_dir: Path = Path("results/v2/profile"), kv_connector: str | None = "P2pNcclConnector",
                 mixed_freqs: Sequence[int] = (1500, 2100, 2520), decode_repeats: int = 3,
                 decode_settle_s: float = 2.0, decode_measure_s: float = 5.0, base_port: int = 8100):
        self.model, self.gpus, self.tp, self.freqs, self.window_s = model, list(gpus), tp, tuple(freqs), window_s
        self.mixed_freqs = tuple(mixed_freqs)
        if decode_repeats < 3 or decode_settle_s < 2 or decode_measure_s < 5:
            raise ValueError("profile-v2 requires >=3 repeats, >=2 s settle and >=5 s measurement")
        self.decode_repeats = int(decode_repeats)
        self.decode_settle_s = float(decode_settle_s)
        self.decode_measure_s = float(decode_measure_s)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.meter = Gpus(self.gpus)
        self.specs = make_specs(model, self.gpus, tp=tp, base_port=base_port, kv_connector=kv_connector)
        self.transfer = PDTransfer(kv_connector, {s.instance_id: s.zmq_address for s in self.specs})
        self.kv_bpt = kv_bytes_per_token(self.specs[0].model_path)
        self.raw: dict = dict(schema=RAW_SCHEMA, model=model, gpus=self.gpus, tp=tp, freqs=list(self.freqs),
                              prefill=[], decode=[], mixed=[], static={}, transfer=[], freq_switch_s=[],
                              kv_bytes_per_token=self.kv_bpt,
                              config=dict(decode_repeats=self.decode_repeats,
                                           decode_settle_s=self.decode_settle_s,
                                           decode_measure_s=self.decode_measure_s,
                                           mixed_freqs=list(self.mixed_freqs), base_port=base_port,
                                           decode_batches=list(DECODE_BATCHES)),
                              environment=self._environment_metadata(self.gpus))

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
                    image_digest=os.environ.get("PDBLEND_IMAGE_DIGEST"),
                    source_hash=os.environ.get("PDBLEND_SOURCE_HASH"),
                    gpu_uuids=uuids or os.environ.get("NVIDIA_VISIBLE_DEVICES"))

    def _lock(self, freq: int, gpus: Sequence[int]) -> float:
        started = time.time()
        for g in gpus:
            self.meter.set_clock(g, freq)
        return time.time() - started

    # ---- prefill -------------------------------------------------------------------------------
    async def _prefill(self, client: EngineClient, gpus: Sequence[int]) -> None:
        for f in self.freqs:
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
                                                power_w=m["mean_power_w"] / len(gpus) if self.tp == 1 else m["mean_power_w"],
                                                runs=len(times), busy_fraction=busy))
                print(f"prefill f={f} n={n}: {statistics.median(times)*1e3:.1f} ms {m['mean_power_w']:.0f} W", flush=True)
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

    async def _wait_progress(self, live, tasks, targets, timeout_s=120.0):
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
        max_tokens = max(512, DECODE_STEPS + 32)
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
                                 steps: int, tag: str) -> dict:
        async with self._background(client, batch, ctx, tag) as (live, tasks):
            # Settle under continuous decode, not idle. NVML is a trailing ~1 s average.
            await asyncio.sleep(self.decode_settle_s)
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
            return dict(batch=batch, context_tokens=ctx, effective_context_tokens=effective_context,
                step_seconds=(end - start) / statistics.median(counts),
                power_w=sum(power) if self.tp > 1 else statistics.fmean(power),
                steady_window_s=end - start, start_s=start, end_s=end,
                steps=int(statistics.median(counts)), min_steps=min(counts),
                power_samples=len(power_samples), frequency_samples=len(clocks),
                mean_freq_mhz=statistics.fmean(v for _, values in clocks for v in values),
                samples_file=str(evidence.relative_to(self.out_dir)))

    async def _decode_batch(self, client: EngineClient, gpus: Sequence[int], batch: int, ctx: int,
                            steps: int, tag: str) -> dict:
        warm = await client.complete(random_prompt(ctx, 700000 + batch * 1000 + ctx), 4, f"{tag}-warmup")
        if warm.error:
            raise RuntimeError(f"decode warmup {tag}: {warm.error}")
        repeats = [await self._decode_batch_once(client, gpus, batch, ctx, steps, f"{tag}-r{r}")
                   for r in range(self.decode_repeats)]
        return dict(batch=batch, context_tokens=ctx,
            effective_context_tokens=statistics.median(r["effective_context_tokens"] for r in repeats),
            step_seconds=statistics.median(r["step_seconds"] for r in repeats),
            power_w=statistics.median(r["power_w"] for r in repeats),
            power_repeats=[r["power_w"] for r in repeats],
            step_repeats=[r["step_seconds"] for r in repeats],
            steady_window_s=min(r["steady_window_s"] for r in repeats),
            steps=min(r["steps"] for r in repeats), repeats=repeats,
            power_samples=sum(r["power_samples"] for r in repeats),
            frequency_samples=sum(r["frequency_samples"] for r in repeats))

    def _decode_grid(self) -> list[tuple[int, int, int]]:
        """(freq, ctx, batch) combos still missing from raw["decode"], within 90% of the KV capacity."""
        cap = self.raw.get("kv_capacity_tokens") or 0
        have = {(d["freq_mhz"], d["context_tokens"], d["batch"]) for d in self.raw["decode"] if len(d.get("repeats", [])) >= self.decode_repeats}
        return [(f, ctx, b) for f in self.freqs for ctx in DECODE_CONTEXTS for b in DECODE_BATCHES
                if not (cap and b * (ctx + DECODE_STEPS) > 0.9 * cap) and (f, ctx, b) not in have]

    async def _decode(self, client: EngineClient, gpus: Sequence[int]) -> None:
        locked = None
        for f, ctx, b in self._decode_grid():
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
            self._checkpoint()

    # ---- mixed interference ----------------------------------------------------------------------
    async def _mixed(self, client: EngineClient, gpus: Sequence[int]) -> None:
        have = {(r["freq_mhz"], r["batch"], r["chunk_tokens"]) for r in self.raw["mixed"] if r.get("valid")}
        for f in self.mixed_freqs:
            if f not in self.freqs:
                continue
            self._lock(f, gpus)
            for batch, chunk in MIXED_PROBES:
                if (f, batch, chunk) in have:
                    continue
                ctx = 1024
                base = next((r for r in self.raw["decode"] if r["freq_mhz"] == f
                             and r["batch"] == batch and r["context_tokens"] == ctx), None)
                alone = next((r["seconds"] for r in self.raw["prefill"]
                              if r["freq_mhz"] == f and r["input_tokens"] == chunk), None)
                row = dict(freq_mhz=f, batch=batch, chunk_tokens=chunk, context_tokens=ctx,
                           base_step_s=base["step_seconds"] if base else None, alone_prefill_s=alone)
                if base is None or alone is None:
                    row.update(valid=False, invalid_reason="missing_base_step_or_prefill")
                    self.raw["mixed"].append(row)
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
                self.raw["mixed"].append(row)
                print(f"mixed f={f} B={batch} chunk={chunk}: {row}", flush=True)
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
        async with EngineClient(p_inst.spec.instance_id, p_inst.spec.base_url) as pc, \
                EngineClient(d_inst.spec.instance_id, d_inst.spec.base_url) as dc:
            for n in TRANSFER_INPUTS:
                rows = []
                for r in range(3):
                    prompt = random_prompt(n, 900 + r)
                    mixed = await dc.complete(prompt, 4, f"tm-{n}-{r}")
                    pre, dec = await pd_complete(self.transfer, pc, dc, prompt, 4, f"tp-{n}-{r}")
                    if mixed.error or mixed.ttft_s is None or pre.error or dec is None or dec.error:
                        raise RuntimeError(mixed.error or pre.error or (dec.error if dec else "missing decode"))
                    rows.append((dec.first_token_s - pre.submitted_s) - mixed.ttft_s)
                self.raw["transfer"].append(dict(input_tokens=n, overhead_s=statistics.median(rows), runs=len(rows)))
                print(f"transfer n={n}: overhead {statistics.median(rows)*1e3:.1f} ms", flush=True)

    # ---- driver ----------------------------------------------------------------------------------
    def run(self, sections=("prefill", "decode", "mixed", "static", "transfer")) -> Path:
        started = time.time()
        with Fleet(self.specs, self.out_dir / "logs") as fleet:
            fleet.start_all()
            inst = fleet[self.specs[0].instance_id]
            gpus = list(inst.spec.gpus)
            self.raw["kv_capacity_tokens"] = self._kv_capacity(inst)

            async def online():
                async with EngineClient(inst.spec.instance_id, inst.spec.base_url) as client:
                    await client.complete(random_prompt(256, 1), 8, "warmup")
                    for name, section in (("prefill", self._prefill), ("decode", self._decode), ("mixed", self._mixed)):
                        if name in sections:
                            await section(client, gpus)
                            self._checkpoint()
            asyncio.run(online())
            self.meter.reset_all()
            if "transfer" in sections and len(self.specs) >= 2:
                asyncio.run(self._transfer(inst, fleet[self.specs[1].instance_id]))
                self._checkpoint()
            if "static" in sections:
                self._static(inst, gpus)
        self.meter.reset_all()
        self.raw["elapsed_s"] = self.raw.get("elapsed_s", 0.0) + time.time() - started
        self._checkpoint()
        model = self.to_model()
        model.save(self.out_dir / "profile.json")
        print(f"profile written to {self.out_dir} in {self.raw['elapsed_s']/60:.1f} min; residuals: "
              + json.dumps({k: round(v, 3) for k, v in model.residuals.items()}), flush=True)
        return self.out_dir / "profile.json"

    def _checkpoint(self) -> None:
        (self.out_dir / "raw.json").write_text(json.dumps(self.raw, indent=1, default=str))

    def resume(self) -> tuple[str, ...]:
        """Load an earlier raw.json and return the sections it already holds, so a rerun can skip them."""
        path = self.out_dir / "raw.json"
        if not path.exists():
            return ()
        old = json.loads(path.read_text())
        if old.get("schema") != RAW_SCHEMA or old.get("config") != self.raw.get("config"):
            raise ValueError("incompatible raw schema/settings: use a new profile output directory")
        for key in ("source_hash", "image_digest", "gpu_uuids"):
            if old.get("environment", {}).get(key) != self.raw.get("environment", {}).get(key):
                raise ValueError(f"cannot resume profile with changed {key}")
        if (old.get("model"), old.get("tp"), old.get("freqs")) != (self.model, self.tp, list(self.freqs)):
            return ()
        self.raw.update(old)
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
                   model=self.model, tp=self.tp)
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
    return p.to_model()
