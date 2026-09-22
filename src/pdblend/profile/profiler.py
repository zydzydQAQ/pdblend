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
import statistics
import time
from pathlib import Path
from typing import Sequence

from ..bench.gates import kv_bytes_per_token, random_prompt
from ..bench.metering import FREQUENCY_TIERS, Gpus
from ..engine.client import EngineClient, PDTransfer, pd_complete
from ..engine.launcher import Fleet, make_specs
from .model import DecodePoint, PrefillPoint, StaticState, fit

PREFILL_INPUTS = (128, 512, 1024, 2048, 4096, 7168)
DECODE_BATCHES = (1, 4, 16, 32, 64, 128, 256)
DECODE_CONTEXTS = (256, 1024, 4096)
DECODE_STEPS = 64
MIXED_PROBES = ((8, 512), (8, 2048), (32, 512), (32, 2048))
MIXED_FREQS = (1500, 2520)
TRANSFER_INPUTS = (512, 2048, 7168)


def window_mean_power(samples, start_s: float, end_s: float, gpu_index: int = 0) -> float | None:
    inside = [row[1][gpu_index] for row in samples if start_s <= row[0] <= end_s]
    return statistics.fmean(inside) if inside else None


class Profiler:
    def __init__(self, model: str, gpus: Sequence[int], tp: int = 1, freqs: Sequence[int] = FREQUENCY_TIERS,
                 window_s: float = 2.0, out_dir: Path = Path("results/v2/profile"), kv_connector: str | None = "P2pNcclConnector"):
        self.model, self.gpus, self.tp, self.freqs, self.window_s = model, list(gpus), tp, tuple(freqs), window_s
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.meter = Gpus(self.gpus)
        self.specs = make_specs(model, self.gpus, tp=tp, kv_connector=kv_connector)
        self.transfer = PDTransfer(kv_connector, {s.instance_id: s.zmq_address for s in self.specs})
        self.kv_bpt = kv_bytes_per_token(self.specs[0].model_path)
        self.raw: dict = dict(model=model, gpus=self.gpus, tp=tp, freqs=list(self.freqs),
                              prefill=[], decode=[], mixed=[], static={}, transfer=[], freq_switch_s=[])

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
    async def _decode_batch(self, client: EngineClient, gpus: Sequence[int], batch: int, ctx: int,
                            steps: int, tag: str) -> dict:
        prompts = [random_prompt(ctx, 1000 * batch + i) for i in range(batch)]
        # early requests decode while later prompts are still being prefilled chunk by chunk; ask for
        # enough tokens that every request is still running once the last prefill has finished
        prefill_chunks = -(-batch * ctx // self.specs[0].max_num_batched_tokens)
        max_tokens = steps + prefill_chunks
        sampler = self.meter.sampler(gpus)
        sampler.start()
        try:
            results = await asyncio.gather(*(client.complete(p, max_tokens, f"{tag}-{i}") for i, p in enumerate(prompts)))
        finally:
            sampler.stop()
        errors = [r.error for r in results if r.error]
        if errors:
            raise RuntimeError(f"decode {tag}: {errors[0]}")
        start = max(r.first_token_s for r in results)
        end = min(r.finished_s for r in results)
        in_window = [sum(1 for t in r.token_times_s if start < t <= end) for r in results]
        window_steps = statistics.median(in_window)
        if end <= start or window_steps < 8:
            raise RuntimeError(f"decode {tag}: steady window too short ({end - start:.3f} s, {window_steps} steps)")
        power = [window_mean_power(sampler.samples, start, end, i) for i in range(len(gpus))]
        power = [p for p in power if p is not None]
        return dict(batch=batch, context_tokens=ctx, step_seconds=(end - start) / window_steps,
                    power_w=sum(power) if self.tp > 1 else statistics.fmean(power) if power else None,
                    steady_window_s=end - start, steps=int(window_steps), max_tokens=max_tokens,
                    prefill_burst_s=start - min(r.submitted_s for r in results))

    def _decode_grid(self) -> list[tuple[int, int, int]]:
        """(freq, ctx, batch) combos still missing from raw["decode"], within 90% of the KV capacity."""
        cap = self.raw.get("kv_capacity_tokens") or 0
        have = {(d["freq_mhz"], d["context_tokens"], d["batch"]) for d in self.raw["decode"]}
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
                await asyncio.sleep(1.5)
            row = await self._decode_batch(client, gpus, b, ctx, DECODE_STEPS, f"d-{f}-{ctx}-{b}")
            row["freq_mhz"] = f
            self.raw["decode"].append(row)
            print(f"decode f={f} ctx={ctx} B={b}: {row['step_seconds']*1e3:.2f} ms/step {row['power_w']:.0f} W", flush=True)
            self._checkpoint()

    # ---- mixed interference ----------------------------------------------------------------------
    async def _mixed(self, client: EngineClient, gpus: Sequence[int]) -> None:
        for f in MIXED_FREQS:
            if f not in self.freqs:
                continue
            self._lock(f, gpus)
            for batch, chunk in MIXED_PROBES:
                ctx = 1024
                prompts = [random_prompt(ctx, 5000 + i) for i in range(batch)]
                probe = random_prompt(chunk, 77)

                async def decoders():
                    return await asyncio.gather(*(client.complete(p, 96, f"m-{f}-{batch}-{chunk}-{i}")
                                                  for i, p in enumerate(prompts)))

                async def prober(delay: float):
                    await asyncio.sleep(delay)
                    return await client.complete(probe, 1, f"mp-{f}-{batch}-{chunk}")

                dec_task = asyncio.create_task(decoders())
                base = self.raw["decode"]
                base_step = next((r["step_seconds"] for r in base
                                  if r["freq_mhz"] == f and r["batch"] == batch and r["context_tokens"] == ctx), None)
                delay = (base_step or 0.03) * 30
                probe_result = await prober(delay)
                results = await dec_task
                errors = [e for e in [probe_result.error] + [r.error for r in results] if e]
                if errors:
                    self.raw["mixed"].append(dict(freq_mhz=f, batch=batch, chunk_tokens=chunk, context_tokens=ctx,
                                                  error=errors[0]))
                    print(f"mixed f={f} B={batch} chunk={chunk}: FAILED {errors[0][:200]}", flush=True)
                    continue
                p_ttft = probe_result.first_token_s - probe_result.submitted_s
                stalls = []
                for r in results:
                    gaps = [b - a for a, b in zip(r.token_times_s, r.token_times_s[1:])]
                    inside = [g for a, g in zip(r.token_times_s, gaps)
                              if probe_result.submitted_s <= a <= probe_result.finished_s + 0.05]
                    if inside:
                        stalls.append(max(inside))
                alone_prefill = next((p["seconds"] for p in self.raw["prefill"]
                                      if p["freq_mhz"] == f and p["input_tokens"] == chunk), None)
                self.raw["mixed"].append(dict(freq_mhz=f, batch=batch, chunk_tokens=chunk, context_tokens=ctx,
                                              probe_ttft_s=p_ttft, decode_max_stall_s=max(stalls) if stalls else None,
                                              base_step_s=base_step, alone_prefill_s=alone_prefill,
                                              additive_pred_s=(base_step or 0) + (alone_prefill or 0)))
                print(f"mixed f={f} B={batch} chunk={chunk}: probe ttft {p_ttft*1e3:.1f} ms, "
                      f"max stall {max(stalls)*1e3 if stalls else -1:.1f} ms", flush=True)

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
                    if dec is None or dec.error:
                        raise RuntimeError(pre.error or dec.error)
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
        if (old.get("model"), old.get("tp"), old.get("freqs")) != (self.model, self.tp, list(self.freqs)):
            return ()
        self.raw.update(old)
        done = [k for k in ("prefill", "mixed", "transfer") if old.get(k)]
        if old.get("decode") and not self._decode_grid():
            done.append("decode")
        return tuple(done) + (("static",) if old.get("static") else ())

    def _kv_capacity(self, inst) -> int:
        log = self.out_dir / "logs" / f"{inst.spec.instance_id}.log"
        try:
            for line in log.read_text(errors="ignore").splitlines():
                if "Maximum concurrency" in line or "GPU KV cache size:" in line:
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
        return fit([PrefillPoint(p["freq_mhz"], p["input_tokens"], p["seconds"], p["power_w"]) for p in r["prefill"]],
                   [DecodePoint(d["freq_mhz"], d["batch"], d["context_tokens"], d["step_seconds"], d["power_w"])
                    for d in r["decode"] if d["power_w"] is not None],
                   static, [(t["input_tokens"], t["overhead_s"]) for t in r["transfer"]],
                   kv_bytes_per_token=self.kv_bpt, kv_capacity_tokens=r.get("kv_capacity_tokens", 0),
                   freq_switch_s=statistics.median(r["freq_switch_s"]) if r["freq_switch_s"] else 0.15,
                   model=self.model, tp=self.tp)


def load_raw(path: Path, model: str, tp: int, kv_bpt: int):
    """Rebuild a PerfModel from raw.json without hardware (used by tests and re-fits)."""
    p = Profiler.__new__(Profiler)
    p.raw = json.loads(Path(path).read_text())
    p.freqs = tuple(p.raw["freqs"])
    p.model, p.tp, p.kv_bpt = model, tp, kv_bpt
    return p.to_model()
