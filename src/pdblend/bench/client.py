"""Open-loop load generation against the proxy: Poisson, staged Gamma-renewal and Azure 2024 replay."""
from __future__ import annotations

import asyncio
import csv
import json
import math
import multiprocessing
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import aiohttp

from ..proxy.sse import StreamScan

MAX_INPUT, MAX_OUTPUT, CONTEXT = 7168, 512, 8192
SLOS = {"alpaca": (1.0, 0.10), "sharegpt": (5.0, 0.15), "longbench": (15.0, 0.20)}


@dataclass
class Request:
    idx: int
    arrival_s: float
    prompt: list
    max_tokens: int
    source: str = ""

    @property
    def input_tokens(self) -> int:
        return len(self.prompt)


@dataclass
class Outcome:
    idx: int
    arrival_s: float
    input_tokens: int
    max_tokens: int
    submitted_s: float
    first_token_s: Optional[float] = None
    finished_s: Optional[float] = None
    completion_tokens: int = 0
    path: str = ""
    prefill: str = ""
    decode: str = ""
    error: Optional[str] = None
    sampling_seed: Optional[int] = None

    @property
    def ttft_s(self):
        return None if self.first_token_s is None else self.first_token_s - self.submitted_s

    @property
    def tpot_s(self):
        if self.first_token_s is None or self.finished_s is None or self.completion_tokens < 2:
            return None
        return (self.finished_s - self.first_token_s) / (self.completion_tokens - 1)


# ---- corpus -------------------------------------------------------------------------------------
def load_split(corpus_root: Path, dataset: str, split: str = "evaluation") -> list[dict]:
    data = json.loads((Path(corpus_root) / f"{dataset}.json").read_text())
    return [r for r in data[split] if r["output_tokens"] >= 2]


def _fit_prompt(records: list[dict], rng: random.Random, n: int) -> list:
    """A prompt of exactly n tokens built from corpus prompts (for shape-driven traces)."""
    tokens: list = []
    while len(tokens) < n:
        tokens += rng.choice(records)["prompt"]
    return tokens[:n]


# ---- traces -------------------------------------------------------------------------------------
def poisson_trace(records: list[dict], rate_rps: float, duration_s: float, seed: int, source: str = "") -> list[Request]:
    content, arrival = random.Random(seed * 7919 + 1), random.Random(seed)
    out, t = [], arrival.expovariate(rate_rps)
    while t < duration_s:
        r = content.choice(records)
        out.append(Request(len(out), t, r["prompt"], min(r["output_tokens"], MAX_OUTPUT), source))
        t += arrival.expovariate(rate_rps)
    return out


def staged_trace(records: list[dict], base_rate_rps: float, stages: Sequence[tuple[float, float]], cv: float,
                 seed: int, source: str = "") -> list[Request]:
    """stages: [(duration_s, rate_scale), ...]; Gamma renewal inter-arrivals with the given CV."""
    content, arrival = random.Random(seed * 7919 + 1), random.Random(seed)
    k = 1.0 / (cv * cv)
    out, t0 = [], 0.0
    for duration, scale in stages:
        rate = base_rate_rps * scale
        t = t0 + arrival.gammavariate(k, 1.0 / (rate * k))
        while t < t0 + duration:
            r = content.choice(records)
            out.append(Request(len(out), t, r["prompt"], min(r["output_tokens"], MAX_OUTPUT), source))
            t += arrival.gammavariate(k, 1.0 / (rate * k))
        t0 += duration
    return out


def read_azure_window(csv_path: Path, offset_s: float, duration_s: float) -> list[tuple[float, int, int]]:
    """(relative_time_s, context_tokens, generated_tokens) rows within [offset, offset+duration)."""
    rows, base = [], None
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)
        for ts, ctx, gen in reader:
            t = datetime.fromisoformat(ts).timestamp()
            base = t if base is None else base
            rel = t - base
            if rel < offset_s:
                continue
            if rel >= offset_s + duration_s:
                break
            rows.append((rel - offset_s, int(ctx), int(gen)))
    return rows


def azure_trace(rows: list[tuple[float, int, int]], records: list[dict], peak_rps: float, seed: int,
                bin_s: float = 60.0, source: str = "azure") -> tuple[list[Request], dict]:
    """Thin the Azure arrivals so the busiest `bin_s` bin runs at `peak_rps`; shapes come from the trace,
    prompt content from the corpus, lengths clipped to the 8192 context."""
    if not rows:
        return [], {}
    duration = rows[-1][0]
    bins = [0] * (int(duration // bin_s) + 1)
    for t, _, _ in rows:
        bins[int(t // bin_s)] += 1
    peak = max(bins) / bin_s
    keep = min(1.0, peak_rps / peak)
    rng, content = random.Random(seed), random.Random(seed * 7919 + 1)
    out = []
    for t, ctx, gen in rows:
        if rng.random() > keep:
            continue
        n_in = max(1, min(ctx, MAX_INPUT))
        n_out = max(2, min(gen, MAX_OUTPUT, CONTEXT - n_in))
        out.append(Request(len(out), t, _fit_prompt(records, content, n_in), n_out, source))
    meta = dict(source_rows=len(rows), kept=len(out), thinning=keep, source_peak_rps=peak,
                target_peak_rps=peak_rps, bin_s=bin_s, duration_s=duration)
    return out, meta


def trace_summary(reqs: list[Request]) -> dict:
    if not reqs:
        return dict(requests=0)
    ins = sorted(r.input_tokens for r in reqs)
    outs = sorted(r.max_tokens for r in reqs)
    q = nearest_rank
    return dict(requests=len(reqs), duration_s=reqs[-1].arrival_s, mean_rps=len(reqs) / max(reqs[-1].arrival_s, 1e-9),
                input_mean=sum(ins) / len(ins), input_p50=q(ins, 0.5), input_p95=q(ins, 0.95),
                input_min=min(ins), input_max=max(ins), output_min=min(outs), output_max=max(outs),
                output_mean=sum(outs) / len(outs), output_p50=q(outs, 0.5), output_p95=q(outs, 0.95))


def nearest_rank(values: Sequence[float], percentile: float):
    """Nearest-rank percentile (1-indexed), shared by all reports."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(float(percentile) * len(ordered)))
    return ordered[min(len(ordered), rank) - 1]


# ---- replay -------------------------------------------------------------------------------------
class LoadClient:
    def __init__(self, proxy_url: str, timeout_s: float = 300.0, concurrency: int = 2048,
                 sampling_seed: int | None = None):
        self.args = (proxy_url, timeout_s, concurrency, sampling_seed)
        self.url = proxy_url.rstrip("/") + "/v1/completions"
        self.timeout = aiohttp.ClientTimeout(total=timeout_s, sock_read=timeout_s)
        self.sem = asyncio.Semaphore(concurrency)
        self.outcomes: list[Outcome] = []

    async def _one(self, session: aiohttp.ClientSession, req: Request, t0: float) -> Outcome:
        body = dict(model="m", prompt=req.prompt, max_tokens=req.max_tokens, temperature=0.0,
                    ignore_eos=True, stream=True, request_id=f"r{req.idx}")
        sampling_seed = None if self.args[3] is None else int(self.args[3])
        if sampling_seed is not None:
            body["seed"] = sampling_seed
        out = Outcome(req.idx, req.arrival_s, req.input_tokens, req.max_tokens, time.time(),
                      sampling_seed=sampling_seed)
        try:
            async with self.sem, session.post(self.url, json=body) as resp:
                out.path = resp.headers.get("X-PDBlend-Path", "")
                out.prefill = resp.headers.get("X-PDBlend-Prefill", "")
                out.decode = resp.headers.get("X-PDBlend-Decode", "")
                if resp.status != 200:
                    out.error = f"{resp.status}: {(await resp.text())[:200]}"
                else:
                    scan = StreamScan()
                    async for chunk in resp.content.iter_any():
                        _, n = scan.feed(chunk)
                        if n and out.first_token_s is None:
                            out.first_token_s = time.time()
                        if b'"error"' in chunk:
                            for line in chunk.split(b"\n"):
                                if line.startswith(b"data:") and b'"error"' in line:
                                    try:
                                        event = json.loads(line[5:].strip())
                                    except ValueError:
                                        continue
                                    if "error" in event and not event.get("choices"):
                                        out.error = str(event["error"])[:200]
                        if scan.done:
                            break
                    out.completion_tokens = scan.completion_tokens()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            out.error = repr(exc)
        out.finished_s = time.time()
        return out

    async def replay(self, reqs: list[Request], progress_every_s: float = 30.0) -> list[Outcome]:
        t0 = time.time()
        tasks = []
        last = t0
        async with aiohttp.ClientSession(timeout=self.timeout, connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=3.0)) as session:
            for req in reqs:
                delay = t0 + req.arrival_s - time.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                tasks.append(asyncio.create_task(self._one(session, req, t0)))
                if time.time() - last >= progress_every_s:
                    done = sum(1 for t in tasks if t.done())
                    print(f"[load] t={time.time()-t0:6.1f}s sent={len(tasks)} done={done}", flush=True)
                    last = time.time()
            self.outcomes = list(await asyncio.gather(*tasks))
        return self.outcomes

    async def replay_detached(self, reqs: list[Request], progress_every_s: float = 30.0) -> list[Outcome]:
        """replay() in a spawned process, so stream parsing does not share the proxy's event loop."""
        ctx = multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_replay_worker, args=(self.args, reqs, progress_every_s, child), daemon=True)
        proc.start()
        child.close()

        def receive():
            while not parent.poll(1.0):
                if not proc.is_alive():
                    raise RuntimeError(f"load process exited with code {proc.exitcode} before reporting outcomes")
            return parent.recv()

        try:
            self.outcomes = await asyncio.get_running_loop().run_in_executor(None, receive)
        finally:
            proc.join()
        return self.outcomes


def _replay_worker(args: tuple, reqs: list[Request], progress_every_s: float, conn) -> None:
    conn.send(asyncio.run(LoadClient(*args).replay(reqs, progress_every_s)))
    conn.close()


def slo_attainment(outcomes: list[Outcome], ttft_slo: float, tpot_slo: float) -> dict:
    n = len(outcomes)
    ok = [o for o in outcomes if o.error is None and o.first_token_s is not None and o.finished_s is not None]
    joint = [o for o in ok if o.ttft_s is not None and o.tpot_s is not None and
             o.ttft_s <= ttft_slo and o.tpot_s <= tpot_slo]
    ttfts = sorted(o.ttft_s for o in ok)
    tpots = sorted(o.tpot_s for o in ok if o.tpot_s is not None)
    def stats(values):
        return {f"p{p}": nearest_rank(values, p / 100) for p in (50, 90, 95, 99)} | {
            "max": max(values) if values else None, "samples": len(values), "missing": n - len(values)}
    errors = sum(1 for o in outcomes if o.error is not None)
    missing_first = sum(1 for o in outcomes if o.error is None and o.first_token_s is None)
    missing_tpot = sum(1 for o in ok if o.tpot_s is None)
    incomplete = sum(1 for o in outcomes if o.error is None and (o.finished_s is None or o.completion_tokens < 2))
    return dict(offered=n, succeeded=len(ok), error=errors, errors=errors, rejected=0,
                missing_first_token=missing_first, missing_tpot=missing_tpot, incomplete=incomplete,
                joint_slo=len(joint), joint_slo_requests=len(joint), joint_slo_rate=len(joint) / n if n else 0.0,
                success_rate=len(ok) / n if n else 0.0,
                joint_output_tokens=sum(o.completion_tokens for o in joint),
                **{f"ttft_{k}": v for k, v in stats(ttfts).items()},
                **{f"tpot_{k}": v for k, v in stats(tpots).items()},
                output_tokens=sum(o.completion_tokens for o in ok),
                paths={p: sum(1 for o in outcomes if o.path == p) for p in {o.path for o in outcomes}})


def dump_outcomes(outcomes: list[Outcome], path: Path) -> None:
    Path(path).write_text("\n".join(json.dumps(asdict(o)) for o in outcomes) + "\n")
