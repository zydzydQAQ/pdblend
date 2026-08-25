#!/usr/bin/env python3
"""Deterministic streaming load generator with TTFT and TBT timestamps."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer


@dataclass
class RequestTrace:
    request_id: int
    scheduled_ns: int
    started_ns: int
    finished_ns: int
    input_tokens: int
    requested_output_tokens: int
    output_tokens: int
    token_times_ns: list[int]
    status: int
    error: str | None

    @property
    def ttft_ms(self) -> float | None:
        if not self.token_times_ns:
            return None
        return (self.token_times_ns[0] - self.started_ns) / 1_000_000

    @property
    def tbt_ms(self) -> list[float]:
        return [
            (right - left) / 1_000_000
            for left, right in zip(self.token_times_ns, self.token_times_ns[1:])
        ]

    @property
    def e2e_ms(self) -> float:
        return (self.finished_ns - self.started_ns) / 1_000_000

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "ttft_ms": self.ttft_ms,
                "tbt_ms": self.tbt_ms,
                "e2e_ms": self.e2e_ms,
            }
        )
        return value


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarize_traces(traces: list[RequestTrace]) -> dict[str, Any]:
    successful = [trace for trace in traces if trace.error is None]
    ttfts = [trace.ttft_ms for trace in successful if trace.ttft_ms is not None]
    tbts = [gap for trace in successful for gap in trace.tbt_ms]
    tpots = [
        statistics.mean(trace.tbt_ms)
        for trace in successful
        if trace.tbt_ms
    ]
    start_ns = min(trace.scheduled_ns for trace in traces)
    end_ns = max(trace.finished_ns for trace in traces)
    duration_s = (end_ns - start_ns) / 1_000_000_000
    output_tokens = sum(trace.output_tokens for trace in successful)
    return {
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_s": duration_s,
        "requests": len(traces),
        "successful_requests": len(successful),
        "failed_requests": len(traces) - len(successful),
        "request_throughput_rps": len(successful) / duration_s,
        "output_throughput_tps": output_tokens / duration_s,
        "input_tokens": sum(trace.input_tokens for trace in successful),
        "output_tokens": output_tokens,
        "mean_ttft_ms": statistics.mean(ttfts) if ttfts else None,
        "p50_ttft_ms": percentile(ttfts, 50),
        "p90_ttft_ms": percentile(ttfts, 90),
        "p99_ttft_ms": percentile(ttfts, 99),
        "mean_tbt_ms": statistics.mean(tbts) if tbts else None,
        "p50_tbt_ms": percentile(tbts, 50),
        "p90_tbt_ms": percentile(tbts, 90),
        "p99_tbt_ms": percentile(tbts, 99),
        "mean_tpot_ms": statistics.mean(tpots) if tpots else None,
    }


def _make_prompts(
    tokenizer: Any, count: int, input_len: int, seed: int
) -> list[list[int]]:
    rng = random.Random(seed)
    special = set(tokenizer.all_special_ids)
    low = min(1000, max(0, tokenizer.vocab_size - 1))
    valid = [
        token_id
        for token_id in range(low, tokenizer.vocab_size)
        if token_id not in special
    ]
    if not valid:
        raise RuntimeError("tokenizer exposes no non-special tokens")
    return [
        [valid[rng.randrange(len(valid))] for _ in range(input_len)]
        for _ in range(count)
    ]


async def _stream_one(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    tokenizer: Any,
    prompt: list[int],
    output_len: int,
    request_id: int,
    scheduled_ns: int,
    semaphore: asyncio.Semaphore,
) -> RequestTrace:
    async with semaphore:
        started_ns = time.monotonic_ns()
        token_times: list[int] = []
        counted_tokens = 0
        status = 0
        error: str | None = None
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": output_len,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
        }
        try:
            async with session.post(url, json=payload) as response:
                status = response.status
                if response.status != 200:
                    error = (await response.text())[:2000]
                else:
                    async for raw_line in response.content:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        body = line[5:].strip()
                        if not body or body == "[DONE]":
                            continue
                        event = json.loads(body)
                        choices = event.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        if (
                            not choice.get("text", "")
                            and choice.get("finish_reason") is not None
                        ):
                            continue
                        token_times.append(time.monotonic_ns())
                        counted_tokens += 1
        except Exception as exc:  # captured in trace, not hidden
            error = f"{type(exc).__name__}: {exc}"
        finished_ns = time.monotonic_ns()
        return RequestTrace(
            request_id=request_id,
            scheduled_ns=scheduled_ns,
            started_ns=started_ns,
            finished_ns=finished_ns,
            input_tokens=len(prompt),
            requested_output_tokens=output_len,
            output_tokens=counted_tokens,
            token_times_ns=token_times,
            status=status,
            error=error,
        )


async def run_workload(
    *,
    base_url: str,
    model: str,
    input_len: int,
    output_len: int,
    num_requests: int,
    request_rate: float,
    max_concurrency: int,
    seed: int,
    tokenizer: Any | None = None,
) -> tuple[list[RequestTrace], dict[str, Any]]:
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        model, local_files_only=True, trust_remote_code=True
    )
    prompts = _make_prompts(tokenizer, num_requests, input_len, seed)
    rng = random.Random(seed + 1)
    arrivals = [0.0]
    for _ in range(1, num_requests):
        arrivals.append(
            arrivals[-1]
            + (
                0.0
                if math.isinf(request_rate)
                else rng.expovariate(request_rate)
            )
        )

    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=max_concurrency)
    semaphore = asyncio.Semaphore(max_concurrency)
    traces: list[RequestTrace] = []
    benchmark_start_ns = time.monotonic_ns()
    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector
    ) as session:

        async def scheduled_request(
            index: int, arrival_s: float
        ) -> RequestTrace:
            delay = (
                benchmark_start_ns / 1_000_000_000
                + arrival_s
                - time.monotonic()
            )
            if delay > 0:
                await asyncio.sleep(delay)
            scheduled_ns = time.monotonic_ns()
            return await _stream_one(
                session,
                f"{base_url.rstrip('/')}/v1/completions",
                model,
                tokenizer,
                prompts[index],
                output_len,
                index,
                scheduled_ns,
                semaphore,
            )

        traces = list(
            await asyncio.gather(
                *[
                    scheduled_request(index, arrival)
                    for index, arrival in enumerate(arrivals)
                ]
            )
        )
    traces.sort(key=lambda trace: trace.request_id)
    return traces, summarize_traces(traces)


def write_results(
    output_dir: Path,
    traces: list[RequestTrace],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "requests.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for trace in traces:
            stream.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
    (output_dir / "latency_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--request-rate", type=float, default=1.0)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    traces, summary = asyncio.run(
        run_workload(
            base_url=args.base_url,
            model=args.model,
            input_len=args.input_len,
            output_len=args.output_len,
            num_requests=args.num_requests,
            request_rate=args.request_rate,
            max_concurrency=args.max_concurrency,
            seed=args.seed,
        )
    )
    write_results(args.output_dir, traces, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
