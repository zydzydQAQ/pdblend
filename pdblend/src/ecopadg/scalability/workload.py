"""Reproducible open-loop workloads sampled from an explicit local corpus."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

from . import protocol as p


def _normalize_record(record):
    if not isinstance(record, dict):
        raise ValueError("pool rows must be objects")
    prompt = record.get("prompt_token_ids", record.get("prompt"))
    plen = record.get("prompt_len", record.get("input_tokens"))
    olen = record.get("output_len", record.get("output_tokens"))
    if type(plen) is not int or plen <= 0 or type(olen) is not int or olen < 2:
        raise ValueError("positive integer prompt_len and output_len >= 2 required")
    if isinstance(prompt, list):
        if len(prompt) != plen or any(type(t) is not int or t < 0 for t in prompt):
            raise ValueError("prompt token IDs must agree with the declared input length")
    elif not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("real prompt text or prompt token IDs required")
    return dict(prompt=copy.deepcopy(prompt), prompt_len=plen, output_len=olen,
                source_id=record.get("source_id", record.get("source_index")),
                content_sha256=p.digest(dict(prompt=prompt, prompt_len=plen, output_len=olen)))


def load_pool(path):
    """Read JSON rows or an explicit {records: [...]} object, without synthesis."""
    payload = json.loads(Path(path).read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError("pool must contain a nonempty explicit list of records")
    return [_normalize_record(row) for row in records]


def trace_hash(trace):
    """Hash the full canonical trace, which deliberately excludes system identity."""
    return p.digest(trace)


def build_trace(pool, *, dataset, n_gpus, seed, rate_rps, stage="formal",
                content_seed=p.CONTENT_SEED, duration_s=None):
    """Produce the real bench_vllm.run_trace {requests, prompts} wire shape.

    System is not an input: every system consumes identical bytes. Content and
    arrival RNGs are independent; the former is also independent of rate and N,
    so compared traces use the same content prefix without length-based bias.
    Each request retains its corpus output cap, served with ignore_eos=True.
    """
    row = p.validate_row(dict(system="pdblend", dataset=dataset, n_gpus=n_gpus,
                              seed=seed, rate_rps=rate_rps, stage=stage))
    expected_duration = p.WARMUP_S if stage == "warmup" else p.ARRIVAL_WINDOW_S
    if duration_s is not None and duration_s != expected_duration:
        raise ValueError("trace duration must equal the frozen stage window")
    if type(content_seed) is not int or content_seed < 0:
        raise ValueError("content_seed must be an explicit nonnegative integer")
    if content_seed != p.CONTENT_SEED and stage != "diagnostic":
        raise ValueError("measurement content seed differs from the frozen protocol")
    records = [_normalize_record(r) for r in pool]
    if not records:
        raise ValueError("cannot generate a trace from an empty pool")
    pool_hash = p.digest(records)
    # Separate domains keep a warm-up corpus realization out of formal traces.
    content_key = p.digest(dict(content_seed=content_seed, arrival_seed=seed,
                                dataset=dataset, phase="warmup" if stage == "warmup" else "measurement"))
    content_rng = random.Random(int(content_key, 16))
    arrival_key = seed if stage != "warmup" else int(p.digest(["warmup", seed]), 16)
    arrival_rng = random.Random(arrival_key)
    requests, prompts, indices, content_hashes = [], [], [], []
    arrival = arrival_rng.expovariate(row["rate_rps"])
    while arrival < expected_duration:
        index = content_rng.randrange(len(records))
        record = records[index]
        request = dict(idx=len(requests), arrival_s=arrival,
                       prompt_len=record["prompt_len"], output_len=record["output_len"],
                       timeout_s=p.request_timeout_s(dataset, record["output_len"]),
                       ignore_eos=True, source_pool_index=index)
        requests.append(request)
        prompts.append(record["prompt"])
        indices.append(index)
        content_hashes.append(record["content_sha256"])
        arrival += arrival_rng.expovariate(row["rate_rps"])
    return dict(schema=2, protocol_id=p.PROTOCOL_ID, protocol_sha256=p.protocol_hash(),
                model=p.MODEL, dataset=dataset, n_gpus=n_gpus, seed=seed,
                arrival_seed=seed, content_seed=content_seed, stage=stage,
                rate_rps=row["rate_rps"], arrival_window_s=expected_duration,
                duration_s=expected_duration, n_requests=len(requests),
                requests=requests, prompts=prompts, source_pool_indices=indices,
                pool_sha256=pool_hash, content_pairing_sha256=p.digest(content_hashes),
                arrival_process="unconditioned Poisson; first arrival exponential",
                content_sampling="uniform with replacement; independent content RNG",
                ignore_eos=True, output_cap="each source record output_len",
                last_request_deadline_offset_s=max(
                    (r["arrival_s"] + r["timeout_s"] for r in requests), default=expected_duration),
                within_trace_resampling=len(set(indices)) < len(indices),
                formal_eligible=False,
                formal_evidence="trace declaration only; actual requests and physical qualification required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--dataset", choices=p.SLOS, required=True)
    parser.add_argument("--n-gpus", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rate-rps", type=float, required=True)
    parser.add_argument("--stage", choices=p.STAGES, default="formal")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    trace = build_trace(load_pool(args.pool), dataset=args.dataset, n_gpus=args.n_gpus,
                        seed=args.seed, rate_rps=args.rate_rps, stage=args.stage)
    with args.out.open("xb") as handle:
        handle.write(p.canonical_bytes(trace))
    print(json.dumps(dict(trace_path=str(args.out.resolve()), trace_sha256=trace_hash(trace),
                         requests=trace["n_requests"])))


if __name__ == "__main__":
    main()
