"""Short PDBlend development smoke wrapper.

This exercises the current policy and an independent profile with a fixed
seed-701 trace.  It records operational outcomes and power/controller files,
but intentionally never claims formal campaign or energy-comparison status.
Dynamic TP and complete KV validation are outside this smoke's scope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from ..control.planner import SLO
from .smoke_trace import build_smoke_trace
from .run import run_point


SEED = 701
WARMUP_SEED = 9701


def run_smoke(*, model: str, gpus: list[int], tp: int, profile: Path,
              corpus: Path | None, out: Path, base_port: int = 8100,
              proxy_port: int | None = None, dataset: str = "sharegpt",
              rate: float = 0.2, duration: float = 100.0, seed: int = SEED) -> dict:
    if seed != SEED:
        raise ValueError(f"smoke only supports seed {SEED}")
    # Keep the request shapes short/medium/long and independent of the
    # workload corpus; this makes the smoke trace reproducible across systems.
    trace = build_smoke_trace(duration_s=duration, seed=SEED, rate_rps=rate)
    warmup = build_smoke_trace(duration_s=min(8.0, duration), seed=WARMUP_SEED,
                               rate_rps=0.5)[:8]
    out.mkdir(parents=True, exist_ok=True)
    proxy_port = base_port + 80 if proxy_port is None else proxy_port
    trace_payload = [asdict(request) for request in trace]
    trace_hash = hashlib.sha256(json.dumps(trace_payload, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
    (out / "trace.json").write_text(json.dumps({"seed": seed, "trace_sha256": trace_hash,
                                                 "requests": trace_payload}, indent=2) + "\n")
    completion = {"status": "failed", "complete": False, "errors": [], "seed": SEED,
                  "scope": {"dynamic_tp": False, "complete_kv": False},
                  "formal_eligible": False, "energy_comparable": False}
    try:
        result = run_point(
            model, gpus, tp, "pdblend", profile, trace,
            SLO(*{"alpaca": (1.0, 0.10), "sharegpt": (5.0, 0.15),
                   "longbench": (15.0, 0.20)}[dataset]), out, warmup,
            proxy_port=proxy_port, base_port=base_port,
            trace_meta={"seed": seed, "source": "smoke-seed701",
                        "warmup_seed": WARMUP_SEED,
                        "input_lengths": [128, 512, 2048],
                        "trace_sha256": trace_hash},
            sampling_seed=seed)
        errors = []
        outcomes = out / "outcomes.jsonl"
        rows = [json.loads(line) for line in outcomes.read_text().splitlines() if line.strip()] if outcomes.is_file() else []
        if len(rows) != len(trace) or not trace:
            errors.append(f"outcomes count {len(rows)} != trace count {len(trace)}")
        for row in rows:
            if (row.get("error") or row.get("completion_tokens") != 16 or
                    row.get("first_token_s") is None or row.get("finished_s") is None or
                    not row.get("path") or row.get("sampling_seed") != seed):
                errors.append(f"invalid outcome {row.get('idx')}")
        for name in ("controller.jsonl", "power.jsonl"):
            if not (out / name).is_file() or not (out / name).read_text().strip():
                errors.append(f"missing {name}")
        metering_path = out / "metering.json"
        if not metering_path.is_file() or json.loads(metering_path.read_text()).get("error"):
            errors.append("metering error or missing metering.json")
        completion.update({"status": "passed" if not errors else "failed",
                           "complete": not errors,
                           "errors": errors, "outcomes": str(outcomes),
                           "controller": str(out / "controller.jsonl"),
                           "power": str(out / "power.jsonl"),
                           "summary": result})
    except Exception as exc:  # retain a machine-readable failed completion
        completion["errors"] = [f"{type(exc).__name__}: {exc}"]
    (out / "completion.json").write_text(json.dumps(completion, indent=2, default=str) + "\n")
    return completion


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", required=True, help="comma-separated physical ordinals")
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--base-port", type=int, default=8100)
    parser.add_argument("--proxy-port", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", choices=("alpaca", "sharegpt", "longbench"), default="sharegpt")
    parser.add_argument("--rate", type=float, default=0.2)
    parser.add_argument("--duration", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    result = run_smoke(model=args.model, gpus=[int(x) for x in args.gpus.split(",") if x],
                       tp=args.tp, profile=args.profile, corpus=args.corpus,
                       out=args.out, base_port=args.base_port, proxy_port=args.proxy_port,
                       dataset=args.dataset, rate=args.rate, duration=args.duration, seed=args.seed)
    print(json.dumps({"status": result["status"], "out": str(args.out), "seed": SEED}, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
