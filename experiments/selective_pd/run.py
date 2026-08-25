#!/usr/bin/env python3
"""Run one configured Selective-PD experiment suite inside the container."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import vllm
import yaml
from transformers import AutoTokenizer

from .launch import Topology
from .loadgen import run_workload, write_results
from .power_sampler import PowerSampler, integrate_energy


def _run_command(command: list[str], timeout: float = 30) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout.strip()


class ClockController:
    def __init__(self) -> None:
        self.touched: set[int] = set()

    def set(self, clocks: dict[str, int] | dict[int, int]) -> None:
        for raw_gpu, raw_clock in clocks.items():
            gpu = int(raw_gpu)
            clock = int(raw_clock)
            try:
                _run_command(
                    [
                        "nvidia-smi",
                        "-i",
                        str(gpu),
                        "-lgc",
                        f"{clock},{clock}",
                    ]
                )
            except (subprocess.SubprocessError, OSError) as exc:
                raise PermissionError(
                    f"failed to lock GPU {gpu} at {clock} MHz; "
                    "clock-controlled experiments require host-driver "
                    f"permission: {exc}"
                ) from exc
            self.touched.add(gpu)

    def reset(self) -> None:
        errors: list[str] = []
        for gpu in sorted(self.touched):
            try:
                _run_command(["nvidia-smi", "-i", str(gpu), "-rgc"])
            except (subprocess.SubprocessError, OSError) as exc:
                errors.append(f"GPU {gpu}: {exc}")
        self.touched.clear()
        if errors:
            raise RuntimeError("failed to restore GPU clocks: " + "; ".join(errors))


def _config_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def collect_manifest(config: dict[str, Any], suite_name: str) -> dict[str, Any]:
    gpu_query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,"
            "power.min_limit,power.max_limit,power.default_limit,"
            "clocks.max.sm,clocks.max.memory,pci.bus_id",
            "--format=csv,noheader",
        ]
    )
    try:
        nixl_version = importlib.metadata.version("nixl")
    except importlib.metadata.PackageNotFoundError:
        nixl_version = None
    return {
        "created_wall_time_ns": time.time_ns(),
        "suite": suite_name,
        "config_hash": _config_hash(config),
        "image_hint": os.environ.get("PDBLEND_IMAGE", "pdblend:vllm-cu128"),
        "python": platform.python_version(),
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nixl": nixl_version,
        "cuda_peer_access": {
            "0_to_1": torch.cuda.can_device_access_peer(0, 1),
            "1_to_0": torch.cuda.can_device_access_peer(1, 0),
        },
        "prefix_caching": False,
        "hostname": platform.node(),
        "gpu_query": gpu_query.splitlines(),
        "gpu_topology": _run_command(["nvidia-smi", "topo", "-m"]),
        "config": config,
    }


def _request_rate(value: Any) -> float:
    if isinstance(value, str) and value.lower() in {"inf", "infinity"}:
        return math.inf
    return float(value)


def _idle_measurement(
    output_dir: Path, gpu_indices: list[int], seconds: float, interval_s: float
) -> dict[str, Any]:
    sampler = PowerSampler(gpu_indices, interval_s)
    sampler.start()
    time.sleep(max(interval_s * 2, 0.2))
    start_ns = time.monotonic_ns()
    time.sleep(seconds)
    end_ns = time.monotonic_ns()
    time.sleep(max(interval_s * 2, 0.2))
    sampler.stop()
    sampler.write_csv(output_dir / "idle_power.csv")
    summary = integrate_energy(sampler.samples, start_ns, end_ns)
    (output_dir / "idle_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _execute_case(
    *,
    case: dict[str, Any],
    repeat: int,
    topology: Topology,
    model: str,
    tokenizer: Any,
    output_dir: Path,
    interval_s: float,
    warmup_requests: int,
    cooldown_s: float,
) -> dict[str, Any]:
    case_dir = output_dir / f"{case['name']}__r{repeat}"
    completion_marker = case_dir / "complete.json"
    if completion_marker.exists():
        return json.loads(completion_marker.read_text(encoding="utf-8"))
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case.json").write_text(
        json.dumps(case, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if warmup_requests > 0:
        warmup_traces, _ = asyncio.run(
            run_workload(
                base_url=topology.base_url,
                model=model,
                input_len=int(case["input_len"]),
                output_len=min(int(case["output_len"]), 32),
                num_requests=warmup_requests,
                request_rate=math.inf,
                max_concurrency=min(
                    warmup_requests,
                    int(case.get("max_concurrency", warmup_requests)),
                ),
                seed=int(case.get("seed", 1)) + 100_000 + repeat,
                tokenizer=tokenizer,
            )
        )
        failures = [trace.error for trace in warmup_traces if trace.error]
        if failures:
            raise RuntimeError(f"warmup failed: {failures[0]}")
        time.sleep(cooldown_s)

    sampler = PowerSampler(topology.measured_gpus, interval_s)
    sampler.start()
    time.sleep(max(interval_s * 2, 0.2))
    try:
        traces, latency = asyncio.run(
            run_workload(
                base_url=topology.base_url,
                model=model,
                input_len=int(case["input_len"]),
                output_len=int(case["output_len"]),
                num_requests=int(case["num_requests"]),
                request_rate=_request_rate(case["request_rate"]),
                max_concurrency=int(case.get("max_concurrency", 16)),
                seed=int(case.get("seed", 1)),
                tokenizer=tokenizer,
            )
        )
        time.sleep(max(interval_s * 2, 0.2))
    finally:
        sampler.stop()

    write_results(case_dir, traces, latency)
    sampler.write_csv(case_dir / "power.csv")
    energy = integrate_energy(
        sampler.samples, int(latency["start_ns"]), int(latency["end_ns"])
    )
    successful = int(latency["successful_requests"])
    output_tokens = int(latency["output_tokens"])
    combined = {
        "case": case["name"],
        "repeat": repeat,
        "topology": topology.mode,
        "clocks_mhz": case.get("clocks_mhz", {}),
        "latency": latency,
        "energy": energy,
        "energy_per_successful_request_j": (
            energy["total_energy_j"] / successful if successful else None
        ),
        "energy_per_output_token_j": (
            energy["total_energy_j"] / output_tokens if output_tokens else None
        ),
    }
    completion_marker.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return combined


def run_suite(config_path: Path, suite_name: str, output_root: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if suite_name not in config["suites"]:
        raise KeyError(f"suite {suite_name!r} is not in {config_path}")
    defaults = config["defaults"]
    suite = config["suites"][suite_name]
    run_id = (
        f"{time.strftime('%Y%m%d-%H%M%S')}-{suite_name}-"
        f"{_config_hash(suite)}"
    )
    suite_dir = output_root / run_id
    suite_dir.mkdir(parents=True, exist_ok=False)
    (suite_dir / "manifest.json").write_text(
        json.dumps(
            collect_manifest(config, suite_name),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    topology = Topology(
        mode=suite["mode"],
        model=defaults["model"],
        log_dir=suite_dir / "logs",
        profile_gpu=int(suite.get("profile_gpu", 0)),
        prefill_gpu=int(suite.get("prefill_gpu", 0)),
        decode_gpu=int(suite.get("decode_gpu", 1)),
        weights=tuple(int(v) for v in suite.get("weights", [1, 1])),
        max_model_len=int(defaults.get("max_model_len", 4096)),
        gpu_memory_utilization=float(
            defaults.get("gpu_memory_utilization", 0.75)
        ),
        kv_buffer_size=float(defaults.get("kv_buffer_size", 3e9)),
        kv_connector=str(defaults.get("kv_connector", "nixl")),
        startup_timeout_s=float(defaults.get("startup_timeout_s", 900)),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        defaults["model"], local_files_only=True, trust_remote_code=True
    )
    clocks = ClockController()
    summaries: list[dict[str, Any]] = []
    cases = list(suite["cases"])
    if defaults.get("randomize_case_order", True):
        random.Random(int(defaults.get("order_seed", 2026))).shuffle(cases)

    try:
        with topology:
            idle_clocks = suite.get("idle_clocks_mhz")
            if idle_clocks:
                clocks.set(idle_clocks)
            _idle_measurement(
                suite_dir,
                topology.measured_gpus,
                float(defaults.get("idle_seconds", 60)),
                float(defaults.get("sample_interval_s", 0.05)),
            )
            for case in cases:
                clocks.set(case.get("clocks_mhz", {}))
                repeats = int(case.get("repeats", defaults.get("repeats", 2)))
                for repeat in range(repeats):
                    print(
                        f"RUN suite={suite_name} case={case['name']} "
                        f"repeat={repeat}",
                        flush=True,
                    )
                    summaries.append(
                        _execute_case(
                            case=case,
                            repeat=repeat,
                            topology=topology,
                            model=defaults["model"],
                            tokenizer=tokenizer,
                            output_dir=suite_dir,
                            interval_s=float(
                                defaults.get("sample_interval_s", 0.05)
                            ),
                            warmup_requests=int(
                                defaults.get("warmup_requests", 2)
                            ),
                            cooldown_s=float(defaults.get("cooldown_s", 2)),
                        )
                    )
    except Exception as exc:
        (suite_dir / "FAILED.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        raise
    finally:
        clocks.reset()

    (suite_dir / "suite_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"COMPLETE {suite_dir}", flush=True)
    return suite_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/selective_pd/configs/pilot.yaml"),
    )
    parser.add_argument("--suite", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/experiments"),
    )
    args = parser.parse_args()
    try:
        run_suite(args.config, args.suite, args.output_root)
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
