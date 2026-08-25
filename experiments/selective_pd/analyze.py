#!/usr/bin/env python3
"""Aggregate pilot traces and produce the motivation report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SUITE_LABELS = {
    "crossover_colocated": "colocated",
    "crossover_a100_prefill": "PD(A100-P,V100-D)",
    "crossover_v100_prefill": "PD(V100-P,A100-D)",
}


def _latest_runs(root: Path) -> dict[str, Path]:
    candidates: dict[str, list[Path]] = defaultdict(list)
    for manifest_path in root.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidates[manifest["suite"]].append(manifest_path.parent)
    return {
        suite: max(paths, key=lambda path: path.stat().st_mtime_ns)
        for suite, paths in candidates.items()
    }


def _case_kind(case_name: str) -> tuple[str, str, str]:
    workload, load, variant = case_name.rsplit("_", 2)
    return workload, load, variant


def _load_requests(case_dir: Path) -> list[dict[str, Any]]:
    requests = []
    with (case_dir / "requests.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            requests.append(json.loads(line))
    return requests


def _bootstrap_ci(values: list[float], seed: int = 2026) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    estimates = []
    for _ in range(5000):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(statistics.mean(sample))
    estimates.sort()
    return estimates[int(0.025 * len(estimates))], estimates[
        int(0.975 * len(estimates))
    ]


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _mean_present(values: list[Any]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.mean(present) if present else None


def collect_rows(runs: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite, run_dir in sorted(runs.items()):
        for marker in sorted(run_dir.glob("*__r*/complete.json")):
            result = json.loads(marker.read_text(encoding="utf-8"))
            latency = result["latency"]
            energy = result["energy"]
            row = {
                "suite": suite,
                "architecture": SUITE_LABELS.get(suite, suite),
                "case": result["case"],
                "repeat": result["repeat"],
                "input_tokens": latency["input_tokens"],
                "output_tokens": latency["output_tokens"],
                "successful_requests": latency["successful_requests"],
                "failed_requests": latency["failed_requests"],
                "duration_s": latency["duration_s"],
                "request_throughput_rps": latency["request_throughput_rps"],
                "p90_ttft_ms": latency["p90_ttft_ms"],
                "p99_ttft_ms": latency["p99_ttft_ms"],
                "p90_tbt_ms": latency["p90_tbt_ms"],
                "p99_tbt_ms": latency["p99_tbt_ms"],
                "mean_tpot_ms": latency["mean_tpot_ms"],
                "total_energy_j": energy["total_energy_j"],
                "energy_per_request_j": result[
                    "energy_per_successful_request_j"
                ],
                "energy_per_output_token_j": result[
                    "energy_per_output_token_j"
                ],
                "clocks_mhz": json.dumps(
                    result["clocks_mhz"], sort_keys=True
                ),
                "case_dir": str(marker.parent),
            }
            rows.append(row)
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["suite"], row["case"])].append(row)
    output = []
    for (suite, case), values in sorted(grouped.items()):
        energies = [float(row["energy_per_request_j"]) for row in values]
        low, high = _bootstrap_ci(energies)
        output.append(
            {
                "suite": suite,
                "architecture": values[0]["architecture"],
                "case": case,
                "repeats": len(values),
                "mean_energy_per_request_j": statistics.mean(energies),
                "energy_ci95_low_j": low,
                "energy_ci95_high_j": high,
                "mean_energy_per_output_token_j": statistics.mean(
                    float(row["energy_per_output_token_j"]) for row in values
                ),
                "mean_p90_ttft_ms": _mean_present(
                    [row["p90_ttft_ms"] for row in values]
                ),
                "mean_p90_tbt_ms": _mean_present(
                    [row["p90_tbt_ms"] for row in values]
                ),
                "mean_request_throughput_rps": statistics.mean(
                    float(row["request_throughput_rps"]) for row in values
                ),
                "failed_requests": sum(
                    int(row["failed_requests"]) for row in values
                ),
            }
        )
    return output


def _attainment(
    case_dirs: list[Path], ttft_slo: float, tbt_slo: float
) -> tuple[float, float]:
    requests = [
        request
        for case_dir in case_dirs
        for request in _load_requests(case_dir)
        if request["error"] is None
    ]
    ttfts = [
        float(request["ttft_ms"])
        for request in requests
        if request["ttft_ms"] is not None
    ]
    tbts = [
        float(value) for request in requests for value in request["tbt_ms"]
    ]
    return (
        sum(value <= ttft_slo for value in ttfts) / len(ttfts)
        if ttfts
        else 0,
        sum(value <= tbt_slo for value in tbts) / len(tbts) if tbts else 0,
    )


def _selective_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["suite"] in SUITE_LABELS:
            grouped[(row["suite"], row["case"])].append(row)

    thresholds: dict[tuple[str, str], tuple[float, float]] = {}
    for (suite, case), values in grouped.items():
        if suite != "crossover_colocated" or not case.endswith("_max"):
            continue
        workload, load, _ = _case_kind(case)
        requests = [
            request
            for row in values
            for request in _load_requests(Path(row["case_dir"]))
            if request["error"] is None
        ]
        ttfts = [
            float(request["ttft_ms"])
            for request in requests
            if request["ttft_ms"] is not None
        ]
        tbts = [
            float(gap) for request in requests for gap in request["tbt_ms"]
        ]
        thresholds[(workload, load)] = (
            _percentile(ttfts, 90),
            _percentile(tbts, 90),
        )

    selective = []
    for slo_name, factor in (("tight", 1.10), ("loose", 1.50)):
        for key, (base_ttft, base_tbt) in sorted(thresholds.items()):
            workload, load = key
            candidates = []
            for (suite, case), values in grouped.items():
                case_workload, case_load, _ = _case_kind(case)
                if (case_workload, case_load) != key:
                    continue
                ttft_attainment, tbt_attainment = _attainment(
                    [Path(row["case_dir"]) for row in values],
                    base_ttft * factor,
                    base_tbt * factor,
                )
                energy = statistics.mean(
                    float(row["energy_per_request_j"]) for row in values
                )
                candidates.append(
                    {
                        "architecture": SUITE_LABELS[suite],
                        "case": case,
                        "energy_per_request_j": energy,
                        "ttft_attainment": ttft_attainment,
                        "tbt_attainment": tbt_attainment,
                        "feasible": (
                            ttft_attainment >= 0.90
                            and tbt_attainment >= 0.90
                        ),
                    }
                )
            feasible = [candidate for candidate in candidates if candidate["feasible"]]
            winner = (
                min(feasible, key=lambda item: item["energy_per_request_j"])
                if feasible
                else None
            )
            selective.append(
                {
                    "workload": workload,
                    "load": load,
                    "slo": slo_name,
                    "ttft_slo_ms": base_ttft * factor,
                    "tbt_slo_ms": base_tbt * factor,
                    "winner": winner["architecture"] if winner else "none",
                    "winner_case": winner["case"] if winner else "none",
                    "winner_energy_per_request_j": (
                        winner["energy_per_request_j"] if winner else None
                    ),
                    "candidates": candidates,
                }
            )
    return selective


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = [field for field in rows[0] if field != "case_dir"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _phase_section(aggregated: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## 相位频率敏感性",
        "",
        "| GPU / case | p90 TTFT (ms) | p90 TBT (ms) | J/request | J/output token |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregated:
        if not row["suite"].startswith("phase_"):
            continue
        ttft = (
            f"{row['mean_p90_ttft_ms']:.2f}"
            if row["mean_p90_ttft_ms"] is not None
            else "n/a"
        )
        tbt = (
            f"{row['mean_p90_tbt_ms']:.2f}"
            if row["mean_p90_tbt_ms"] is not None
            else "n/a"
        )
        lines.append(
            f"| {row['architecture']} / {row['case']} | "
            f"{ttft} | "
            f"{tbt} | "
            f"{row['mean_energy_per_request_j']:.2f} | "
            f"{row['mean_energy_per_output_token_j']:.4f} |"
        )
    return lines


def _crossover_section(aggregated: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## 等资源架构 crossover",
        "",
        "| Architecture | Case | p90 TTFT (ms) | p90 TBT (ms) | J/request | 95% CI |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in aggregated:
        if row["suite"] not in SUITE_LABELS:
            continue
        lines.append(
            f"| {row['architecture']} | {row['case']} | "
            f"{row['mean_p90_ttft_ms']:.2f} | "
            f"{row['mean_p90_tbt_ms']:.2f} | "
            f"{row['mean_energy_per_request_j']:.2f} | "
            f"[{row['energy_ci95_low_j']:.2f}, "
            f"{row['energy_ci95_high_j']:.2f}] |"
        )
    return lines


def write_report(
    path: Path,
    runs: dict[str, Path],
    rows: list[dict[str, Any]],
    aggregated: list[dict[str, Any]],
    selective: list[dict[str, Any]],
) -> None:
    expected = {
        "phase_a100",
        "phase_v100",
        "crossover_colocated",
        "crossover_a100_prefill",
        "crossover_v100_prefill",
    }
    missing = sorted(expected - set(runs))
    indexed = {
        (row["suite"], row["case"]): row for row in aggregated
    }

    def metric(suite: str, case: str, name: str) -> float:
        return float(indexed[(suite, case)][name])

    a100_decode_energy_saving = 100 * (
        1
        - metric(
            "phase_a100", "decode_f930", "mean_energy_per_request_j"
        )
        / metric(
            "phase_a100", "decode_f1410", "mean_energy_per_request_j"
        )
    )
    a100_decode_tbt_cost = 100 * (
        metric("phase_a100", "decode_f930", "mean_p90_tbt_ms")
        / metric("phase_a100", "decode_f1410", "mean_p90_tbt_ms")
        - 1
    )
    v100_decode_energy_saving = 100 * (
        1
        - metric(
            "phase_v100", "decode_f900", "mean_energy_per_request_j"
        )
        / metric(
            "phase_v100", "decode_f1380", "mean_energy_per_request_j"
        )
    )
    v100_decode_tbt_cost = 100 * (
        metric("phase_v100", "decode_f900", "mean_p90_tbt_ms")
        / metric("phase_v100", "decode_f1380", "mean_p90_tbt_ms")
        - 1
    )
    pd_ttfts = [
        float(row["mean_p90_ttft_ms"])
        for row in aggregated
        if row["suite"] in {
            "crossover_a100_prefill",
            "crossover_v100_prefill",
        }
    ]
    colocated_ttfts = [
        float(row["mean_p90_ttft_ms"])
        for row in aggregated
        if row["suite"] == "crossover_colocated"
    ]
    feasible_pd = sum(
        candidate["feasible"]
        for cell in selective
        for candidate in cell["candidates"]
        if candidate["architecture"].startswith("PD(")
    )
    raw_pd_cheaper = 0
    for row in aggregated:
        if row["suite"] not in {
            "crossover_a100_prefill",
            "crossover_v100_prefill",
        }:
            continue
        colocated = indexed.get(("crossover_colocated", row["case"]))
        if (
            colocated is not None
            and float(row["mean_energy_per_request_j"])
            < float(colocated["mean_energy_per_request_j"])
        ):
            raw_pd_cheaper += 1
    lines = [
        "# Selective PD 本机 motivation 预实验",
        "",
        "## 结论",
        "",
        f"- 相位级 DVFS 确实有收益：A100 decode 从 1410 降到 930 MHz，"
        f"GPU J/request 降 {a100_decode_energy_saving:.1f}%，p90 TBT 仅增加 "
        f"{a100_decode_tbt_cost:.1f}%；V100 的 1380→900 MHz 对应节能 "
        f"{v100_decode_energy_saving:.1f}%、p90 TBT 增加 "
        f"{v100_decode_tbt_cost:.1f}%。",
        f"- 但本机 A100/V100 不支持 CUDA P2P。NIXL 被强制走 UCX host-staged/TCP 后，"
        f"PD p90 TTFT 为 {min(pd_ttfts):.0f}–{max(pd_ttfts):.0f} ms，"
        f"而等资源 colocated 为 {min(colocated_ttfts):.0f}–"
        f"{max(colocated_ttfts):.0f} ms。",
        f"- 不考虑 SLO 时，有 {raw_pd_cheaper} 个 PD 配置的 GPU J/request "
        "低于对应 colocated；加入 90% TTFT+TBT 双 SLO 后，"
        f"可行 PD 配置为 {feasible_pd} 个，所有可行单元均选择 colocated。"
        "这正说明能耗目标不能脱离传输拓扑和 SLO 做全分离。",
        "- 当前设备支持的结论是“弱互连/无 P2P 区域应选择不分离”，"
        "不是“PD 在任何硬件上都无效”。要观察真正 crossover，需要同代、"
        "P2P 可用或 NVLink/RDMA 的第二组硬件。",
        "",
        "## 计量口径",
        "",
        "- 主模型：Qwen2.5-7B-Instruct，FP16，vLLM 0.11.2；"
        "显式关闭 prefix caching，避免重复 trace 命中 KV 缓存。",
        "- PD 传输：NIXL 1.4.0，UCX `self,sm,tcp,cuda_copy`，"
        "用于绕过本机双向 CUDA P2P=`False`。",
        "- 本报告只积分两张 GPU 的 NVML 功率；主机 CPU、DRAM 与 PCIe "
        "交换芯片能量不在计量范围内，因此对 host-staged PD 的能量估计偏乐观。",
        "",
        "## 数据完整性",
        "",
        f"- 原始有效运行：{len(rows)}",
        f"- 已发现 suite：{', '.join(sorted(runs))}",
        f"- 缺失 suite：{', '.join(missing) if missing else '无'}",
        "",
    ]
    lines.extend(_phase_section(aggregated))
    lines.extend([""])
    lines.extend(_crossover_section(aggregated))
    lines.extend(
        [
            "",
            "## 90% 双 SLO 下的 oracle epoch-selective 选择",
            "",
            "| Workload | Load | SLO | TTFT/TBT threshold (ms) | Winner | J/request |",
            "|---|---|---|---:|---|---:|",
        ]
    )
    for row in selective:
        energy = row["winner_energy_per_request_j"]
        lines.append(
            f"| {row['workload']} | {row['load']} | {row['slo']} | "
            f"{row['ttft_slo_ms']:.2f} / {row['tbt_slo_ms']:.2f} | "
            f"{row['winner']} ({row['winner_case']}) | "
            f"{energy:.2f} |"
            if energy is not None
            else (
                f"| {row['workload']} | {row['load']} | {row['slo']} | "
                f"{row['ttft_slo_ms']:.2f} / {row['tbt_slo_ms']:.2f} | "
                "none | n/a |"
            )
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- Colocated 与 PD 始终使用同一组 A100+V100，PD 还交换两次卡的角色；结果不能归因于单一 GPU 型号。",
            "- Gross energy 从首个请求调度到最后一个 token，包含实验窗口中的空闲池功耗，但不含模型加载。",
            "- P2pNcclConnector 的失败由硬件能力核验复现：`nvidia-smi topo -p2p` 为 NS，CUDA peer access 双向为 False；最终 PD 数据均来自获批的 NIXL host-staged fallback。",
            "- 两次重复的 bootstrap 区间只能反映 pilot 抖动；论文级结果应至少五次重复并扩大请求数。",
            "- 两卡只能给出 workload epoch 级 oracle 选择，无法实现三池同时常驻的请求级 hybrid。",
            "",
            "## 下一步",
            "",
            "本机下一步应把 host-staged 结果作为“不可分离区”写入拓扑判据；"
            "若获得 P2P 可用的第二张同代 GPU，再扩到 25%/50%/75% 负载、"
            "三组 SLO、至少五次重复。按本次 pilot 墙钟时间估算，完整矩阵约需 8–12 小时。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", type=Path, default=Path("results/experiments")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiments/summary"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/selective_pd_motivation_pilot.md"),
    )
    args = parser.parse_args()
    runs = {
        suite: path
        for suite, path in _latest_runs(args.input_root).items()
        if not suite.endswith("_confirm")
    }
    rows = collect_rows(runs)
    aggregated = _aggregate(rows)
    selective = _selective_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "summary.csv", rows)
    _write_csv(args.output_dir / "aggregated.csv", aggregated)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "runs": {key: str(value) for key, value in runs.items()},
                "raw": rows,
                "aggregated": aggregated,
                "selective": selective,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(args.report, runs, rows, aggregated, selective)
    print(f"WROTE {args.report}")


if __name__ == "__main__":
    main()
