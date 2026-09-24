#!/usr/bin/env python3
"""Report strict all-baseline goals from a frozen cohort points.json snapshot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdblend.bench.cohort_dominance import analyze_points
from pdblend.bench.measurement_compatibility import hydrate_receipt_evidence, load_compatibility


def csv_file(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--measurement-compatibility", type=Path, action='append', default=[],
                        help='Explicit reviewed source pair; repeat for additional frozen source pairs')
    args = parser.parse_args()
    source = args.input_dir.resolve()
    out = args.output_dir.resolve() if args.output_dir else source / "all-baseline-dominance"
    blob = (source / "points.json").read_bytes()
    reviews = [load_compatibility(path) for path in args.measurement_compatibility]
    points = [hydrate_receipt_evidence(row) for row in json.loads(blob)]
    result = analyze_points(points, measurement_compatibility=reviews)
    snapshot = source / "snapshot.json"
    result["source"] = dict(points_path=str(source / "points.json"),
                           points_sha256=hashlib.sha256(blob).hexdigest(),
                           snapshot=json.loads(snapshot.read_text()) if snapshot.exists() else None)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    csv_file(out / "comparisons.csv", result["comparisons"])
    csv_file(out / "cases.csv", result["cases"])
    lines = ["# 全 baseline 能耗与 SLO 目标检查", "",
             "目标：PD 全部到达请求成功且收尾后无未决请求；联合达标请求数不少于四 baseline 的最大值，attainment ≥90%，TTFT/TPOT P99 均不过线。",
             "能耗目标：八卡服务+尾部能耗严格低于四 baseline 的最小值。请求失败或 SLO 不合格的 baseline 仍保留在能耗目标中；缺任一 baseline 或完整能耗，结论为 incomplete。",
             "旧、新 PD revision 分开统计。小于 3% 的正节能幅度需要同配置、同输入的至少 3 次独立配对重复且每次满足目标，才标 stable_observed_win；这不等价于统计显著或正式验收。",
             "energy_complete、measurement qualification 与 formal qualification 分列；快照未携带独立测量资格字段时记 unknown，不从能耗完整或历史 evidence_valid 推断。150秒窗口末请求允许在尾部完成，检查的是收尾后未决请求。", "",
             "numerical_goal_met 只表示数值目标；observed_goal_met 还要求候选独立测量验收通过。跨计量源码哈希必须显式绑定兼容审查，双方原始八卡计量验收通过且服务+尾部能耗完整。baseline 频率失败仍保留作能耗目标，并显示其验收状态；正式 profile 资格不因此升级。", "",
             "状态计数：`" + json.dumps(result["status_counts"], ensure_ascii=False, sort_keys=True) + "`", "",
             "| 系列 / revision | 场景 | 状态 | 观测数 | 最小节能% | 能耗完整 | 测量资格 | 正式资格 |",
             "|---|---|---|---:|---:|---|---|---|"]
    for row in result["cases"]:
        margin = "NA" if row["min_saving_pct"] is None else f'{row["min_saving_pct"]:.3f}'
        lines.append(f'| {row["series"]} / {str(row["revision"])[:10]} | {row["model"]} {row["dataset"]} ×{row["rate_scale"]:g} | {row["status"]} | {row["observations"]} | {margin} | {row["energy_complete"]} | {row["measurement_qualification"]} | {row["formal_eligible"]} |')
    lines.extend(["", "逐点缺失项、未通过原因、baseline 请求成功状态、revision 与 receipt 见 comparisons.csv；来源哈希与全量结果见 summary.json。", ""])
    (out / "report.md").write_text("\n".join(lines))
    print(json.dumps(dict(output=str(out), cases=len(result["cases"]), statuses=result["status_counts"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
