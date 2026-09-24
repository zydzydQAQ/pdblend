#!/usr/bin/env python3
"""Export frozen comparison observations to 18 wide plotting CSV files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

SYSTEMS = [("pdblend", "PDBlend"), ("mixed", "Mixed"),
           ("distserve", "DistServe"), ("dynamollm", "DynamoLLM"),
           ("ecoserve", "EcoServe")]
MODELS = ["7B", "14B", "32B"]
DATASETS = [("alpaca", "Alpaca"), ("sharegpt", "ShareGPT"), ("longbench", "LongBench")]
HEADER = ["rate"] + [label for _, label in SYSTEMS]


def number(value):
    value = Decimal(value)
    if not value.is_finite():
        raise ValueError(f"Nonfinite value: {value}")
    return value


def fmt(value):
    result = format(value, "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export(source_dir, out):
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    snapshot = json.loads((source_dir / "snapshot.json").read_text())
    raw = (source_dir / "compare-snapshot.csv").read_bytes()
    if hashlib.sha256(raw).hexdigest() != snapshot["source_sha256"]:
        raise ValueError("Snapshot hash mismatch")
    with (source_dir / "compare-snapshot.csv").open() as f:
        all_rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    other_status = defaultdict(set)
    for row in all_rows:
        key = (row["model_id"], row["dataset"], row["system"], number(row["offered_rps"]))
        if row["status"] == "measured":
            if row["measurement_usable"].lower() != "true":
                raise ValueError("Measured row lacks usable measurement: " + row["point_id"])
            groups[key].append(row)
        else:
            other_status[key].add(row["status"])
    selected = {}
    for key, candidates in groups.items():
        selected[key] = max(candidates, key=lambda r: (
            snapshot["receipt_mtime_ns"][r["receipt_path"]], r["receipt_path"]))
    out.mkdir(parents=True)
    metadata = out / "metadata"
    metadata.mkdir()
    provenance, file_info, checks = [], [], []
    for model in MODELS:
        model_id = f"Qwen2.5-{model}-Instruct"
        for dataset, dataset_label in DATASETS:
            rates = sorted({k[3] for k in selected if k[:2] == (model_id, dataset)})
            energy_rows, slo_rows = [], []
            for rate in rates:
                energy_row, slo_row = {"rate": fmt(rate)}, {"rate": fmt(rate)}
                for system, label in SYSTEMS:
                    key = (model_id, dataset, system, rate)
                    row = selected.get(key)
                    energy, slo, energy_status = "", "", "no_completed_observation"
                    prov = dict(model=model, dataset=dataset_label, rate=fmt(rate), system=label,
                                energy_service_tail_kJ="", slo_attainment_pct="",
                                observation_status="no_completed_observation",
                                energy_status=energy_status, other_record_statuses=";".join(sorted(other_status[key])),
                                revision="", point_id="", rate_scale="", seed="", duration_s="",
                                offered_requests="", successful_requests="", joint_slo_requests="",
                                service_energy_j="", tail_energy_j="", slo_pass="",
                                formal_eligible="", strict_rank_eligible="",
                                receipt_path="", receipt_sha256="", receipt_mtime_ns="",
                                measured_candidates=0, excluded_receipts="")
                    if row is not None:
                        receipt_path = Path(row["receipt_path"])
                        receipt_bytes = receipt_path.read_bytes()
                        if hashlib.sha256(receipt_bytes).hexdigest() != row["receipt_sha256"]:
                            raise ValueError("Receipt hash mismatch: " + str(receipt_path))
                        receipt = json.loads(receipt_bytes)
                        # Historical receipts predate this marker; their bound
                        # result and measured status supply completion evidence.
                        if receipt.get("recorded_window_complete") is False:
                            raise ValueError("Window incomplete: " + str(receipt_path))
                        result_bytes = (receipt_path.parent / "result.json").read_bytes()
                        if hashlib.sha256(result_bytes).hexdigest() != receipt["artifacts"]["result.json"]:
                            raise ValueError("Result hash mismatch: " + str(receipt_path))
                        result = json.loads(result_bytes)
                        metrics = result["metrics"]
                        for field in ["energy_service_j", "energy_tail_j", "energy_service_tail_j", "joint_slo_rate"]:
                            expected = metrics.get(field)
                            actual = row[field]
                            if expected is None:
                                if actual:
                                    raise ValueError("Unexpected numeric value for " + field)
                            elif not actual or not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-9):
                                raise ValueError("Metric mismatch: " + row["point_id"] + " " + field)
                        slo_fraction = number(row["joint_slo_rate"])
                        if not 0 <= slo_fraction <= 1:
                            raise ValueError("SLO outside [0, 1]")
                        slo = fmt(slo_fraction * 100)
                        service, tail = row["energy_service_j"], row["energy_tail_j"]
                        if service and tail and row["energy_comparable"].lower() == "true":
                            total = number(service) + number(tail)
                            if total < 0 or not math.isclose(float(total), float(row["energy_service_tail_j"]), rel_tol=1e-12, abs_tol=1e-9):
                                raise ValueError("Invalid energy sum: " + row["point_id"])
                            energy, energy_status = fmt(total / 1000), "complete_service_and_tail"
                        elif not service or not tail:
                            energy_status = "missing_" + "_and_".join(name for name, val in [("service", service), ("tail", tail)] if not val)
                        else:
                            energy_status = "energy_not_comparable"
                        prov.update({field: row[field] for field in [
                            "revision", "point_id", "rate_scale", "seed", "duration_s",
                            "offered_requests", "successful_requests", "joint_slo_requests",
                            "slo_pass", "formal_eligible", "strict_rank_eligible", "receipt_path", "receipt_sha256"]})
                        prov.update(observation_status="measured", energy_status=energy_status,
                                    energy_service_tail_kJ=energy, slo_attainment_pct=slo,
                                    service_energy_j=service, tail_energy_j=tail,
                                    receipt_mtime_ns=snapshot["receipt_mtime_ns"][row["receipt_path"]],
                                    measured_candidates=len(groups[key]),
                                    excluded_receipts=";".join(r["receipt_path"] for r in groups[key] if r is not row))
                        checks.append(row["receipt_path"])
                    energy_row[label], slo_row[label] = energy, slo
                    provenance.append(prov)
                energy_rows.append(energy_row)
                slo_rows.append(slo_row)
            for directory, records, unit in [
                ("energy_kJ", energy_rows, "kJ, eight-GPU service plus tail"),
                ("slo_attainment_pct", slo_rows, "percent, 0 to 100")]:
                rel = Path(directory) / f"{model}_{dataset_label}.csv"
                write_csv(out / rel, records, HEADER)
                file_info.append(dict(path=str(rel), rows=len(records), unit=unit,
                                      numeric_cells=sum(bool(r[label]) for r in records for _, label in SYSTEMS)))
    write_csv(metadata / "point_sources.csv", provenance, list(provenance[0]))
    manifest = dict(
        schema="pdblend-wide-plot-csv/v1", snapshot_time=snapshot["snapshot_time"],
        source_mtime=snapshot["source_mtime"], source_sha256=snapshot["source_sha256"],
        source_rows=len(all_rows), source_status_counts=dict(Counter(r["status"] for r in all_rows)),
        selected_observations=len(selected), receipt_hashes_verified=len(checks),
        result_hashes_verified=len(checks), files=file_info, columns=HEADER,
        selection="Per model/dataset/system/configured rate, latest measured receipt_mtime_ns; receipt_path lexicographic tie-break; never by metric.",
        energy_scope="150-second eight-GPU service window plus recorded request tail; both segments must be available and energy_comparable=true.",
        energy_unit="kJ", rate_unit="configured offered requests per second",
        slo_unit="percent (0-100)", slo_definition="successful requests satisfying both TTFT and TPOT thresholds / all offered requests * 100",
        blank="No completed observation, missing energy segment, or energy not comparable; see metadata/point_sources.csv.",
        retest_policy="Separate 2026-09-25 lowload candidate retest is not substituted into the main comparison matrix.",
        mixed_revision_notice="PDBlend: 7B/14B use d4fe except 7B LongBench rate 5.859375 uses 498e; 32B uses 5f09. See per-point provenance.",
        qualification="Observed single-seed results; does not grant formal or strict ranking qualification.",
    )
    (metadata / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (out / "README.md").write_text(
        "# 五系统绘图 CSV\n\n"
        f"冻结时间：{snapshot['snapshot_time']}；源表更新时间：{snapshot['source_mtime']}。\n\n"
        "共 18 份绘图 CSV：energy_kJ/ 下 9 份能耗表，slo_attainment_pct/ 下 9 份 SLO attainment 表。"
        "文件按模型_数据集命名；UTF-8 BOM，逗号分隔。\n\n"
        "所有文件列顺序：`rate,PDBlend,Mixed,DistServe,DynamoLLM,EcoServe`，rate 按数值升序排列。\n\n"
        "- rate：配置到达率，单位请求/s；不是负载倍率、完成吞吐或请求数/150。\n"
        "- energy_kJ：150 秒服务窗口＋请求收尾的八卡 GPU 能耗，单位 kJ。两段任一缺失即留空。"
        "未包含模型加载、启动、窗口间 reset 及 CPU/主机能耗。\n"
        "- slo_attainment_pct：成功且同时满足 TTFT、TPOT 的请求数 / 全部到达请求数 ×100，取值 0–100%。"
        "失败请求计入分母；这是请求级 attainment，区别于整点 SLO 是否通过。\n"
        "- SLO 门槛：Alpaca TTFT 1s / TPOT 100ms；ShareGPT 5s / 150ms；LongBench 15s / 200ms。\n"
        "- 空单元格表示没有完整观测或指标缺失；0 表示实际测得的零。不插值、不以零补缺失、"
        "不从旧版本补能量。能耗与 SLO 始终取自同一所选 receipt。\n\n"
        "## 选点和版本\n\n"
        "仅选主比较表 status=measured 的记录；同模型、数据集、系统、rate 按冻结 receipt 落盘时间取最新一次，"
        "同时间按 receipt_path 排序。保留 SLO 失败观测，不按能耗或 SLO 高低择优。"
        "本包是截至快照时间的现有观测汇总。\n\n"
        "PDBlend 7B/14B 主要取当前 d4fe2c6ac5 修订；7B LongBench rate=5.859375 仅有旧 498e446c2f 修订，"
        "该行应标为旧版观测、不要与 d4fe 曲线直接连线。32B 使用历史完整轮 5f09841bc8。"
        "其他系统也可能存在不同修订，完整逐点版本见 metadata/point_sources.csv。\n\n"
        "今日 ShareGPT ×0.25 的 PD2100/Eco/PD1500 候选配置对照属于独立实验，未覆盖主矩阵中的同名系统数值。"
        "这些数据均为 seed701 的观测；代码修订和配置重跑不作为独立重复样本平均，不提供误差条或正式排名资格。\n\n"
        "## 可追溯性\n\n"
        "metadata/point_sources.csv 逐单元格列出来源、版本、缺失原因、被排除的重测及两个指标；"
        "metadata/manifest.json 记录快照、规则、单位与验证计数；metadata/SHA256SUMS 校验包内文件。\n",
        encoding="utf-8")
    print(json.dumps(dict(selected=len(selected), files=file_info), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.source.resolve(), args.out.resolve())
