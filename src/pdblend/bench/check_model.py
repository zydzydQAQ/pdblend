"""Planner model vs. measured matrix points: does admission agree with observed SLO attainment?"""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

from ..control.planner import SLO
from .matrix import _planner, parse_kv


def check_model(profile: Path, root: Path, corpus: Path, slo_ok: float = 0.9, out: Path | None = None) -> dict:
    spec = json.loads((root / "spec.json").read_text())
    points = {p["name"]: dict(spec.get("defaults", {}), **p) for p in spec["points"]}
    rows = []
    for summary in sorted(root.glob("*/summary.json")):
        p = points.get(summary.parent.name)
        if p is None or not p.get("layout"):
            continue
        d = json.loads(summary.read_text())
        s = d["slo"]
        planner, counts, fc = _planner(profile, corpus, p["dataset"], p["layout"], p.get("split", "evaluation"))
        fc = replace(fc, rate_rps=float(p["rate"]))
        c = parse_kv(p.get("clocks", "P=2520,D=2520,M=2520"))
        f_P, f_D, f_M, tau = c.get("P", 2520), c.get("D", 2520), c.get("M", 2520), int(p.get("tau", 0))
        admitted = planner.evaluate(counts, f_P, f_D, f_M, tau, fc) is not None
        loose = copy.copy(planner)
        loose.cfg = replace(planner.cfg, slo=SLO(1e9, 1e9), rho_max=0.999)
        pred = loose.evaluate(counts, f_P, f_D, f_M, tau, fc)
        measured_ok = s["joint_slo_rate"] >= slo_ok
        rows.append(dict(name=p["name"], rate=p["rate"], admitted=admitted, measured_ok=measured_ok,
                         mismatch=admitted != measured_ok, slo=s["joint_slo_rate"],
                         ttft_p90=s["ttft_p90"], tpot_p90=s["tpot_p90"], power_w=d["window_mean_power_w"],
                         pred_ttft=pred.ttft_s if pred else None, pred_tpot=pred.tpot_s if pred else None,
                         pred_power_w=pred.power_w if pred else None))
    agree = [r for r in rows if r["pred_power_w"] and r["measured_ok"]]
    power_err = [abs(r["pred_power_w"] - r["power_w"]) / r["power_w"] for r in agree]
    result = dict(profile=str(profile), root=str(root), points=len(rows), mismatches=sum(r["mismatch"] for r in rows),
                  false_admit=[r["name"] for r in rows if r["admitted"] and not r["measured_ok"]],
                  false_reject=[r["name"] for r in rows if not r["admitted"] and r["measured_ok"]],
                  power_rel_err_max=max(power_err, default=0.0),
                  power_rel_err_mean=sum(power_err) / len(power_err) if power_err else 0.0, rows=rows)
    if out:
        out.write_text(json.dumps(result, indent=1))
    return result


def format_rows(result: dict) -> str:
    lines = []
    for r in result["rows"]:
        pred = (f"ttft={r['pred_ttft']:.3f} tpot={r['pred_tpot']:.3f} P={r['pred_power_w']:.0f}"
                if r["pred_power_w"] else "unstable")
        flag = "  <-- MISMATCH" if r["mismatch"] else ""
        lines.append(f"{r['name']:30s} r={r['rate']:7.2f} slo={r['slo']:.2f} ttft90={r['ttft_p90']:6.3f} "
                     f"tpot90={r['tpot_p90']:.3f} P={r['power_w']:4.0f} | admit={'Y' if r['admitted'] else 'N'} {pred}{flag}")
    return "\n".join(lines)
