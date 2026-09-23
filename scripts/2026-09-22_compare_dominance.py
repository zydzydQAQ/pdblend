#!/usr/bin/env python3
"""Fail-closed paired comparison for one PDblend screening matrix."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdblend.bench.dominance import BASELINES, PAIR_FIELDS, load_evidence  # noqa: E402
from pdblend.seed_config import SINGLE_SEED, SEED_POLICY  # noqa: E402


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def m2_used(directory: Path) -> bool:
    path = directory / "controller.jsonl"
    if not path.exists():
        return False
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") == "plan" and 0 < row.get("counts", {}).get("M", 0) < 4:
            return True
    return False


def compare(candidate_root: Path, baseline_root: Path, out: Path) -> dict:
    rows = []
    for cdir in sorted(candidate_root.glob("*-pdblend_dominance")):
        name = cdir.name.removesuffix("-pdblend_dominance")
        candidate = load_evidence(cdir)
        row = {"name": name, "candidate_dir": str(cdir), "status": "inconclusive",
               "m2_experimental": m2_used(cdir), "single_seed": True,
               "seed_policy": SEED_POLICY}
        reasons = []
        if candidate is None:
            reasons.append("candidate_missing_or_stale")
            row["reasons"] = ";".join(reasons)
            rows.append(row)
            continue
        cm = candidate["metrics"]
        row.update({"candidate_slo": cm["joint_slo_rate"], "candidate_j_per_token": cm["j_per_token"],
                    "candidate_ttft_p99": cm["ttft_p99"], "candidate_tpot_p99": cm["tpot_p99"]})
        if candidate["identity"].get("seed") != SINGLE_SEED:
            reasons.append("unsupported_seed:" + str(candidate["identity"].get("seed")))
        if cm["joint_slo_rate"] < .9 or not finite(cm["j_per_token"]):
            reasons.append("candidate_slo_or_energy")
        wins = []
        for policy in BASELINES:
            bdir = baseline_root / f"{name}-{policy}"
            baseline = load_evidence(bdir)
            prefix = f"baseline_{policy}"
            row[prefix + "_present"] = baseline is not None
            if baseline is None:
                reasons.append(prefix + "_missing_or_stale"); wins.append(False); continue
            bm = baseline["metrics"]
            row[prefix + "_slo"] = bm["joint_slo_rate"]
            row[prefix + "_j_per_token"] = bm["j_per_token"]
            mismatches = [key for key in PAIR_FIELDS
                          if candidate["identity"].get(key) != baseline["identity"].get(key)]
            if baseline["identity"].get("seed") != SINGLE_SEED:
                mismatches.append("unsupported_seed")
            if mismatches:
                reasons.append(prefix + "_mismatch:" + ",".join(mismatches)); wins.append(False); continue
            win = finite(bm["j_per_token"]) and float(cm["j_per_token"]) < float(bm["j_per_token"])
            row[prefix + "_energy_win"] = win
            wins.append(win)
            if not win:
                reasons.append(prefix + "_energy_loss")
        row["baseline_wins"] = sum(wins)
        if not reasons and len(wins) == len(BASELINES):
            row["status"] = "screen_m2" if row["m2_experimental"] else "screen_pass"
        row["reasons"] = ";".join(reasons)
        rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader()
        writer.writerows(rows)
    summary = {"points": len(rows), "screen_pass": sum(r["status"] == "screen_pass" for r in rows),
               "screen_m2": sum(r["status"] == "screen_m2" for r in rows),
               "inconclusive": sum(r["status"] == "inconclusive" for r in rows),
               "baseline_scope": list(BASELINES), "single_seed": True,
               "seed_policy": SEED_POLICY}
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    return {"summary": summary, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    compare(args.candidate_root, args.baseline_root, args.out)
