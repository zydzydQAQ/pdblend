#!/usr/bin/env python3
"""Strict per-point PDblend vs four independent baseline comparison.

Missing, mismatched or SLO-failing baseline evidence is inconclusive; it is
never treated as an energy win.
"""
import argparse
import csv
import json
import math
from pathlib import Path

BASELINES = ("mixed", "distserve_static", "dynamollm", "ecoserve")


def load(path):
    p = Path(path)
    return json.loads((p / "summary.json").read_text()) if (p / "summary.json").exists() else None


def compatible(a, b):
    if not a or not b:
        return False
    return (a.get("model") == b.get("model") and a.get("tp") == b.get("tp")
            and a.get("gpus") == b.get("gpus")
            and a.get("requests") == b.get("requests")
            and a.get("trace_meta", {}).get("dataset") == b.get("trace_meta", {}).get("dataset")
            and a.get("trace_meta", {}).get("seed") == b.get("trace_meta", {}).get("seed")
            and a.get("trace_meta", {}).get("scale") == b.get("trace_meta", {}).get("scale")
            and a.get("window_s", 0) >= 295 and b.get("window_s", 0) >= 295)


def finite(v):
    return v is not None and math.isfinite(float(v))


def compare(candidate_root, baseline_root, out):
    candidate_root, baseline_root = Path(candidate_root), Path(baseline_root)
    rows = []
    for cdir in sorted(candidate_root.glob("*-pdblend_dominance")):
        c = load(cdir)
        if not c:
            continue
        name = cdir.name.removesuffix("-pdblend_dominance")
        row = dict(name=name, candidate_dir=str(cdir), candidate_slo=c["slo"]["joint_slo_rate"],
                   candidate_j_per_token=c.get("j_per_token"), status="inconclusive")
        candidate_ok = c["slo"]["joint_slo_rate"] >= .9 and finite(c.get("j_per_token"))
        reasons = [] if candidate_ok else ["candidate_slo_or_energy"]
        wins = []
        for baseline in BASELINES:
            b = load(baseline_root / f"{name}-{baseline}")
            prefix = f"baseline_{baseline}"
            row[prefix + "_present"] = bool(b)
            if not b:
                reasons.append(prefix + "_missing")
                wins.append(False)
                continue
            row[prefix + "_slo"] = b["slo"]["joint_slo_rate"]
            row[prefix + "_j_per_token"] = b.get("j_per_token")
            if not compatible(c, b):
                reasons.append(prefix + "_mismatch")
                wins.append(False)
                continue
            if b["slo"]["joint_slo_rate"] < .9 or not finite(b.get("j_per_token")):
                reasons.append(prefix + "_not_slo_valid")
                wins.append(False)
                continue
            win = float(c["j_per_token"]) < float(b["j_per_token"])
            row[prefix + "_energy_win"] = win
            wins.append(win)
            if not win:
                reasons.append(prefix + "_energy_loss")
        if candidate_ok and all(wins) and len(wins) == len(BASELINES):
            row["status"] = "win"
        row["reasons"] = ";".join(reasons)
        row["baseline_wins"] = sum(wins)
        rows.append(row)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            for k, v in list(r.items()):
                if isinstance(v, (list, dict)):
                    r[k] = json.dumps(v, sort_keys=True)
            w.writerow(r)
    summary = {"points": len(rows), "wins": sum(r["status"] == "win" for r in rows),
               "inconclusive": sum(r["status"] != "win" for r in rows),
               "baseline_scope": list(BASELINES)}
    out.with_suffix(".json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("candidate_root")
    p.add_argument("baseline_root")
    p.add_argument("out")
    a = p.parse_args()
    compare(a.candidate_root, a.baseline_root, a.out)
