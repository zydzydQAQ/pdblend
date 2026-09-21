#!/usr/bin/env python3
"""Compare pdblend against the four baselines at every dataset x scale point.

Pass = pdblend joint >= 0.9 AND mean_power <= min(mean_power of baselines with joint >= 0.9).
If no baseline passes SLO at a point, pdblend passes with joint >= 0.9 and power <= min baseline power.
"""
import json
import sys
from pathlib import Path

BASELINES = ("mixed", "distserve_static", "dynamollm", "ecoserve")
SLO_OK = 0.9


def load(root: Path):
    pts = {}
    for d in sorted(root.iterdir()):
        f = d / "summary.json"
        if not f.is_file():
            continue
        parts = d.name.split("-")
        if len(parts) < 3 or not parts[1].startswith("x"):
            continue
        ds, sc, pol = parts[0], parts[1], "-".join(parts[2:])
        s = json.loads(f.read_text())
        pts[(ds, sc, pol)] = dict(joint=s["slo"]["joint_slo_rate"], w=s["mean_power_w"],
                                  jtok=s["energy_j"] / max(1, s["slo"]["output_tokens"]),
                                  ttft90=s["slo"]["ttft_p90"], tpot90=s["slo"]["tpot_p90"])
    return pts


def main(root: Path) -> int:
    pts = load(root)
    cells = sorted({(ds, sc) for ds, sc, _ in pts}, key=lambda c: (c[0], float(c[1][1:])))
    fails = []
    header = f"{'point':<22}{'mixed':>9}{'distserve':>10}{'dynamollm':>10}{'ecoserve':>10}{'pdblend':>9}  verdict"
    print(header)
    print("-" * len(header))
    for ds, sc in cells:
        row = f"{ds}-{sc:<16}"
        base_ok, base_any = [], []
        for pol in BASELINES:
            r = pts.get((ds, sc, pol))
            if r is None:
                row += f"{'--':>9 if pol=='mixed' else 10}"
                continue
            mark = "" if r["joint"] >= SLO_OK else f"({r['joint']:.2f})"
            row += f"{r['w']:>9.0f}{mark:<1}" if pol == "mixed" else f"{r['w']:>10.0f}{mark:<1}"
            (base_ok if r["joint"] >= SLO_OK else base_any).append(r)
        p = pts.get((ds, sc, "pdblend"))
        if p is None:
            row += f"{'--':>9}  (pending)"
            print(row)
            continue
        mark = "" if p["joint"] >= SLO_OK else f"({p['joint']:.2f})"
        row += f"{p['w']:>9.0f}{mark:<1}"
        ref = base_ok or base_any
        if not ref:
            verdict = "no-baseline"
        elif p["joint"] < SLO_OK:
            verdict = "FAIL joint"
        elif p["w"] <= min(r["w"] for r in ref):
            verdict = f"PASS (-{(1 - p['w']/min(r['w'] for r in ref))*100:.1f}%)"
        else:
            verdict = f"LOSE (+{(p['w']/min(r['w'] for r in ref) - 1)*100:.1f}%)"
            fails.append((ds, sc))
        print(row + "  " + verdict)
    print(f"\npdblend loses at {len(fails)} points: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "results/v2/eval-7b-v2")))
