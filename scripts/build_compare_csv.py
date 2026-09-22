#!/usr/bin/env python3
"""Build compare.csv: one row per measured point, all metrics + pdblend-vs-baselines verdict.

Everything is recomputed from the raw artifacts (summary/outcomes/power/util/freq/controller),
so points measured before and after the util patch are directly comparable.
Idempotent: rebuilds the whole CSV on every run. Run inside the container:

    python scripts/build_compare_csv.py results/v2/eval-7b-v2
"""
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

BASELINES = ("mixed", "distserve_static", "dynamollm", "ecoserve")
POLICY_ORDER = {"mixed": 0, "distserve_static": 1, "dynamollm": 2, "ecoserve": 3, "pdblend": 4}
SLO_FLOOR = 0.9
TDP_W = 275.0
ROLE_ORDER = ("P", "D", "M", "L1", "off")
ACTIVE = ("P", "D", "M")

_cap_cache: dict = {}


def result_path(path: Path) -> Path | None:
    """Resolve a result stream in plain or gzip-compressed form."""
    if path.exists():
        return path
    gz = Path(str(path) + ".gz")
    return gz if gz.exists() else None


def open_result(path: Path):
    resolved = result_path(path)
    if resolved is None:
        raise FileNotFoundError(path)
    return gzip.open(resolved, "rt") if resolved.name.endswith(".gz") else resolved.open()


def q(vals, p):
    return vals[min(len(vals) - 1, int(p * len(vals)))] if vals else None


def r(x, n=4):
    return round(x, n) if isinstance(x, float) else x


def layout_str(roles: dict) -> str:
    c = Counter(roles.values())
    return ",".join(f"{k}={c[k]}" for k in ROLE_ORDER if c.get(k))


def capacity_util(spec_defaults, ds, final_roles, last_plan, mean_rps):
    if not final_roles or not last_plan or not mean_rps:
        return None
    layout = layout_str(final_roles)
    clocks = "P={},D={},M={}".format(last_plan.get("f_P") or 2520, last_plan.get("f_D") or 2520,
                                     last_plan.get("f_M") or 2520)
    key = (ds, layout, clocks, last_plan.get("tau", 0))
    if key not in _cap_cache:
        try:
            # Keep CSV rebuilding usable on a small analysis host without the
            # optional numerical stack; the capacity column can remain empty.
            from pdblend.bench.matrix import layout_capacity

            _cap_cache[key] = layout_capacity(Path(spec_defaults["profile"]), Path(spec_defaults["corpus"]),
                                              ds, layout, clocks, key[3])
        except Exception:
            _cap_cache[key] = None
    cap = _cap_cache[key]
    return mean_rps / cap if cap else None


def window_bounds(outcomes, window_s):
    t0 = min((o["submitted_s"] for o in outcomes if o.get("submitted_s")), default=None)
    return (t0, t0 + window_s) if t0 and window_s else (None, None)


def series_stats(path, t0, t1):
    if result_path(path) is None:
        return None, None
    vals = []
    with open_result(path) as fh:
        for line in fh:
            t, v = json.loads(line)
            if (t0 is None or t >= t0) and (t1 is None or t < t1):
                vals.append(sum(v) / len(v))
    if not vals:
        return None, None
    vals.sort()
    return sum(vals) / len(vals), q(vals, 0.9)


def active_slot_share(ctl_path, t0, t1):
    if result_path(ctl_path) is None or t0 is None:
        return None
    with open_result(ctl_path) as fh:
        plans = [json.loads(l) for l in fh if '"kind": "plan"' in l]
    plans = [p for p in plans if p.get("kind") == "plan" and p.get("roles")]
    if not plans:
        return None
    plans.sort(key=lambda p: p["t"])
    busy = 0.0
    for i, p in enumerate(plans):
        a = max(p["t"], t0)
        b = min(plans[i + 1]["t"] if i + 1 < len(plans) else t1, t1)
        if b > a:
            n = sum(1 for role in p["roles"].values() if role in ACTIVE)
            busy += (b - a) * n / len(p["roles"])
    return busy / (t1 - t0) if t1 > t0 else None


def point_row(d: Path, root, spec_defaults) -> dict | None:
    try:
        s = json.loads((d / "summary.json").read_text())
        with open_result(d / "outcomes.jsonl") as fh:
            outcomes = [json.loads(l) for l in fh]
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    att = s["slo"]
    meta = s.get("trace_meta", {})
    ttft_lo, tpot_lo = s["slo_ttft_s"], s["slo_tpot_s"]

    ok = [o for o in outcomes if o.get("error") is None and o.get("first_token_s") is not None]
    ttfts = sorted(o["first_token_s"] - o["submitted_s"] for o in ok)
    tpots = sorted((o["finished_s"] - o["first_token_s"]) / max(o["completion_tokens"] - 1, 1)
                   for o in ok if o.get("finished_s") is not None and o["completion_tokens"] >= 2)
    passing = [o for o in ok
               if o["first_token_s"] - o["submitted_s"] <= ttft_lo
               and (o["completion_tokens"] < 2
                    or (o["finished_s"] - o["first_token_s"]) / max(o["completion_tokens"] - 1, 1) <= tpot_lo)]
    win_s = s.get("window_s") or 0
    out_tok = att["output_tokens"]
    t0, t1 = window_bounds(outcomes, win_s)

    ctl = d / "controller.jsonl"
    plans = []
    if result_path(ctl) is not None:
        with open_result(ctl) as fh:
            plans = [p for p in (json.loads(l) for l in fh if '"kind": "plan"' in l)
                     if p.get("kind") == "plan"]
    last_plan = plans[-1] if plans else None
    ev = s.get("controller", {}).get("events", {})
    per_gpu = [float(v) for v in s.get("per_gpu_mean_w", {}).values()]
    sm_mean, sm_p90 = series_stats(d / "util.jsonl", t0, t1)
    f_mean, _ = series_stats(d / "freq.jsonl", t0, t1)

    row = dict(
        name=d.name, dataset=meta.get("dataset"), scale=meta.get("scale"),
        rate=r(s["trace"].get("mean_rps"), 3), policy=s["policy"]["name"],
        offered=att["offered"], success_rate=r(att["success_rate"]), joint_slo_rate=r(att["joint_slo_rate"]),
        ttft_mean=r(sum(ttfts) / len(ttfts) if ttfts else None), ttft_p50=r(q(ttfts, 0.5)),
        ttft_p90=r(q(ttfts, 0.9)), ttft_p95=r(q(ttfts, 0.95)), ttft_p99=r(q(ttfts, 0.99)),
        tpot_mean=r(sum(tpots) / len(tpots) if tpots else None), tpot_p50=r(q(tpots, 0.5)),
        tpot_p90=r(q(tpots, 0.9)), tpot_p95=r(q(tpots, 0.95)), tpot_p99=r(q(tpots, 0.99)),
        goodput_req_s=r(len(passing) / win_s if win_s else None, 3),
        goodput_tok_s=r(sum(o["completion_tokens"] for o in passing) / win_s if win_s else None, 1),
        window_s=r(win_s, 2), window_energy_j=r(s["window_energy_j"], 1), energy_j=r(s["energy_j"], 1),
        energy_kwh=r(s["energy_j"] / 3.6e6, 6), mean_power_w=r(s["mean_power_w"], 1),
        window_mean_power_w=r(s["window_mean_power_w"], 1),
        gpu_w_mean=r(sum(per_gpu) / len(per_gpu) if per_gpu else None, 1),
        gpu_w_max=r(max(per_gpu) if per_gpu else None, 1),
        j_per_token=r(s["energy_j"] / out_tok if out_tok else None),
        window_j_per_token=r(s["window_energy_j"] / out_tok if out_tok else None),
        j_per_request=r(s["energy_j"] / max(att["succeeded"], 1), 2),
        sm_util_mean=r(sm_mean, 2), sm_util_p90=r(sm_p90, 2), freq_mean_mhz=r(f_mean, 0),
        active_slot_share=r(active_slot_share(ctl, t0, t1), 4),
        power_pct_tdp=r(s["mean_power_w"] / (len(per_gpu) * TDP_W) if per_gpu else None, 4),
        capacity_util=r(capacity_util(spec_defaults, meta.get("dataset"),
                                      s.get("final_roles"), last_plan, s["trace"].get("mean_rps")), 4),
        plans=ev.get("plan", 0), wakes=ev.get("wake", 0), parks=ev.get("park", 0),
        shield_events=len(s.get("controller", {}).get("shield_events", [])),
        final_layout=layout_str(s.get("final_roles", {})),
        jpt_check=r(s.get("j_per_token")),
    )
    return row


def add_verdicts(rows):
    groups: dict = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["scale"]), []).append(row)
    for (ds, sc), members in sorted(groups.items(), key=str):
        bases = [m for m in members if m["policy"] in BASELINES and m["j_per_token"] is not None]
        passing = [b for b in bases if (b["joint_slo_rate"] or 0) >= SLO_FLOOR]
        pool = passing or bases
        best = min(pool, key=lambda b: b["j_per_token"], default=None)
        for m in members:
            if best is None:
                m.update(best_baseline="", best_baseline_jpt="", delta_pct_vs_best="", verdict="")
                continue
            delta = (m["j_per_token"] - best["j_per_token"]) / best["j_per_token"] * 100
            m.update(best_baseline=best["policy"], best_baseline_jpt=best["j_per_token"],
                     delta_pct_vs_best=r(delta, 2))
            if m["policy"] == "pdblend":
                ok = (m["joint_slo_rate"] or 0) >= SLO_FLOOR and m["j_per_token"] <= best["j_per_token"]
                m["verdict"] = "PASS" if ok else "LOSE"
            elif m["policy"] in BASELINES:
                m["verdict"] = "slo_fail" if (m["joint_slo_rate"] or 0) < SLO_FLOOR else ""
            else:
                m["verdict"] = ""


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/v2/eval-7b-v2")
    spec = json.loads((root / "spec.json").read_text())
    rows = []
    for d in sorted(root.iterdir()):
        if d.is_dir():
            row = point_row(d, root, spec["defaults"])
            if row:
                rows.append(row)
    add_verdicts(rows)
    rows.sort(key=lambda r: (str(r["dataset"]), float(r["scale"] or 0),
                             POLICY_ORDER.get(r["policy"], 9), r["policy"]))
    out = root / "compare.csv"
    tmp = root / ".compare.csv.tmp"
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["name"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(out)
    losers = [r["name"] for r in rows if r["verdict"] == "LOSE"]
    print(f"wrote {out} rows={len(rows)} pdblend_pass={sum(1 for r in rows if r['verdict'] == 'PASS')} "
          f"lose={len(losers)} {losers}")


if __name__ == "__main__":
    main()
