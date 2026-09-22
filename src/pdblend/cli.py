"""pdblend command line: gates, profiling, serving and benchmarks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="pdblend")
    sub = parser.add_subparsers(dest="command", required=True)

    g0 = sub.add_parser("gate-kv", help="G0: cross-GPU KV transfer on two instances")
    g0.add_argument("--model", default="Qwen2.5-7B-Instruct")
    g0.add_argument("--gpus", default="0,1")
    g0.add_argument("--connector", default="P2pNcclConnector", choices=["NixlConnector", "P2pNcclConnector"])
    g0.add_argument("--lengths", default="512,2048,7168")
    g0.add_argument("--repeats", type=int, default=3)
    g0.add_argument("--out", type=Path, default=Path("results/v2/gate-g0.json"))

    g1 = sub.add_parser("gate-park", help="G1: parking state power and wake latency")
    g1.add_argument("--model", default="Qwen2.5-7B-Instruct")
    g1.add_argument("--gpu", type=int, default=0)
    g1.add_argument("--window", type=float, default=8.0)
    g1.add_argument("--out", type=Path, default=Path("results/v2/gate-g1.json"))

    pr = sub.add_parser("profile", help="fit the affine perf/power model from a ~150-point grid")
    pr.add_argument("--model", default="Qwen2.5-7B-Instruct")
    pr.add_argument("--gpus", default="0,1", help="two instances: the first is profiled, the second is the KV transfer peer")
    pr.add_argument("--tp", type=int, default=1)
    pr.add_argument("--freqs", default=",".join(map(str, (900, 1200, 1500, 1800, 2100, 2520))))
    pr.add_argument("--sections", default="prefill,decode,mixed,static,transfer")
    pr.add_argument("--window", type=float, default=2.0)
    pr.add_argument("--out", type=Path, default=Path("results/v2/profile"))
    pr.add_argument("--resume", action="store_true", help="skip sections already present in <out>/raw.json")
    pr.add_argument("--decode-repeats", type=int, default=3)
    pr.add_argument("--decode-settle", type=float, default=2.0)
    pr.add_argument("--decode-measure", type=float, default=5.0)
    pr.add_argument("--mixed-freqs", default="1500,2100,2520")
    pr.add_argument("--base-port", type=int, default=8100)

    be = sub.add_parser("bench", help="run one benchmark point: fleet + proxy + controller + open-loop load")
    be.add_argument("--model", default="Qwen2.5-7B-Instruct")
    be.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    be.add_argument("--tp", type=int, default=1)
    be.add_argument("--policy", default="pdblend")
    be.add_argument("--profile", type=Path, required=True)
    be.add_argument("--corpus", type=Path, default=Path("datasets/prepared/2026-09-13-7b-v1"))
    be.add_argument("--dataset", default="sharegpt", choices=["alpaca", "sharegpt", "longbench"])
    be.add_argument("--split", default="evaluation")
    be.add_argument("--rate", type=float, default=5.0, help="Poisson rate (or base rate for --stages)")
    be.add_argument("--duration", type=float, default=100.0)
    be.add_argument("--stages", default="", help="staged Gamma trace: 'dur:scale,dur:scale,...'")
    be.add_argument("--cv", type=float, default=1.5)
    be.add_argument("--azure", default="", choices=["", "conv", "code"], help="replay an Azure 2024 window")
    be.add_argument("--azure-offset", type=float, default=0.0)
    be.add_argument("--azure-peak", type=float, default=5.0)
    be.add_argument("--seed", type=int, default=701)
    be.add_argument("--scale", type=float, default=None, help="label: rate as a multiple of the frozen capacity")
    be.add_argument("--connector", default="P2pNcclConnector")
    be.add_argument("--period", type=float, default=10.0)
    be.add_argument("--layout", default="", help="manual policy: 'M=4' or 'P=1,D=3,off=4'")
    be.add_argument("--clocks", default="P=2520,D=2520,M=2520", help="manual policy clocks per role")
    be.add_argument("--tau", type=int, default=0)
    be.add_argument("--out", type=Path, required=True)

    mx = sub.add_parser("matrix", help="run every point of a JSON matrix spec, skipping finished ones")
    mx.add_argument("spec", type=Path)
    mx.add_argument("--only", default="", help="substring filter on point names")
    mx.add_argument("--dry", action="store_true")
    mx.add_argument("--shard", default="", help="i/n: run points[i::n] so runners can share a spec on disjoint GPUs")
    mx.add_argument("--gpus", default="", help="override the spec's default GPU list for this runner")

    m3 = sub.add_parser("m3-spec", help="write the M3 co-located vs disaggregated matrix spec")
    m3.add_argument("--profile", type=Path, required=True)
    m3.add_argument("--gpus", default="4,5,6,7")
    m3.add_argument("--root", type=Path, default=Path("results/v2/m3"))
    m3.add_argument("--duration", type=float, default=120.0)

    ev = sub.add_parser("eval-spec", help="write the P4 evaluation matrix spec (controlled, ported, ablations, traces)")
    ev.add_argument("--profile", type=Path, required=True)
    ev.add_argument("--model", default="Qwen2.5-7B-Instruct")
    ev.add_argument("--tp", type=int, default=1)
    ev.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ev.add_argument("--corpus", type=Path, default=Path("datasets/prepared/2026-09-13-7b-v1"))
    ev.add_argument("--root", type=Path, default=Path("results/v2/eval-7b"))
    ev.add_argument("--duration", type=float, default=300.0)
    ev.add_argument("--scales", default="0.25,0.5,0.75")
    ev.add_argument("--stages", default="300:0.25,300:0.75,300:0.5,300:1.0", help="'' disables the staged group")
    ev.add_argument("--azure", default="conv,code", help="'' disables the Azure group")
    ev.add_argument("--azure-duration", type=float, default=1800.0)

    cp = sub.add_parser("capacity", help="max Poisson rate the planner model admits for a fixed layout")
    cp.add_argument("--profile", type=Path, required=True)
    cp.add_argument("--corpus", type=Path, default=Path("datasets/prepared/2026-09-13-7b-v1"))
    cp.add_argument("--dataset", default="sharegpt", choices=["alpaca", "sharegpt", "longbench"])
    cp.add_argument("--layout", default="M=4")
    cp.add_argument("--clocks", default="P=2520,D=2520,M=2520")
    cp.add_argument("--tau", type=int, default=0)

    mv = sub.add_parser("motivation", help="M1/M2 figures from a profile raw.json and/or the M3 crossover heat-map")
    mv.add_argument("--raw", type=Path, default=None, help="profile raw.json for M1/M2")
    mv.add_argument("--m3-root", type=Path, default=None, help="matrix root holding the M3 points")
    mv.add_argument("--out", type=Path, default=Path("results/v2/motivation"))

    rp = sub.add_parser("report", help="aggregate summary.json files under a results root")
    rp.add_argument("root", type=Path)
    rp.add_argument("--reference", default="static_best")
    rp.add_argument("--out", type=Path, default=None)

    g5 = sub.add_parser("gate-g5", help="G5: baseline ports vs frozen pdblend_baselines, decision-by-decision on CPU")
    g5.add_argument("--profile", type=Path, required=True)
    g5.add_argument("--dataset", default="sharegpt", choices=["alpaca", "sharegpt", "longbench"])
    g5.add_argument("--trials", type=int, default=300)
    g5.add_argument("--seed", type=int, default=7)
    g5.add_argument("--out", type=Path, default=Path("results/v2/gate-g5.json"))

    cm = sub.add_parser("check-model", help="planner admission vs measured SLO over a finished manual-layout matrix")
    cm.add_argument("root", type=Path)
    cm.add_argument("--profile", type=Path, required=True)
    cm.add_argument("--corpus", type=Path, default=Path("datasets/prepared/2026-09-13-7b-v1"))
    cm.add_argument("--slo-ok", type=float, default=0.9, help="measured joint SLO rate counted as 'met'")
    cm.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "check-model":
        from .bench.check_model import check_model, format_rows
        r = check_model(args.profile, args.root, args.corpus, args.slo_ok, args.out)
        print(format_rows(r))
        print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=1))
    elif args.command == "gate-g5":
        from .bench.gate_g5 import gate_g5
        r = gate_g5(args.profile, args.out, args.trials, args.seed, args.dataset)
        print(json.dumps({k: {kk: vv for kk, vv in r[k].items() if kk != "mismatches"}
                          for k in ("eco_admission", "eco_scaling", "dynamo_scale_freq")}, indent=1))
    elif args.command == "report":
        from .bench.report import report
        result = report(args.root, args.reference, args.out)
        print(json.dumps(result["summary"], indent=1, default=str))
    elif args.command == "bench":
        from .bench.matrix import bench_point, brief
        print(json.dumps(brief(bench_point(vars(args))), indent=1))
    elif args.command == "matrix":
        from .bench.matrix import run_matrix
        for row in run_matrix(args.spec, args.only, args.dry, args.shard, args.gpus):
            print(json.dumps(row, default=str))
    elif args.command == "motivation":
        from .bench import motivation as mv
        if args.raw:
            r = mv.plot_m1_m2(args.raw, args.out)
            print(json.dumps(dict(m1=r["m1"], m2=r["m2"]), indent=1))
        if args.m3_root:
            print(json.dumps(mv.plot_m3(args.m3_root, args.out)["winners"], indent=1))
    elif args.command == "m3-spec":
        from .bench.matrix import m3_spec
        spec = m3_spec(args.profile, args.gpus, args.root, duration=args.duration)
        print(json.dumps(dict(capacity_rps=spec["capacity_rps"], points=len(spec["points"]), root=spec["root"])))
    elif args.command == "eval-spec":
        from .bench.matrix import eval_spec
        spec = eval_spec(args.profile, args.gpus, args.root, corpus=args.corpus, model=args.model, tp=args.tp,
                         scales=tuple(float(s) for s in args.scales.split(",")), duration=args.duration,
                         stages=args.stages, azure=tuple(a for a in args.azure.split(",") if a),
                         azure_duration=args.azure_duration)
        print(json.dumps(dict(capacity_rps=spec["capacity_rps"], groups=spec["groups"], points=len(spec["points"]),
                              root=spec["root"])))
    elif args.command == "capacity":
        from .bench.matrix import layout_capacity
        print(json.dumps(dict(dataset=args.dataset, layout=args.layout, clocks=args.clocks, tau=args.tau,
                              capacity_rps=layout_capacity(args.profile, args.corpus, args.dataset, args.layout,
                                                           args.clocks, args.tau))))
    elif args.command == "profile":
        from .profile.profiler import Profiler
        prof = Profiler(args.model, [int(g) for g in args.gpus.split(",")], tp=args.tp,
                        freqs=[int(f) for f in args.freqs.split(",")], window_s=args.window, out_dir=args.out,
                        decode_repeats=args.decode_repeats, decode_settle_s=args.decode_settle,
                        decode_measure_s=args.decode_measure,
                        mixed_freqs=tuple(int(f) for f in args.mixed_freqs.split(",")),
                        base_port=args.base_port)
        sections = tuple(args.sections.split(","))
        if args.resume:
            have = prof.resume()
            sections = tuple(s for s in sections if s not in have)
            print(f"resuming: skipping {have}, running {sections}", flush=True)
        prof.run(sections)
    elif args.command == "gate-kv":
        from .bench.gates import gate_kv
        result = gate_kv(args.model, tuple(int(g) for g in args.gpus.split(",")), args.connector,
                         tuple(int(n) for n in args.lengths.split(",")), args.repeats, out=args.out)
        print(json.dumps(result["summary"], indent=1))
    elif args.command == "gate-park":
        from .bench.gates import gate_park
        result = gate_park(args.model, args.gpu, args.window, out=args.out)
        for s in result["states"]:
            print(s["state"], {k: v for k, v in s.items() if k in ("mean_power_w", "freq_mhz", "sleep_s", "wake_s", "start_to_ready_s")})


if __name__ == "__main__":
    main()
