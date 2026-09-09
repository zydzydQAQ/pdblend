#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_trace.py — 生成 Poisson / Gamma 到达过程 trace(复用 DistServe 源码逻辑)。

来源(原样提取,算法与参数语义一致):
  - DistServe simdistserve/base/workload.py:
      get_poisson_interarrival / get_gamma_interarrival
      (shape = 1/cv^2, scale = cv^2/rate, x1000 → 毫秒; poisson 即 cv=1)
  - DistServe evaluation/2-benchmark-serving/2-benchmark-serving.py::get_request:
      同一 gamma 采样用作秒级 asyncio.sleep 间隔;请求 i 的绝对到达时刻
      arrival(i) = Σ_{j<i} interval(j),第一个请求 t=0。
  - 请求采样:同 2-benchmark-serving.py::sample_requests(random.sample(.ds 请求, n))。

用法(容器内或宿主机均可,structs.py 与本脚本同目录):
  python3 make_trace.py --ds datasets/sharegpt.ds --n 100 --process poisson \
      --rate 2 --seed 0 --out traces/sharegpt_poisson_r2.json
  python3 make_trace.py --ds datasets/sharegpt.ds --n 100 --process gamma \
      --rate 2 --cv 2 --seed 0 --out traces/sharegpt_gamma_r2_cv2.json
"""
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structs import Dataset  # noqa: E402  (DistServe 原码)


def sample_requests(dataset_path: str, num_prompts: int, seed: int):
    """DistServe 2-benchmark-serving.py::sample_requests 同款(random.sample)。"""
    rng = random.Random(seed)
    dataset = Dataset.load(dataset_path)
    if num_prompts > len(dataset.reqs):
        raise ValueError(
            f"n={num_prompts} 大于数据集大小 ({len(dataset.reqs)}), 请调小 --n")
    return rng.sample(dataset.reqs, num_prompts)


def gamma_intervals(n: int, rate: float, cv: float, seed: int):
    """DistServe get_gamma_interarrival 原样:gamma(shape=1/cv^2, scale=cv^2/rate)。
    返回 n 个间隔(秒;第 0 个请求在 t=0,故 intervals 从下标 0 起逐请求消耗)。"""
    if seed is not None:
        np.random.seed(seed)
    shape = 1.0 / (cv * cv)
    scale = cv * cv / rate
    return list(np.random.gamma(shape, scale, size=n))


def diurnal_arrivals(rate_lo: float, rate_hi: float, cycle_s: float,
                     duration_s: float, seed: int):
    """日巡负载:非齐次 Poisson,rate(t) 在 [lo, hi] 间正弦变化(t=0 为谷)。

    Ogata 稀疏化:按 hi 齐次采样,以 rate(t)/hi 概率接受。
    """
    import math
    rng = np.random.default_rng(seed)
    t, out = 0.0, []
    while True:
        t += rng.exponential(1.0 / rate_hi)
        if t >= duration_s:
            break
        r = rate_lo + (rate_hi - rate_lo) * 0.5 * (
            1.0 - math.cos(2.0 * math.pi * t / cycle_s))
        if rng.random() < r / rate_hi:
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser(description="生成 Poisson/Gamma 到达 trace(对齐 DistServe)")
    ap.add_argument("--ds", required=True, help="0-prepare-dataset.py 生成的 .ds 文件")
    ap.add_argument("--n", type=int, default=100, help="采样请求数")
    ap.add_argument("--process", choices=["poisson", "gamma", "uniform"], default="poisson")
    ap.add_argument("--rate", type=float, default=2.0, help="请求到达率 (req/s)")
    ap.add_argument("--cv", type=float, default=1.0,
                    help="gamma 变异系数 (poisson 恒为 1;uniform 忽略)")
    ap.add_argument("--profile", choices=["constant", "diurnal"],
                    default="constant",
                    help="diurnal:非齐次 Poisson 日巡(--rate-lo/hi/cycle/duration)")
    ap.add_argument("--rate-lo", type=float, default=0.0, help="日巡谷速率")
    ap.add_argument("--rate-hi", type=float, default=0.0, help="日巡峰速率")
    ap.add_argument("--cycle-s", type=float, default=360.0, help="日巡周期(s)")
    ap.add_argument("--duration-s", type=float, default=720.0,
                    help="日巡总时长(s,默认 2 周期)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.profile == "diurnal":
        if args.rate_lo <= 0 or args.rate_hi <= args.rate_lo:
            raise SystemExit("--profile diurnal 需 0 < rate-lo < rate-hi")
        arr = diurnal_arrivals(args.rate_lo, args.rate_hi, args.cycle_s,
                               args.duration_s, args.seed)
        if not arr:
            raise SystemExit("diurnal 未产生任何到达,检查参数")
        args.n = len(arr)
        mean_rate = args.n / args.duration_s
        reqs = sample_requests(args.ds, args.n, args.seed)
        arrivals = arr
        cv = None
        meta_extra = dict(profile="diurnal", rate_lo=args.rate_lo,
                          rate_hi=args.rate_hi, cycle_s=args.cycle_s,
                          duration_s=args.duration_s)
        eff_rate = round(mean_rate, 4)
    else:
        if args.process == "poisson":
            cv = 1.0
        elif args.process == "gamma":
            cv = args.cv
        else:
            cv = None  # uniform: 固定间隔 1/rate

        reqs = sample_requests(args.ds, args.n, args.seed)
        if cv is not None:
            intervals = gamma_intervals(args.n, args.rate, cv, args.seed)
        else:
            intervals = [1.0 / args.rate] * args.n

        arrivals = [0.0]
        for itv in intervals[:-1]:  # 最后一个间隔无人消费(与 DistServe 循环一致)
            arrivals.append(arrivals[-1] + itv)
        meta_extra = dict(profile="constant")
        eff_rate = args.rate

    trace = {
        "meta": {
            "dataset": os.path.basename(args.ds).replace(".ds", ""),
            "ds": os.path.abspath(args.ds),
            "process": args.process,
            "rate": eff_rate,
            "cv": cv,
            "seed": args.seed,
            "n": args.n,
            "unit": "s",
            "source": "DistServe: workload.py::get_gamma_interarrival + "
                      "2-benchmark-serving.py::get_request",
            **meta_extra,
        },
        "requests": [
            {"idx": i, "arrival_s": round(arrivals[i], 6),
             "prompt_len": r.prompt_len, "output_len": r.output_len}
            for i, r in enumerate(reqs)
        ],
        "prompts": [r.prompt for r in reqs],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    if args.profile == "diurnal":
        print(f"[trace] diurnal lo={args.rate_lo} hi={args.rate_hi} "
              f"cycle={args.cycle_s}s dur={args.duration_s}s "
              f"n={args.n} mean_rate={eff_rate} → {args.out}")
    else:
        print(f"[trace] {args.process} rate={args.rate} cv={cv} "
              f"n={args.n} → {args.out}")
        print(f"[trace] 首请求 t=0; 平均间隔 {np.mean(intervals):.3f}s "
              f"(理论 1/rate={1/args.rate:.3f}s)")


if __name__ == "__main__":
    main()
