#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""d2d_bandwidth.py — L20 卡间 D2D 拷贝带宽实测(P4 动机数据, 无需 pynvml)。

用法(容器内, --gpus all):
  python3 d2d_bandwidth.py --pairs 0-1,0-4,4-5 --gib 1 --iters 20
输出 CSV: pair, topology, gib, gbps, ms
"""
from __future__ import annotations

import argparse
import csv
import time

import torch


def bench_d2d(src: int, dst: int, nbytes: int, iters: int):
    """D2D 拷贝带宽: 在目标设备流上发拷贝并用该设备的 CUDA Event 计时。"""
    with torch.cuda.device(dst):
        x = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{src}")
        y = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{dst}")
        for _ in range(5):  # 预热
            y.copy_(x, non_blocking=True)
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(iters):
            y.copy_(x, non_blocking=True)
        ev1.record()
        torch.cuda.synchronize()
        dt_s = ev0.elapsed_time(ev1) / 1000.0
        gbps = (nbytes * iters) / dt_s / 1e9
        return gbps, dt_s / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="0-1,0-4,4-5")
    ap.add_argument("--gib", type=float, default=1.0, help="拷贝字节数 (GiB)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="d2d.csv")
    args = ap.parse_args()

    nbytes = int(args.gib * 1024**3)
    rows = []
    print(f"GPU count: {torch.cuda.device_count()}")
    for pair in args.pairs.split(","):
        s, d = (int(x) for x in pair.split("-"))
        gbps, ms = bench_d2d(s, d, nbytes, args.iters)
        print(f"{s}->{d}: {gbps:.1f} GB/s, {ms:.2f} ms / {args.gib} GiB")
        rows.append({"pair": pair, "gib": args.gib, "gbps": round(gbps, 1),
                     "ms": round(ms, 3)})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "gib", "gbps", "ms"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
