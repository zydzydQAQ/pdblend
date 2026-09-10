#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""power_record.py — 独立功率采样驱动(复用 pdblend.motivations.common 的 PowerSampler)。

用于与 bench_vllm.py 并行采集:bench 测时延/SLO,本脚本积分 GPU 能耗。
用法(容器内):
  python3 power_record.py --gpus 0,1,2,3 --seconds 300 --out power.csv \
      --freq 1500            # 可选: 采样前同步锁频, 结束后解锁
产出 power.csv: 列 [t_s, gpu0_w, gpu1_w, ...]; stdout 打印 total_energy_j / mean_w。
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
import time

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, "/workspace")  # 容器内挂载路径
sys.path.insert(0, _ROOT)  # 宿主机: /root/workspace
sys.path.insert(0, os.path.join(_ROOT, "pdblend", "src"))  # ecopadg
from pdblend.motivations.common.backends import get_backend  # noqa: E402
from pdblend.motivations.common.gpu_utils import (  # noqa: E402
    PowerSampler, reset_all_gpus, set_all_gpus_clock)


def main():
    ap = argparse.ArgumentParser(description="GPU 功率采样(20ms 间隔, 梯形积分)")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--interval", type=float, default=0.02)
    ap.add_argument("--out", default="power.csv")
    ap.add_argument("--freq", type=int, default=0, help=">0 时采样前同步锁频(需 SYS_ADMIN)")
    ap.add_argument("--backend", default="pynvml")
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",")]
    backend = get_backend(args.backend)
    if args.freq > 0:
        set_all_gpus_clock(gpus, args.freq, backend)
        print(f"[power] 已锁频 {args.freq} MHz, settle 1.5s ...")
        time.sleep(1.5)

    sampler = PowerSampler(gpus=gpus, interval=args.interval, backend=backend)
    sampler.start()

    def _sigterm(signum, frame):
        raise KeyboardInterrupt  # 走 finally 落盘(SIGTERM 默认会跳过 finally)

    signal.signal(signal.SIGTERM, _sigterm)
    try:
        time.sleep(args.seconds)
    finally:
        sampler.stop()
        if args.freq > 0:
            reset_all_gpus(gpus, backend)
        # 落盘放在 finally:SIGTERM(KeyboardInterrupt)也保证写出
        samples = list(getattr(sampler, "samples", None) or getattr(sampler, "rows", []))
        util = dict(sampler.utilization_samples)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s"] + [f"gpu{i}_w" for i in gpus]
                       + [f"gpu{i}_util_pct" for i in gpus])
            for t, ws in samples:
                w.writerow([float(t)] + [float(x) for x in ws]
                           + util.get(t, [float("nan")] * len(gpus)))
        if sampler.error:
            raise SystemExit("[power] sampling failed: " + sampler.error)
        if len(samples) < 2:
            raise SystemExit("[power] too few samples: %d" % len(samples))
        energy = sampler.total_energy_j()
        mean = sampler.mean_power_w()
        print(f"[power] {args.seconds}s × {len(gpus)} GPU: "
              f"total_energy={energy:.1f} J, mean_power={mean:.1f} W, "
              f"rows={len(samples)}")
        print(f"[power] saved: {args.out}")


if __name__ == "__main__":
    main()
