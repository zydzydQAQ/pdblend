# -*- coding: utf-8 -*-
"""placement:DistServe max-rate 仿真(仅 --mode capacity-envelope)。

选配默认目标在 planner.choose_iso_load(min J s.t. SLO 非劣)。
实验启动只起 mixed;空间 PD 由线上控制面决定,不在这里 inherit。
本模块禁止当作默认优化入口。

源码参照(DistServe,Apache-2.0):
  simdistserve/benchmarks/search_configs.py::get_distserve_configs
    配置空间 (pp_cross, tp_prefill, pp_prefill, tp_decode, pp_decode);
  simdistserve/benchmarks/parallel_bisect.py::run_binary_search
    对每个配置二分搜索 per-GPU rate(P90 TTFT/P90 TPOT containment 约束);
  simdistserve/simulate.py::find_best_config
    最优 per-GPU rate,GPU 数少者优先;
  simdistserve/estimators/time_estimator.py::get_prefill_time / get_decode_time
    成本公式:prefill = (a + b*Σtok + c*Σtok²)/pp + pp;
              decode = (a + b*Σgen + c*bs)/pp。

与上游的差异:仿真器为移植的简化事件仿真(prefill FCFS token 批 + 桥接 +
continuous batching decode + 逐请求 KV 传输时延),成本模型完全复用上游公式,
画像 JSON 由本机 OpModel 导出(同构)。
"""
from __future__ import annotations

import heapq
import json
import random
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def load_profile_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _params(profile: dict, model: str, tp: int) -> dict:
    return profile[model][str(tp)]


def prefill_time_ms(profile: dict, model: str, tp: int, pp: int,
                    tokens_list: Sequence[int]) -> float:
    """DistServe get_prefill_time 原公式。"""
    a, b, c = _params(profile, model, tp)["prefill"]
    num_total = sum(int(x) for x in tokens_list)
    sq = sum(int(x) ** 2 for x in tokens_list)
    delay = a + b * num_total + c * sq
    return delay / pp + pp  # pp_factor=1/pp, pp_const=pp(上游原样)


def decode_time_ms(profile: dict, model: str, tp: int, pp: int,
                   batch_size: int,
                   token_generated_list: Optional[Sequence[int]] = None) -> float:
    """DistServe get_decode_time 原公式(pp_const=0)。"""
    if token_generated_list is None:
        token_generated_list = [1] * int(batch_size)
    params = _params(profile, model, tp)
    thr = params["decoding_large_small_bs_threshold"]
    a, b, c = (params["decoding_smallbs"] if batch_size < thr
               else params["decoding_largebs"])
    delay = a + b * sum(int(x) for x in token_generated_list) + c * int(batch_size)
    return delay / pp


def distserve_configs(tps: Sequence[int], pps: Sequence[int], total_gpus: int,
                      pp_cross_options: Sequence[int] = (1,)) -> List[Tuple[int, int, int, int, int]]:
    """get_distserve_configs 同款过滤(单节点版)。"""
    tps = [int(t) for t in tps]
    pps = [int(p) for p in pps]
    cfgs: List[Tuple[int, int, int, int, int]] = []
    for pp_cross in pp_cross_options:
        for tp_p, pp_p, tp_d, pp_d in product(tps, pps, tps, pps):
            if pp_cross * (tp_p * pp_p + tp_d * pp_d) > total_gpus:
                continue
            if pp_cross * pp_p not in pps:
                continue
            if pp_cross * pp_d not in pps:
                continue
            cfgs.append((int(pp_cross), tp_p, pp_p, tp_d, pp_d))
    return cfgs


def _num_gpus(config: Tuple[int, int, int, int, int]) -> int:
    pp_cross, tp_p, pp_p, tp_d, pp_d = config
    return pp_cross * (tp_p * pp_p + tp_d * pp_d)

class Simulator:
    """DS-PD 段(1P1D)的简化事件仿真:prefill FCFS token 批 + 桥接 +
    continuous batching decode + 逐请求 KV 传输时延。

    确定性:到达用 Poisson(rng seed),长度按 workload 顺序循环取用。
    """

    def __init__(self, profile: dict, model: str,
                 workload_lengths: Sequence[Tuple[int, int]],
                 tp_p: int, pp_p: int, tp_d: int, pp_d: int,
                 transfer_ms_per_kv_token: float = 0.0, seed: int = 0,
                 max_tokens_per_batch: int = 8192,
                 max_batch_size: int = 128, max_decode_batch: int = 1024,
                 max_sim_s: float = 7200.0):
        self.profile = profile
        self.model = model
        self.lengths = list(workload_lengths)
        self.tp_p, self.pp_p, self.tp_d, self.pp_d = tp_p, pp_p, tp_d, pp_d
        self.kv_ms = float(transfer_ms_per_kv_token)
        self.seed = seed
        self.max_tokens = max_tokens_per_batch
        self.max_bs = max_batch_size
        self.max_dbs = max_decode_batch
        self.max_sim_s = max_sim_s

    def _arrivals(self, rate: float, n: int) -> List[float]:
        if rate <= 0:
            return [0.0] + [float("inf")] * (n - 1)
        rng = random.Random(self.seed)
        out = [0.0]
        t = 0.0
        for _ in range(n - 1):
            t += rng.expovariate(rate)
            out.append(t)
        return out

    def _prefill_batch(self, queue: List[dict]) -> Tuple[List[dict], float]:
        """FCFS 贪心组批(两个上限:batch 数与 token 数,对齐 DistServe 默认)。"""
        batch: List[dict] = []
        toks = 0
        while queue and len(batch) < self.max_bs:
            r = queue[0]
            if toks + r["plen"] > self.max_tokens and batch:
                break
            batch.append(queue.pop(0))
            toks += r["plen"]
        dt = prefill_time_ms(self.profile, self.model, self.tp_p, self.pp_p,
                             [r["plen"] for r in batch])
        return batch, dt

    def run(self, rate: float, N: int, ttft_target_ms: float,
            tpot_target_ms: float, trace=None) -> dict:
        """trace 可选:[(arrival_s, plen, olen), ...] —— 提供时按外部 trace 仿真
        (对齐实测验证);否则 Poisson 生成。"""
        if trace is not None:
            arrivals = [float(x[0]) for x in trace]
            self._trace_lengths = [(int(x[1]), int(x[2])) for x in trace]
            N = len(trace)
        else:
            arrivals = self._arrivals(rate, N)
            self._trace_lengths = None
        # 事件堆:(t, i, kind, payload);kind: arrival / pf_done / iter
        events: List[Tuple[float, int, str, object]] = []
        for i in range(N):
            if arrivals[i] == arrivals[i]:
                heapq.heappush(events, (arrivals[i], i, "arrival", i))
        pf_queue: List[dict] = []
        pf_busy_until = -1.0
        bridge: List[Tuple[float, dict]] = []   # (transfer_done_ms, req)
        active: List[dict] = []
        first: Dict[int, float] = {}
        last: Dict[int, float] = {}
        count: Dict[int, int] = {}
        results: Dict[int, dict] = {}
        t = 0.0
        iter_queued = False
        while events and len(results) < N and t < self.max_sim_s:
            t, _, kind, payload = heapq.heappop(events)
            if kind == "arrival":
                i = payload
                if self._trace_lengths is not None:
                    plen, olen = self._trace_lengths[i]
                else:
                    plen, olen = self.lengths[i % len(self.lengths)]
                pf_queue.append(dict(rid=i, plen=plen, olen=olen, arr=t))
                if t >= pf_busy_until and pf_queue:
                    batch, dt = self._prefill_batch(pf_queue)
                    pf_busy_until = t + dt / 1000.0  # dt 为 ms,时间轴为 s
                    heapq.heappush(events, (pf_busy_until, 0, "pf_done", batch))
            elif kind == "pf_done":
                for r in payload:
                    bridge.append((t + r["plen"] * self.kv_ms / 1000.0, r))
                if pf_queue:
                    batch, dt = self._prefill_batch(pf_queue)
                    pf_busy_until = t + dt / 1000.0
                    heapq.heappush(events, (pf_busy_until, 0, "pf_done", batch))
            elif kind == "iter":
                iter_queued = False
                for r in list(active):
                    last[r["rid"]] = t
                    count[r["rid"]] = count.get(r["rid"], 0) + 1
                    if count[r["rid"]] >= r["olen"]:
                        active.remove(r)
                        f = first.get(r["rid"], t)
                        l = last[r["rid"]]
                        tpot = (l - f) / (r["olen"] - 1) if r["olen"] > 1 else 0.0
                        results[r["rid"]] = dict(
                            ttft_ms=max((f - r["arr"]) * 1000.0, 0.0),
                            tpot_ms=tpot * 1000.0)
                ready = [x for x in bridge if x[0] <= t]
                bridge = [x for x in bridge if x[0] > t]
                for _, r in ready:
                    if len(active) < self.max_dbs:
                        active.append(r)
                        first[r["rid"]] = t  # 下个迭代完成首 token
                    else:
                        bridge.insert(0, (t, r))
            if (active or bridge) and not iter_queued:
                iter_queued = True
                bs = max(len(active), 1)
                heapq.heappush(events, (t + decode_time_ms(
                    self.profile, self.model, self.tp_d, self.pp_d, bs) / 1000.0,
                    0, "iter", None))
        ttfts = [r["ttft_ms"] for r in results.values()]
        tpots = [r["tpot_ms"] for r in results.values()]
        p50_ttft = float(np.percentile(ttfts, 50)) if ttfts else float("nan")
        p90_ttft = float(np.percentile(ttfts, 90)) if ttfts else float("nan")
        p90_tpot = float(np.percentile(tpots, 90)) if tpots else float("nan")
        containment = (len(results) == N and p90_ttft <= ttft_target_ms
                       and p90_tpot <= tpot_target_ms)
        return dict(completed=len(results), N=N,
                    p50_ttft_ms=p50_ttft, p90_ttft_ms=p90_ttft,
                    p90_tpot_ms=p90_tpot,
                    ttft_target_ms=ttft_target_ms,
                    tpot_target_ms=tpot_target_ms,
                    containment=containment)


def bisect_rate(profile: dict, model: str, workload_lengths,
                tp_p: int, pp_p: int, tp_d: int, pp_d: int,
                transfer_ms_per_kv_token: float, seed: int, N: int,
                ttft_target_ms: float, tpot_target_ms: float,
                max_per_gpu_rate: float = 5.0, esp: float = 0.25) -> float:
    """run_binary_search 同款:对 per-GPU rate 二分,返回最大可达率。

    num_gpu = pp_cross*(tp_p*pp_p + tp_d*pp_d);rate_total = per_gpu × num_gpu。
    """
    num_gpu = _num_gpus((1, tp_p, pp_p, tp_d, pp_d))
    sim = Simulator(profile=profile, model=model,
                    workload_lengths=workload_lengths,
                    tp_p=tp_p, pp_p=pp_p, tp_d=tp_d, pp_d=pp_d,
                    transfer_ms_per_kv_token=transfer_ms_per_kv_token,
                    seed=seed)

    def feasible(per_gpu: float) -> bool:
        if per_gpu <= 1e-9:
            return True  # 空载平凡可行
        r = sim.run(rate=per_gpu * num_gpu, N=N,
                    ttft_target_ms=ttft_target_ms,
                    tpot_target_ms=tpot_target_ms)
        return bool(r["containment"])

    # 注意:high 不可行时不能直接返 0——真实可达率可能远低于上界,
    # 二分区间自然收敛到可行边界(DistServe 原逻辑同样在 [0, max] 上二分)。
    low, high = 0.0, float(max_per_gpu_rate)
    while high - low > esp:
        mid = (low + high) / 2.0
        if feasible(mid):
            low = mid
        else:
            high = mid
    return low


def find_best_config(results: Dict[Tuple[int, int, int, int, int], float]) -> Tuple[int, int, int, int, int]:
    """find_best_config 同款:最优 per-GPU rate;平局取 GPU 数最少。"""
    if not results:
        raise ValueError("results 不能为空")
    best_rate = max(results.values())
    cands = [c for c, r in results.items() if r == best_rate]
    return min(cands, key=_num_gpus)
