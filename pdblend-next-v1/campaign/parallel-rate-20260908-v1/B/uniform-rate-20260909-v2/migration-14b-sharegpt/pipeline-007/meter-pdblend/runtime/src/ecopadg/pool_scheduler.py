# -*- coding: utf-8 -*-
"""pool_scheduler:P/D 池在线调度器 + 可选 selective 路由。

SelectiveRouter 只在 mixed 与 PD 同时启动时生效(静态 A9)。
有 OpModel 时走 path_gross_j;无模型回退 mean-pin 阈值(不是写死 512)。
默认 EcoSPD 单池,路由整段倒入该池。

批处理统一约定(与 runner/ds_proxy 对齐):
  - prefill 池:默认 FCFS + token 预算派发(DistServe ContextStageFCFSScheduler
    语义;实例内 vLLM 按 token 批执行),池忙时锁 prefill_freq(高频:
    能效+TTFT 双优,E1b/M1-C),空闲降至 idle_freq;
    可选 pack / sjf 只作控制面消融,不改默认;
  - decode 池:continuous batching(vLLM 原生);准入沿用 DistServe
    waiting_block_prop_threshold 桥接门控(ds_bridge.should_accept);
    逐实例 slack 选频(select_decode_freq,E2 slack 预算);
    可选 hold 撑批,不改默认;
  - mixed 池:sarathi 混批 + M̃1 窗口(online_scheduler.PaDGWindowScheduler,
    本模块不重复实现;不是严格 PaDG)。

全部纯逻辑,实例负载/名单由调用方(控制面)注入。
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from ecopadg.ds_bridge import blocks_needed, should_accept
from ecopadg.online_scheduler import select_decode_freq
from ecopadg.types import (
    B_REF, CTX_REF, DECODE_B_CAP, KV_XFER_S_PER_TOK, OUT_REF, RHO_TPUT,
    TOKEN_BUDGET,
)

PD_SCHED_POLICIES = ("fcfs", "pack", "sjf", "hold")
DEFAULT_PACK_STARVE_S = 0.2
DEFAULT_SJF_STARVE_S = 0.5
DEFAULT_HOLD_MS = 20.0
DEFAULT_HOLD_B = B_REF


def normalize_pd_sched_policy(value: object) -> str:
    policy = str(value or "fcfs").strip().lower()
    if policy not in PD_SCHED_POLICIES:
        raise ValueError("unknown pd sched policy: %r" % value)
    return policy


def pd_sched_ctor_kwargs(args=None, **overrides) -> dict:
    """Controller / tests 共用的 policy 构造参数。"""
    def _get(name: str, default):
        if name in overrides:
            return overrides[name]
        if args is None:
            return default
        return getattr(args, name, default)

    return dict(
        policy=normalize_pd_sched_policy(_get("pd_sched_policy", "fcfs")),
        pack_starve_s=float(_get("pack_starve_s", DEFAULT_PACK_STARVE_S)),
        sjf_starve_s=float(_get("sjf_starve_s", DEFAULT_SJF_STARVE_S)),
        hold_ms=float(_get("decode_hold_ms", DEFAULT_HOLD_MS)),
        hold_b=int(_get("decode_hold_b", DEFAULT_HOLD_B)),
    )


@dataclass
class _PrefillQueued:
    rid: int
    prompt_len: int
    arrival_s: float


@dataclass
class _BridgeQueued:
    rid: int
    prompt_len: int
    output_len: int
    arrival_s: float


@dataclass
class PrefillSchedStats:
    dispatched: int = 0
    hol_skips: int = 0
    dispatch_token_sum: int = 0
    dispatch_waves: int = 0
    busy_samples: int = 0
    busy_true: int = 0

    def snapshot(self) -> dict:
        waves = max(int(self.dispatch_waves), 0)
        samples = max(int(self.busy_samples), 0)
        return {
            "dispatched": int(self.dispatched),
            "hol_skips": int(self.hol_skips),
            "mean_p_tokens": (
                float(self.dispatch_token_sum) / waves if waves else 0.0),
            "p_busy_frac": (
                float(self.busy_true) / samples if samples else 0.0),
        }


@dataclass
class DecodeSchedStats:
    admitted: int = 0
    hold_wait_sum_s: float = 0.0
    hold_wait_n: int = 0
    b_hat_sum: int = 0
    b_hat_n: int = 0

    def snapshot(self) -> dict:
        n_wait = max(int(self.hold_wait_n), 0)
        n_b = max(int(self.b_hat_n), 0)
        return {
            "admitted": int(self.admitted),
            "mean_hold_wait_s": (
                float(self.hold_wait_sum_s) / n_wait if n_wait else 0.0),
            "mean_b_hat": (
                float(self.b_hat_sum) / n_b if n_b else 0.0),
        }


def merge_decode_stats(stats: Sequence[DecodeSchedStats]) -> dict:
    merged = DecodeSchedStats()
    for item in stats:
        merged.admitted += int(item.admitted)
        merged.hold_wait_sum_s += float(item.hold_wait_sum_s)
        merged.hold_wait_n += int(item.hold_wait_n)
        merged.b_hat_sum += int(item.b_hat_sum)
        merged.b_hat_n += int(item.b_hat_n)
    return merged.snapshot()


class PrefillPoolScheduler:
    """P 池:默认 FCFS + token 预算 + least-loaded 派发。

    pack / sjf 只改派发顺序,不改 token 预算与独占超长请求。
    hold 在 P 侧仍走 FCFS。
    """

    def __init__(self, opmodel, prefill_freq: int = 2520,
                 idle_freq: int = 600, token_budget: int = TOKEN_BUDGET,
                 policy: str = "fcfs",
                 pack_starve_s: float = DEFAULT_PACK_STARVE_S,
                 sjf_starve_s: float = DEFAULT_SJF_STARVE_S,
                 hold_ms: float = DEFAULT_HOLD_MS,
                 hold_b: int = DEFAULT_HOLD_B,
                 clock=time.time):
        self.opmodel = opmodel
        self.prefill_freq = int(prefill_freq)
        self.idle_freq = int(idle_freq)
        self.token_budget = int(token_budget)
        self.policy = normalize_pd_sched_policy(policy)
        self.pack_starve_s = float(pack_starve_s)
        self.sjf_starve_s = float(sjf_starve_s)
        self.hold_ms = float(hold_ms)
        self.hold_b = int(hold_b)
        self._clock = clock
        self._q: Deque[_PrefillQueued] = deque()
        self._dispatched_tokens: Dict[str, int] = {}  # 本调度器派出的在途 token
        self._done_len: Dict[int, int] = {}           # rid → prompt_len 记账
        self.stats = PrefillSchedStats()

    def submit(self, rid: int, prompt_len: int) -> None:
        self._q.append(_PrefillQueued(
            rid=int(rid), prompt_len=int(prompt_len),
            arrival_s=float(self._clock())))

    def cancel(self, rid: int) -> None:
        """请求方在派发前放弃(超时/断开):出队,避免幽灵派发占用 P 池。"""
        rid = int(rid)
        self._q = deque(item for item in self._q if item.rid != rid)
        self._done_len.pop(rid, None)

    def queued(self) -> int:
        return len(self._q)

    def queued_prompt_lens(self) -> List[int]:
        return [item.prompt_len for item in self._q]

    def on_dispatched_done(self, rid: int, instance: str,
                           prompt_len: Optional[int] = None) -> None:
        """prefill 完成:释放该实例在途 token 记账。"""
        if prompt_len is None:
            prompt_len = self._done_len.pop(int(rid), 0)
        self._dispatched_tokens[instance] = max(
            0, self._dispatched_tokens.get(instance, 0) - int(prompt_len))

    def sample(self) -> None:
        self.stats.busy_samples += 1
        if self.busy():
            self.stats.busy_true += 1

    def _can_take(self, tokens: Dict[str, int], name: str, plen: int) -> bool:
        return tokens[name] + plen <= self.token_budget or tokens[name] == 0

    def _least_loaded(self, instances: Sequence[str],
                      tokens: Dict[str, int]) -> str:
        return min(instances, key=lambda n: (tokens[n], n))

    def _assign(self, item: _PrefillQueued, name: str,
                tokens: Dict[str, int],
                out: List[Tuple[int, str]]) -> None:
        tokens[name] += item.prompt_len
        self._dispatched_tokens[name] = (
            self._dispatched_tokens.get(name, 0) + item.prompt_len)
        self._done_len[item.rid] = item.prompt_len
        out.append((item.rid, name))
        self.stats.dispatched += 1
        self.stats.dispatch_token_sum += int(item.prompt_len)

    def _dispatch_fcfs(self, instances: Sequence[str],
                       tokens: Dict[str, int]) -> List[Tuple[int, str]]:
        out: List[Tuple[int, str]] = []
        while self._q:
            item = self._q[0]
            name = self._least_loaded(instances, tokens)
            if not self._can_take(tokens, name, item.prompt_len):
                break
            self._q.popleft()
            self._assign(item, name, tokens, out)
        return out

    def _force_head_if_starved(self, instances: Sequence[str],
                               tokens: Dict[str, int], now: float,
                               starve_s: float) -> Optional[Tuple[int, str]]:
        if not self._q:
            return None
        head = self._q[0]
        if now - head.arrival_s < float(starve_s):
            return None
        name = self._least_loaded(instances, tokens)
        if not self._can_take(tokens, name, head.prompt_len):
            return None
        self._q.popleft()
        out: List[Tuple[int, str]] = []
        self._assign(head, name, tokens, out)
        return out[0]

    def _dispatch_pack(self, instances: Sequence[str],
                       tokens: Dict[str, int],
                       now: float) -> List[Tuple[int, str]]:
        out: List[Tuple[int, str]] = []
        forced = self._force_head_if_starved(
            instances, tokens, now, self.pack_starve_s)
        if forced is not None:
            out.append(forced)
        remaining = list(self._q)
        kept: List[_PrefillQueued] = []
        for item in remaining:
            name = self._least_loaded(instances, tokens)
            if self._can_take(tokens, name, item.prompt_len):
                if kept:
                    self.stats.hol_skips += 1
                self._assign(item, name, tokens, out)
            else:
                kept.append(item)
        self._q = deque(kept)
        return out

    def _dispatch_sjf(self, instances: Sequence[str],
                      tokens: Dict[str, int],
                      now: float) -> List[Tuple[int, str]]:
        items = list(self._q)

        def _sjf_key(item: _PrefillQueued) -> Tuple[int, float, int, int]:
            starved = now - item.arrival_s >= self.sjf_starve_s
            if starved:
                return (0, item.arrival_s, item.rid, item.prompt_len)
            return (1, float(item.prompt_len), item.rid, 0)

        items.sort(key=_sjf_key)
        out: List[Tuple[int, str]] = []
        kept: List[_PrefillQueued] = []
        for item in items:
            name = self._least_loaded(instances, tokens)
            if self._can_take(tokens, name, item.prompt_len):
                self._assign(item, name, tokens, out)
            else:
                kept.append(item)
        # 未派出的保持到达序,避免下一轮反复打乱 starve 记账以外的相对序
        kept.sort(key=lambda item: (item.arrival_s, item.rid))
        self._q = deque(kept)
        return out

    def dispatch(self, instances: Sequence[str],
                 inflight_tokens: Optional[Dict[str, int]] = None
                 ) -> List[Tuple[int, str]]:
        """按 policy 派发:每实例在途 token 不超预算;单条超预算请求独占派发。

        inflight_tokens:调用方可注入实例当前在途 token(engine 侧口径);
        缺省用本调度器自记账。返回 [(rid, instance)]。
        """
        if not instances:
            return []
        tokens = dict(self._dispatched_tokens)
        if inflight_tokens:
            for k, v in inflight_tokens.items():
                tokens[k] = max(tokens.get(k, 0), int(v))
        for name in instances:
            tokens.setdefault(name, 0)
        now = float(self._clock())
        if self.policy == "pack":
            out = self._dispatch_pack(instances, tokens, now)
        elif self.policy == "sjf":
            out = self._dispatch_sjf(instances, tokens, now)
        else:
            out = self._dispatch_fcfs(instances, tokens)
        if out:
            self.stats.dispatch_waves += 1
        return out

    def busy(self) -> bool:
        """P 池忙 = 有排队或 prefill 在途(派发未收尾)。
        不含 decode 流阶段:整请求口径会把 P 池永久锁在高频。"""
        return (len(self._q) > 0
                or any(v > 0 for v in self._dispatched_tokens.values()))

    def pool_freq(self, busy: bool) -> int:
        return self.prefill_freq if busy else self.idle_freq


class DecodePoolScheduler:
    """D 池:桥接准入(DistServe 语义)+ 逐实例 slack 选频。

    hold 只推迟准入以攒批;pack/sjf 在 D 侧仍 FCFS。
    """

    def __init__(self, opmodel, slo_tpot_s: float = 0.1,
                 tpot_margin: float = 0.35,
                 freq_candidates: Sequence[int] = (2520, 1800, 1500,
                                                   1200, 900, 600),
                 idle_freq: int = 600, max_blocks: int = 2048,
                 waiting_threshold: float = 0.5, block_size: int = 16,
                 clock=time.time, lam_alpha: float = 0.3,
                 busy_floor_mhz: int = 1200,
                 lookup_decode_freq: int = 0,
                 policy: str = "fcfs",
                 pack_starve_s: float = DEFAULT_PACK_STARVE_S,
                 sjf_starve_s: float = DEFAULT_SJF_STARVE_S,
                 hold_ms: float = DEFAULT_HOLD_MS,
                 hold_b: int = DEFAULT_HOLD_B):
        self.opmodel = opmodel
        self.slo_tpot_s = float(slo_tpot_s)
        self.tpot_margin = float(tpot_margin)
        self.freq_candidates = tuple(int(f) for f in freq_candidates)
        self.idle_freq = int(idle_freq)
        self.max_blocks = int(max_blocks)
        self.waiting_threshold = float(waiting_threshold)
        self.block_size = int(block_size)
        self._clock = clock
        self.lam_alpha = float(lam_alpha)
        self.busy_floor_mhz = int(busy_floor_mhz)
        self.lookup_decode_freq = int(lookup_decode_freq or 0)
        self.policy = normalize_pd_sched_policy(policy)
        self.pack_starve_s = float(pack_starve_s)
        self.sjf_starve_s = float(sjf_starve_s)
        self.hold_ms = float(hold_ms)
        self.hold_b = int(hold_b)
        self._bridge: Deque[_BridgeQueued] = deque()
        # 每实例活跃集合:rid -> (prompt_len, output_len)
        self._active: Dict[str, Dict[int, Tuple[int, int]]] = {}
        self._rr = 0
        # slack 记账:(instance, rid) → (首 token 时刻, n)
        self._tok: Dict[Tuple[str, int], Tuple[float, int]] = {}
        # 入流速率 EMA(on_prefill_done 口径),吞吐下限选频用
        self.lam_hat: float = 0.0
        self._last_rate_t: Optional[float] = None
        self._inflow_since: int = 0
        self.stats = DecodeSchedStats()

    # ------------------------------------------------------------------
    def on_prefill_done(self, rid: int, prompt_len: int,
                        output_len: int) -> None:
        now = float(self._clock())
        self._bridge.append(_BridgeQueued(
            rid=int(rid), prompt_len=int(prompt_len),
            output_len=int(output_len), arrival_s=now))
        self._inflow_since += 1
        if self._last_rate_t is None:
            self._last_rate_t = now
            return
        dt = now - self._last_rate_t
        if dt >= 1.0:
            inst = self._inflow_since / dt
            self.lam_hat = (self.lam_alpha * inst
                            + (1.0 - self.lam_alpha) * self.lam_hat)
            self._inflow_since = 0
            self._last_rate_t = now

    def bridge_depth(self) -> int:
        return len(self._bridge)

    def _inst_blocks(self, name: str) -> int:
        return sum(blocks_needed(plen, self.block_size)
                   for plen, _ in self._active.get(name, {}).values())

    def _hold_blocks(self, instances: Sequence[str], now: float) -> bool:
        if self.policy != "hold" or not self._bridge:
            return False
        name = min(instances, key=lambda n: (self._inst_blocks(n), n))
        b_hat = self._b_hat(name)
        ready = len(self._bridge)
        wait_s = now - self._bridge[0].arrival_s
        if b_hat >= self.hold_b or ready >= self.hold_b:
            return False
        return wait_s < (self.hold_ms / 1000.0)

    def try_admit(self, instances: Sequence[str]) -> List[Tuple[int, str]]:
        """桥队列 → decode 实例:least-loaded;should_accept 门控;拒则停排。"""
        if not instances:
            return []
        now = float(self._clock())
        if self._hold_blocks(instances, now):
            return []
        out: List[Tuple[int, str]] = []
        while self._bridge:
            item = self._bridge[0]
            need = blocks_needed(item.prompt_len, self.block_size)
            # least-loaded(按占用块),平局 round-robin
            order = sorted(instances,
                           key=lambda n: (self._inst_blocks(n), n))
            name = order[0]
            wb = self._inst_blocks(name)
            if not should_accept(wb, self.max_blocks, self.waiting_threshold,
                                 need, self.max_blocks - wb):
                break  # FCFS:队头被拒即停(DistServe post_process 语义)
            self._bridge.popleft()
            self._active.setdefault(name, {})[item.rid] = (
                item.prompt_len, item.output_len)
            out.append((item.rid, name))
            self.stats.admitted += 1
            self.stats.hold_wait_sum_s += max(0.0, now - item.arrival_s)
            self.stats.hold_wait_n += 1
        return out

    def sample(self, instances: Sequence[str]) -> None:
        for name in instances:
            self.stats.b_hat_sum += int(self._b_hat(name))
            self.stats.b_hat_n += 1

    def on_token(self, rid: int, instance: str, now: float) -> None:
        """逐 token 回调(slack 记账,语义同 PaDGWindowScheduler.on_token)。"""
        key = (str(instance), int(rid))
        first, n = self._tok.get(key, (float(now), 0))
        self._tok[key] = (first, n + 1)

    def _inst_slack(self, name: str, now: float):
        act = self._active.get(name, {})
        vals = [first + n * self.slo_tpot_s - now
                for (inst, rid), (first, n) in self._tok.items()
                if inst == name and rid in act and n > 0]
        return min(vals) if vals else None

    def complete(self, rid: int, instance: str) -> None:
        self._active.get(instance, {}).pop(int(rid), None)
        self._tok.pop((str(instance), int(rid)), None)

    def cancel(self, rid: int, instance: str) -> None:
        """请求方放弃(准入超时/断开):桥队列 + 活跃集 + slack 记账全清。

        只 complete 不清桥:泵稍后会把幽灵 rid 准入进活跃集且无人再
        complete → 永久泄漏,门控逐渐锁死(高压级联雪崩根因)。"""
        rid = int(rid)
        self._bridge = deque(item for item in self._bridge if item.rid != rid)
        self.complete(rid, instance)

    # ------------------------------------------------------------------
    def _b_hat(self, name: str) -> int:
        return len(self._active.get(name, {}))

    def _ctx_hat(self, name: str) -> float:
        act = self._active.get(name, {})
        if not act:
            return CTX_REF
        return float(sum(p + o / 2.0 for p, o in act.values()) / len(act))

    def _out_hat(self, name: str) -> float:
        act = self._active.get(name, {})
        if not act:
            return OUT_REF
        return float(sum(o for _, o in act.values()) / len(act))

    def pool_freqs(self, instances: Sequence[str],
                   now: Optional[float] = None) -> Dict[str, int]:
        """逐实例 slack+均衡批吞吐选频;空闲实例降至 idle_freq。"""
        out: Dict[str, int] = {}
        n_act = max(sum(1 for n in instances if self._b_hat(n) > 0), 1)
        if now is None:
            now = float(self._clock())
        # 准入上限对应的最大批量(块口径换算成请求数,按均值 prompt 估)
        b_cap = DECODE_B_CAP
        # 忙时频率下限(默认 1200,72B TP4 用 1500):E1b 能耗 U 曲线最优
        # ≈1500,更低档每 token 能耗更高、且持续慢跑会重开 KV 背压楔死
        # (日巡实测 att 0.31);TP4 通信占比高,1200 档实测 p90 贴穿 SLO。
        busy_freqs = tuple(f for f in self.freq_candidates
                           if f >= self.busy_floor_mhz) \
            or self.freq_candidates
        for name in instances:
            b = self._b_hat(name)
            if b == 0:
                out[name] = self.idle_freq
                continue
            # 吞吐判据用均衡批(continuous batching 自平衡),ρ 预算 0.7
            demand = (self.lam_hat / n_act) * self._out_hat(name) / RHO_TPUT
            budget = None
            s = self._inst_slack(name, now)
            if s is not None:
                budget = min((s + self.slo_tpot_s) * 0.8,
                             3.0 * self.slo_tpot_s) * 1000.0
            out[name] = int(select_decode_freq(
                self.opmodel, b, self._ctx_hat(name), self.slo_tpot_s,
                busy_freqs, margin=self.tpot_margin,
                budget_ms_override=budget,
                demand_tok_per_s=demand, b_cap=b_cap,
                prefer_mhz=int(getattr(self, "lookup_decode_freq", 0) or 0)))
        return out


class SelectiveRouter:
    """先 SLO 可行,再比同负载 gross J;无模型才用长度阈值。

    无模型回退用调用方传入的 mean-pin 阈值;构造默认 512 只给单测。
    过载外溢仍保留。两池都不行回 mixed,禁止再降频。
    """

    def __init__(self, pd_prompt_threshold: int = 512,
                 load_spill: float = 0.9, opmodel=None,
                 cost_based: bool = True,
                 kv_xfer_s_per_tok: float = KV_XFER_S_PER_TOK,
                 slo_ttft_s: float = 5.0,
                 output_hat: int = 200,
                 att_mixed: float = 1.0):
        self.pd_prompt_threshold = int(pd_prompt_threshold)
        self.load_spill = float(load_spill)
        self.opmodel = opmodel
        self.cost_based = bool(cost_based)
        self.kv_xfer_s_per_tok = float(kv_xfer_s_per_tok)
        self.slo_ttft_s = float(slo_ttft_s)
        self.output_hat = int(output_hat)
        self.att_mixed = float(att_mixed)
        self.last_both_infeasible = False

    def route_cost(self, prompt_len: int, mixed_load: float,
                   pd_load: float) -> Tuple[float, float]:
        """兼容旧测试:返回 (cost_mixed, cost_pd)。有模型时用 path_gross_j。"""
        from ecopadg.constraints import path_gross_j
        plen = max(int(prompt_len), 1)
        if self.opmodel is not None:
            cm = path_gross_j(self.opmodel, "mixed", plen, self.output_hat,
                              mixed_load, kv_xfer_s_per_tok=0.0)
            cp = path_gross_j(self.opmodel, "pd", plen, self.output_hat,
                              pd_load,
                              kv_xfer_s_per_tok=self.kv_xfer_s_per_tok)
            return cm, cp
        t_pre = 2.0e-4 * plen
        cost_m = float(mixed_load) * t_pre + t_pre
        cost_p = float(pd_load) * t_pre + t_pre + self.kv_xfer_s_per_tok * plen
        return cost_m, cost_p

    def _pool_ok(self, load: float, prompt_len: int,
                 pending_prefill_s: float = 0.0,
                 tpot_slack_s: float = 1e9,
                 free_blocks: int = 10 ** 9) -> bool:
        from ecopadg.constraints import PoolView, pool_slo_ok
        if self.opmodel is not None:
            # SLO/TTFT:该请求自己的 L
            need = self.opmodel.prefill_time_ms(max(int(prompt_len), 1)) / 1000.0
        else:
            need = 2.0e-4 * max(int(prompt_len), 1)
        view = PoolView(name="x", pending_prefill_s=pending_prefill_s,
                        tpot_slack_s=tpot_slack_s, free_blocks=free_blocks,
                        load=load)
        return pool_slo_ok(view, need, s_ttft=self.slo_ttft_s,
                           need_blocks=max(int(prompt_len) + 15, 0) // 16)

    def route(self, prompt_len: int, has_mixed: bool, has_pd: bool,
              mixed_load: float, pd_load: float,
              mixed_pending_s: float = 0.0, pd_pending_s: float = 0.0,
              mixed_slack_s: float = 1e9, pd_slack_s: float = 1e9) -> str:
        if not has_mixed and not has_pd:
            raise RuntimeError("无可用池")
        self.last_both_infeasible = False
        if not has_pd:
            return "mixed"
        if not has_mixed:
            return "pd"
        ok_m = self._pool_ok(mixed_load, prompt_len, mixed_pending_s,
                             mixed_slack_s)
        ok_p = self._pool_ok(pd_load, prompt_len, pd_pending_s, pd_slack_s)
        if ok_m and not ok_p:
            return "mixed"
        if ok_p and not ok_m:
            return "pd"
        if not ok_m and not ok_p:
            self.last_both_infeasible = True
            return "mixed"  # 都不可行:回退 mixed,禁止为节能再降频
        if self.cost_based and self.opmodel is not None:
            cm, cp = self.route_cost(prompt_len, mixed_load, pd_load)
            primary = "pd" if cp < cm else "mixed"
        else:
            primary = "pd" if int(prompt_len) >= self.pd_prompt_threshold \
                else "mixed"
        loads = {"mixed": float(mixed_load), "pd": float(pd_load)}
        other = "mixed" if primary == "pd" else "pd"
        if loads[primary] >= self.load_spill and loads[other] < self.load_spill:
            return other
        return primary


def pool_saturated(mean_inflight: float, queued: int, n_active: int,
                   b_ref: int = B_REF) -> bool:
    """L2 应急 unpark 的饱和判定(可单测)。"""
    return (mean_inflight >= 3.0 * b_ref
            or queued >= 4 * max(int(n_active), 1))
