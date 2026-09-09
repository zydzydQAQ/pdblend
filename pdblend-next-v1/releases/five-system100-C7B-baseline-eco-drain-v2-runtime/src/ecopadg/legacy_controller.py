#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ecospd_controller.py — 三层在线控制面。

pdblend 启动布局由 boot-slo-layout 在 4×mixed / 2×1P1D / A9 中选
(可信:att≥0.90 再 min J;不可信:长短混杂且分流够则静态 A9,否则
λ < α·μ_P 才允许纯 2×1P1D)。不线上改拓扑。
历史运行器保留兼容入口；新三池实现位于 ecopadg.serving。
--goodput-gate 在饱和或 SLO 回退时恢复服务容量，不再抑制满频。
--force-continuous --no-park。空 --lookup 不加载 lookup.json,prefer=0,
可行集内按 J/token 选频。显式 --lookup 的 freq_d 只作 slack prefer,不改
boot layout。L1/L2 仅当切换代价已测且摊还通过。不查 inherit 表。
L0:Alg 1 准入;mixed 路径可滚动 prefill 角色(接 prefill 钉 2520)。
L1:到达 CV / rate_fast 扩环;SLO 违约满频全 unpark。
L2:低载 park(pdblend 默认关)。可选 P2 通过宿主 supervisor 慢速事务化 mixed↔PD。

时间分离不是严格 PaDG。过载 503(计入窗口 att)。
"""
from __future__ import annotations

# 防御:直跑时 ecopadg/ 目录会进 sys.path[0],包内 types.py 遮蔽标准库 types。
import os as _os
import sys as _sys
_PKG_DIR = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:] = [p for p in _sys.path
                if _os.path.abspath(p or _os.getcwd()) != _PKG_DIR]

import argparse
import asyncio
import csv
import json
import math
import os
import signal
import sys
import threading
import time
from dataclasses import replace as dataclass_replace
from typing import Dict, List, Optional

try:
    import aiohttp
    from aiohttp import web
except ModuleNotFoundError:  # pure controller tests do not need the HTTP plane
    aiohttp = None
    web = None

WS_ROOT = os.environ.get("PDBLEND_WS", "/root/workspace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WS_ROOT)

from ecopadg.dvfs import DvfsController
from ecopadg.engine_telemetry import FileTelemetryRegistry
from ecopadg.learned_controller import (
    LearnedController, ModelBundle, load_model_bundle,
)
from ecopadg.logging_schema import DecisionJSONLLogger
from ecopadg.mpc.types import (
    MODE_NODG,
    MODE_STRICT_PADG,
    ControlAction,
    FastAction,
    SlowAction,
)
from ecopadg.shadow_runner import ShadowRunner
from ecopadg.strict_padg_executor import (
    EcoSpdTemporalExecutor,
    ExecutionResult,
)
from ecopadg.telemetry import ControlState
from ecopadg.global_scheduler import (
    Action, GlobalScheduler, GlobalState, build_partition_space,
)
from ecopadg.online_scheduler import (
    DECODE_RELATIVE_WINDOW, MODE_CONTINUOUS, MODE_PREFILL_WINDOW,
    PaDGWindowScheduler, decode_occupancy_blocks_dvfs,
    decode_window_blocks_dvfs,
)
from ecopadg.pool_scheduler import (
    DecodePoolScheduler, PrefillPoolScheduler, SelectiveRouter,
    merge_decode_stats, pd_sched_ctor_kwargs, pool_saturated,
)
from ecopadg.control_plane import (
    burst_wanted, next_prefill_role, pick_prefill_role,
    refuse_new_pd, want_pd_mode, want_shrink_ring,
)
from ecopadg.layout_oracle import (
    UNTRUSTED_DECODE_FLOOR_MHZ, cliff_lock_wanted,
)
from ecopadg.opmodel import freq_model_trustworthy, load_opmodel, model_runtime_cfg
from ecopadg.router import prompt_tokens
from ecopadg.types import (
    B_REF, GOODPUT_ATT_GATE, GOODPUT_KNEE_BAND_PP, LOAD_CAP_PER_INST,
    PREFILL_PIN_MHZ, ROLE_DECODE, ROLE_MIXED, ROLE_PREFILL,
    SLO_NON_INFERIOR_PP, TOKEN_BUDGET, freqs_for_model,
)
from ecopadg.planner import (
    PD_PREFILL_RHO_GUARD, estimate_decode_s, estimate_mu_decode_single,
)
from ecopadg.predictor import (
    ApproxPredictor, LoadMonitor, WorkloadStats, calibration_scales,
    capacity_model_trustworthy,
)
from ecopadg.temporal_coordinator import (
    MODE_CONTINUOUS as TEMPORAL_CONTINUOUS,
    MODE_TEMPORAL,
    GlobalTemporalCoordinator,
    TemporalConfig,
)
from ecopadg.profiler import (
    classify_size_bucket, dataset_size_bucket, interp_lookup_freq,
    load_lookup_table, load_switch_cost, rematerialization_open,
)
from ecopadg.types import Partition, SloSpec, SystemConfig
from ecopadg.measure.backends import get_backend

MODEL_CFG = {
    "14b": model_runtime_cfg("14b", WS_ROOT),
    "32b": model_runtime_cfg("32b", WS_ROOT),
    "72b": model_runtime_cfg("72b", WS_ROOT),
}

SLO_SAMPLE_WINDOW = 64
SLO_MIN_SAMPLES = 8
SLO_TRIP_CONSECUTIVE_WINDOWS = 2
DEFAULT_SLO_TRIP_HOLD_S = 5.0
DEFAULT_TEMPORAL_ACK_TIMEOUT_S = 5.0
DEFAULT_TEMPORAL_DECISION_PERIOD_S = 3.0


def _parse_endpoint(s: str):
    """'http://host:port@0,1' → (url, [0,1])"""
    url, _, gpus = s.partition("@")
    return url.strip(), [int(g) for g in gpus.split(",") if g.strip()]


def _normalize_baseline_att(value) -> tuple:
    """只接受 [0,1] 内的实测 attainment；缺失时返回 NaN/False。"""
    try:
        att = float(value)
    except (TypeError, ValueError):
        return float("nan"), False
    trusted = att == att and 0.0 <= att <= 1.0
    return (att, True) if trusted else (float("nan"), False)


def _capacity_untrusted_continuous_fallback(
        *, capacity_trusted: bool, controller_mode: str, strict_padg: bool,
        explicit_force_continuous: bool, no_roll: bool) -> bool:
    """Whether ordinary rule pdblend must avoid capacity-based windows."""
    return bool(
        not capacity_trusted
        and str(controller_mode) == "rule"
        and not strict_padg
        and not explicit_force_continuous
        and not no_roll
    )


def _validate_temporal_actuation_args(args) -> None:
    """Reject CLI layouts that cannot satisfy the atomic four-engine path."""
    if not bool(getattr(args, "enable_temporal_actuation", False)):
        return
    errors = []
    if not bool(getattr(args, "joint_temporal", False)):
        errors.append("--enable-temporal-actuation requires --joint-temporal")
    if str(getattr(args, "controller", "rule")) != "mpc":
        errors.append("--enable-temporal-actuation requires --controller mpc")
    if not str(getattr(args, "telemetry_dir", "") or "").strip():
        errors.append("--telemetry-dir is required")
    if not str(getattr(args, "engine_control_dir", "") or "").strip():
        errors.append("--engine-control-dir is required")
    try:
        joint_capacity = float(
            getattr(args, "joint_capacity_per_replica", 0.0) or 0.0)
    except (TypeError, ValueError):
        joint_capacity = 0.0
    if not math.isfinite(joint_capacity) or joint_capacity <= 0.0:
        errors.append(
            "--joint-capacity-per-replica must be a positive frozen profile")
    mixed_specs = [
        value
        for value in str(getattr(args, "mixed", "") or "").split(";")
        if value.strip()
    ]
    if len(mixed_specs) != 4:
        errors.append("exactly four --mixed engines are required")
    if str(getattr(args, "segments", "") or "").strip():
        errors.append("--segments is incompatible with temporal actuation")
    if bool(getattr(args, "enable_rematerialization", False)):
        errors.append(
            "--enable-rematerialization is incompatible with temporal actuation"
        )
    if errors:
        raise ValueError("; ".join(errors))


class MixedInstance:
    def __init__(self, url: str, gpus: List[int], opmodel, cfg: SystemConfig,
                 capacity_req_s: float, name: str = "",
                 handle=None):
        self.url = url
        self.gpus = gpus
        self.name = str(name)
        self.handle = handle
        self.sched = PaDGWindowScheduler(opmodel, cfg,
                                         mixed_capacity_req_s=capacity_req_s)
        self.inflight = 0
        self.parked = False
        self.draining = False
        self.freq = int(cfg.prefill_freq)


class Segment:
    def __init__(self, purl: str, pgpus: List[int], durl: str,
                 dgpus: List[int], opmodel, cfg: SystemConfig,
                 busy_floor_mhz: int = 1200, prefill_name: str = "",
                 decode_name: str = "", prefill_handle=None,
                 decode_handle=None, sched_kwargs=None):
        self.purl, self.pgpus = purl, pgpus
        self.durl, self.dgpus = durl, dgpus
        self.prefill_name = str(prefill_name)
        self.decode_name = str(decode_name)
        self.prefill_handle = prefill_handle
        self.decode_handle = decode_handle
        self.inflight = 0
        self.parked = False
        self.draining = False
        self.pfreq = 0
        self.dfreq = 0
        # 每段独立 D 池调度器(KV 亲和:准入只看本段 decode 实例)。
        # max_blocks=3600×0.5 → 上限 ~40 请求/实例(E1b-XL:b=40、真实 ctx
        # 下 iter@2520 仍 ≤ TPOT 预算);默认 2048 时峰值 22 请求即关门,
        # 准入排队 p90 达 26s(日巡实测)。
        self.dsched = DecodePoolScheduler(
            opmodel, slo_tpot_s=cfg.slo.tpot_s, tpot_margin=cfg.tpot_margin,
            freq_candidates=cfg.freq_candidates,
            idle_freq=min(cfg.freq_candidates), max_blocks=3600,
            busy_floor_mhz=busy_floor_mhz,
            lookup_decode_freq=int(
                getattr(cfg, "lookup_decode_freq", 0) or 0),
            **dict(sched_kwargs or {}))


class EcoSpdController:
    def __init__(self, args, mcfg):
        _validate_temporal_actuation_args(args)
        self.args = args
        self.mcfg = mcfg
        self._pd_sched_kwargs = pd_sched_ctor_kwargs(args)
        self.controller_mode = str(
            getattr(args, "controller", "rule") or "rule")
        self.shadow_mode = str(getattr(args, "shadow", "off") or "off")
        self._cliff_lock = bool(getattr(args, "cliff_lock", False))
        self.strict_padg = bool(getattr(args, "strict_padg", False))
        self.no_park = bool(getattr(args, "no_park", False))
        self.no_dvfs = bool(getattr(args, "no_dvfs", False))
        self.goodput_gate = bool(getattr(args, "goodput_gate", False))
        self.no_roll = bool(getattr(args, "no_roll", False))
        self.joint_temporal = bool(
            getattr(args, "joint_temporal", False))
        self.enable_temporal_actuation = bool(
            getattr(args, "enable_temporal_actuation", False))
        self.temporal_actuation_enabled = self.enable_temporal_actuation
        self.temporal_ack_timeout_s = float(getattr(
            args,
            "temporal_ack_timeout",
            DEFAULT_TEMPORAL_ACK_TIMEOUT_S,
        ))
        self.temporal_decision_period_s = float(getattr(
            args,
            "temporal_decision_period",
            DEFAULT_TEMPORAL_DECISION_PERIOD_S,
        ))
        self.joint_capacity_per_replica = float(getattr(
            args, "joint_capacity_per_replica", 0.0) or 0.0)
        self._joint_capacity_profile_trusted = bool(
            math.isfinite(self.joint_capacity_per_replica)
            and self.joint_capacity_per_replica > 0.0)
        explicit_force_continuous = bool(
            getattr(args, "force_continuous", False))
        self._explicit_force_continuous = explicit_force_continuous
        self.slo_trip_hold_s = float(getattr(
            args, "slo_trip_hold_s", DEFAULT_SLO_TRIP_HOLD_S))
        self.backend = get_backend("pynvml")
        # PE2 实测 settle p90=1.18s(空载 5×3 对),dwell 1.5s 即可摊销
        self.dvfs = DvfsController(backend=self.backend, min_dwell_s=1.5)
        self.opmodel = load_opmodel(mcfg["tables"])
        if self.opmodel is None:
            raise SystemExit("[ecospd] 缺操作点表: %s" % mcfg["tables"])
        self.slo = SloSpec(ttft_s=args.slo_ttft, tpot_s=args.slo_tpot)
        self.baseline_att, self._baseline_trusted = _normalize_baseline_att(
            getattr(args, "baseline_att", float("nan")))
        self.baseline_source = (
            "measured" if self._baseline_trusted else "missing")
        freqs = freqs_for_model(args.model_key)
        tpot = float(args.slo_tpot)
        floor72 = (2520 if args.model_key == "72b" and tpot <= 0.1 else 0)
        self.freq_trusted = freq_model_trustworthy(self.opmodel, freqs)
        decode_floor = 2520 if (not self.freq_trusted or floor72) else 0
        plan_p = int(getattr(args, "freq_prefill", 0) or 0)
        plan_d = int(getattr(args, "freq_decode", 0) or 0)
        plan_m = int(getattr(args, "freq_mixed", 0) or 0)
        # 规划/运行一致:P 永不低于满频
        prefill_mhz = max(plan_p, PREFILL_PIN_MHZ)
        if plan_d and decode_floor and plan_d < decode_floor:
            decode_floor = decode_floor
        elif plan_d and not decode_floor:
            decode_floor = int(plan_d)
        self.plan_freq_prefill = prefill_mhz
        self.plan_freq_decode = plan_d
        self.plan_freq_mixed = plan_m or prefill_mhz
        self.cfg = SystemConfig(model=args.model_key, slo=self.slo,
                                gpu_count=int(args.gpu_count),
                                freq_candidates=freqs,
                                prefill_freq=prefill_mhz,
                                global_period_s=args.global_period,
                                force_continuous=bool(
                                    explicit_force_continuous or self.no_roll),
                                decode_floor_mhz=decode_floor,
                                baseline_att=self.baseline_att,
                                role_switch_cost_j=float(getattr(
                                    args, "role_switch_cost_j",
                                    float("nan"))),
                                freq_model_trusted=self.freq_trusted,
                                lookup_decode_freq=0)
        self.switch_cost = load_switch_cost(
            getattr(args, "switch_cost", "") or "")
        arg_l2 = float(getattr(args, "role_switch_cost_j", float("nan")))
        if arg_l2 == arg_l2 and arg_l2 >= 0.0:
            self.switch_cost["role_switch_cost_j"] = arg_l2
        self.lookup_table = load_lookup_table(
            getattr(args, "lookup", "") or "",
            model=str(getattr(args, "model_key", "14b")))
        self._refresh_lookup_pref(log=False)
        self.tokenizer = None
        tok_name = getattr(args, "tokenizer", "") or ""
        if tok_name:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tok_name, trust_remote_code=True)
            except Exception as exc:
                print("[ecospd] tokenizer 加载失败,回退启发式: %s" % exc,
                      flush=True)
        tp = int(mcfg["tp"])

        self.mixed: List[MixedInstance] = []
        for part in (args.mixed or "").split(";"):
            if not part.strip():
                continue
            url, gpus = _parse_endpoint(part)
            self.mixed.append(MixedInstance(url, gpus, self.opmodel, self.cfg,
                                            mcfg["capacity_req_s"]))
        self.segments: List[Segment] = []
        for part in (args.segments or "").split(";"):
            if not part.strip():
                continue
            pspec, _, dspec = part.partition("|")
            purl, pgpus = _parse_endpoint(pspec)
            durl, dgpus = _parse_endpoint(dspec)
            # 72B@100ms / 不可信:钉 2520。可信时地板 = max(1200, 规划 D 频)。
            tpot = float(getattr(args, "slo_tpot", 0.1))
            if (not self.freq_trusted
                    or (args.model_key == "72b" and tpot <= 0.1)):
                floor = 2520
            else:
                base = 1200 if args.model_key != "72b" else 2520
                floor = max(base, int(self.plan_freq_decode or 0))
            self.segments.append(Segment(purl, pgpus, durl, dgpus,
                                         self.opmodel, self.cfg,
                                         busy_floor_mhz=floor,
                                         sched_kwargs=self._pd_sched_kwargs))
        if str(getattr(args, "pools", "all")).lower() == "mixed":
            self.segments = []
        if not self.mixed and not self.segments:
            raise SystemExit("[ecospd] 至少需要一个 mixed 实例或一个段")
        if self.strict_padg and (not self.mixed or self.segments):
            raise SystemExit(
                "[ecospd] --strict-padg 只支持 V0 mixed-only 执行面")
        self.engine_telemetry = FileTelemetryRegistry(
            getattr(args, "telemetry_dir", "") or "",
            ("mixed-%d" % i for i in range(len(self.mixed))))
        self._engine_phase_last: Dict[str, str] = {}
        self._telemetry_last = self.engine_telemetry.aggregate()
        print("[ecospd] plan freq P=%d D=%s M=%s mixed=%d seg=%d"
              % (self.plan_freq_prefill, self.plan_freq_decode or "-",
                 self.plan_freq_mixed, len(self.mixed), len(self.segments)),
              flush=True)
        print("[ecospd] pd_sched_policy=%s hold_ms=%s hold_b=%s"
              % (self._pd_sched_kwargs.get("policy"),
                 self._pd_sched_kwargs.get("hold_ms"),
                 self._pd_sched_kwargs.get("hold_b")),
              flush=True)

        self.psched = PrefillPoolScheduler(
            self.opmodel, prefill_freq=self.cfg.prefill_freq,
            idle_freq=min(self.cfg.freq_candidates),
            token_budget=args.prefill_token_budget,
            **self._pd_sched_kwargs)
        self.router = SelectiveRouter(pd_prompt_threshold=args.pd_threshold,
                                      load_spill=0.9, opmodel=self.opmodel,
                                      cost_based=True,
                                      slo_ttft_s=self.slo.ttft_s,
                                      att_mixed=self.baseline_att)
        self.monitor = LoadMonitor(window_s=30.0, fast_window_s=1.0,
                                   pd_prompt_threshold=args.pd_threshold)
        self.predictor = ApproxPredictor(
            opmodel=self.opmodel, stats=WorkloadStats(), slo=self.slo,
            freq_candidates=self.cfg.freq_candidates,
            pd_prompt_threshold=args.pd_threshold)
        # PE1 容量校准(与 choose_partition 同一口径);缺网格时 scale=1
        grid_path = os.path.join(WS_ROOT, "pdblend", "script", "bench",
                                 "results", "pe1_capacity", "rate_grid.json")
        pe1_ok = False
        rank_s = None
        if os.path.exists(grid_path):
            with open(grid_path, encoding="utf-8") as f:
                grid = json.load(f)
            cres = calibration_scales(
                self.predictor, grid, args.model_key,
                args.gpu_count // tp, args.gpu_count // (2 * tp))
            m_scale, s_scale, att_m, att_s = cres
            pe1_ok = bool(cres.calibrated)
            rank_s = cres.rank_spearman
            self.predictor = ApproxPredictor(
                opmodel=self.opmodel, stats=WorkloadStats(), slo=self.slo,
                freq_candidates=self.cfg.freq_candidates,
                pd_prompt_threshold=args.pd_threshold,
                mixed_scale=m_scale, seg_scale=s_scale,
                att_high_mixed=att_m, att_high_seg=att_s)
            # 过载护栏改用校准后的单实例容量(静态 MODEL_CFG 值偏乐观,
            # 会让 _overload 永不触发)
            cap1 = self.predictor.rate_mixed(1)
            for mi in self.mixed:
                mi.sched.capacity = cap1
            # 14B lite 只有 att 点、无 μ_S90 → calibrated=False,不得开深 DVFS
            # 深 DVFS 只在容量也可信时打开;保序打结时保持浅地板
            if cres.calibrated:
                self.freq_trusted = True
                self.cfg.freq_model_trusted = True
                if (self.cfg.decode_floor_mhz == 2520 and floor72 == 0
                        and capacity_model_trustworthy(
                            True, rank_s, True)):
                    self.cfg.decode_floor_mhz = 0
            print("[ecospd] PE1 校准: mixed_scale=%.3f seg_scale=%.3f "
                  "cap/inst=%.2f req/s trusted=%s" % (
                      m_scale, s_scale, cap1, self.freq_trusted),
                  flush=True)
        self.capacity_trusted = capacity_model_trustworthy(
            pe1_ok, rank_s, self.freq_trusted)
        self._capacity_untrusted_continuous = (
            _capacity_untrusted_continuous_fallback(
                capacity_trusted=self.capacity_trusted,
                controller_mode=self.controller_mode,
                strict_padg=self.strict_padg,
                explicit_force_continuous=explicit_force_continuous,
                no_roll=self.no_roll,
            )
        )
        if self._capacity_untrusted_continuous:
            # MixedInstance schedulers retain this mutable config reference.
            # Flip it before either the fast pump or control plane can rely on
            # untrusted capacity estimates.
            self.cfg.force_continuous = True
            print("[ecospd] capacity untrusted -> continuous mixed full ring",
                  flush=True)
        if self._cliff_lock:
            self.cfg.decode_floor_mhz = PREFILL_PIN_MHZ
        elif not self.capacity_trusted:
            # 不可信:允许浅 DVFS,禁止把地板打到 0(32B r=1.2 空心)
            self.cfg.decode_floor_mhz = max(
                int(self.cfg.decode_floor_mhz or 0),
                UNTRUSTED_DECODE_FLOOR_MHZ)
        d_floor = int(self.cfg.decode_floor_mhz or 0)
        for seg in self.segments:
            if hasattr(seg, "dsched") and d_floor:
                seg.dsched.busy_floor_mhz = max(
                    int(getattr(seg.dsched, "busy_floor_mhz", 0) or 0),
                    d_floor)
        self._refresh_lookup_pref(log=True)
        # park/unpark 可达子空间:tp 固定,n_mixed<=|mixed|,pairs<=|segments|
        space = [p for p in build_partition_space(
            {"mixed": [tp], "prefill": [tp], "decode": [tp]}, args.gpu_count)
            if p.n_mixed <= len(self.mixed)
            and p.n_prefill == p.n_decode
            and p.n_prefill <= len(self.segments)]
        if self.cfg.force_continuous:
            # Continuous baselines and the untrusted-capacity fallback stay on
            # the complete started layout; GlobalScheduler cannot select a
            # parked subset.
            space = [p for p in space
                     if p.n_mixed == len(self.mixed)
                     and p.n_prefill == len(self.segments)
                     and p.n_decode == len(self.segments)]
        self.gsched = GlobalScheduler(space, self.predictor, self.cfg)
        tp_p = max((len(s.pgpus) for s in self.segments), default=tp)
        tp_d = max((len(s.dgpus) for s in self.segments), default=tp)
        self.partition = Partition(
            n_mixed=len(self.mixed), n_prefill=len(self.segments),
            n_decode=len(self.segments), tp_mixed=tp, tp_prefill=tp_p,
            tp_decode=tp_d, gpus_total=args.gpu_count)
        self.last_switch = -1e9
        self._control_plane = (
            not bool(self.cfg.force_continuous)
            and not self.joint_temporal
        )
        self._prefill_role = 0
        self._ring_min = (
            max(len(self.mixed), 0)
            if self.no_park or self.joint_temporal
            else (2 if self._control_plane and len(self.mixed) >= 4
                  else max(len(self.mixed), 0)))
        if (self._control_plane and len(self.mixed) >= 4
                and not self._cliff_lock and self.capacity_trusted
                and self._baseline_trusted
                and not bool(getattr(self, "no_park", False))):
            for mi in self.mixed[self._ring_min:]:
                mi.parked = True
            self.partition = Partition(
                n_mixed=self._ring_min, n_prefill=len(self.segments),
                n_decode=len(self.segments), tp_mixed=tp, tp_prefill=tp_p,
                tp_decode=tp_d, gpus_total=args.gpu_count)
            print("[ecospd] L0 ring=%d parked=%d" % (
                self._ring_min, len(self.mixed) - self._ring_min), flush=True)

        self._enable_joint_local_admission()
        self.temporal_coordinator = None
        self._joint_temporal_initial_config = None
        if self.joint_temporal:
            initial_mode = (
                TEMPORAL_CONTINUOUS
                if (
                    self.enable_temporal_actuation
                    or bool(self.cfg.force_continuous)
                )
                else MODE_TEMPORAL
            )
            initial_n_prefill = (
                4 if len(self.mixed) >= 4
                else (2 if len(self.mixed) >= 2 else 1)
            )
            self._joint_temporal_initial_config = TemporalConfig(
                mode=initial_mode,
                n_prefill_active=(
                    initial_n_prefill
                    if initial_mode == TEMPORAL_CONTINUOUS else 1
                ),
                window_s=max(
                    float(self.cfg.window_min_s),
                    (
                        self.temporal_decision_period_s
                        if self.enable_temporal_actuation else 1.5
                    ),
                ),
                token_budget=max(
                    int(getattr(args, "prefill_token_budget",
                                TOKEN_BUDGET)), 1),
                fP=(
                    max(int(value) for value in freqs)
                    if self.enable_temporal_actuation
                    else int(self.plan_freq_prefill or 0) or None
                ),
                fD=(
                    max(int(value) for value in freqs)
                    if self.enable_temporal_actuation
                    else int(self.plan_freq_decode or 0) or None
                ),
            )
            self.temporal_coordinator = GlobalTemporalCoordinator(
                len(self.mixed),
                self._joint_temporal_initial_config,
                active=self.active_mixed(),
            )

        self._rid = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        if not self._baseline_trusted:
            print("[ecospd] baseline_att 缺省/非法 → full-ring + fmax; "
                  "禁止 predictor shrink", flush=True)
        else:
            print("[ecospd] baseline_att=%.3f source=%s"
                  % (self.baseline_att, self.baseline_source), flush=True)
        self.rows: List[dict] = []
        self.sched_rows: List[dict] = []
        self.temporal_rows: List[dict] = []
        self.engine_phase_rows: List[dict] = []
        self._last_engine_phase_log_s = -1e9
        self._last_engine_phase_key = None
        self._ttft_samples: List[float] = []
        self._tpot_samples: List[float] = []
        self._slo_outcomes: List[int] = []
        self._overload_rejects = 0
        self._slo_high_windows = {"ttft": 0, "tpot": 0}
        self._slo_trip_reason = ""
        self._prefill_pin_mask = [0 for _ in self.mixed]
        self._prefill_pin_reasons = ["not-evaluated" for _ in self.mixed]
        self._learned_queue_last: Optional[float] = None
        self._last_mpc_fast_switch = -1e9
        self._last_joint_mpc_step = -1e9
        self._last_joint_mpc_dispatch = -1e9
        self._controller_decision = None
        self._controller_execution_result = None
        self._temporal_fail_closed_pending = False
        self._shadow_decision = None
        self._learned_init_error = ""
        self.learned_controller = None
        self.temporal_executor = None
        self.shadow_runner = None
        if self.controller_mode == "mpc" or self.shadow_mode == "mpc":
            bundle_path = str(getattr(args, "model_bundle", "") or "")
            try:
                if bundle_path:
                    bundle = load_model_bundle(bundle_path)
                else:
                    budget = max(
                        int(getattr(args, "prefill_token_budget",
                                    TOKEN_BUDGET)), 1)
                    budgets = tuple(sorted({max(budget // 2, 1), budget}))
                    offsets = tuple(range(max(len(self.mixed), 1)))
                    full = max(len(self.mixed) + len(self.segments), 1)
                    bundle = ModelBundle(
                        version="builtin-conservative-v1",
                        frequencies_mhz=tuple(int(f) for f in freqs),
                        token_budgets=budgets,
                        rolling_offsets=offsets,
                        prefill_counts=(
                            (1, 2, 4)
                            if self.enable_temporal_actuation else ()
                        ),
                        window_seconds=(
                            (max(
                                float(self.cfg.window_min_s),
                                DEFAULT_TEMPORAL_DECISION_PERIOD_S,
                            ),)
                            if self.enable_temporal_actuation else ()
                        ),
                        prefill_frequencies_mhz=(
                            tuple(int(f) for f in freqs)
                            if self.enable_temporal_actuation else ()
                        ),
                        decode_frequencies_mhz=(
                            tuple(int(f) for f in freqs)
                            if self.enable_temporal_actuation else ()
                        ),
                        replica_counts=(
                            (4,)
                            if self.enable_temporal_actuation
                            else tuple(range(1, full + 1))
                        ))
            except (OSError, TypeError, ValueError) as exc:
                self._learned_init_error = type(exc).__name__
                bundle = ModelBundle(
                    version="invalid-bundle-fallback",
                    frequencies_mhz=(max(int(f) for f in freqs),),
                    token_budgets=(int(TOKEN_BUDGET),),
                    rolling_offsets=(0,),
                    replica_counts=(
                        max(len(self.mixed) + len(self.segments), 1),))
                print("[ecospd] model bundle invalid → MPC fail-closed: %s"
                      % exc, flush=True)
            if self.enable_temporal_actuation:
                bundle = dataclass_replace(
                    bundle,
                    replica_counts=(4,),
                    prefill_counts=(
                        bundle.prefill_counts or (1, 2, 4)
                    ),
                    window_seconds=(
                        bundle.window_seconds
                        or (max(
                            float(self.cfg.window_min_s),
                            float(bundle.shield.min_fast_dwell_s),
                            float(bundle.shield.min_dvfs_dwell_s),
                        ),)
                    ),
                    prefill_frequencies_mhz=(
                        bundle.prefill_frequencies_mhz
                        or tuple(int(f) for f in freqs)
                    ),
                    decode_frequencies_mhz=(
                        bundle.decode_frequencies_mhz
                        or tuple(int(f) for f in freqs)
                    ),
                )
            decision_log = os.path.join(
                str(args.out), "controller_decisions.jsonl")
            decision_logger = DecisionJSONLLogger(decision_log)
            executor = None
            if self.enable_temporal_actuation:
                self.temporal_executor = EcoSpdTemporalExecutor(
                    self,
                    self.temporal_coordinator,
                    self.engine_telemetry,
                    str(getattr(args, "engine_control_dir", "") or ""),
                    ack_timeout_s=self.temporal_ack_timeout_s,
                )
                executor = self.temporal_executor
            self.learned_controller = LearnedController(
                bundle=bundle, executor=executor, logger=decision_logger)
            if self.shadow_mode == "mpc":
                self.shadow_runner = ShadowRunner(
                    self.learned_controller, decision_logger)
        self._force_max_until = 0.0
        self._force_max_freq = bool(
            self._cliff_lock or not self._baseline_trusted)
        self._events: Dict[int, asyncio.Event] = {}     # mixed 窗释放
        self._pd_dispatch: Dict[int, asyncio.Event] = {}  # P 池派发
        self._pd_seg: Dict[int, int] = {}
        self._pd_admit: Dict[int, asyncio.Event] = {}    # D 池准入
        self._topology_admissions = 0
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.rematerialization_enabled = rematerialization_open(
            bool(getattr(args, "enable_rematerialization", False)),
            getattr(self, "switch_cost", None),
        )
        if (bool(getattr(args, "enable_rematerialization", False))
                and not self.rematerialization_enabled):
            print("[ecospd] L2 rematerialization blocked: "
                  "role_switch_cost unmeasured", flush=True)
        self.topology_adapter = None
        self.topology_coordinator = None
        self.topology = None  # compatibility/status alias
        self._last_topology_eval = time.time()
        if self.rematerialization_enabled:
            from ecopadg.controller_topology import (
                ControllerTopologyAdapter,
                degraded_topology,
            )
            try:
                self.topology_adapter = ControllerTopologyAdapter(self)
                self.topology_coordinator = self.topology_adapter.topology
            except Exception as exc:  # supervisor ownership boundary
                error = "%s:%s" % (type(exc).__name__, exc)
                self.topology_coordinator = degraded_topology(self, error)
                print(
                    "[ecospd] P2 attach failed -> DEGRADED: %s" % error,
                    flush=True,
                )
            self.topology = self.topology_coordinator

    def _refresh_lookup_pref(self, log: bool = False):
        """显式查表 freq_d → slack prefer。空表保持 0(能量最优)。"""
        table = getattr(self, "lookup_table", None) or {}
        args = getattr(self, "args", None)
        model = str(getattr(args, "model_key", "14b") if args else "14b")
        slo = str(getattr(args, "dataset", "sharegpt") if args else "sharegpt")
        stats = getattr(getattr(self, "predictor", None), "stats", None)
        pin = float(getattr(stats, "prompt_len", 0) or 0) if stats else 0.0
        pout = float(getattr(stats, "output_len", 0) or 0) if stats else 0.0
        if pin > 0 and pout > 0:
            bucket = classify_size_bucket(pin, pout)
        else:
            bucket = dataset_size_bucket(slo)
        b_hat = 1.0
        for mi in getattr(self, "mixed", None) or []:
            sched = getattr(mi, "sched", None)
            fn = getattr(sched, "_b_hat", None) if sched is not None else None
            if callable(fn):
                try:
                    b_hat = max(b_hat, float(fn()))
                except (TypeError, ValueError):
                    pass
        for seg in getattr(self, "segments", None) or []:
            dsched = getattr(seg, "dsched", None)
            if dsched is None:
                continue
            for name in getattr(dsched, "_active", {}) or {}:
                try:
                    b_hat = max(b_hat, float(len(dsched._active.get(name) or {})))
                except (TypeError, ValueError):
                    pass
        pref = interp_lookup_freq(table, bucket, model, slo, b_hat)
        mhz = int(pref or 0)
        cfg = getattr(self, "cfg", None)
        if cfg is not None:
            cfg.lookup_decode_freq = mhz
        for seg in getattr(self, "segments", None) or []:
            dsched = getattr(seg, "dsched", None)
            if dsched is not None:
                dsched.lookup_decode_freq = mhz
        if log:
            print("[ecospd] lookup L0 prefer decode=%s bucket=%s"
                  % (mhz or "-", bucket), flush=True)
        return pref

    # ------------------------------------------------------------------
    # P2 topology adapter / gates
    # ------------------------------------------------------------------

    def _topology_blocks_requests(self) -> bool:
        if not bool(getattr(self, "rematerialization_enabled", False)):
            return False
        coordinator = getattr(self, "topology_coordinator", None)
        return coordinator is None or coordinator.blocks_requests

    def _topology_queues_empty(self) -> bool:
        mixed_empty = all(
            m.inflight == 0
            and not m.draining
            and not m.sched.buffer
            and not m.sched.active
            for m in self.mixed
        )
        segments_empty = all(
            s.inflight == 0 and not s.draining for s in self.segments
        )
        dispatch_empty = not (
            self.psched.queued()
            or self._events
            or self._pd_dispatch
            or self._pd_admit
            or self._pd_seg
            or int(getattr(self, "_topology_admissions", 0))
        )
        return bool(mixed_empty and segments_empty and dispatch_empty)

    def _topology_slo_safe(self) -> bool:
        if self._want_max_freq():
            return False
        ttft = self._sample_p90(self._ttft_samples)
        tpot = self._sample_p90(self._tpot_samples)
        if ttft is not None and ttft >= 0.9 * self.slo.ttft_s:
            return False
        if tpot is not None and tpot >= 0.9 * self.slo.tpot_s:
            return False
        return True

    def _topology_trusted(self) -> bool:
        return bool(
            self.capacity_trusted
            and self._baseline_trusted
            and self.freq_trusted
        )

    def _pd_busy_floor(self) -> int:
        tpot = float(getattr(self.args, "slo_tpot", 0.1))
        model_key = str(getattr(self.args, "model_key", "14b"))
        if (
            not getattr(self, "freq_trusted", False)
            or (model_key == "72b" and tpot <= 0.1)
        ):
            return 2520
        base = 1200 if model_key != "72b" else 2520
        return max(base, int(getattr(self, "plan_freq_decode", 0) or 0))

    def _rebuild_runtime_from_pool(
        self, pool_manager, partition: Partition
    ) -> None:
        """Publish only a validated supervisor-backed physical layout."""
        from ecopadg.pool_manager import STATE_PARKED, STATE_READY

        live = pool_manager._live()
        if any(
            item.state not in (STATE_READY, STATE_PARKED) for item in live
        ):
            raise RuntimeError("cannot publish non-ready topology")
        mixed_records = sorted(
            (item for item in live if item.spec.role == ROLE_MIXED),
            key=lambda item: (min(item.spec.gpus), item.name),
        )
        prefill_records = {
            item.segment: item
            for item in live if item.spec.role == ROLE_PREFILL
        }
        decode_records = {
            item.segment: item
            for item in live if item.spec.role == ROLE_DECODE
        }
        if set(prefill_records) != set(decode_records):
            raise RuntimeError("cannot publish unpaired P/D topology")

        new_mixed = []
        for record in mixed_records:
            instance = MixedInstance(
                record.endpoint(),
                list(record.spec.gpus),
                self.opmodel,
                self.cfg,
                self.mcfg["capacity_req_s"],
                name=record.name,
                handle=record.handle,
            )
            instance.parked = record.state == STATE_PARKED
            new_mixed.append(instance)
        new_segments = []
        for segment_id in sorted(prefill_records):
            prefill = prefill_records[segment_id]
            decode = decode_records[segment_id]
            segment = Segment(
                prefill.endpoint(),
                list(prefill.spec.gpus),
                decode.endpoint(),
                list(decode.spec.gpus),
                self.opmodel,
                self.cfg,
                busy_floor_mhz=self._pd_busy_floor(),
                prefill_name=prefill.name,
                decode_name=decode.name,
                prefill_handle=prefill.handle,
                decode_handle=decode.handle,
                sched_kwargs=self._pd_sched_kwargs,
            )
            segment.parked = (
                prefill.state == STATE_PARKED
                or decode.state == STATE_PARKED
            )
            new_segments.append(segment)
        if (
            sum(not item.parked for item in new_mixed)
            != int(partition.n_mixed)
            or sum(not item.parked for item in new_segments)
            != int(partition.pairs())
        ):
            raise RuntimeError("published topology does not match partition")

        # Queues are guaranteed empty by the topology gate.  Replace all
        # request-plane references atomically while requests remain blocked.
        new_psched = PrefillPoolScheduler(
            self.opmodel,
            prefill_freq=self.cfg.prefill_freq,
            idle_freq=min(self.cfg.freq_candidates),
            token_budget=self.args.prefill_token_budget,
            **self._pd_sched_kwargs,
        )
        self.mixed = new_mixed
        self.segments = new_segments
        self.psched = new_psched
        self.partition = partition
        self._enable_joint_local_admission()
        self._prefill_role = 0
        self._events.clear()
        self._pd_dispatch.clear()
        self._pd_seg.clear()
        self._pd_admit.clear()
        self.engine_telemetry = FileTelemetryRegistry(
            getattr(self.args, "telemetry_dir", "") or "",
            (item.name or "mixed-%d" % index
             for index, item in enumerate(self.mixed)),
        )
        self._engine_phase_last = {}
        self._telemetry_last = self.engine_telemetry.aggregate()
        self._prefill_pin_mask = [0 for _ in self.mixed]
        self._prefill_pin_reasons = ["not-evaluated" for _ in self.mixed]
        temporal = getattr(self, "temporal_coordinator", None)
        if temporal is not None:
            temporal.configure(
                active=self.active_mixed(),
                now=time.time(),
                reason="topology-rebuild",
            )

    def _topology_target(self, mode: str) -> Partition:
        mode = str(mode).strip().lower()
        tp = int(self.mcfg["tp"])
        gpus = int(self.args.gpu_count)
        if mode == "mixed":
            return Partition(
                n_mixed=gpus // tp,
                tp_mixed=tp,
                tp_prefill=tp,
                tp_decode=tp,
                gpus_total=gpus,
            )
        if mode in ("pd", "p/d", "spatial-pd"):
            pairs = gpus // max(2 * tp, 1)
            return Partition(
                n_prefill=pairs,
                n_decode=pairs,
                tp_mixed=tp,
                tp_prefill=tp,
                tp_decode=tp,
                gpus_total=gpus,
            )
        raise ValueError("topology mode must be mixed or pd")

    @staticmethod
    def _partition_from_payload(payload: dict) -> Partition:
        allowed = {
            "n_mixed", "n_prefill", "n_decode",
            "tp_mixed", "tp_prefill", "tp_decode",
            "pp_mixed", "pp_prefill", "pp_decode", "gpus_total",
        }
        values = {
            key: int(value) for key, value in payload.items()
            if key in allowed
        }
        if set(payload) - allowed:
            raise ValueError("unknown partition fields")
        return Partition(**values)

    def _projected_topology_savings(
        self, target: Partition, now: Optional[float] = None
    ) -> float:
        now = time.time() if now is None else float(now)
        lam = float(self.monitor.rate(now))
        try:
            current_prediction = self.predictor.predict(
                self.partition, lam, None
            )
            target_prediction = self.predictor.predict(target, lam, None)
            threshold = (
                self.baseline_att
                - float(self.cfg.slo_non_inferior_pp)
            )
            if (
                not target_prediction.attainable
                or target_prediction.attainment < threshold
            ):
                return float("nan")
            return float(
                current_prediction.energy_j_per_s
                - target_prediction.energy_j_per_s
            )
        except Exception:
            return float("nan")

    def topology_status_payload(self) -> dict:
        if not bool(getattr(self, "rematerialization_enabled", False)):
            return {
                "enabled": False,
                "state": "STEADY",
                "steady": True,
                "blocks_requests": False,
                "reason": "feature-disabled",
            }
        coordinator = getattr(self, "topology_coordinator", None)
        if coordinator is None:
            return {
                "enabled": True,
                "state": "DEGRADED",
                "steady": False,
                "blocks_requests": True,
                "error": "coordinator-unavailable",
            }
        payload = coordinator.status().to_dict()
        payload["enabled"] = True
        return payload

    def request_topology(self, payload: dict) -> dict:
        if not bool(getattr(self, "rematerialization_enabled", False)):
            return {
                "accepted": False,
                "state": "STEADY",
                "reason": "feature-disabled",
            }
        if not isinstance(payload, dict):
            raise ValueError("topology request must be an object")
        coordinator = self.topology_coordinator
        if str(payload.get("action", "")).lower() == "abort":
            return coordinator.abort().to_dict()
        if "partition" in payload:
            target = self._partition_from_payload(payload["partition"])
        else:
            target = self._topology_target(payload.get("mode", ""))
        savings = payload.get("projected_savings_w")
        if savings is None:
            savings = self._projected_topology_savings(target)
        with coordinator.topology_lock:
            if self.topology_adapter is not None:
                self.topology_adapter.reconcile_controller()
            return coordinator.request_transition(
                target,
                projected_savings_w=float(savings),
                reason="manual",
            ).to_dict()

    def abort_topology(self) -> dict:
        if not bool(getattr(self, "rematerialization_enabled", False)):
            return {
                "accepted": False,
                "state": "STEADY",
                "reason": "feature-disabled",
            }
        return self.topology_coordinator.abort().to_dict()

    def _maybe_request_topology(self, now: float) -> None:
        # Spatial rematerialization is deliberately absent from FastMPC and
        # active learned control.  It is evaluated only on this slow rule path.
        if (
            not self.rematerialization_enabled
            or self.controller_mode != "rule"
            or not self.topology_coordinator.is_steady
        ):
            return
        lam = self.monitor.rate(now)
        lam_fast = self.monitor.rate_fast(now)
        stats = self.monitor.stats(now, default=self.predictor.stats)
        cv = self.monitor.arrival_cv(now)
        want_pd = want_pd_mode(
            stats.frac_long,
            cv,
            lam_fast,
            lam,
            self._topology_slo_safe(),
        )
        if self.partition.n_mixed > 0 and not self.partition.pairs():
            if not want_pd:
                return
            target = self._topology_target("pd")
        elif self.partition.pairs() and not self.partition.n_mixed:
            if want_pd:
                return
            target = self._topology_target("mixed")
        else:
            return
        coordinator = self.topology_coordinator
        with coordinator.topology_lock:
            if self.topology_adapter is not None:
                self.topology_adapter.reconcile_controller()
            decision = coordinator.request_transition(
                target,
                projected_savings_w=self._projected_topology_savings(
                    target, now
                ),
                now=now,
                reason="slow-rule",
            )
        if decision.accepted:
            print(
                "[ecospd] P2 topology accepted -> %s" % target,
                flush=True,
            )

    def _topology_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(0.5)
            coordinator = self.topology_coordinator
            if coordinator is None:
                continue
            now = time.time()
            if coordinator.is_steady:
                if (
                    now - self._last_topology_eval
                    >= float(self.args.topology_period)
                ):
                    self._last_topology_eval = now
                    self._maybe_request_topology(now)
                continue
            decision = coordinator.step(now)
            if decision.committed or decision.rolled_back:
                status = coordinator.status()
                self.partition = status.current_partition
                self.last_switch = status.last_switch

    # ------------------------------------------------------------------
    # 池视图
    # ------------------------------------------------------------------

    def _enable_joint_local_admission(self) -> None:
        """Leave engine-local schedulers in immediate continuous mode."""
        if not bool(getattr(self, "joint_temporal", False)):
            return
        for instance in self.mixed:
            instance.sched.cfg = dataclass_replace(
                instance.sched.cfg, force_continuous=True)
            instance.sched.mode = MODE_CONTINUOUS
            instance.sched.window_end_s = 0.0

    def _temporal_pressure(
            self, now: float, *, burst: bool,
            slo_trip: bool = False) -> None:
        coordinator = getattr(self, "temporal_coordinator", None)
        if not bool(getattr(self, "joint_temporal", False)) \
                or coordinator is None or not burst:
            return
        if bool(getattr(self, "enable_temporal_actuation", False)):
            # The active set may only change at the executor's acknowledged
            # publication boundary. Frequency safety is handled immediately
            # by _dvfs_step while the fast MPC emits the continuous fallback.
            self._unpark_all()
            return
        active = self.active_mixed()
        if not active:
            return
        if slo_trip:
            reason = str(getattr(self, "_slo_trip_reason", "") or "active")
            coordinator.emergency_continuous(
                reason="slo-trip:%s" % reason,
                active=active,
                now=now,
            )
            return
        coordinator.emergency_continuous(
            reason="burst",
            active=active,
            now=now,
            temporary=True,
            hold_s=coordinator.snapshot().window_s,
        )

    def _temporal_sched_fields(self) -> dict:
        coordinator = getattr(self, "temporal_coordinator", None)
        empty_engine_fields = {
            "engine_control_generation": "",
            "engine_control_requested_generation": "",
            "engine_control_applied_generation": "",
            "engine_control_rollback_generation": "",
            "engine_telemetry_requested_generations": "",
            "engine_telemetry_applied_generations": "",
            "engine_telemetry_pending_generations": "",
            "engine_telemetry_applied_modes": "",
            "engine_telemetry_control_errors": "",
            "temporal_execution_applied": "",
            "temporal_execution_reason": "",
            "temporal_fail_closed_pending": "",
            "executor_generation": "",
            "ack_latency": "",
            "rolled_back": "",
            "reason": "",
        }
        if not bool(getattr(self, "joint_temporal", False)) \
                or coordinator is None:
            return {
                "temporal_generation": "",
                "temporal_mode": "",
                "temporal_n_prefill_active": "",
                "temporal_active_set": "",
                "temporal_window_s": "",
                "temporal_window_end_s": "",
                "temporal_fP_target": "",
                "temporal_fD_target": "",
                "temporal_fallback_reason": "",
                **empty_engine_fields,
            }
        state = coordinator.snapshot()
        aggregate = dict(getattr(self, "_telemetry_last", {}) or {})
        executor = getattr(self, "temporal_executor", None)
        execution = getattr(self, "_controller_execution_result", None)
        status = executor.status() if executor is not None else {}
        engine_fields = {
            "engine_control_generation": status.get("generation", ""),
            "engine_control_requested_generation": status.get(
                "requested_generation", ""),
            "engine_control_applied_generation": status.get(
                "applied_generation", ""),
            "engine_control_rollback_generation": status.get(
                "rollback_generation", ""),
            "engine_telemetry_requested_generations": json.dumps(
                aggregate.get("requested_generations", {}),
                sort_keys=True,
            ),
            "engine_telemetry_applied_generations": json.dumps(
                aggregate.get("applied_generations", {}),
                sort_keys=True,
            ),
            "engine_telemetry_pending_generations": json.dumps(
                aggregate.get("pending_generations", {}),
                sort_keys=True,
            ),
            "engine_telemetry_applied_modes": json.dumps(
                aggregate.get("applied_modes", {}),
                sort_keys=True,
            ),
            "engine_telemetry_control_errors": json.dumps(
                aggregate.get("control_errors", {}),
                sort_keys=True,
            ),
            "temporal_execution_applied": (
                "" if execution is None else int(bool(execution.applied))
            ),
            "temporal_execution_reason": (
                "" if execution is None
                else str(getattr(execution, "reason", ""))
            ),
            "temporal_fail_closed_pending": int(bool(getattr(
                self, "_temporal_fail_closed_pending", False))),
            "executor_generation": (
                "" if execution is None
                or getattr(execution, "generation", None) is None
                else int(execution.generation)
            ),
            "ack_latency": (
                "" if execution is None
                or getattr(execution, "ack_latency_s", None) is None
                else float(execution.ack_latency_s)
            ),
            "rolled_back": (
                "" if execution is None
                else int(bool(getattr(execution, "rolled_back", False)))
            ),
            "reason": (
                "" if execution is None
                else str(getattr(execution, "reason", ""))
            ),
        }
        return {
            "temporal_generation": state.generation,
            "temporal_mode": state.mode,
            "temporal_n_prefill_active": state.n_prefill_active,
            "temporal_active_set": ";".join(
                str(index) for index in state.active_set),
            "temporal_window_s": state.window_s,
            "temporal_window_end_s": state.window_end_s,
            "temporal_fP_target": (
                "" if state.fP is None else state.fP),
            "temporal_fD_target": (
                "" if state.fD is None else state.fD),
            "temporal_fallback_reason": state.fallback_reason,
            **engine_fields,
        }

    def active_mixed(self) -> List[int]:
        return [i for i, m in enumerate(self.mixed)
                if not m.parked and not m.draining]

    def active_segments(self) -> List[int]:
        return [i for i, s in enumerate(self.segments)
                if not s.parked and not s.draining]

    def _mixed_load(self) -> float:
        act = self.active_mixed()
        if not act:
            return 1.0
        cap = float(LOAD_CAP_PER_INST) * len(act)
        return min(sum(self.mixed[i].inflight for i in act) / cap, 1.0)

    def _pd_load(self) -> float:
        act = self.active_segments()
        if not act:
            return 1.0
        cap = float(LOAD_CAP_PER_INST) * len(act)
        infl = sum(self.segments[i].inflight for i in act) \
            + self.psched.queued()
        return min(infl / cap, 1.0)

    def _unpark_mixed_emergency(self) -> bool:
        """SLO 回退:禁止为节能拒收,先唤醒 mixed。"""
        if self._topology_blocks_requests():
            return False
        for mi in self.mixed:
            if mi.parked or mi.draining:
                mi.parked = False
                mi.draining = False
                self._sync_partition_counts()
                return True
        return False

    def _unpark_all(self) -> None:
        if self._topology_blocks_requests():
            return
        for mi in self.mixed:
            mi.parked = False
            mi.draining = False
        for seg in self.segments:
            seg.parked = False
            seg.draining = False
        if hasattr(self, "partition"):
            self._sync_partition_counts()

    def _sync_partition_counts(self) -> None:
        if self._topology_blocks_requests():
            return
        n_m = len(self.active_mixed())
        n_s = len(self.active_segments())
        tp = self.mcfg["tp"]
        self.partition = Partition(
            n_mixed=n_m, n_prefill=n_s, n_decode=n_s,
            tp_mixed=tp, tp_prefill=tp, tp_decode=tp,
            gpus_total=self.args.gpu_count)

    def _l1_expand_ring(self, unpark_all: bool = False) -> None:
        if self._topology_blocks_requests():
            return
        if unpark_all or self._want_max_freq():
            if any(m.parked or m.draining for m in self.mixed):
                self._unpark_all()
                self._sync_partition_counts()
            return
        for mi in self.mixed:
            if mi.parked or mi.draining:
                mi.parked = False
                mi.draining = False
                self._sync_partition_counts()
                print("[ecospd] L1 expand ring -> %d"
                      % len(self.active_mixed()), flush=True)
                return

    def reset_runtime(self) -> None:
        """档间隔离:清滑窗、回默认环、解满频锁。"""
        self.monitor.reset()
        self.predictor.stats = WorkloadStats()
        self._ttft_samples.clear()
        self._tpot_samples.clear()
        outcomes = getattr(self, "_slo_outcomes", None)
        if outcomes is not None:
            outcomes.clear()
        self._overload_rejects = 0
        self._learned_queue_last = None
        self._controller_decision = None
        self._controller_execution_result = None
        self._shadow_decision = None
        self._last_joint_mpc_step = -1e9
        self._last_joint_mpc_dispatch = -1e9
        self._force_max_until = 0.0
        self._slo_high_windows = {"ttft": 0, "tpot": 0}
        self._slo_trip_reason = ""
        self._force_max_freq = bool(
            self._cliff_lock or not self._baseline_trusted)
        self._prefill_role = 0
        self._unpark_all()
        if (self._control_plane and len(self.mixed) >= 4
                and not self._cliff_lock and self.capacity_trusted
                and self._baseline_trusted
                and not bool(getattr(self, "no_park", False))):
            for mi in self.mixed[self._ring_min:]:
                mi.parked = True
            self._sync_partition_counts()
        coordinator = getattr(self, "temporal_coordinator", None)
        initial = getattr(self, "_joint_temporal_initial_config", None)
        if coordinator is not None and initial is not None:
            if bool(getattr(self, "enable_temporal_actuation", False)):
                executor = getattr(self, "temporal_executor", None)
                if executor is not None:
                    fmax = max(int(f) for f in self.cfg.freq_candidates)
                    action = ControlAction(
                        fast=FastAction(
                            mode=TEMPORAL_CONTINUOUS,
                            frequency_mhz=fmax,
                            token_budget=initial.token_budget,
                            rolling_offset=0,
                            n_prefill_active=4,
                            window_s=initial.window_s,
                            prefill_freq_mhz=fmax,
                            decode_freq_mhz=fmax,
                        ),
                        slow=SlowAction(active_replicas=4),
                    )
                    self._controller_execution_result = executor.apply(action)
            else:
                coordinator.configure(
                    initial,
                    active=self.active_mixed(),
                    now=time.time(),
                    reason="runtime-reset",
                )
        print("[ecospd] runtime reset ring=%d" % len(self.active_mixed()),
              flush=True)

    def _slo_trip_active(self, now: Optional[float] = None) -> bool:
        current = time.time() if now is None else float(now)
        until = float(getattr(self, "_force_max_until", 0.0) or 0.0)
        if current < until:
            return True
        if until > 0.0:
            self._force_max_until = 0.0
            self._slo_trip_reason = ""
            self._slo_high_windows = {"ttft": 0, "tpot": 0}
        return False

    def _note_slo_sample(
            self, metric: str, value_s: float, slo_s: float,
            now: Optional[float] = None) -> None:
        try:
            value = float(value_s)
            limit = float(slo_s)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value) or value <= 0.0 or limit <= 0.0:
            return
        current = time.time() if now is None else float(now)
        self._slo_trip_active(current)
        samples = (
            self._tpot_samples if metric == "tpot" else self._ttft_samples)
        samples.append(value)
        if len(samples) > SLO_SAMPLE_WINDOW:
            del samples[:-SLO_SAMPLE_WINDOW]
        if len(samples) < SLO_MIN_SAMPLES:
            return

        p90 = self._sample_p90(samples)
        baseline = float(getattr(self, "baseline_att", float("nan")))
        attainment_low = False
        if math.isfinite(baseline) and 0.0 <= baseline <= 1.0:
            attainment = (
                sum(1 for sample in samples if sample < limit) / len(samples))
            attainment_low = (
                attainment + 1e-12 < max(baseline - 0.01, 0.0))
        signals = []
        if p90 is not None and p90 >= 0.9 * limit:
            signals.append("p90")
        if attainment_low:
            signals.append("attainment")

        counters = getattr(self, "_slo_high_windows", None)
        if not isinstance(counters, dict):
            counters = {"ttft": 0, "tpot": 0}
            self._slo_high_windows = counters
        # A rolling window may remain high after one tail sample. Requiring
        # the new sample to carry pressure prevents that same outlier from
        # being counted as two consecutive high windows.
        if signals and value >= 0.9 * limit:
            counters[metric] = int(counters.get(metric, 0)) + 1
        else:
            counters[metric] = 0
        if counters[metric] < SLO_TRIP_CONSECUTIVE_WINDOWS:
            return

        hold_s = max(float(getattr(
            self, "slo_trip_hold_s", DEFAULT_SLO_TRIP_HOLD_S)), 0.0)
        self._force_max_until = max(
            float(getattr(self, "_force_max_until", 0.0) or 0.0),
            current + hold_s,
        )
        self._slo_trip_reason = "%s-%s" % (metric, "+".join(signals))
        self._unpark_all()
        if hasattr(self, "psched"):
            self.psched.prefill_freq = PREFILL_PIN_MHZ
        coordinator = getattr(self, "temporal_coordinator", None)
        if bool(getattr(self, "joint_temporal", False)) \
                and coordinator is not None \
                and not bool(getattr(
                    self, "enable_temporal_actuation", False)):
            coordinator.emergency_continuous(
                reason="slo-trip:%s" % self._slo_trip_reason,
                active=self.active_mixed(),
                now=current,
            )

    def _note_tpot(
            self, tpot_s: float, now: Optional[float] = None) -> None:
        """Record bounded TPOT evidence and trip only on sustained pressure."""
        self._note_slo_sample("tpot", tpot_s, self.slo.tpot_s, now)

    def _note_ttft(
            self, ttft_s: float, now: Optional[float] = None) -> None:
        """Record bounded TTFT evidence and trip only on sustained pressure."""
        self._note_slo_sample("ttft", ttft_s, self.slo.ttft_s, now)
        slo = getattr(self, "slo", None)
        if slo is None:
            return
        try:
            self._note_slo_outcome(float(ttft_s) < float(slo.ttft_s))
        except (TypeError, ValueError):
            self._note_slo_outcome(False)

    def _note_slo_outcome(self, ok: bool) -> None:
        """Rolling goodput evidence; 503 overloaded counts as a miss."""
        buf = getattr(self, "_slo_outcomes", None)
        if buf is None:
            self._slo_outcomes = buf = []
        buf.append(1 if ok else 0)
        if len(buf) > SLO_SAMPLE_WINDOW:
            del buf[:-SLO_SAMPLE_WINDOW]

    def _window_slo_att(self) -> Optional[float]:
        buf = getattr(self, "_slo_outcomes", None) or []
        if len(buf) < SLO_MIN_SAMPLES:
            return None
        return sum(buf) / float(len(buf))

    def _decode_instance_count(self) -> int:
        return len(self.active_segments()) + len(self.active_mixed())

    def _decode_inflight(self) -> int:
        n = 0
        for index in self.active_segments():
            n += max(int(getattr(self.segments[index], "inflight", 0) or 0), 0)
        for index in self.active_mixed():
            n += max(int(getattr(self.mixed[index], "inflight", 0) or 0), 0)
        return n

    def _decode_occupancy_blocks_dvfs(self) -> bool:
        n_inst = self._decode_instance_count()
        if n_inst <= 0:
            return False
        if decode_occupancy_blocks_dvfs(
                self._decode_inflight(), n_inst, alpha=PD_PREFILL_RHO_GUARD):
            return True
        pred = getattr(self, "predictor", None)
        dec_s = estimate_decode_s(pred, PREFILL_PIN_MHZ)
        if dec_s is None:
            return False
        mon = getattr(self, "monitor", None)
        if mon is None:
            return False
        try:
            rate = float(mon.rate(time.time()))
        except (TypeError, ValueError, AttributeError):
            return False
        return decode_occupancy_blocks_dvfs(
            rate * float(dec_s), n_inst, alpha=PD_PREFILL_RHO_GUARD)

    def _decode_mu_blocks_dvfs(self) -> bool:
        """同 α 的 μ_D 占用:λ ≥ α·μ_D 则钉满。μ_D 未知不猜,不改布局。"""
        n_inst = self._decode_instance_count()
        pred = getattr(self, "predictor", None)
        mu_d = estimate_mu_decode_single(pred, max(int(n_inst), 1), PREFILL_PIN_MHZ)
        try:
            mu = float(mu_d)
        except (TypeError, ValueError):
            return False
        if mu != mu or mu <= 0.0:
            return False
        mon = getattr(self, "monitor", None)
        if mon is None:
            return False
        try:
            rate = float(mon.rate(time.time()))
        except (TypeError, ValueError, AttributeError):
            return False
        if rate != rate or rate <= 0.0:
            return False
        return rate + 1e-12 >= float(PD_PREFILL_RHO_GUARD) * mu

    def _min_decode_slack_s(self, now: Optional[float] = None) -> Optional[float]:
        stamp = time.time() if now is None else float(now)
        slacks: List[float] = []
        for index, seg in enumerate(self.segments):
            if bool(getattr(seg, "parked", False)):
                continue
            dsched = getattr(seg, "dsched", None)
            if dsched is None:
                continue
            slack = dsched._inst_slack("d%d" % index, stamp)
            if slack is not None:
                slacks.append(float(slack))
        for mixed in self.mixed:
            if bool(getattr(mixed, "parked", False)):
                continue
            sched = getattr(mixed, "sched", None)
            if sched is None or not hasattr(sched, "pool_slack"):
                continue
            slack = sched.pool_slack(stamp)
            if slack is not None:
                slacks.append(float(slack))
        if not slacks:
            return None
        return min(slacks)

    def _slo_risk_detected(self) -> bool:
        """Conservative recovery on saturation or observed SLO regression."""
        if not bool(getattr(self, "goodput_gate", False)):
            return False
        baseline = float(getattr(self, "baseline_att", float("nan")))
        trusted = bool(getattr(self, "_baseline_trusted", False))
        if trusted and baseline + 1e-12 < (
                GOODPUT_ATT_GATE + GOODPUT_KNEE_BAND_PP):
            return True
        if self._decode_occupancy_blocks_dvfs():
            return True
        if self._decode_mu_blocks_dvfs():
            return True
        outcomes = getattr(self, "_slo_outcomes", None) or []
        if decode_window_blocks_dvfs(
                outcomes,
                tau=GOODPUT_ATT_GATE, baseline=baseline,
                trusted=trusted, delta=SLO_NON_INFERIOR_PP):
            return True
        short = list(outcomes)[-int(DECODE_RELATIVE_WINDOW):]
        if decode_window_blocks_dvfs(
                short, min_samples=int(DECODE_RELATIVE_WINDOW),
                tau=GOODPUT_ATT_GATE, baseline=baseline,
                trusted=trusted, delta=SLO_NON_INFERIOR_PP):
            return True
        window = self._window_slo_att()
        if window is None:
            return False
        if window + 1e-12 < GOODPUT_ATT_GATE:
            return True
        if trusted and window + 1e-12 < baseline - SLO_NON_INFERIOR_PP:
            return True
        return False

    def _goodput_gate_blocks_dvfs(self) -> bool:
        return self._slo_risk_detected()

    def _want_max_freq(self, now: Optional[float] = None) -> bool:
        return bool(self._goodput_gate_blocks_dvfs()
                    or self._force_max_freq or self._slo_trip_active(now))

    def _mixed_queue_view(self) -> tuple:
        act = self.active_mixed()
        if not act:
            return 0.0, 1e9
        now = time.time()
        pend = 0.0
        slack = 1e9
        for i in act:
            pend += self.mixed[i].sched._predicted_prefill_s()
            sl = self.mixed[i].sched.pool_slack(now)
            if sl is not None:
                slack = min(slack, sl)
        return pend / len(act), slack

    def _pd_pending_s(self) -> float:
        if self.opmodel is None:
            return 0.0
        return sum(self.opmodel.prefill_time_ms(plen) / 1000.0
                   for plen in self.psched.queued_prompt_lens())

    def _mixed_instance_id(self, index: int) -> str:
        instance_ids = tuple(
            getattr(self.engine_telemetry, "instance_ids", ()))
        if 0 <= int(index) < len(instance_ids):
            return str(instance_ids[int(index)])
        name = str(getattr(self.mixed[int(index)], "name", "") or "")
        return name or "mixed-%d" % int(index)

    def _mixed_prefill_pin(
            self, index: int, now: float,
            telemetry_views: Optional[dict] = None) -> tuple:
        """Return the per-instance prefill pin decision and its reason."""
        strict_telemetry = bool(
            self.strict_padg and self.engine_telemetry.enabled)
        if strict_telemetry:
            views = (
                self.engine_telemetry.read_all(now)
                if telemetry_views is None else telemetry_views)
            view = views.get(self._mixed_instance_id(index))
            if view is not None and not view.stale:
                if view.phase == "prefill":
                    return True, "strict-fresh-prefill"
                return False, "strict-fresh-%s" % view.phase
            coordinator = getattr(self, "temporal_coordinator", None)
            if bool(getattr(self, "joint_temporal", False)) \
                    and coordinator is not None:
                suspected = int(index) in coordinator.snapshot().active_set
                source = "temporal-set"
            else:
                role = int(getattr(self, "_prefill_role", -1))
                suspected = int(index) == role
                source = "role"
            if suspected:
                state = "missing" if view is None else "stale"
                return True, "strict-%s-%s-fail-closed" % (
                    source, state)
            state = "missing" if view is None else "stale"
            return False, "strict-non%s-%s" % (source, state)

        mode = str(getattr(self.mixed[int(index)].sched, "mode", "unknown"))
        if mode == MODE_PREFILL_WINDOW:
            return True, "scheduler-prefill-window"
        return False, "scheduler-%s" % mode

    def _refresh_strict_phase_roles(self, now: Optional[float] = None) -> dict:
        """按引擎真实 prefill→decode 边沿轮转，不按请求完成事件猜相位。"""
        aggregate = self.engine_telemetry.aggregate(now)
        self._telemetry_last = aggregate
        if not (self.strict_padg and self.engine_telemetry.enabled):
            return aggregate
        views = self.engine_telemetry.read_all(now)
        active = self.active_mixed()
        if not active:
            return aggregate
        if bool(getattr(self, "joint_temporal", False)):
            for key, current in views.items():
                if not current.stale:
                    self._engine_phase_last[key] = current.phase
            return aggregate
        role = (self._prefill_role
                if self._prefill_role in active else active[0])
        instance_id = self._mixed_instance_id(role)
        view = views.get(instance_id)
        if view is not None and not view.stale:
            previous = self._engine_phase_last.get(instance_id)
            if previous == "prefill" and view.phase in ("decode", "idle"):
                self._prefill_role = next_prefill_role(active, role)
        for key, current in views.items():
            if not current.stale:
                self._engine_phase_last[key] = current.phase
        return aggregate

    # ------------------------------------------------------------------
    # 请求处理
    # ------------------------------------------------------------------

    async def handle_completion(self, request: web.Request) -> web.StreamResponse:
        """Atomically admit only while the physical topology is STEADY."""
        coordinator = getattr(self, "topology_coordinator", None)
        admitted = False
        if (
            bool(getattr(self, "rematerialization_enabled", False))
            and coordinator is not None
        ):
            with coordinator.topology_lock:
                if coordinator.blocks_requests:
                    return web.json_response(
                        {
                            "error": "topology-transition",
                            "topology": coordinator.status().state.value,
                        },
                        status=503,
                    )
                with self._lock:
                    self._topology_admissions = int(
                        getattr(self, "_topology_admissions", 0)
                    ) + 1
                admitted = True
        elif self._topology_blocks_requests():
            return web.json_response(
                {"error": "topology-transition", "topology": "DEGRADED"},
                status=503,
            )
        try:
            return await self._handle_completion_admitted(request)
        finally:
            if admitted:
                with self._lock:
                    self._topology_admissions = max(
                        int(self._topology_admissions) - 1, 0
                    )

    async def _handle_completion_admitted(
        self, request: web.Request
    ) -> web.StreamResponse:
        body = await request.json()
        prompt = body.get("prompt", "")
        output_len = int(body.get("max_tokens", 128))
        plen = prompt_tokens(prompt, body, getattr(self, "tokenizer", None))
        now = time.time()
        with self._lock:
            rid = self._rid
            self._rid += 1
        self.monitor.on_arrival(now, plen, output_len)
        slo_trip = self._want_max_freq(now)
        burst = burst_wanted(
            self.monitor.arrival_cv(now), self.monitor.rate_fast(now),
            self.monitor.rate(now), slo_trip=slo_trip)
        if bool(getattr(self, "joint_temporal", False)):
            self._temporal_pressure(
                now, burst=burst, slo_trip=slo_trip)
        elif self._control_plane and burst:
            self._l1_expand_ring(unpark_all=self._want_max_freq())
        act_m, act_s = self.active_mixed(), self.active_segments()
        if not act_m and not act_s:
            self._note_slo_outcome(False)
            self._overload_rejects = int(
                getattr(self, "_overload_rejects", 0)) + 1
            return web.json_response(
                dict(error="no-capacity", rid=rid), status=503)
        sat_m = (not act_m) or self._mixed_load() >= 1.0
        sat_p = (not act_s) or self._pd_load() >= 1.0
        if sat_m and sat_p:
            if self._unpark_mixed_emergency():
                act_m = self.active_mixed()
            if not act_m:
                self._note_slo_outcome(False)
                self._overload_rejects = int(
                    getattr(self, "_overload_rejects", 0)) + 1
                return web.json_response(
                    dict(error="overloaded", rid=rid), status=503)
        m_pend, m_slack = self._mixed_queue_view()
        p_pend = self._pd_pending_s()
        if refuse_new_pd(burst, bool(act_m)):
            path = "mixed"
        else:
            path = self.router.route(plen, has_mixed=bool(act_m),
                                     has_pd=bool(act_s),
                                     mixed_load=self._mixed_load(),
                                     pd_load=self._pd_load(),
                                     mixed_pending_s=m_pend, pd_pending_s=p_pend,
                                     mixed_slack_s=m_slack)
            if getattr(self.router, "last_both_infeasible", False):
                self._force_max_freq = True
        row = dict(rid=rid, path=path, segment=-1, prompt_len=plen,
                   output_len=output_len, gate_wait_s="", prefill_s="",
                   error="")
        self.rows.append(row)
        if path == "mixed":
            resp = await self._serve_mixed(request, body, rid, plen,
                                           output_len, row)
        else:
            resp = await self._serve_pd(request, body, rid, plen,
                                        output_len, row)
        return resp

    async def _stream_from(self, resp: web.StreamResponse, url: str,
                           body: dict, row: dict, on_token=None) -> None:
        """转发引擎 SSE 流;on_token(now) 逐 token 回调(slack 记账)。"""
        try:
            timeout = aiohttp.ClientTimeout(total=900, sock_read=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url + "/v1/completions",
                                        json=body) as r:
                    if r.status != 200:
                        row["error"] = "engine-http-%d" % r.status
                        await resp.write(b"data: {\"error\": \"engine\"}\n\n")
                    else:
                        async for raw in r.content:
                            await resp.write(raw)
                            if on_token is not None and raw.startswith(b"data:") \
                                    and b"[DONE]" not in raw:
                                on_token(time.time())
        except Exception as e:  # noqa: BLE001
            row["error"] = "stream-error:%s" % type(e).__name__

    async def _serve_mixed(self, request, body, rid, plen, olen,
                           row) -> web.StreamResponse:
        act = self.active_mixed()
        coordinator = getattr(self, "temporal_coordinator", None)
        if (bool(getattr(self, "joint_temporal", False))
                and coordinator is not None and act):
            inflight = [m.inflight for m in self.mixed]
            mi = coordinator.choose_prefill_target(act, inflight)
        elif self._control_plane and act:
            inflight = [m.inflight for m in self.mixed]
            mi = pick_prefill_role(
                act, getattr(self, "_prefill_role", act[0]),
                inflight, int(LOAD_CAP_PER_INST))
            self._prefill_role = mi
        else:
            mi = min(act, key=lambda i: self.mixed[i].inflight) if act else 0
        inst = self.mixed[mi]
        inst.inflight += 1
        ev = asyncio.Event()
        with self._lock:
            self._events[rid] = ev
        arrival_wall = time.time()
        inst.sched.submit(rid, arrival_wall, plen, olen)
        t0 = time.perf_counter()
        tok_t: List[float] = []
        # 计数/调度器登记的回收必须在 finally:客户端断开(CancelledError)
        # 否则 inflight 与窗口活跃集泄漏,负载估计与路由逐渐失真。
        try:
            try:
                await asyncio.wait_for(ev.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                row["error"] = "gate-timeout"
            row["gate_wait_s"] = round(time.perf_counter() - t0, 4)
            resp = web.StreamResponse()
            resp.headers["Content-Type"] = "text/event-stream"
            resp.headers["Cache-Control"] = "no-cache"
            await resp.prepare(request)

            def _on_tok(now, _t=tok_t):
                inst.sched.on_token(rid, now)
                if not _t:
                    self._note_ttft(now - arrival_wall, now)
                _t.append(now)

            await self._stream_from(resp, inst.url, body, row,
                                    on_token=_on_tok)
        finally:
            if len(tok_t) >= 2:
                self._note_tpot((tok_t[-1] - tok_t[0]) / (len(tok_t) - 1))
            inst.sched.complete(rid)
            inst.inflight = max(0, inst.inflight - 1)
            if (self._control_plane
                    and not bool(getattr(self, "joint_temporal", False))
                    and not (self.strict_padg
                             and self.engine_telemetry.enabled)):
                self._prefill_role = next_prefill_role(
                    self.active_mixed() or [mi], mi)
            with self._lock:
                self._events.pop(rid, None)
        await resp.write_eof()
        return resp

    async def _serve_pd(self, request, body, rid, plen, olen,
                        row) -> web.StreamResponse:
        # 阶段 0:P 池 FCFS 派发(选段)
        ev_d = asyncio.Event()
        with self._lock:
            self._pd_dispatch[rid] = ev_d
        self.psched.submit(rid, plen)
        t0 = time.perf_counter()
        # 全程 try/finally 对账:客户端断开(CancelledError)或任一阶段超时,
        # P 队列/在途 token/D 活跃集必须回收,否则池门控逐渐收死。
        si = -1
        seg = None
        pf_acct_done = False
        tok_t: List[float] = []
        try:
            try:
                await asyncio.wait_for(ev_d.wait(), timeout=120.0)
            except asyncio.TimeoutError:
                row["error"] = "prefill-dispatch-timeout"
            si = self._pd_seg.get(rid, 0)
            seg = self.segments[si]
            seg.inflight += 1
            row["segment"] = si
            resp = web.StreamResponse()
            resp.headers["Content-Type"] = "text/event-stream"
            resp.headers["Cache-Control"] = "no-cache"
            await resp.prepare(request)
            if not row["error"]:
                # 阶段 0.5:D 池准入门在 prefill 之前(关键顺序)。
                # 若门在 prefill 之后:decode 活跃集满 → 准入关门 → 已
                # prefill 未 decode 的 KV 堆满 PyNccl 生产者缓冲(~10 请求)
                # → P 引擎插入阻塞整体卡死 → phase-1 全体超时级联崩溃
                # (r3.4 纯段布局实测 att 0.022 的根因)。先过门再 prefill,
                # 过渡态 KV ≈ 0(phase-2 紧跟 phase-1)。
                ev_a = asyncio.Event()
                with self._lock:
                    self._pd_admit[rid] = ev_a
                seg.dsched.on_prefill_done(rid, plen, olen)
                try:
                    await asyncio.wait_for(ev_a.wait(), timeout=300.0)
                except asyncio.TimeoutError:
                    row["error"] = "decode-admit-timeout"
                row["gate_wait_s"] = round(time.perf_counter() - t0, 4)
            if not row["error"]:
                # 阶段 1:prefill(max_tokens=1,KV → 段内 consumer)
                pf = dict(body)
                pf["max_tokens"] = 1
                pf["stream"] = False
                t_pf = time.perf_counter()
                try:
                    timeout = aiohttp.ClientTimeout(total=600, sock_read=120)
                    async with aiohttp.ClientSession(timeout=timeout) as s:
                        async with s.post(seg.purl + "/v1/completions",
                                          json=pf) as pr:
                            await pr.read()
                            if pr.status != 200:
                                row["error"] = "prefill-http-%d" % pr.status
                except Exception as e:  # noqa: BLE001
                    row["error"] = "prefill-error:%s" % type(e).__name__
                row["prefill_s"] = round(time.perf_counter() - t_pf, 4)
                try:
                    self._note_ttft(float(row["prefill_s"])
                                    + float(row.get("gate_wait_s") or 0))
                except (TypeError, ValueError):
                    pass
                self.psched.on_dispatched_done(rid, "s%d" % si, plen)
                pf_acct_done = True
            if not row["error"]:
                # 阶段 2:decode 流式(紧跟 phase-1,KV 即刻被消费)
                await self._stream_from(
                    resp, seg.durl, body, row,
                    on_token=lambda now: (tok_t.append(now),
                                          seg.dsched.on_token(
                                              rid, "d%d" % si, now)))
        finally:
            if len(tok_t) >= 2:
                self._note_tpot((tok_t[-1] - tok_t[0]) / (len(tok_t) - 1))
            if seg is None:
                # 派发等待期间被取消:出队防幽灵派发
                self.psched.cancel(rid)
            else:
                if not pf_acct_done and rid in self._pd_seg:
                    # 已派发但 prefill 未完成收尾:释放在途 token 记账
                    self.psched.on_dispatched_done(rid, "s%d" % si, plen)
                elif not pf_acct_done:
                    self.psched.cancel(rid)
                # cancel(含 complete):桥队列 + 活跃集 + slack 记账全清,
                # 防准入超时幽灵泄漏
                seg.dsched.cancel(rid, "d%d" % si)
                seg.inflight = max(0, seg.inflight - 1)
            with self._lock:
                self._pd_dispatch.pop(rid, None)
                self._pd_admit.pop(rid, None)
                self._pd_seg.pop(rid, None)
        await resp.write_eof()
        return resp

    # ------------------------------------------------------------------
    # 派发/准入泵(asyncio task,100ms)
    # ------------------------------------------------------------------

    def _record_engine_phase(self, now: float) -> None:
        if not bool(getattr(self, "joint_temporal", False)):
            return
        aggregate_fn = getattr(self.engine_telemetry, "aggregate", None)
        if not callable(aggregate_fn):
            return
        aggregate = aggregate_fn(now)
        coordinator = getattr(self, "temporal_coordinator", None)
        state = coordinator.snapshot() if coordinator is not None else None
        phases = aggregate.get("phases", {})
        key = (
            "" if state is None else state.mode,
            "" if state is None else state.n_prefill_active,
            tuple(sorted(phases.items())),
            round(float(aggregate.get("omega", 0.0) or 0.0), 6),
            aggregate.get("applied_generation"),
        )
        if (
            key == self._last_engine_phase_key
            and now - self._last_engine_phase_log_s < 1.0
        ):
            return
        self._last_engine_phase_key = key
        self._last_engine_phase_log_s = now
        self.engine_phase_rows.append({
            "t": round(float(now), 4),
            "mode": "" if state is None else state.mode,
            "n_prefill_active": (
                "" if state is None else state.n_prefill_active),
            "active_set": (
                "" if state is None else
                ";".join(str(index) for index in state.active_set)),
            "phases": json.dumps(phases, sort_keys=True),
            "omega": round(float(
                aggregate.get("omega", 0.0) or 0.0), 6),
            "telemetry_available": int(
                aggregate.get("available", 0) or 0),
            "requested_generation": aggregate.get("requested_generation"),
            "applied_generation": aggregate.get("applied_generation"),
            "ack_complete": int(bool(aggregate.get("ack_complete", False))),
            "ack_reason": aggregate.get("ack_reason", ""),
        })

    async def _pump(self):
        while not self._stop.is_set():
            now = time.time()
            self._refresh_strict_phase_roles(now)
            coordinator = getattr(self, "temporal_coordinator", None)
            if (bool(getattr(self, "joint_temporal", False))
                    and coordinator is not None):
                active_mixed = self.active_mixed()
                coordinator.eligible_indices(active_mixed)
                views = self.engine_telemetry.read_all(now)
                coordinator.step(
                    now,
                    {
                        index: views.get(self._mixed_instance_id(index))
                        for index in active_mixed
                    },
                )
                self._record_engine_phase(now)
            act = self.active_segments()
            if act:
                names = ["s%d" % i for i in act]
                for rid, name in self.psched.dispatch(instances=names):
                    si = int(name[1:])
                    with self._lock:
                        self._pd_seg[rid] = si
                        ev = self._pd_dispatch.get(rid)
                    if ev is not None:
                        ev.set()
                for si in act:
                    seg = self.segments[si]
                    for rid, _ in seg.dsched.try_admit(["d%d" % si]):
                        with self._lock:
                            ev = self._pd_admit.get(rid)
                        if ev is not None:
                            ev.set()
                    seg.dsched.sample(["d%d" % si])
            self.psched.sample()
            # mixed 释放泵(100ms):释放节奏若挂在 1s DVFS 循环上,
            # 准入延迟 p90≈1s,紧 TTFT(alpaca 1s)必然贴线违约
            for mi in self.mixed:
                if mi.parked:
                    continue
                d = mi.sched.step(now)
                for rid in d.released:
                    with self._lock:
                        ev = self._events.get(rid)
                    if ev is not None:
                        ev.set()
            await asyncio.sleep(0.1)

    # ------------------------------------------------------------------
    # DVFS 线程(周期 period,默认 1s)
    # ------------------------------------------------------------------

    def _emergency_unpark(self) -> None:
        """饱和快路径(1s 级):任一池饱和且存在 parked/draining 实例时,
        立即全部解除该池 park。

        30s 周期调度对负载阶跃反应太慢(λ̂ EMA 滞后 + 每周期一步),
        r3.4 冷启动实测爬坡 ~90s,毁掉短 trace 一半请求的 SLO。
        unpark 是解锁级操作(实例常驻热身),秒级生效;回落再 park
        由周期调度器带滞回执行,不影响稳态省能。"""
        if self._topology_blocks_requests():
            return
        changed = False

        def _per(pool_act):
            if not pool_act:
                return float("inf")
            return sum(x.inflight for x in pool_act) / len(pool_act)

        if any(m.parked or m.draining for m in self.mixed):
            act = [m for m in self.mixed if not m.parked and not m.draining]
            if pool_saturated(_per(act), 0, max(len(act), 1),
                              b_ref=self.predictor.b_ref):
                for m in self.mixed:
                    m.parked = False
                    m.draining = False
                changed = True
                print("[ecospd] emergency unpark mixed -> %d"
                      % len(self.mixed), flush=True)
        if any(s.parked or s.draining for s in self.segments):
            act_s = [s for s in self.segments
                     if not s.parked and not s.draining]
            if pool_saturated(_per(act_s), self.psched.queued(),
                              max(len(act_s), 1),
                              b_ref=self.predictor.b_ref):
                for s in self.segments:
                    s.parked = False
                    s.draining = False
                changed = True
                print("[ecospd] emergency unpark segments -> %d"
                      % len(self.segments), flush=True)
        if changed:
            n_m = sum(1 for m in self.mixed
                      if not m.parked and not m.draining)
            n_s = sum(1 for s in self.segments
                      if not s.parked and not s.draining)
            tp = self.mcfg["tp"]
            self.partition = Partition(
                n_mixed=n_m, n_prefill=n_s, n_decode=n_s,
                tp_mixed=tp, tp_prefill=tp, tp_decode=tp,
                gpus_total=self.args.gpu_count)
            self.last_switch = time.time()

    def _dvfs_loop(self):
        idle_f = min(self.cfg.freq_candidates)
        max_f = max(self.cfg.freq_candidates)
        while not self._stop.is_set():
            time.sleep(self.args.period)
            now = time.time()
            coordinator = getattr(self, "topology_coordinator", None)
            if (
                bool(getattr(self, "rematerialization_enabled", False))
                and coordinator is not None
            ):
                with coordinator.topology_lock:
                    if coordinator.blocks_requests:
                        continue
                    self._dvfs_step(now, idle_f, max_f)
            elif not self._topology_blocks_requests():
                self._dvfs_step(now, idle_f, max_f)

    def _joint_temporal_dvfs_step(
            self, now: float, idle_f: int, max_f: int,
            force: bool) -> None:
        """Execute acknowledged fP/fD targets from fresh engine phases."""
        self._unpark_all()
        coordinator = getattr(self, "temporal_coordinator", None)
        if coordinator is None:
            return
        state = coordinator.snapshot()
        prefill_target = (
            max_f if state.fP is None else int(state.fP))
        decode_target = (
            max_f if state.fD is None else int(state.fD))
        views = self.engine_telemetry.read_all(now)
        executor = getattr(self, "temporal_executor", None)
        executor_status = executor.status() if executor is not None else {}
        expected_generation = (
            executor_status.get("rollback_generation")
            if executor_status.get("rollback_generation") is not None
            else executor_status.get("applied_generation")
        )
        control_failure = any(
            not bool(getattr(view, "stale", True))
            and (
                bool(getattr(view, "control_error", None))
                or (
                    expected_generation is not None
                    and getattr(view, "applied_generation", None)
                    != expected_generation
                )
                or (
                    getattr(view, "applied_mode", None) is not None
                    and getattr(view, "applied_mode", None) != state.mode
                )
            )
            for view in views.values()
        )
        if control_failure:
            force = True
            self._force_max_freq = True
            self._temporal_fail_closed_pending = True
        active_set = set(int(index) for index in state.active_set)
        desired: Dict[int, int] = {}
        pin_mask: List[int] = []
        reasons: List[str] = []
        for index, mixed in enumerate(self.mixed):
            view = views.get(self._mixed_instance_id(index))
            fresh = view is not None and not bool(
                getattr(view, "stale", True))
            pin = False
            if force:
                frequency = max_f
                reason = "global-max-override"
            elif fresh:
                phase = str(getattr(view, "phase", "unknown")).lower()
                if phase == "prefill":
                    frequency = prefill_target
                    pin = True
                    reason = "temporal-fresh-prefill-fP"
                elif phase == "decode":
                    frequency = decode_target
                    reason = "temporal-fresh-decode-fD"
                elif phase == "idle":
                    frequency = idle_f
                    reason = "temporal-fresh-idle-min"
                elif phase == "overlap" and state.mode == TEMPORAL_CONTINUOUS:
                    frequency = max(prefill_target, decode_target)
                    reason = "continuous-overlap-max-phase-target"
                else:
                    frequency = max_f
                    pin = True
                    reason = "temporal-phase-%s-max-fail-closed" % phase
            elif (
                state.mode == TEMPORAL_CONTINUOUS
                and int(getattr(mixed, "inflight", 0) or 0) == 0
                and not bool(getattr(
                    getattr(mixed, "sched", None), "buffer", ()))
                and not bool(getattr(
                    getattr(mixed, "sched", None), "active", ()))
            ):
                frequency = idle_f
                freshness = "missing" if view is None else "stale"
                reason = "controller-idle-%s-min" % freshness
            elif state.mode == TEMPORAL_CONTINUOUS:
                frequency = max(prefill_target, decode_target)
                freshness = "missing" if view is None else "stale"
                reason = "continuous-%s-max-phase-target" % freshness
            elif index in active_set:
                frequency = prefill_target
                pin = True
                freshness = "missing" if view is None else "stale"
                reason = (
                    "temporal-active-set-%s-fP-fail-closed" % freshness)
            else:
                frequency = decode_target
                freshness = "missing" if view is None else "stale"
                reason = "temporal-nonactive-%s-fD" % freshness
            mixed.parked = False
            mixed.draining = False
            mixed.freq = int(frequency)
            pin_mask.append(int(pin))
            reasons.append(reason)
            for gpu in mixed.gpus:
                desired[gpu] = int(frequency)
        self._prefill_pin_mask = pin_mask
        self._prefill_pin_reasons = reasons
        self._telemetry_last = self.engine_telemetry.aggregate(now)
        self.dvfs.apply(desired, now)

    def _dvfs_step(self, now: float, idle_f: int, max_f: int) -> None:
        """Apply one DVFS decision while holding the topology lock."""
        slo_trip = self._want_max_freq(now)
        burst = burst_wanted(
            self.monitor.arrival_cv(now),
            self.monitor.rate_fast(now),
            self.monitor.rate(now),
            slo_trip=slo_trip,
        )
        if bool(getattr(self, "joint_temporal", False)):
            self._temporal_pressure(
                now, burst=burst, slo_trip=slo_trip)
        elif self._control_plane and burst:
            self._l1_expand_ring(unpark_all=slo_trip)
        self._emergency_unpark()
        if bool(getattr(self, "enable_temporal_actuation", False)):
            decision_period = float(getattr(
                self,
                "temporal_decision_period_s",
                DEFAULT_TEMPORAL_DECISION_PERIOD_S,
            ))
            last_dispatch = float(getattr(
                self, "_last_joint_mpc_dispatch", -1e9))
            if now - last_dispatch >= decision_period:
                self._last_joint_mpc_dispatch = now
                self._joint_mpc_step(now)
        self._refresh_lookup_pref()
        force = bool(self._cliff_lock or self._want_max_freq(now))
        desired: Dict[int, int] = {}
        if bool(getattr(self, "no_dvfs", False)):
            self._prefill_pin_mask = [0 for _ in self.mixed]
            self._prefill_pin_reasons = [
                "no-dvfs-max-override" for _ in self.mixed]
            for mi in self.mixed:
                mi.freq = max_f
                for gpu in mi.gpus:
                    desired[gpu] = mi.freq
            for seg in self.segments:
                freq = max_f
                seg.pfreq = seg.dfreq = freq
                for gpu in seg.pgpus + seg.dgpus:
                    desired[gpu] = freq
            self.dvfs.apply(desired, now)
            return
        if bool(getattr(self, "enable_temporal_actuation", False)):
            self._joint_temporal_dvfs_step(
                now, idle_f, max_f, force)
            return
        strict_telemetry = bool(
            self.strict_padg and self.engine_telemetry.enabled)
        telemetry_views = (
            self.engine_telemetry.read_all(now) if strict_telemetry else None)
        pin_mask: List[int] = []
        pin_reasons: List[str] = []
        for i, mi in enumerate(self.mixed):
            pin = False
            if mi.parked:
                f = idle_f
                reason = "parked"
            elif force:
                f = max_f
                reason = "global-max-override"
            else:
                pin, reason = self._mixed_prefill_pin(
                    i, now, telemetry_views)
                if pin:
                    f = max_f
                else:
                    # Strict telemetry is authoritative: fresh decode/idle
                    # phases use the scheduler's frequency even when a host
                    # buffer is non-empty. Missing/stale data fails closed
                    # only for the suspected current prefill role.
                    f = mi.sched.freq
                    if (not strict_telemetry
                            and not (mi.sched.buffer
                                     or mi.sched.active or mi.inflight)):
                        f = idle_f
            pin_mask.append(int(pin))
            pin_reasons.append(reason)
            if not pin and not force and not mi.parked:
                # 释放已移至 100ms _pump;此处只读调度器当前频率决策
                f = int(f)
            mi.freq = f
            for g in mi.gpus:
                desired[g] = f
        self._prefill_pin_mask = pin_mask
        self._prefill_pin_reasons = pin_reasons
        # P 池忙 = 排队或 prefill 在途;decode 流阶段不锁 P 池高频
        p_busy = self.psched.busy()
        for si, seg in enumerate(self.segments):
            if seg.parked:
                fp = fd = idle_f
            elif force:
                fp = fd = max_f
            else:
                fp = max(self.psched.pool_freq(busy=p_busy),
                         PREFILL_PIN_MHZ if p_busy else 0)
                fd = seg.dsched.pool_freqs(["d%d" % si])["d%d" % si]
            seg.pfreq, seg.dfreq = fp, fd
            for g in seg.pgpus:
                desired[g] = fp
            for g in seg.dgpus:
                desired[g] = fd
        self.dvfs.apply(desired, now)

    # ------------------------------------------------------------------
    # 周期调度线程(global_period,默认 30s):park/unpark 级迁移
    # ------------------------------------------------------------------

    def _apply_partition(self, target: Partition) -> None:
        if self._topology_blocks_requests():
            return
        if bool(getattr(self, "joint_temporal", False)):
            self._unpark_all()
            return
        if bool(getattr(self, "_capacity_untrusted_continuous", False)):
            self._unpark_all()
            return
        if bool(getattr(self, "no_park", False)):
            self._unpark_all()
            return
        if self._cliff_lock or self._want_max_freq():
            self._unpark_all()
            return
        # mixed:超出目标数的实例进入 draining → 排空后 parked
        for i, mi in enumerate(self.mixed):
            want_active = i < target.n_mixed
            if want_active:
                mi.parked = False
                mi.draining = False
            elif not mi.parked:
                mi.draining = True
        for i, seg in enumerate(self.segments):
            want_active = i < target.pairs()
            if want_active:
                seg.parked = False
                seg.draining = False
            elif not seg.parked:
                seg.draining = True

    def _settle_drains(self) -> None:
        if self._topology_blocks_requests():
            return
        if bool(getattr(self, "joint_temporal", False)):
            self._unpark_all()
            return
        for mi in self.mixed:
            if mi.draining and mi.inflight == 0 \
                    and not mi.sched.buffer and not mi.sched.active:
                mi.draining = False
                mi.parked = True
        for seg in self.segments:
            if seg.draining and seg.inflight == 0:
                seg.draining = False
                seg.parked = True

    @staticmethod
    def _sample_p90(values: List[float]) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(float(value) for value in values)
        index = max(int((0.9 * len(ordered)) + 0.999999) - 1, 0)
        return ordered[min(index, len(ordered) - 1)]

    def _learned_control_state(
            self, now: float, lam: float, lam_fast: float) -> ControlState:
        """Aggregate only measurements available at this controller tick."""
        queue_depth = float(
            sum(len(m.sched.buffer) for m in self.mixed)
            + int(self.psched.queued()))
        kv_demand = int(queue_depth) + sum(
            max(int(m.inflight), 0) for m in self.mixed) + sum(
            max(int(s.inflight), 0) for s in self.segments)
        previous = self._learned_queue_last
        growth = 0.0 if previous is None else queue_depth - previous
        self._learned_queue_last = queue_depth
        active = len(self.active_mixed()) + len(self.active_segments())
        full = max(len(self.mixed) + len(self.segments), 1)
        frequencies = [
            int(m.freq) for m in self.mixed if not m.parked
        ] + [
            max(int(s.pfreq), int(s.dfreq))
            for s in self.segments if not s.parked
        ]
        current_frequency = max(
            frequencies or [max(int(f) for f in self.cfg.freq_candidates)])
        temporal = getattr(self, "temporal_coordinator", None)
        temporal_state = (
            temporal.snapshot()
            if bool(getattr(self, "joint_temporal", False))
            and temporal is not None
            else None
        )
        executor = getattr(self, "temporal_executor", None)
        executor_status = executor.status() if executor is not None else {}
        engine_generation = (
            executor_status.get("rollback_generation")
            if executor_status.get("rollback_generation") is not None
            else executor_status.get("applied_generation")
        )
        if temporal_state is None:
            current_mode = (
                MODE_STRICT_PADG if self.strict_padg else MODE_NODG)
            current_budget = max(
                int(getattr(self.args, "prefill_token_budget",
                            TOKEN_BUDGET)), 1)
            current_n_prefill = 4
            current_window = 3.0
            current_fP = None
            current_fD = None
            last_temporal_change = float("-inf")
        else:
            current_mode = temporal_state.mode
            current_budget = temporal_state.token_budget
            current_n_prefill = temporal_state.n_prefill_active
            current_window = temporal_state.window_s
            current_fP = temporal_state.fP
            current_fD = temporal_state.fD
            last_temporal_change = float(getattr(
                self, "_last_temporal_change_s", float("-inf")))
        cap1 = 0.0
        if bool(getattr(
                self, "_joint_capacity_profile_trusted", False)):
            cap1 = float(self.joint_capacity_per_replica)
        else:
            try:
                cap1 = float(self.predictor.rate_mixed(1))
            except (AttributeError, TypeError, ValueError):
                cap1 = 0.0
        return ControlState.aggregate(
            timestamp_s=now,
            engine=self._telemetry_last,
            baseline_attainment=self.baseline_att,
            baseline_measured=self._baseline_trusted,
            capacity_trusted=bool(
                (self.capacity_trusted or self._joint_capacity_profile_trusted)
                and not self._learned_init_error),
            frequency_trusted=self.freq_trusted,
            arrival_rate_rps=lam,
            arrival_rate_fast_rps=lam_fast,
            queue_depth=queue_depth,
            queue_growth=growth,
            queue_observed=previous is not None,
            ttft_p90_s=self._sample_p90(self._ttft_samples),
            tpot_p90_s=self._sample_p90(self._tpot_samples),
            ttft_slo_s=self.slo.ttft_s,
            tpot_slo_s=self.slo.tpot_s,
            kv_required_blocks=max(kv_demand, 0),
            active_replicas=max(active, 1),
            full_replicas=full,
            fmin_mhz=min(int(f) for f in self.cfg.freq_candidates),
            fmax_mhz=max(int(f) for f in self.cfg.freq_candidates),
            max_token_budget=max(
                int(getattr(self.args, "prefill_token_budget",
                            TOKEN_BUDGET)), 1),
            current_mode=current_mode,
            current_frequency_mhz=current_frequency,
            current_token_budget=current_budget,
            current_rolling_offset=0,
            current_n_prefill_active=current_n_prefill,
            current_window_s=current_window,
            current_prefill_freq_mhz=current_fP,
            current_decode_freq_mhz=current_fD,
            last_fast_change_s=self._last_mpc_fast_switch,
            last_temporal_change_s=last_temporal_change,
            last_slow_change_s=self.last_switch,
            temporal_generation=engine_generation,
            capacity_per_replica_rps=max(cap1, 0.0),
            mean_prompt_tokens=self.predictor.stats.prompt_len,
            mean_output_tokens=self.predictor.stats.output_len,
            metadata={
                "bundle_error": self._learned_init_error,
                "capacity_source": (
                    "frozen-joint-profile"
                    if self._joint_capacity_profile_trusted
                    else "pe1-predictor"),
            },
        )

    def _joint_mpc_step(
            self, now: float) -> Optional[ExecutionResult]:
        """Run at most one acknowledged joint MPC action per fast period."""
        if not (
            bool(getattr(self, "joint_temporal", False))
            and bool(getattr(self, "enable_temporal_actuation", False))
            and str(getattr(self, "controller_mode", "rule")) == "mpc"
        ):
            return None
        current = float(now)
        period = max(
            float(getattr(
                self,
                "temporal_decision_period_s",
                DEFAULT_TEMPORAL_DECISION_PERIOD_S,
            )),
            0.0,
        )
        previous = float(getattr(self, "_last_joint_mpc_step", -1e9))
        if current - previous < period:
            return None
        self._last_joint_mpc_step = current
        learned = getattr(self, "learned_controller", None)
        if learned is None:
            self._force_max_freq = True
            self._unpark_all()
            return None

        self._telemetry_last = self.engine_telemetry.aggregate(current)
        expected_engines = len(self.mixed)
        if (
            int(self._telemetry_last.get("available", 0) or 0)
            < expected_engines
            or not bool(self._telemetry_last.get("ack_complete", False))
        ):
            # Engines may still be loading or idle before their first
            # scheduler step. Do not create failed generations while waiting
            # for the initial continuous generation acknowledgement.
            return None
        lam = self.monitor.rate(current)
        lam_fast = self.monitor.rate_fast(current)
        self.predictor.stats = self.monitor.stats(
            current, default=self.predictor.stats)
        state = self._learned_control_state(current, lam, lam_fast)
        decision = None
        fail_closed_pending = bool(getattr(
            self, "_temporal_fail_closed_pending", False))
        try:
            decision = learned.decide(state)
            if fail_closed_pending:
                decision = dataclass_replace(
                    decision,
                    chosen=learned.shield.fallback(state),
                    fallback=True,
                    reasons=(
                        "executor-fail-closed-continuous-fmax",
                    ) + tuple(decision.reasons),
                )
            learned.log_decision(state, decision, shadow=False)
            result = learned.apply(state, decision)
        except Exception as exc:
            self._learned_init_error = type(exc).__name__
            if decision is None:
                self._force_max_freq = True
                self._temporal_fail_closed_pending = True
                self._unpark_all()
                return None
            result = ExecutionResult(
                applied=False,
                action=decision.chosen,
                reason="mpc-execution-error:%s:%s"
                % (type(exc).__name__, str(exc)),
            )

        self._controller_decision = decision
        self._controller_execution_result = result
        if result.applied:
            if str(getattr(result, "reason", "")) != "no-change":
                self._last_mpc_fast_switch = current
            recovered = bool(
                fail_closed_pending
                and decision.chosen.fast.mode == TEMPORAL_CONTINUOUS
            )
            if recovered:
                self._temporal_fail_closed_pending = False
            if bool(getattr(decision, "fallback", False)) and not recovered:
                self._force_max_freq = True
                self._unpark_all()
            elif (
                not self._cliff_lock
                and not self._slo_trip_active(current)
                and bool(getattr(self, "_baseline_trusted", False))
                and bool(
                    getattr(self, "capacity_trusted", False)
                    or getattr(
                        self, "_joint_capacity_profile_trusted", False))
                and bool(getattr(self, "freq_trusted", False))
            ):
                self._force_max_freq = False
        else:
            self._force_max_freq = True
            self._temporal_fail_closed_pending = True
            self._unpark_all()
        return result

    def _periodic_step(self, now: Optional[float] = None) -> Action:
        """Serialize L3 inplace decisions against physical rematerialization."""
        now = time.time() if now is None else float(now)
        coordinator = getattr(self, "topology_coordinator", None)
        if (
            bool(getattr(self, "rematerialization_enabled", False))
            and coordinator is not None
        ):
            with coordinator.topology_lock:
                if coordinator.blocks_requests:
                    state = coordinator.status().state.value
                    return Action(
                        migrate=False,
                        partition=self.partition,
                        reasons=["topology:%s" % state.lower()],
                    )
                return self._periodic_step_steady(now)
        if self._topology_blocks_requests():
            state = (
                coordinator.status().state.value
                if coordinator is not None else "DEGRADED"
            )
            return Action(
                migrate=False,
                partition=self.partition,
                reasons=["topology:%s" % state.lower()],
            )
        return self._periodic_step_steady(now)

    def _periodic_step_steady(self, now: float) -> Action:
        """Execute one existing slow inplace decision while topology is steady."""
        self._settle_drains()
        lam = self.monitor.rate(now)
        lam_fast = self.monitor.rate_fast(now)
        self.predictor.stats = self.monitor.stats(
            now, default=self.predictor.stats)
        for seg in self.segments:
            seg.dsched.lam_hat = lam_fast
        st = GlobalState(partition=self.partition, lam_hat=lam,
                         baseline_attainment=self.baseline_att,
                         last_switch=self.last_switch)
        n_full = max(
            int(self.args.gpu_count) // max(int(self.mcfg["tp"]), 1), 1)
        mu = 0.0
        if bool(getattr(
                self, "_joint_capacity_profile_trusted", False)):
            mu = float(self.joint_capacity_per_replica) * n_full
        else:
            try:
                mu = float(self.predictor.rate_mixed(n_full))
                scale = float(
                    getattr(self.predictor, "mixed_scale", 1.0) or 1.0)
                if scale > 1.0:
                    mu = mu / scale
            except (TypeError, ValueError, ZeroDivisionError):
                mu = 0.0
        cliff = cliff_lock_wanted(
            max(lam, lam_fast), mu, static_cliff=self._cliff_lock,
            slo_trip=self._want_max_freq())
        if cliff:
            self._force_max_freq = True
            self._unpark_all()

        safe_dynamic = self.capacity_trusted and self._baseline_trusted
        controller_mode = str(getattr(self, "controller_mode", "rule"))
        shadow_mode = str(getattr(self, "shadow_mode", "off"))
        learned_state = None
        if controller_mode == "mpc" or shadow_mode == "mpc":
            learned_state = self._learned_control_state(now, lam, lam_fast)
        if controller_mode == "mpc":
            if bool(getattr(
                    self, "enable_temporal_actuation", False)) \
                    and bool(getattr(self, "joint_temporal", False)):
                execution = getattr(
                    self, "_controller_execution_result", None)
                reasons = ["mpc-joint-fast-loop"]
                if execution is not None and not execution.applied:
                    reasons.extend(["last-execution-failed", "max-freq"])
                act = Action(
                    migrate=False,
                    partition=self.partition,
                    reasons=reasons,
                )
            elif self.learned_controller is None or learned_state is None:
                act = Action(
                    migrate=False, partition=self.partition,
                    reasons=["mpc-unavailable", "max-freq"])
                self._force_max_freq = True
                self._unpark_all()
            else:
                try:
                    self._controller_decision = self.learned_controller.run(
                        learned_state, shadow=False, actuate=True)
                except Exception as exc:  # fail closed at backend boundary
                    self._controller_decision = None
                    self._learned_init_error = type(exc).__name__
                    act = Action(
                        migrate=False, partition=self.partition,
                        reasons=["mpc-error:%s" % type(exc).__name__,
                                 "max-freq"])
                    self._force_max_freq = True
                    self._unpark_all()
                else:
                    if self._controller_decision.fallback:
                        self._force_max_freq = True
                        self._unpark_all()
                        reasons = ["mpc-fallback", "max-freq"]
                    else:
                        # The default executor is intentionally a no-op until
                        # the patched engine exposes an atomic action endpoint.
                        reasons = ["mpc-feature-gated-noop"]
                    reasons.extend(list(self._controller_decision.reasons))
                    act = Action(
                        migrate=False, partition=self.partition,
                        reasons=reasons)
        elif safe_dynamic:
            act = self.gsched.step(st, now)
        else:
            if bool(getattr(
                    self, "_capacity_untrusted_continuous", False)):
                reason = "capacity-untrusted-continuous-mixed"
            else:
                reason = ("capacity-untrusted" if self._baseline_trusted
                          else "baseline-missing")
            act = Action(migrate=False, partition=self.partition,
                         reasons=[reason])
            self._unpark_all()

        if (shadow_mode == "mpc" and self.shadow_runner is not None
                and learned_state is not None):
            try:
                self._shadow_decision = self.shadow_runner.evaluate(
                    learned_state, active_action=learned_state.current_action)
            except Exception as exc:  # shadow must not disturb rule control
                self._shadow_decision = None
                print("[ecospd] shadow MPC skipped: %s" % type(exc).__name__,
                      flush=True)

        if "max-freq" in act.reasons or "no-feasible-candidate" in act.reasons:
            self._force_max_freq = True
            self._unpark_all()
        elif (controller_mode == "rule" and safe_dynamic
              and not self._cliff_lock and not cliff
              and now >= self._force_max_until):
            self._force_max_freq = False
        if cliff and act.migrate:
            smaller = (
                act.partition.total_gpus() < self.partition.total_gpus()
                or act.partition.pairs() < self.partition.pairs()
                or act.partition.n_mixed < self.partition.n_mixed)
            if smaller:
                act.migrate = False
                act.reasons = ["cliff-lock"] + list(act.reasons)
        if (bool(getattr(self, "no_park", False))
                or bool(getattr(self, "joint_temporal", False))) \
                and act.migrate:
            self._unpark_all()
            act.migrate = False
            reason = (
                "joint-temporal-fixed-engines"
                if bool(getattr(self, "joint_temporal", False))
                else "no-park-ablation"
            )
            act.reasons = [reason] + list(act.reasons)
        if act.migrate:
            self._apply_partition(act.partition)
            self.partition = act.partition
            self.last_switch = now

        stw = self.monitor.stats(now, default=self.predictor.stats)
        cv = self.monitor.arrival_cv(now)
        slack_ok = not self._want_max_freq()
        want_pd = want_pd_mode(stw.frac_long, cv, lam_fast, lam, slack_ok)
        if want_pd and not self.segments:
            # P2 is evaluated only by _topology_loop (>=300 s), never here in
            # the ordinary L3 or FastMPC path.
            note = (
                "deferred-to-slow-p2"
                if bool(getattr(
                    self, "rematerialization_enabled", False
                )) else "feature-disabled"
            )
            print("[ecospd] topology want-pd (%s)" % note, flush=True)
        queues_empty = all(
            m.inflight == 0 and not m.sched.buffer and not m.sched.active
            for m in self.mixed)
        if (controller_mode == "rule" and safe_dynamic
                and self._control_plane
                and want_shrink_ring(cv, lam_fast, lam, queues_empty,
                                     slo_trip=self._want_max_freq())
                and len(self.active_mixed()) > self._ring_min
                and not cliff
                and not bool(getattr(self, "no_park", False))):
            target = Partition(
                n_mixed=self._ring_min,
                n_prefill=len(self.active_segments()),
                n_decode=len(self.active_segments()),
                tp_mixed=self.mcfg["tp"], tp_prefill=self.mcfg["tp"],
                tp_decode=self.mcfg["tp"],
                gpus_total=self.args.gpu_count)
            self._apply_partition(target)
            self.partition = target
            self.last_switch = now
            act.migrate = True
            act.reasons = ["l2-shrink-ring"] + list(act.reasons)
        controller_decision = getattr(self, "_controller_decision", None)
        shadow_decision = getattr(self, "_shadow_decision", None)
        controller_action = (
            json.dumps(controller_decision.chosen.to_dict(), sort_keys=True)
            if controller_decision is not None else "")
        shadow_action = (
            json.dumps(shadow_decision.chosen.to_dict(), sort_keys=True)
            if shadow_decision is not None else "")
        model_version = ""
        learned = getattr(self, "learned_controller", None)
        if learned is not None:
            model_version = str(learned.model_version)
        fallback_continuous = bool(getattr(
            self, "_capacity_untrusted_continuous", False))
        execution_mode = (
            "joint-temporal"
            if bool(getattr(self, "joint_temporal", False))
            else (
                "capacity-untrusted-continuous-mixed"
                if fallback_continuous
                else (
                    "strict-padg" if self.strict_padg
                    else ("continuous-no-roll"
                          if bool(getattr(self, "no_roll", False))
                          else "phase-biased-mixed")
                )
            )
        )
        slo_trip_active = self._slo_trip_active(now)
        slo_trip_reason = (
            str(getattr(self, "_slo_trip_reason", ""))
            if slo_trip_active else "")
        temporal_fields = self._temporal_sched_fields()
        self.sched_rows.append(dict(
            t=round(now, 1), lam_hat=round(lam, 4),
            n_mixed_active=len(self.active_mixed()),
            n_seg_active=len(self.active_segments()),
            partition=repr(self.partition), migrate=int(act.migrate),
            reasons=";".join(act.reasons)[:120],
            controller=controller_mode, shadow=shadow_mode,
            controller_mode=controller_mode, shadow_mode=shadow_mode,
            controller_action=controller_action,
            shadow_action=shadow_action,
            model_version=model_version,
            mixed_freqs=";".join(str(m.freq) for m in self.mixed),
            mixed_modes=";".join(str(getattr(m.sched, "mode", "unknown"))
                                 for m in self.mixed),
            execution_mode=execution_mode,
            capacity_trusted=int(bool(self.capacity_trusted)),
            force_continuous=int(bool(getattr(
                getattr(self, "cfg", None), "force_continuous", False))),
            slo_trip_active=int(slo_trip_active),
            slo_trip_reason=slo_trip_reason,
            prefill_pin_mask=";".join(str(int(value)) for value in getattr(
                self, "_prefill_pin_mask", [])),
            prefill_pin_reasons=";".join(str(value) for value in getattr(
                self, "_prefill_pin_reasons", [])),
            no_park=int(bool(getattr(self, "no_park", False))),
            no_dvfs=int(bool(getattr(self, "no_dvfs", False))),
            no_roll=int(bool(getattr(self, "no_roll", False))),
            engine_phases=json.dumps(
                self._telemetry_last.get("phases", {}), sort_keys=True),
            omega_hat=round(float(
                self._telemetry_last.get("omega", 0.0) or 0.0), 6),
            telemetry_available=int(
                self._telemetry_last.get("available", 0) or 0),
            seg_freqs=";".join("%d|%d" % (s.pfreq, s.dfreq)
                               for s in self.segments),
            **temporal_fields))
        print("[ecospd] t=%.0f lam=%.3f part=%s migrate=%s %s"
              % (now, lam, self.partition, act.migrate,
                 act.reasons[:1]), flush=True)
        return act

    def _periodic_loop(self):
        while not self._stop.is_set():
            time.sleep(self.cfg.global_period_s)
            self._periodic_step()

    # ------------------------------------------------------------------
    def _flush(self):
        os.makedirs(self.args.out, exist_ok=True)
        coordinator = getattr(self, "temporal_coordinator", None)
        temporal_rows = getattr(self, "temporal_rows", None)
        if temporal_rows is None:
            temporal_rows = []
            self.temporal_rows = temporal_rows
        if coordinator is not None:
            temporal_rows.extend(
                event.to_row() for event in coordinator.drain_events())
        if self.rows:
            with open(os.path.join(self.args.out, "ctrl_requests.csv"), "w",
                      newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
                w.writeheader()
                w.writerows(self.rows)
        if self.sched_rows:
            with open(os.path.join(self.args.out, "sched.csv"), "w",
                      newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f,
                                   fieldnames=list(self.sched_rows[0].keys()))
                w.writeheader()
                w.writerows(self.sched_rows)
        if temporal_rows:
            with open(os.path.join(
                    self.args.out, "temporal_windows.csv"), "w",
                    newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f, fieldnames=list(temporal_rows[0].keys()))
                w.writeheader()
                w.writerows(temporal_rows)
        engine_phase_rows = getattr(self, "engine_phase_rows", [])
        if engine_phase_rows:
            with open(os.path.join(
                    self.args.out, "engine_phase_history.csv"), "w",
                    newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f, fieldnames=list(engine_phase_rows[0].keys()))
                w.writeheader()
                w.writerows(engine_phase_rows)
        self._flush_sched_policy()
        print("[ecospd] flushed %d requests, %d sched rows -> %s"
              % (len(self.rows), len(self.sched_rows), self.args.out),
              flush=True)

    def _sched_policy_payload(self) -> dict:
        kwargs = dict(getattr(self, "_pd_sched_kwargs", {}) or {})
        p_stats = self.psched.stats.snapshot()
        d_stats = merge_decode_stats(
            [seg.dsched.stats for seg in self.segments])
        return {
            "policy": kwargs.get("policy", getattr(self.psched, "policy", "fcfs")),
            "pack_starve_s": kwargs.get("pack_starve_s"),
            "sjf_starve_s": kwargs.get("sjf_starve_s"),
            "decode_hold_ms": kwargs.get("hold_ms"),
            "decode_hold_b": kwargs.get("hold_b"),
            **p_stats,
            **d_stats,
        }

    def _flush_sched_policy(self) -> None:
        payload = self._sched_policy_payload()
        with open(os.path.join(self.args.out, "sched_policy.json"), "w",
                  encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        with open(os.path.join(self.args.out, "sched_stats.csv"), "w",
                  newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
            writer.writeheader()
            writer.writerow(payload)

    def shutdown(self):
        self._stop.set()
        all_gpus = sorted({g for m in self.mixed for g in m.gpus}
                          | {g for s in self.segments
                             for g in s.pgpus + s.dgpus})
        self.dvfs.force_reset(all_gpus)
        self._flush()
        print("[ecospd] shutdown:频率已解锁", flush=True)


def _topology_period(value: str) -> float:
    period = float(value)
    if not math.isfinite(period) or period < 300.0:
        raise argparse.ArgumentTypeError(
            "topology period must be at least 300 seconds"
        )
    return period


def _nonnegative_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0.0:
        raise argparse.ArgumentTypeError(
            "value must be finite and non-negative"
        )
    return seconds


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise argparse.ArgumentTypeError(
            "value must be finite and positive"
        )
    return seconds


class _EcoSpdArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args=args, namespace=namespace)
        try:
            _validate_temporal_actuation_args(parsed)
        except ValueError as exc:
            self.error(str(exc))
        return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    ap = _EcoSpdArgumentParser(description="EcoSPD 三池控制器")
    ap.add_argument("--model-key", default="32b")
    ap.add_argument("--dataset", default="sharegpt",
                    help="PE1 网格键(sharegpt 顶层,其余数据集嵌套)")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mixed", default="",
                    help="分号分隔 'url@g0,g1' 列表")
    ap.add_argument("--segments", default="",
                    help="分号分隔 'purl@g0,g1|durl@g2,g3' 列表")
    ap.add_argument("--slo-ttft", type=float, default=5.0)
    ap.add_argument("--slo-tpot", type=float, default=0.1)
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--global-period", type=float, default=30.0)
    ap.add_argument(
        "--slo-trip-hold-s", "--slo-trip-hold",
        dest="slo_trip_hold_s", type=_nonnegative_seconds,
        default=DEFAULT_SLO_TRIP_HOLD_S,
        help="seconds to hold the sustained-SLO global frequency trip",
    )
    ap.add_argument("--pd-threshold", type=int, default=512)
    ap.add_argument("--prefill-token-budget", type=int, default=8192)
    ap.add_argument(
        "--pd-sched-policy",
        choices=("fcfs", "pack", "sjf", "hold"),
        default="fcfs",
        help="control-plane PD dispatch/admit policy; default FCFS",
    )
    ap.add_argument("--pack-starve-s", type=float, default=0.2,
                    help="pack: force head after this wait (seconds)")
    ap.add_argument("--sjf-starve-s", type=float, default=0.5,
                    help="sjf: promote aged head after this wait (seconds)")
    ap.add_argument("--decode-hold-ms", type=float, default=20.0,
                    help="hold: max wait before admitting a short decode batch")
    ap.add_argument("--decode-hold-b", type=int, default=8,
                    help="hold: admit once ready batch reaches this size")
    ap.add_argument("--gpu-count", type=int, default=8)
    ap.add_argument("--baseline-att", type=float, default=float("nan"),
                    help="同负载 mixed@2520 实测 attainment;缺省则满频满环且禁止缩容")
    ap.add_argument("--tokenizer", default="",
                    help="HF tokenizer 名或路径;空则用请求体 token / 启发式")
    ap.add_argument("--pools", default="all",
                    help="all | mixed:mixed-only 消融")
    ap.add_argument("--force-continuous", action="store_true",
                    help="M0-D:mixed 池只做 slack DVFS,不开时间窗")
    ap.add_argument("--goodput-gate", action="store_true",
                    help="pdblend:窗口 att 或 baseline <0.90 则钉 2520;"
                         "mixed_dvfs 不加此旗标")
    ap.add_argument("--no-park", action="store_true",
                    help="ablation: keep every started instance active")
    ap.add_argument("--no-dvfs", action="store_true",
                    help="ablation: pin active instances at maximum frequency")
    ap.add_argument("--no-roll", action="store_true",
                    help="ablation: continuous admission without rolling windows")
    ap.add_argument(
        "--joint-temporal",
        action="store_true",
        help=("gate new mixed requests with the global KV-local temporal "
              "coordinator"),
    )
    ap.add_argument(
        "--enable-temporal-actuation",
        action="store_true",
        help="actively apply acknowledged joint MPC controls to mixed engines",
    )
    ap.add_argument(
        "--engine-control-dir",
        default="",
        help="per-engine patched vLLM runtime control JSON directory",
    )
    ap.add_argument(
        "--temporal-ack-timeout",
        type=_nonnegative_seconds,
        default=DEFAULT_TEMPORAL_ACK_TIMEOUT_S,
        help="seconds to await all four fresh engine control acknowledgements",
    )
    ap.add_argument(
        "--temporal-decision-period",
        type=_positive_seconds,
        default=DEFAULT_TEMPORAL_DECISION_PERIOD_S,
        help="minimum seconds between joint MPC decisions in the fast loop",
    )
    ap.add_argument(
        "--joint-capacity-per-replica",
        type=float,
        default=0.0,
        help=("frozen seed0 profile capacity used by joint temporal MPC; "
              "required for active actuation"),
    )
    ap.add_argument("--strict-padg", action="store_true",
                    help="执行面为 V0 且禁 chunked prefill；记录 strict PaDG 模式")
    ap.add_argument("--telemetry-dir", default="",
                    help="patched vLLM 原子 phase/KV JSON 目录")
    ap.add_argument("--freq-mixed", type=int, default=0,
                    help="choose_partition 的 freq_mixed;0=用满频")
    ap.add_argument("--freq-prefill", type=int, default=0,
                    help="规划 P 频;运行时仍钉 ≥2520")
    ap.add_argument("--freq-decode", type=int, default=0,
                    help="规划 D 频;可信时作为忙时地板")
    ap.add_argument("--cliff-lock", action="store_true",
                    help="容量崖:全程 f_max,禁止 park")
    ap.add_argument("--controller", choices=("rule", "mpc"), default="rule",
                    help="active controller; rule preserves existing behavior")
    ap.add_argument("--shadow", choices=("off", "mpc"), default="off",
                    help="evaluate/log MPC without actuation")
    ap.add_argument("--model-bundle", default="",
                    help="optional learned-controller JSON bundle")
    ap.add_argument(
        "--supervisor-url",
        default="http://127.0.0.1:8765",
        help="host engine supervisor JSON RPC base URL",
    )
    ap.add_argument(
        "--enable-rematerialization",
        action="store_true",
        help="enable slow supervisor-backed mixed↔P/D role rebuilding",
    )
    ap.add_argument(
        "--topology-period",
        type=_topology_period,
        default=300.0,
        help="slow spatial evaluation period in seconds (minimum 300)",
    )
    ap.add_argument(
        "--topology-min-dwell",
        type=_nonnegative_seconds,
        default=600.0,
        help="minimum seconds between validated spatial role switches",
    )
    ap.add_argument(
        "--role-switch-cost-j",
        type=float,
        default=float("nan"),
        help="measured restart energy; unknown/NaN blocks role switching",
    )
    ap.add_argument(
        "--lookup",
        default="",
        help="explicit lookup.json only; empty = no freq_d prefer",
    )
    ap.add_argument(
        "--switch-cost",
        default="",
        help="measured switch_cost.json; missing L1/L2 keys keep that layer off",
    )
    return ap


async def main():
    args = build_arg_parser().parse_args()
    if aiohttp is None or web is None:
        raise SystemExit("[ecospd] runtime requires the aiohttp package")
    if str(args.pools).lower() == "mixed":
        args.segments = ""
        if not args.mixed:
            raise SystemExit("[ecospd] --pools mixed 需要 --mixed")
    ctrl = EcoSpdController(args, model_runtime_cfg(args.model_key, WS_ROOT))

    async def _reset(_request):
        if ctrl._topology_blocks_requests():
            return web.json_response(
                {"ok": False, "error": "topology-transition"},
                status=503,
            )
        ctrl.reset_runtime()
        return web.json_response(dict(ok=True))

    async def _topology_get(_request):
        return web.json_response(ctrl.topology_status_payload())

    async def _topology_post(request):
        try:
            payload = await request.json()
            result = ctrl.request_topology(payload)
        except (TypeError, ValueError) as exc:
            return web.json_response(
                {"accepted": False, "error": str(exc)}, status=400
            )
        status = 202 if result.get("accepted") else 409
        return web.json_response(result, status=status)

    async def _topology_abort(_request):
        result = ctrl.abort_topology()
        status = 202 if result.get("accepted") else 409
        return web.json_response(result, status=status)

    app = web.Application()
    app.router.add_post("/v1/completions", ctrl.handle_completion)
    app.router.add_post("/v1/reset", _reset)
    app.router.add_get("/v1/topology", _topology_get)
    app.router.add_post("/v1/topology", _topology_post)
    app.router.add_post("/v1/topology/abort", _topology_abort)
    app.router.add_get("/v1/models",
                       lambda r: web.json_response(dict(data=[dict(id="ecospd")])))
    ctrl.loop = asyncio.get_running_loop()
    threading.Thread(target=ctrl._dvfs_loop, daemon=True).start()
    threading.Thread(target=ctrl._periodic_loop, daemon=True).start()
    if ctrl.rematerialization_enabled:
        threading.Thread(target=ctrl._topology_loop, daemon=True).start()
    asyncio.get_running_loop().create_task(ctrl._pump())

    def _sig(signum, frame):
        ctrl.shutdown()
        os._exit(0)
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()
    print("[ecospd] ready :%d mixed=%d segments=%d"
          % (args.port, len(ctrl.mixed), len(ctrl.segments)), flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        ctrl.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
