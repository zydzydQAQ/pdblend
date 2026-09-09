# -*- coding: utf-8 -*-
"""核心数据结构:Instance / Segment / Partition / SLO / SystemConfig。"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

ROLE_PREFILL = "prefill"
ROLE_DECODE = "decode"
ROLE_MIXED = "mixed"
ROLES = (ROLE_PREFILL, ROLE_DECODE, ROLE_MIXED)
INSTANCE_KINDS = (ROLE_MIXED, ROLE_PREFILL, ROLE_DECODE)

# DistServe goodput: att≥90% 才允许 DVFS / 用能耗排布局
GOODPUT_ATT_GATE = 0.90
# 膝点带宽:可信 baseline ∈ [τ, τ+δ) 时关 slack。τ 是地板不是膝点保护。
GOODPUT_KNEE_BAND_PP = 0.02
# P/max(goodput,1e-9) 在 att→0 时会炸到 1e13;超过此值禁止用来排名
ENERGY_RANK_MAX_J_PER_S = 1e8
# 同负载 SLO 非劣:单 seed 点估计不低于 baseline − 1pp
SLO_NON_INFERIOR_PP = 0.01
# 缩容(少开卡)额外裕度:pred_att ≥ att_mixed + 3pp 且 ρ < 0.4 才许 shrink
SHRINK_ATT_MARGIN_PP = 0.03
SHRINK_RHO_MAX = 0.4
# 搜索与运行时禁止把 prefill 降到满频以下(32B r=3.4 根因)
PREFILL_PIN_MHZ = 2520
ENERGY_PAD_S = 2.0
PLANNER_MODE_ISO_LOAD = "min_j_s.t. slo_non_inferior"
PLANNER_MODE_CAPACITY = "capacity-envelope"

DEFAULT_FREQS = (2520, 2100, 1800, 1500, 1200, 1050, 900, 600)
FREQS_14B = (2520, 1500, 900)
FREQS_72B = (2520, 1800, 1500, 1200, 1050)
FREQ_TRUST_RATIO = 1.15  # iter(fmin)/iter(fmax) 低于此则频率模型不可信


def freqs_for_model(model_key: str) -> Tuple[int, ...]:
    """忙时候选频档:14b 不含 600,与 choose_partition 对齐。"""
    key = str(model_key).lower()
    if key in ("14b", "14"):
        return FREQS_14B
    if key in ("72b", "72"):
        return FREQS_72B
    return DEFAULT_FREQS

# 统一 batch / token 口径(选频、预测、P 池、窗口、后处理共用;禁止再写 4/4096)
# TOKEN_BUDGET:P 池 DistServe 组批上限(Σtok);不是 prefill 的独立 K 维
B_REF = 8
TOKEN_BUDGET = 8192
DECODE_B_CAP = 48
CTX_REF = 272.0
OUT_REF = 200.0
RHO_TPUT = 0.7
LOAD_CAP_PER_INST = 32          # 路由负载分母(并发请求上限代理)
KV_XFER_S_PER_TOK = 2.0e-6     # P4 量级:KV 传输时延代理,不进方法定义
# B5 仅 4096 token 墙钟可信。短于该长度的传输时延记 0 并标 untrusted。
# 只作 TTFT 约束,不进能耗目标;默认放置同 NUMA。
TRUSTED_KV_TOKENS = 4096
KV_TTFT_S_SAME_NUMA = 0.0123
KV_TTFT_S_CROSS_NUMA = 0.0283

# 两层 SLO。发表主表冻死为用户档；硬件档只解释 72B / 相对裕度，不覆盖 09-01。
SLO_TIER_USER = "user"
SLO_TIER_HW = "hw"
PUBLICATION_SLO_TIER = SLO_TIER_USER
HW_SLO_MULT = 5.0
_SLO_HW_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "slo_hw.json")

# 用户档：TTFT 按数据集；TPOT 按数据集阅读速率（不再三套共用 100 ms）
USER_DATASET_SLO: Dict[str, Tuple[float, float]] = {
    "sharegpt": (5.0, 0.15),
    "longbench": (15.0, 0.2),
    "alpaca": (1.0, 0.1),
}


def normalize_dataset(name: str) -> str:
    key = str(name).strip().lower().replace("_", "-")
    if key in ("alpaca-gpt4", "alpacagpt4", "alpaca"):
        return "alpaca"
    if key in ("long-bench", "longbench"):
        return "longbench"
    if key in ("share-gpt", "sharegpt"):
        return "sharegpt"
    return key


def model_key_of(model: str) -> str:
    s = str(model).strip().lower()
    if "72" in s:
        return "72b"
    if "32" in s:
        return "32b"
    if "14" in s:
        return "14b"
    if s in ("7b", "7"):
        return "7b"
    return s


def default_tp_of(model: str) -> int:
    key = model_key_of(model)
    if key == "72b":
        return 4
    return 2


def load_slo_hw(path: Optional[str] = None) -> Dict[str, Any]:
    """读 5× 空载标定表。缺文件则空 dict，不抛。"""
    p = path or os.environ.get("SLO_HW_JSON") or _SLO_HW_JSON
    if not p or not os.path.isfile(p):
        return {}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


@dataclass(frozen=True)
class SloSpec:
    """TTFT/TPOT SLO。默认用户档 ShareGPT（5s / 0.15s）。"""
    ttft_s: float = 5.0
    tpot_s: float = 0.15

    @classmethod
    def from_dataset(cls, dataset: str) -> "SloSpec":
        """用户档：TTFT 与 TPOT 都按数据集（Alpaca 100 / ShareGPT 150 / LongBench 200 ms）。"""
        key = normalize_dataset(dataset)
        if key not in USER_DATASET_SLO:
            raise KeyError("未知数据集 SLO: %s" % dataset)
        ttft, tpot = USER_DATASET_SLO[key]
        return cls(ttft_s=float(ttft), tpot_s=float(tpot))

    @classmethod
    def from_hw(cls, model: str, tp: int, dataset: str,
                table: Optional[Mapping[str, Any]] = None) -> "SloSpec":
        """硬件档：5× 空载 ITL / 该数据集典型 prompt TTFT。"""
        data = dict(table) if table is not None else load_slo_hw()
        mkey = model_key_of(model)
        ds = normalize_dataset(dataset)
        models = data.get("models") or {}
        rec = models.get(mkey) or models.get("%s-tp%s" % (mkey, tp))
        if not rec:
            raise KeyError("slo_hw 无模型 %s tp=%s" % (mkey, tp))
        rec_tp = int(rec.get("tp") or tp)
        if rec_tp != int(tp):
            raise KeyError("slo_hw %s 的 tp=%s，不是 %s" % (mkey, rec_tp, tp))
        tpot = rec.get("tpot_hw_s")
        ttfts = rec.get("ttft_hw_s") or {}
        ttft = ttfts.get(ds)
        if tpot is None or ttft is None:
            raise KeyError("slo_hw 缺 %s/tp%s/%s" % (mkey, tp, ds))
        return cls(ttft_s=float(ttft), tpot_s=float(tpot))

    @classmethod
    def from_tier(cls, tier: str, dataset: str,
                  model: str = "", tp: int = 0) -> "SloSpec":
        t = str(tier or PUBLICATION_SLO_TIER).strip().lower()
        if t in ("", SLO_TIER_USER, "user-tier"):
            return cls.from_dataset(dataset)
        if t in (SLO_TIER_HW, "hw-tier", "hardware"):
            m = model or "32b"
            ntp = int(tp) or default_tp_of(m)
            return cls.from_hw(m, ntp, dataset)
        raise KeyError("未知 SLO 档: %s" % tier)


@dataclass
class InstanceSpec:
    """一个 vLLM 实例规格。"""
    role: str
    model: str
    tp: int = 1
    gpus: Tuple[int, ...] = ()
    max_model_len: int = 8192
    gpu_mem_util: float = 0.85

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError("未知实例角色: %r" % self.role)


@dataclass
class Segment:
    """DistServe 的段:1 个 prefill 实例 + 1 个 decode 实例(1P1D)。"""
    prefill: InstanceSpec
    decode: InstanceSpec

    def gpus(self) -> int:
        return self.prefill.tp + self.decode.tp


@dataclass
class Partition:
    """PD 池划分:(n_mixed, n_prefill, n_decode),各角色 TP。"""
    n_mixed: int = 0
    n_prefill: int = 0
    n_decode: int = 0
    tp_mixed: int = 1
    tp_prefill: int = 1
    tp_decode: int = 1
    pp_mixed: int = 1
    pp_prefill: int = 1
    pp_decode: int = 1
    token_budget_mixed: int = TOKEN_BUDGET
    prefill_max_tokens: int = TOKEN_BUDGET
    decode_max_batch: int = DECODE_B_CAP
    freq_mixed: int = 2520
    freq_prefill: int = 2520
    freq_decode: int = 2520
    gpus_total: int = 8

    def total_gpus(self) -> int:
        return (self.n_mixed * self.tp_mixed * self.pp_mixed
                + self.n_prefill * self.tp_prefill * self.pp_prefill
                + self.n_decode * self.tp_decode * self.pp_decode)

    def pairs(self) -> int:
        """PD 对数(1P1D 段数)。"""
        return min(self.n_prefill, self.n_decode)

    def __repr__(self) -> str:
        return ("Partition(mixed=%d,tp%dxpp%d prefill=%d,tp%dxpp%d "
                "decode=%d,tp%dxpp%d | %d/%d gpus)"
                % (self.n_mixed, self.tp_mixed, self.pp_mixed,
                   self.n_prefill, self.tp_prefill, self.pp_prefill,
                   self.n_decode, self.tp_decode, self.pp_decode,
                   self.total_gpus(), self.gpus_total))


@dataclass
class SystemConfig:
    """系统运行配置(SLO、频率档、窗口/迟滞参数)。"""
    model: str
    gpu_count: int = 8
    freq_candidates: Tuple[int, ...] = DEFAULT_FREQS
    slo: SloSpec = field(default_factory=SloSpec)
    # 实测 p90/模型均值比 ~1.3+(批增长、上下文分布、gate 抖动),
    # 0.15 裕量必然 p90 贴边违约;0.35 → ShareGPT 目标约 98ms(SLO 150ms)。
    tpot_margin: float = 0.35
    prefill_freq: int = 2520
    window_min_s: float = 3.0
    window_max_s: float = 20.0
    global_period_s: float = 30.0
    min_dwell_s: float = 3.0
    partition_min_dwell_s: float = 120.0
    drain_timeout_s: float = 60.0
    hysteresis: float = 0.15
    target_attainment: float = 0.9  # 仅附录/容量包络;选配不用绝对门槛
    attainment_tolerance_pp: float = 0.03  # 兼容旧测试;选配用 slo_non_inferior_pp
    slo_non_inferior_pp: float = SLO_NON_INFERIOR_PP
    baseline_att: float = float("nan")  # 未测则周期调度回退 target_attainment
    migration_cost_j: float = 2000.0  # park/unpark only
    role_switch_cost_j: float = float("nan")  # restart; unknown blocks P2
    downscale_headroom: float = 0.3
    force_continuous: bool = False   # M0-D:只做 slack DVFS,不开时间窗
    decode_floor_mhz: int = 0        # 72B@100ms=2520:关闭 decode 降频
    lookup_decode_freq: int = 0      # 查表 f* 甜点;0=不用,可行则优先
    freq_model_trusted: bool = True  # False:忙时不降频、planner 不缩容
    # D 级旋钮:不进方法定义。slack 折扣/帽/过载阈见 online_scheduler。

    def __post_init__(self):
        if not 0.0 <= self.tpot_margin < 1.0:
            raise ValueError("tpot_margin 必须在 [0,1)")
        if not 0.0 <= self.hysteresis < 1.0:
            raise ValueError("hysteresis 必须在 [0,1)")
