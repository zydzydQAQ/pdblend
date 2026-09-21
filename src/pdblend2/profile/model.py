"""Affine performance/power model fitted from a small profiling grid.

Per frequency tier f:
  prefill time    t_p(n)      = a + b*n + c*n^2         (n prompt tokens)
  prefill power   P_p(n)      = p0 + p1*min(n, PREFILL_SATURATION)/PREFILL_SATURATION
  decode step     t_d(B, ctx) = alpha + beta*B + gamma*B*ctx + delta*B^2   (delta >= 0: compute-bound knee)
  decode power    P_d(B)      = q0 + q1*B
Static states (active_idle per f, active_idle_reset, parked, off) carry a power and a wake latency.
KV transfer time  t_x(bytes) = x0 + bytes/bandwidth.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

PREFILL_SATURATION = 1024


@dataclass
class PrefillPoint:
    freq_mhz: int
    input_tokens: int
    seconds: float
    power_w: float


@dataclass
class DecodePoint:
    freq_mhz: int
    batch: int
    context_tokens: int
    step_seconds: float
    power_w: float


@dataclass
class StaticState:
    power_w: float
    wake_s: float = 0.0


@dataclass
class PerfModel:
    freqs: tuple[int, ...]
    prefill_time: dict[int, tuple[float, float, float]]
    prefill_power: dict[int, tuple[float, float]]
    decode_time: dict[int, tuple[float, float, float, float]]
    decode_power: dict[int, tuple[float, float]]
    static: dict[str, StaticState]                  # "active_idle@2520", "active_idle_reset", "parked", "off"
    transfer: tuple[float, float] = (0.0, 10e9)     # (fixed_s, bytes_per_s)
    freq_switch_s: float = 0.15
    kv_bytes_per_token: int = 0
    kv_capacity_tokens: int = 0
    residuals: dict[str, float] = field(default_factory=dict)
    model: str = ""
    tp: int = 1

    # ---- queries -------------------------------------------------------------------------
    def nearest_freq(self, f: int) -> int:
        return min(self.freqs, key=lambda x: abs(x - f))

    def prefill_seconds(self, n: int, f: int) -> float:
        a, b, c = self.prefill_time[self.nearest_freq(f)]
        return max(a + b * n + c * n * n, 1e-4)

    def prefill_power_w(self, n: int, f: int) -> float:
        p0, p1 = self.prefill_power[self.nearest_freq(f)]
        return p0 + p1 * min(n, PREFILL_SATURATION) / PREFILL_SATURATION

    def prefill_energy_j(self, n: int, f: int) -> float:
        return self.prefill_seconds(n, f) * self.prefill_power_w(n, f)

    def prefill_marginal_seconds(self, n: int, f: int) -> float:
        """Extra step time when n prompt tokens ride along in a decode step (no per-step intercept)."""
        _, b, c = self.prefill_time[self.nearest_freq(f)]
        return max(b * n + c * n * n, 0.0)

    def step_seconds(self, batch: float, ctx: float, f: int) -> float:
        alpha, beta, gamma, delta = self.decode_time[self.nearest_freq(f)]
        return max(alpha + beta * batch + gamma * batch * ctx + delta * batch * batch, 1e-4)

    def decode_power_w(self, batch: float, f: int) -> float:
        q0, q1 = self.decode_power[self.nearest_freq(f)]
        return q0 + q1 * batch

    def token_energy_j(self, batch: float, ctx: float, f: int) -> float:
        """Energy per generated token at steady-state batch B."""
        return self.step_seconds(batch, ctx, f) * self.decode_power_w(batch, f) / max(batch, 1e-6)

    def static_power_w(self, state: str, f: Optional[int] = None) -> float:
        key = f"active_idle@{self.nearest_freq(f)}" if state == "active_idle" else state
        if key == "active_idle_reset" and key not in self.static:
            key = f"active_idle@{min(self.freqs)}"
        return self.static[key].power_w

    def wake_seconds(self, state: str) -> float:
        return self.static[state].wake_s if state in self.static else 0.0

    def transfer_seconds(self, tokens: int) -> float:
        fixed, bw = self.transfer
        return fixed + tokens * self.kv_bytes_per_token / bw

    # ---- (de)serialisation ------------------------------------------------------------------
    def to_json(self) -> str:
        d = asdict(self)
        for key in ("prefill_time", "prefill_power", "decode_time", "decode_power"):
            d[key] = {str(k): list(v) for k, v in d[key].items()}
        return json.dumps(d, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "PerfModel":
        d = json.loads(text)
        for key in ("prefill_time", "prefill_power", "decode_time", "decode_power"):
            d[key] = {int(k): tuple(v) for k, v in d[key].items()}
        d["decode_time"] = {k: v + (0.0,) * (4 - len(v)) for k, v in d["decode_time"].items()}
        d["static"] = {k: StaticState(**v) for k, v in d["static"].items()}
        d["freqs"] = tuple(d["freqs"])
        d["transfer"] = tuple(d["transfer"])
        return cls(**d)

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> "PerfModel":
        return cls.from_json(Path(path).read_text())


def _lstsq(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    rel = np.abs(pred - y) / np.maximum(np.abs(y), 1e-9)
    return coef, float(rel.max()) if len(y) else 0.0


def fit(prefill: Iterable[PrefillPoint], decode: Iterable[DecodePoint], static: dict[str, StaticState],
        transfer_points: Iterable[tuple[int, float]] = (), kv_bytes_per_token: int = 0,
        kv_capacity_tokens: int = 0, freq_switch_s: float = 0.15, model: str = "", tp: int = 1) -> PerfModel:
    prefill, decode = list(prefill), list(decode)
    freqs = tuple(sorted({p.freq_mhz for p in prefill} | {d.freq_mhz for d in decode}))
    pt, pp, dt, dp, residuals = {}, {}, {}, {}, {}
    for f in freqs:
        ps = [p for p in prefill if p.freq_mhz == f]
        if ps:
            n = np.array([p.input_tokens for p in ps], float)
            coef, r = _lstsq(np.stack([np.ones_like(n), n, n * n], 1), np.array([p.seconds for p in ps]))
            pt[f] = tuple(map(float, coef))
            residuals[f"prefill_time@{f}"] = r
            sat = np.minimum(n, PREFILL_SATURATION) / PREFILL_SATURATION
            coef, r = _lstsq(np.stack([np.ones_like(n), sat], 1), np.array([p.power_w for p in ps]))
            pp[f] = tuple(map(float, coef))
            residuals[f"prefill_power@{f}"] = r
        ds = [d for d in decode if d.freq_mhz == f]
        if ds:
            B = np.array([d.batch for d in ds], float)
            ctx = np.array([d.context_tokens for d in ds], float)
            y = np.array([d.step_seconds for d in ds])
            coef, r = _lstsq(np.stack([np.ones_like(B), B, B * ctx, B * B], 1), y)
            if coef[3] < 0 or len(set(B)) < 4:
                coef, r = _lstsq(np.stack([np.ones_like(B), B, B * ctx], 1), y)
                coef = np.append(coef, 0.0)
            dt[f] = tuple(map(float, coef))
            residuals[f"decode_time@{f}"] = r
            coef, r = _lstsq(np.stack([np.ones_like(B), B], 1), np.array([d.power_w for d in ds]))
            dp[f] = tuple(map(float, coef))
            residuals[f"decode_power@{f}"] = r
    transfer = (0.0, 10e9)
    tps = list(transfer_points)
    if len(tps) >= 2 and kv_bytes_per_token:
        b = np.array([t * kv_bytes_per_token for t, _ in tps], float)
        coef, r = _lstsq(np.stack([np.ones_like(b), b], 1), np.array([s for _, s in tps]))
        transfer = (max(float(coef[0]), 0.0), 1.0 / max(float(coef[1]), 1e-12))
        residuals["transfer"] = r
    elif len(tps) == 1 and kv_bytes_per_token:
        transfer = (0.0, tps[0][0] * kv_bytes_per_token / max(tps[0][1], 1e-6))
    return PerfModel(freqs, pt, pp, dt, dp, dict(static), transfer, freq_switch_s,
                     kv_bytes_per_token, kv_capacity_tokens, residuals, model, tp)
