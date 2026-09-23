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
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from .index import finite
from .decode import predict as decode_predict, supported as decode_supported, boundary as decode_boundary

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
    power_repeats: tuple[float, ...] = ()
    steady_window_s: float = 0.0


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
    quality: dict = field(default_factory=dict)
    decode_overrides: dict[int, dict] = field(default_factory=dict)
    # Layout identity is optional for historical synthetic fixtures.  Keeping
    # these fields after legacy defaults preserves positional construction.
    pp: int = 1
    system: str = "pdblend"
    profile_key: dict = field(default_factory=dict)
    bounded_coverage: dict = field(default_factory=dict)
    decode_power_overrides: dict[int, dict] = field(default_factory=dict)

    def __setattr__(self, name, value):
        # Compile on construction/publication, never while serving a query.
        if name == 'freqs':
            from .index import FrequencyIndex
            object.__setattr__(self, '_frequency_index', FrequencyIndex(value))
        elif name == 'decode_power_overrides':
            from .power_table import PowerTableMap
            value = PowerTableMap(value)
        elif name in ('prefill_time', 'prefill_power', 'decode_time', 'decode_power', 'decode_overrides'):
            from .index import FrequencyMap
            value = FrequencyMap(value)
        object.__setattr__(self, name, value)

    def __post_init__(self):
        self.query_qualification  # Reject an oversized aggregate at publication.

    def __getstate__(self):
        from dataclasses import fields
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def __setstate__(self, state):
        from dataclasses import fields
        # Old pickles carry plain dictionaries and no compiled directories.
        # Re-run normal initialization once; never lazily compile on a query.
        self.__init__(**{field.name: state[field.name] for field in fields(self) if field.name in state})

    @property
    def query_qualification(self):
        from .index import IndexQualificationError, MAX_INDEX_BYTES
        size = self._frequency_index.bytes + sum(
            table.bytes for table in self.decode_power_overrides.index if table is not None) + 8 * (
                len(self.decode_power_overrides.index) + sum(len(getattr(self, name).slots) for name in (
                    'prefill_time', 'prefill_power', 'decode_time', 'decode_power', 'decode_overrides')))
        if size > MAX_INDEX_BYTES:
            raise IndexQualificationError('O(1) complete profile index memory limit exceeded')
        return dict(qualified=True, complexity='O(1)', max_breakpoints_per_bucket=4,
                    index_bytes=size,
                    compiled_at='construction_or_table_assignment')

    # ---- queries -------------------------------------------------------------------------
    def nearest_freq(self, f: int) -> int:
        return self._frequency_index.nearest(f)

    def prefill_seconds(self, n: int, f: int) -> float:
        if not finite(n): raise ValueError('missing_profile: non-finite prefill tokens')
        if self.bounded_coverage:
            bounds = self.bounded_coverage['prefill_tokens']
            if not self._frequency_index.contains(f) or not bounds[0] <= n <= bounds[1]:
                raise ValueError('prefill query outside measured coverage')
        a, b, c = self.prefill_time[self.nearest_freq(f)]
        return max(a + b * n + c * n * n, 1e-4)

    def prefill_power_w(self, n: int, f: int) -> float:
        if not finite(n): raise ValueError('missing_profile: non-finite prefill tokens')
        p0, p1 = self.prefill_power[self.nearest_freq(f)]
        return p0 + p1 * min(n, PREFILL_SATURATION) / PREFILL_SATURATION

    def prefill_energy_j(self, n: int, f: int) -> float:
        return self.prefill_seconds(n, f) * self.prefill_power_w(n, f)

    def prefill_marginal_seconds(self, n: int, f: int) -> float:
        """Extra step time when n prompt tokens ride along in a decode step (no per-step intercept)."""
        if not finite(n): raise ValueError('missing_profile: non-finite prefill tokens')
        _, b, c = self.prefill_time[self.nearest_freq(f)]
        return max(b * n + c * n * n, 0.0)

    def step_seconds(self, batch: float, ctx: float, f: int) -> float:
        if not finite(batch) or not finite(ctx): raise ValueError('missing_profile: non-finite decode shape')
        if self.bounded_coverage and not self.decode_supported(batch, ctx, f):
            raise ValueError('decode query outside measured coverage')
        frequency = self.nearest_freq(f)
        alpha, beta, gamma, delta = self.decode_time[frequency]
        legacy = max(alpha + beta * batch + gamma * batch * ctx + delta * batch * batch, 1e-4)
        override = self.decode_overrides.get(frequency)
        if not override:
            return legacy
        value = max(decode_predict(override, batch, ctx), 1e-4)
        if decode_supported(override, batch, ctx):
            return value
        b, c = decode_boundary(override, batch, ctx)
        return max(legacy, value, decode_predict(override, b, c))

    def decode_supported(self, batch: float, ctx: float, f: int) -> bool:
        if not finite(batch) or not finite(ctx) or not finite(f): return False
        if self.bounded_coverage and not self._frequency_index.contains(f):
            return False
        override = self.decode_overrides.get(self.nearest_freq(f))
        if not override:
            return True  # Legacy behavior and baseline admission are unchanged.
        return decode_supported(override, batch, ctx)

    def decode_power_supported(self, batch: float, ctx: float, f: int) -> bool:
        if not finite(batch) or not finite(ctx) or not finite(f): return False
        if not self.decode_power_overrides:
            return True  # Historical affine profiles retain their original API.
        from pdblend.profile.query.power_table import PowerCoverageError
        try:
            self.decode_power_overrides.at(f).predict(batch, ctx)
            return True
        except PowerCoverageError:
            return False

    def decode_power_w(self, batch: float, f: int, *, ctx: float | None = None) -> float:
        if not finite(batch) or (ctx is not None and not finite(ctx)):
            from .power_table import PowerCoverageError
            raise PowerCoverageError('missing_profile: non-finite decode power shape')
        if self.decode_power_overrides:
            from pdblend.profile.query.power_table import PowerCoverageError
            return self.decode_power_overrides.at(f).predict(batch, ctx)
        q0, q1 = self.decode_power[self.nearest_freq(f)]
        return q0 + q1 * batch

    def token_energy_j(self, batch: float, ctx: float, f: int) -> float:
        """Energy per generated token at steady-state batch B."""
        return self.step_seconds(batch, ctx, f) * self.decode_power_w(batch, f, ctx=ctx) / max(batch, 1e-6)

    def static_power_w(self, state: str, f: Optional[int] = None) -> float:
        key = f"active_idle@{self.nearest_freq(f)}" if state == "active_idle" else state
        if key == "active_idle_reset" and key not in self.static:
            key = f"active_idle@{self._frequency_index.axis.low}"
        return self.static[key].power_w

    def wake_seconds(self, state: str) -> float:
        return self.static[state].wake_s if state in self.static else 0.0

    def transfer_seconds(self, tokens: int) -> float:
        fixed, bw = self.transfer
        return fixed + tokens * self.kv_bytes_per_token / bw

    # ---- (de)serialisation ------------------------------------------------------------------
    def to_json(self) -> str:
        d = asdict(self)
        if not self.decode_overrides:
            d.pop('decode_overrides', None)
        if not self.decode_power_overrides:
            d.pop('decode_power_overrides', None)
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
        d['decode_overrides'] = {int(k): v for k, v in d.get('decode_overrides', {}).items()}
        for spec in d['decode_overrides'].values():
            from pdblend.profile.query.decode import predict
            if any(not np.isfinite(x) or (x < 0 and spec['kind'] != 'split_b1_relative') for x in spec['coefficients']):
                raise ValueError('invalid decode override coefficients')
            predict(spec, 1, 1)  # Reject unknown models instead of silently falling back.
        d['decode_power_overrides'] = {int(k): v for k, v in d.get('decode_power_overrides', {}).items()}
        if d['decode_power_overrides']:
            from pdblend.profile.query.power_table import validate
            if set(d['decode_power_overrides']) != set(d['freqs']):
                raise ValueError('decode power override must cover every declared frequency')
            for spec in d['decode_power_overrides'].values():
                validate(spec)
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


def _quality(pred: np.ndarray, y: np.ndarray) -> dict:
    rel = np.abs(pred - y) / np.maximum(np.abs(y), 1e-9)
    return dict(mape=float(np.mean(rel)) if len(rel) else 0.0,
                max=float(np.max(rel)) if len(rel) else 0.0,
                samples=int(len(rel)))


def fit(prefill: Iterable[PrefillPoint], decode: Iterable[DecodePoint], static: dict[str, StaticState],
        transfer_points: Iterable[tuple[int, float]] = (), kv_bytes_per_token: int = 0,
        kv_capacity_tokens: int = 0, freq_switch_s: float = 0.15, model: str = "", tp: int = 1,
        pipeline_parallel: int = 1, system: str = "pdblend", profile_key: dict | None = None) -> PerfModel:
    prefill, decode = list(prefill), list(decode)
    freqs = tuple(sorted({p.freq_mhz for p in prefill} | {d.freq_mhz for d in decode}))
    pt, pp, dt, dp, residuals, quality = {}, {}, {}, {}, {}, {}
    for f in freqs:
        ps = [p for p in prefill if p.freq_mhz == f]
        if ps:
            n = np.array([p.input_tokens for p in ps], float)
            coef, r = _lstsq(np.stack([np.ones_like(n), n, n * n], 1), np.array([p.seconds for p in ps]))
            pt[f] = tuple(map(float, coef))
            residuals[f"prefill_time@{f}"] = r
            quality[f"prefill_time@{f}"] = _quality(np.stack([np.ones_like(n), n, n * n], 1) @ coef,
                                                       np.array([p.seconds for p in ps]))
            sat = np.minimum(n, PREFILL_SATURATION) / PREFILL_SATURATION
            coef, r = _lstsq(np.stack([np.ones_like(n), sat], 1), np.array([p.power_w for p in ps]))
            pp[f] = tuple(map(float, coef))
            residuals[f"prefill_power@{f}"] = r
            quality[f"prefill_power@{f}"] = _quality(np.stack([np.ones_like(n), sat], 1) @ coef,
                                                       np.array([p.power_w for p in ps]))
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
            power_ds, excluded = [], []
            for d in ds:
                repeats = tuple(float(x) for x in d.power_repeats)
                cv = (statistics.stdev(repeats) / statistics.mean(repeats)
                      if len(repeats) >= 2 and statistics.mean(repeats) > 0 else 0.0)
                if d.batch < 4 or cv > 0.10:
                    excluded.append(dict(batch=d.batch, context_tokens=d.context_tokens, cv=cv,
                        reason="diagnostic_b1" if d.batch < 4 else "unstable_repeats"))
                else:
                    power_ds.append(d)
            if len({d.batch for d in power_ds}) < 2:
                # Tiny synthetic/raw fixtures and interrupted profiles may only contain
                # diagnostics. Keep them loadable; production audit will reject the
                # insufficient planner-quality coverage.
                power_ds = ds
                excluded = excluded + [dict(reason="insufficient_stable_batches", fallback=True)]
            PB = np.array([d.batch for d in power_ds], float)
            Py = np.array([d.power_w for d in power_ds])
            pX = np.stack([np.ones_like(PB), PB], 1)
            coef, r = _lstsq(pX, Py)
            dp[f] = tuple(map(float, coef))
            residuals[f"decode_power@{f}"] = r
            quality[f"decode_time@{f}"] = _quality(np.stack([np.ones_like(B), B, B * ctx, B * B], 1) @ dt[f], y)
            quality[f"decode_power@{f}"] = dict(_quality(pX @ coef, Py), excluded=excluded)
            diagnostics = [d for d in ds if d.batch == 1]
            if diagnostics:
                quality[f"decode_power_b1@{f}"] = _quality(
                    np.array([coef[0] + coef[1] for d in diagnostics]),
                    np.array([d.power_w for d in diagnostics]))
    transfer = (0.0, 10e9)
    tps = list(transfer_points)
    if len(tps) >= 2 and kv_bytes_per_token:
        b = np.array([t * kv_bytes_per_token for t, _ in tps], float)
        coef, r = _lstsq(np.stack([np.ones_like(b), b], 1), np.array([s for _, s in tps]))
        transfer = (max(float(coef[0]), 0.0), 1.0 / max(float(coef[1]), 1e-12))
        residuals["transfer"] = r
        quality["transfer"] = _quality(np.stack([np.ones_like(b), b], 1) @ coef,
                                         np.array([s for _, s in tps]))
    elif len(tps) == 1 and kv_bytes_per_token:
        transfer = (0.0, tps[0][0] * kv_bytes_per_token / max(tps[0][1], 1e-6))
    return PerfModel(freqs=freqs, prefill_time=pt, prefill_power=pp, decode_time=dt,
                     decode_power=dp, static=dict(static), transfer=transfer,
                     freq_switch_s=freq_switch_s, kv_bytes_per_token=kv_bytes_per_token,
                     kv_capacity_tokens=kv_capacity_tokens, residuals=residuals, model=model,
                     tp=tp, pp=pipeline_parallel, system=system,
                     profile_key=profile_key or {}, quality=quality)
