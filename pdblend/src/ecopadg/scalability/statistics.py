"""Deterministic statistics for measured scalability evidence.

Capacity stability uses an OLS trend with a Newey--West (Bartlett) covariance,
30 seconds of autocorrelation lags, and a one-sided 95% normal upper limit.
The input must cover the final 300 seconds of the arrival window; drain is not
included in this test. Paired confidence intervals resample complete seed pairs.
"""
import math

import numpy as np


def quantiles(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {f"p{p}": float(np.percentile(values, p)) if values else None
            for p in (50, 95, 99)}


def backlog_stability(rows, arrival_end_s, *, window_s=300., max_gap_s=5.):
    result = dict(slope_rps=None, upper_rps=None, sample_count=0, valid=False,
                  method="OLS with Newey-West Bartlett HAC, 30-second lags; one-sided 95% upper limit",
                  window_s=window_s, error=None)
    try:
        pairs = [(float(r["t_s"]), float(r["pending"])) for r in rows]
        if any(not math.isfinite(t) or not math.isfinite(y) or y < 0 for t, y in pairs):
            raise ValueError("invalid backlog sample")
        if any(b[0] <= a[0] for a, b in zip(pairs, pairs[1:])):
            raise ValueError("backlog timestamps must strictly increase")
        lo = float(arrival_end_s) - window_s
        selected = [(t, y) for t, y in pairs if lo <= t <= arrival_end_s]
        result["sample_count"] = len(selected)
        if len(selected) < 60:
            raise ValueError("at least 60 backlog samples required")
        t, y = np.asarray(selected, dtype=float).T
        gaps = np.diff(t)
        if t[0] - lo > max_gap_s or arrival_end_s - t[-1] > max_gap_s or max(gaps) > max_gap_s:
            raise ValueError("backlog does not cover the final arrival window")
        x = np.column_stack((np.ones(len(t)), t - t.mean()))
        inv = np.linalg.inv(x.T @ x)
        beta = inv @ x.T @ y
        residuals = y - x @ beta
        score = x * residuals[:, None]
        meat = score.T @ score
        lag = min(len(t) // 4, max(1, int(math.ceil(30. / np.median(gaps)))))
        for k in range(1, lag + 1):
            cross = score[k:].T @ score[:-k]
            meat += (1. - k / (lag + 1.)) * (cross + cross.T)
        covariance = inv @ meat @ inv * len(t) / (len(t) - 2)
        standard_error = math.sqrt(max(0., float(covariance[1, 1])))
        result.update(valid=True, slope_rps=float(beta[1]),
                      upper_rps=float(beta[1] + 1.6448536269514722 * standard_error),
                      standard_error_rps=standard_error, hac_lags=lag)
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
        result["error"] = str(exc)
    return result


def paired_ratio_interval(numerators, denominators, *, scale=1., repetitions=5000, seed=20260910):
    """Ratio of paired means, with a percentile 95% bootstrap interval.

Missing pairs are an input error: callers must explicitly disclose excluded
seeds rather than silently substituting or independently resampling systems.
"""
    a, b = np.asarray(numerators, dtype=float), np.asarray(denominators, dtype=float)
    if (a.ndim != 1 or a.shape != b.shape or not len(a) or scale <= 0
            or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))
            or np.any(a < 0) or np.any(b <= 0)):
        raise ValueError("finite nonnegative numerators and positive paired denominators required")
    estimate = float(a.mean() / (scale * b.mean()))
    result = dict(estimate=estimate, ci95_low=None, ci95_high=None, paired_seeds=len(a),
                  method="ratio of seed-paired means; percentile paired bootstrap, 5000 resamples",
                  bootstrap_seed=seed)
    if len(a) >= 2:
        indices = np.random.default_rng(seed).integers(0, len(a), size=(repetitions, len(a)))
        ratios = a[indices].mean(axis=1) / (scale * b[indices].mean(axis=1))
        result.update(ci95_low=float(np.quantile(ratios, .025)),
                      ci95_high=float(np.quantile(ratios, .975)))
    return result


def capacity_interval(rows, *, tolerance=.05):
    """A per-seed observed pass/fail bracket; invalid runs never bound capacity."""
    valid = [r for r in rows if r.get("measurement_valid") and r.get("stage") == "capacity"]
    grouped = {}
    for row in valid:
        grouped.setdefault(float(row["rate_rps"]), []).append(bool(row.get("capacity_pass")))
    # Repeated disagreement and a pass above a failure invalidate monotonic bracketing.
    inconsistent = any(len(set(outcomes)) > 1 for outcomes in grouped.values())
    passes = [rate for rate, outcomes in grouped.items() if all(outcomes)]
    fails = [rate for rate, outcomes in grouped.items() if not all(outcomes)]
    lower, upper = max(passes, default=None), min(fails, default=None)
    inconsistent |= lower is not None and upper is not None and lower >= upper
    relative_width = (upper / lower - 1. if lower and upper and not inconsistent else None)
    return dict(capacity_lower_rps=lower, capacity_upper_rps=upper,
                bracket_relative_width=relative_width, inconsistent=inconsistent,
                bracket_complete=bool(relative_width is not None and relative_width <= tolerance + 1e-12),
                valid_runs=len(valid), invalid_runs=sum(not r.get("measurement_valid", False) for r in rows),
                lower_censored=lower is None, upper_censored=upper is None)
