"""Shared synthetic PerfModel for pdblend CPU tests."""
from pdblend.control.forecast import Forecast
from pdblend.profile.model import PerfModel, StaticState

FREQS = (900, 1200, 1500, 1800, 2100, 2520)


def synthetic_model() -> PerfModel:
    pt, pp, dt, dp, st = {}, {}, {}, {}, {}
    for f in FREQS:
        k = 2520 / f
        pt[f] = (0.01 * k, 4e-5 * k, 1e-9 * k)
        pp[f] = (90.0, 180.0 / k)
        dt[f] = (0.012 * k, 2e-4 * k, 3e-8 * k, 2e-7 * k)
        dp[f] = (95.0, 2.0 / k)
        st[f"active_idle@{f}"] = StaticState(60 + 8 * (f - 900) / 1620)
    st["parked"] = StaticState(35.0, 0.05)
    st["off"] = StaticState(34.0, 30.0)
    return PerfModel(FREQS, pt, pp, dt, dp, st, transfer=(0.005, 5e9), freq_switch_s=0.1,
                     kv_bytes_per_token=57344, kv_capacity_tokens=250000, model="synthetic")


def fc(rate, in_mean=512, out_mean=128, inputs=None, peak_rps=0.0, completed_bins=0):
    inputs = tuple(inputs) if inputs is not None else tuple([in_mean] * 50)
    return Forecast(rate, 0.0, in_mean, in_mean * 1.5, out_mean, 0, inputs, (),
                    peak_rps=peak_rps, completed_bins=completed_bins)
