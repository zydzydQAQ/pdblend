import math

from pdblend2.profile.model import DecodePoint, PerfModel, PrefillPoint, StaticState, fit


def synthetic():
    prefill, decode = [], []
    for f, scale in ((900, 2.0), (2520, 1.0)):
        for n in (128, 512, 1024, 2048, 4096, 7168):
            prefill.append(PrefillPoint(f, n, scale * (0.01 + 4e-5 * n + 1e-9 * n * n), 150 + 100 * min(n, 1024) / 1024 / scale))
        for B in (1, 4, 16, 32, 64):
            for ctx in (256, 1024, 4096):
                decode.append(DecodePoint(f, B, ctx, scale * (0.008 + 1e-4 * B + 2e-8 * B * ctx), 90 + 2.0 * B / scale))
    static = {"active_idle@900": StaticState(60), "active_idle@2520": StaticState(78),
              "parked": StaticState(35, 0.05), "off": StaticState(34, 30.0)}
    return fit(prefill, decode, static, transfer_points=[(512, 0.01), (7168, 0.05)],
               kv_bytes_per_token=57344, kv_capacity_tokens=200_000, model="synthetic")


def test_fit_recovers_synthetic_curves():
    m = synthetic()
    assert m.freqs == (900, 2520)
    assert math.isclose(m.prefill_seconds(2048, 2520), 0.01 + 4e-5 * 2048 + 1e-9 * 2048 ** 2, rel_tol=1e-3)
    assert math.isclose(m.step_seconds(32, 1024, 900), 2 * (0.008 + 1e-4 * 32 + 2e-8 * 32 * 1024), rel_tol=1e-3)
    assert math.isclose(m.decode_power_w(16, 2520), 90 + 32, rel_tol=1e-3)
    assert max(m.residuals.values()) < 1e-3


def test_queries_and_static_states():
    m = synthetic()
    assert m.static_power_w("active_idle", 2400) == 78
    assert m.static_power_w("parked") == 35 and m.wake_seconds("off") == 30
    assert m.nearest_freq(1000) == 900
    fixed, bw = m.transfer
    assert fixed >= 0 and 1e9 < bw < 100e9
    assert m.transfer_seconds(7168) > m.transfer_seconds(512)
    assert m.token_energy_j(64, 1024, 900) < m.token_energy_j(1, 1024, 900)


def test_json_roundtrip(tmp_path):
    m = synthetic()
    m.save(tmp_path / "m.json")
    back = PerfModel.load(tmp_path / "m.json")
    assert back.prefill_time == m.prefill_time and back.decode_power == m.decode_power
    assert back.static["off"].wake_s == 30.0 and back.transfer == m.transfer
