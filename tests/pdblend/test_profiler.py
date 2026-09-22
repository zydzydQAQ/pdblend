import json

from pdblend.profile.profiler import load_raw, window_mean_power


def _raw():
    freqs = [1500, 2520]
    prefill, decode = [], []
    for f in freqs:
        k = 2520 / f
        for n in (128, 512, 1024, 2048, 4096, 7168):
            prefill.append(dict(freq_mhz=f, input_tokens=n, seconds=k * (0.01 + 4e-5 * n + 1e-9 * n * n),
                                power_w=120 + 150 * min(n, 1024) / 1024 / k))
        for ctx in (256, 1024, 4096):
            for b in (1, 4, 16, 32, 64):
                decode.append(dict(freq_mhz=f, batch=b, context_tokens=ctx,
                                   step_seconds=k * (0.012 + 2e-4 * b + 3e-8 * b * ctx), power_w=100 + 2.0 * b / k))
    return dict(freqs=freqs, prefill=prefill, decode=decode, mixed=[],
                static={"active_idle@1500": dict(power_w=70), "active_idle@2520": dict(power_w=78),
                        "parked": dict(power_w=35, wake_s=0.05), "off": dict(power_w=34, wake_s=30)},
                transfer=[dict(input_tokens=512, overhead_s=0.02), dict(input_tokens=2048, overhead_s=0.05),
                          dict(input_tokens=7168, overhead_s=0.15)],
                freq_switch_s=[0.1, 0.12, 0.11], kv_capacity_tokens=200000)


def test_load_raw_fits_and_queries(tmp_path):
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(_raw()))
    model = load_raw(path, "m", 1, 57344)
    assert model.freqs == (1500, 2520)
    assert max(model.residuals.values()) < 0.05
    assert model.prefill_seconds(2048, 2520) < model.prefill_seconds(2048, 1500)
    assert model.step_seconds(32, 1024, 2520) < model.step_seconds(32, 1024, 1500)
    assert model.static_power_w("parked") == 35
    assert model.wake_seconds("off") == 30
    assert model.kv_capacity_tokens == 200000
    assert abs(model.freq_switch_s - 0.11) < 1e-9
    assert 0.04 < model.transfer_seconds(2048) < 0.06


def test_window_mean_power():
    samples = [(0.0, [10.0, 1.0]), (1.0, [20.0, 2.0]), (2.0, [30.0, 3.0]), (3.0, [40.0, 4.0])]
    assert window_mean_power(samples, 1.0, 2.0) == 25.0
    assert window_mean_power(samples, 1.0, 2.0, 1) == 2.5
    assert window_mean_power(samples, 5.0, 6.0) is None
