"""CPU tests for the motivation analysis helpers."""
import json

from pdblend.bench import motivation as mv

FREQS = [900, 1500, 2520]


def fake_raw():
    prefill = [dict(freq_mhz=f, input_tokens=2048, seconds=0.5 * 2520 / f, power_w=150 + f / 20) for f in FREQS]
    decode = [dict(freq_mhz=f, batch=32, context_tokens=1024, step_seconds=0.03 * (1 + 0.2 * (2520 / f - 1)),
                   power_w=80 + f / 10) for f in FREQS]
    static = {"off": dict(power_w=34.0, wake_s=33.0), "parked": dict(power_w=35.0, wake_s=0.03),
              "active_idle_reset": dict(power_w=75.0, wake_s=0.0), "active_idle@900": dict(power_w=63.0, wake_s=0.0)}
    return dict(freqs=FREQS, prefill=prefill, decode=decode, static=static)


def test_m1_m2_rows():
    raw = fake_raw()
    m1 = mv.m1_rows(raw)
    assert [r["freq_mhz"] for r in m1] == FREQS
    # compute-bound prefill saves little per token at low clock; memory-bound decode saves a lot
    assert m1[0]["decode_j_per_token"] < m1[-1]["decode_j_per_token"]
    assert m1[0]["prefill_ms"] > m1[-1]["prefill_ms"]
    assert [r["state"] for r in mv.m2_rows(raw)] == ["off", "parked", "active_idle_reset", "active_idle@900"]


def _point(root, name, dataset, scale, counts, slo, jpr):
    d = root / name
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(dict(
        slo=dict(joint_slo_rate=slo, ttft_p90=0.5, tpot_p90=0.05, offered=10, success_rate=1.0, ttft_p50=0.2, tpot_p50=0.03),
        trace=dict(mean_rps=4.0), trace_meta=dict(dataset=dataset, scale=scale), policy=dict(name="manual"),
        fixed_plan=dict(counts=counts, f_P=2520, f_D=1500, f_M=2520, tau=0), j_per_request=jpr, j_per_token=1.0,
        mean_power_w=500.0, energy_j=1.0, window_energy_j=1.0, model="m", gpus=[4, 5, 6, 7])))


def test_m3_winners(tmp_path):
    _point(tmp_path, "a", "sharegpt", 0.5, {"M": 4}, 0.95, 300.0)
    _point(tmp_path, "b", "sharegpt", 0.5, {"P": 1, "D": 3}, 0.95, 240.0)
    _point(tmp_path, "c", "sharegpt", 1.0, {"P": 1, "D": 3}, 0.5, 100.0)
    _point(tmp_path, "d", "sharegpt", 1.0, {"M": 4}, 0.92, 280.0)
    rows = mv.m3_rows(tmp_path)
    assert {r["layout"] for r in rows} == {"4M", "1P+3D"}
    w = {(x["dataset"], x["scale"]): x for x in mv.m3_winners(rows)}
    assert w[("sharegpt", 0.5)]["winner"] == "1P+3D" and abs(w[("sharegpt", 0.5)]["saving_vs_colocated"] - 0.2) < 1e-9
    assert w[("sharegpt", 1.0)]["winner"] == "4M"
    out = mv.plot_m3(tmp_path, tmp_path / "fig")
    assert (tmp_path / "fig" / "m3_crossover.png").exists() and len(out["winners"]) == 2
