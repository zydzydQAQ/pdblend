# -*- coding: utf-8 -*-
"""P0 测量门禁：14B commit、hybrid schema、prefill README 32b 措辞。"""
from __future__ import annotations

import importlib.util
import os
import sys

_SCRIPTS = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "new-motivations", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
_WS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _WS not in sys.path:
    sys.path.insert(0, _WS)


def _load_probe_mod():
    path = os.path.join(
        os.path.dirname(__file__), "..", "script", "bench", "probe_14b_iter.py")
    spec = importlib.util.spec_from_file_location("probe_14b_iter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

import p0_common as P0  # noqa: E402
import plot_prefill_slo as PS  # noqa: E402


def test_derived_hybrid_formula():
    assert P0.derived_hybrid_ms(20.0, 100.0, 8, 0) == 20.0
    assert P0.derived_hybrid_ms(20.0, 100.0, 8, 256) == 20.0
    assert P0.derived_hybrid_ms(20.0, 200.0, 8, 256) == 25.0


def test_hybrid_schema_ok():
    row = dict(prefill_tokens=0, n_decode=8, kv_tokens=256, freq_mhz=1500,
               iter_p50_ms=40.0, mode="step")
    assert P0.hybrid_schema_ok(row)
    assert not P0.hybrid_schema_ok({"freq_mhz": 1500})


def test_probe_commit_clock_fail():
    rows = [
        dict(freq_mhz=900, req_clock_mhz=900, actual_clock_mhz=2520,
             iter_ms=30.0),
        dict(freq_mhz=1500, req_clock_mhz=1500, actual_clock_mhz=1500,
             iter_ms=28.0),
        dict(freq_mhz=2520, req_clock_mhz=2520, actual_clock_mhz=2520,
             iter_ms=27.0),
    ]
    ok, msg = P0.probe_commit_ok(rows)
    assert not ok
    assert "clock" in msg


def test_probe_commit_not_mono():
    rows = [
        dict(freq_mhz=900, req_clock_mhz=900, actual_clock_mhz=900,
             iter_ms=20.0),
        dict(freq_mhz=1500, req_clock_mhz=1500, actual_clock_mhz=1500,
             iter_ms=28.0),
        dict(freq_mhz=2520, req_clock_mhz=2520, actual_clock_mhz=2520,
             iter_ms=30.0),
    ]
    ok, msg = P0.probe_commit_ok(rows)
    assert not ok
    assert "单调" in msg


def test_probe_commit_ok():
    rows = [
        dict(freq_mhz=900, req_clock_mhz=900, actual_clock_mhz=900,
             iter_ms=27.13),
        dict(freq_mhz=1500, req_clock_mhz=1500, actual_clock_mhz=1502,
             iter_ms=24.60),
        dict(freq_mhz=2520, req_clock_mhz=2520, actual_clock_mhz=2510,
             iter_ms=23.51),
    ]
    ok, msg = P0.probe_commit_ok(rows)
    assert ok and msg == "ok"


def test_probe_commit_flat_ratio_fails():
    rows = [
        dict(freq_mhz=900, req_clock_mhz=900, actual_clock_mhz=900,
             iter_ms=21.0),
        dict(freq_mhz=1500, req_clock_mhz=1500, actual_clock_mhz=1500,
             iter_ms=21.0),
        dict(freq_mhz=2520, req_clock_mhz=2520, actual_clock_mhz=2520,
             iter_ms=21.0),
    ]
    ok, msg = P0.probe_commit_ok(rows)
    assert not ok
    assert "比值" in msg


def test_probe_trust_ratio_matches_opmodel():
    from ecopadg.types import FREQ_TRUST_RATIO
    assert P0.FREQ_TRUST_RATIO == FREQ_TRUST_RATIO


def test_probe_commit_writes_power_and_deletes_synthetic(tmp_path):
    probe = _load_probe_mod()

    table = tmp_path / "E1b_14B_tp2"
    table.mkdir()
    (table / "SYNTHETIC").write_text("keep\n", encoding="utf-8")
    (table / "p1b_latency.csv").write_text(
        "phase,freq_mhz,batch,prompt_len,metric,value_ms,repeat\n"
        "decode,900,8,16,iter_ms,21.0,0\n"
        "decode,900,1,16,iter_ms,21.0,0\n"
        "prefill,2520,1,384,ttft_corrected_ms,67.0,0\n",
        encoding="utf-8")
    (table / "p1b_power.csv").write_text(
        "freq_mhz,phase,batch,prompt_len,mean_w,std_w,energy_j,duration_s,"
        "dynamic_j,j_per_token_gross,j_per_token_dynamic,repeat,note\n"
        "900,idle,0,0,132.8,,,5.0,,,,1,synth\n"
        "900,decode,8,16,326.84,,,,,,20.66,1,synth\n"
        "1500,idle,0,0,138.54,,,5.0,,,,1,synth\n"
        "2520,idle,0,0,163.3,,,5.0,,,,1,synth\n"
        "2520,decode,8,16,688.79,,,,,,28.77,1,synth\n",
        encoding="utf-8")
    rows = [
        dict(freq_mhz=900, req_clock_mhz=900, actual_clock_mhz=900,
             iter_ms=27.13, mean_w=400.0, repeat=0),
        dict(freq_mhz=1500, req_clock_mhz=1500, actual_clock_mhz=1500,
             iter_ms=24.60, mean_w=450.0, repeat=0),
        dict(freq_mhz=2520, req_clock_mhz=2520, actual_clock_mhz=2520,
             iter_ms=23.51, mean_w=500.0, repeat=0),
    ]
    msg = probe.commit_rows(rows, str(table))
    assert "删除 SYNTHETIC" in msg
    assert not (table / "SYNTHETIC").exists()
    lat = (table / "p1b_latency.csv").read_text(encoding="utf-8")
    assert "27.13" in lat
    assert "21.0" in lat  # b=1 拼表行保留
    pw_rows = list(__import__("csv").DictReader(
        (table / "p1b_power.csv").open(encoding="utf-8")))
    measured = [r for r in pw_rows
                if r["phase"] == "decode" and r["batch"] == "8"]
    assert {int(float(r["freq_mhz"])) for r in measured} == {900, 1500, 2520}
    assert all("probe_14b_iter" in r["note"] for r in measured)
    assert all(float(r["j_per_token_dynamic"]) > 0 for r in measured)


def test_probe_commit_keeps_synthetic_when_gate_fails(tmp_path):
    probe = _load_probe_mod()

    table = tmp_path / "E1b_14B_tp2"
    table.mkdir()
    syn = table / "SYNTHETIC"
    syn.write_text("keep\n", encoding="utf-8")
    rows = [
        dict(freq_mhz=900, req_clock_mhz=900, actual_clock_mhz=900,
             iter_ms=21.0, mean_w=400.0, repeat=0),
        dict(freq_mhz=1500, req_clock_mhz=1500, actual_clock_mhz=1500,
             iter_ms=21.0, mean_w=400.0, repeat=0),
        dict(freq_mhz=2520, req_clock_mhz=2520, actual_clock_mhz=2520,
             iter_ms=21.0, mean_w=400.0, repeat=0),
    ]
    msg = probe.commit_rows(rows, str(table))
    assert "保持 SYNTHETIC" in msg
    assert syn.exists()


def test_prefill_readme_mentions_32b_when_present(tmp_path):
    rows = [dict(model="32b", dataset="sharegpt", branch="B")]
    path = str(tmp_path / "README.md")
    PS.write_readme(path, rows, [], meas_path="measurements.csv")
    text = open(path, encoding="utf-8").read()
    assert "仍缺实测" not in text
    assert "32B-TP2 已入表" in text


def test_iso_sumtok_gates():
    assert P0.iso_sumtok_relerr(773.0, 747.0) < 4.0
    assert P0.iso_sumtok_ok(3.0, 6.0)
    assert not P0.iso_sumtok_ok(12.0, 3.0)
    assert not P0.iso_sumtok_ok(None, 1.0)


def test_n4_iso_calib_summary(tmp_path):
    import run_prefill_freq_32b as N4  # noqa: WPS433

    meas = str(tmp_path / "measurements.csv")
    with open(meas, "w", encoding="utf-8") as f:
        f.write(
            "model,batch,freq_mhz,prompt_len,ttft_ms,j_per_ptoken_gross\n"
            "32b,1,2520,3072,773.0,0.089\n"
            "32b,8,2520,384,747.0,0.084\n"
            "32b,1,1500,3072,900.0,0.080\n"
            "32b,8,1500,384,880.0,0.078\n")
    note = N4.summarize_iso_calib(meas)
    assert "过（K 不进表）" in note
    assert "主表保持 K=1" in note


def test_n4_protocol_is_sumtok_not_k_grid(tmp_path):
    import run_prefill_freq_32b as N4  # noqa: WPS433

    out = str(tmp_path)
    N4.write_protocol(out, "/models/Qwen2.5-32B-Instruct", "0,1", ran=False)
    text = open(os.path.join(out, "README.md"), encoding="utf-8").read()
    assert "Σtok" in text
    assert "K=1" in text
    assert "不写回 E1b" in text
    assert "1,4,8" not in text
    assert "--batches 1" in text
    assert "--batches 8" in text


def test_prefill_readme_missing_32b(tmp_path):
    rows = [dict(model="7b", dataset="sharegpt", branch="B")]
    path = str(tmp_path / "README.md")
    PS.write_readme(path, rows, [], meas_path="measurements.csv")
    text = open(path, encoding="utf-8").read()
    assert "32B-TP2 仍缺实测" in text
