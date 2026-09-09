# -*- coding: utf-8 -*-
"""测量栈:FakeBackend、梯形积分、PerfModel、Dataset、第一波 protocol。"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import pytest

from ecopadg.measure.backends import BackendError, FakeBackend, get_backend
from ecopadg.measure.datasets import Dataset, TestRequest
from ecopadg.measure.gpu_clock import nearest_freq
from ecopadg.measure.perfmodel import PerfModel, _interp1d
from ecopadg.measure.power import (
    PowerSampler, trapezoid_energy, trapezoid_mean_power,
)
from ecopadg.measure.prompts import make_prompts


def test_fake_backend_unlocked_is_max_and_invalid_raises():
    be = FakeBackend(gpu_count=2, freqs=(2520, 1500, 900))
    assert be.current_freq(0) == 2520
    be.set_clock(0, 1500)
    assert be.locked == {0: 1500}
    assert be.current_freq(0) == 1500
    assert be.current_freq(1) == 2520
    be.reset_clock(0)
    assert be.locked == {}
    with pytest.raises(BackendError):
        be.set_clock(0, 1111)
    with pytest.raises(BackendError):
        be.current_freq(9)
    assert get_backend("fake").current_freq(0) == 2520


def test_trapezoid_ramp_and_two_gpu_constant():
    # 0→100W / 10s → 500J
    ramp = [(0.0, [0.0]), (10.0, [100.0])]
    assert trapezoid_energy(ramp) == pytest.approx(500.0)
    assert trapezoid_mean_power(ramp) == pytest.approx(50.0)
    # 2×100W 恒 10s → 2000J
    flat = [(0.0, [100.0, 100.0]), (10.0, [100.0, 100.0])]
    assert trapezoid_energy(flat) == pytest.approx(2000.0)
    assert trapezoid_mean_power(flat) == pytest.approx(200.0)
    assert trapezoid_energy([(0.0, [10.0])]) == 0.0


def test_power_sampler_uses_injected_samples():
    s = PowerSampler(gpus=[0], interval=0.02, backend=FakeBackend())
    s.samples = [(0.0, [0.0]), (10.0, [100.0])]
    assert s.total_energy_j() == pytest.approx(500.0)
    s.samples = [(0.0, [100.0, 100.0]), (10.0, [100.0, 100.0])]
    assert s.total_energy_j() == pytest.approx(2000.0)


def test_nearest_freq_ties_prefer_higher():
    levels = (2520, 1500, 900)
    assert nearest_freq(1480, levels) == 1500
    assert nearest_freq(2520, levels) == 2520
    assert nearest_freq(1200, levels) == 1500
    with pytest.raises(BackendError):
        nearest_freq(1500, [])


def test_interp1d_clamp_and_midpoint():
    pts = [(900.0, 30.0), (1500.0, 20.0), (2520.0, 10.0)]
    assert _interp1d(pts, 600) == 30.0
    assert _interp1d(pts, 3000) == 10.0
    assert _interp1d(pts, 1200) == pytest.approx(25.0)


def test_perfmodel_reads_synthetic_csv(tmp_path):
    lat = tmp_path / "p1b_latency.csv"
    pw = tmp_path / "p1b_power.csv"
    with lat.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=("freq_mhz", "phase", "batch", "prompt_len",
                           "iter_ms", "ttft_ms"))
        w.writeheader()
        w.writerow(dict(freq_mhz=2520, phase="decode", batch=1,
                        prompt_len=128, iter_ms=20, ttft_ms=""))
        w.writerow(dict(freq_mhz=1500, phase="decode", batch=1,
                        prompt_len=128, iter_ms=30, ttft_ms=""))
        w.writerow(dict(freq_mhz=2520, phase="prefill", batch=1,
                        prompt_len=128, iter_ms="", ttft_ms=50))
        w.writerow(dict(freq_mhz=1500, phase="prefill", batch=1,
                        prompt_len=128, iter_ms="", ttft_ms=80))
    with pw.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=("freq_mhz", "phase", "batch",
                           "j_per_token_dynamic"))
        w.writeheader()
        w.writerow(dict(freq_mhz=2520, phase="decode", batch=1,
                        j_per_token_dynamic=0.002))
        w.writerow(dict(freq_mhz=1500, phase="decode", batch=1,
                        j_per_token_dynamic=0.003))
    pm = PerfModel(str(lat), str(pw))
    assert pm.iter_time_ms(1, 2520) == pytest.approx(20.0)
    assert pm.prefill_time_ms(128, 2520) == pytest.approx(50.0)
    assert pm.dyn_j_per_token(1, 2520) == pytest.approx(2.0)
    assert pm.iter_time_ms(1, 2010) == pytest.approx(25.0)


def test_perfmodel_reads_probe_metric_schema(tmp_path):
    lat = tmp_path / "p1b_latency.csv"
    lat.write_text(
        "phase,freq_mhz,batch,prompt_len,metric,value_ms,repeat\n"
        "decode,2520,8,16,iter_ms,23.51,0\n"
        "decode,900,8,16,iter_ms,27.13,0\n"
        "prefill,2520,1,384,ttft_corrected_ms,67.3,0\n",
        encoding="utf-8")
    pm = PerfModel(str(lat))
    assert pm.iter_time_ms(8, 2520) == pytest.approx(23.51)
    assert pm.prefill_time_ms(384, 2520) == pytest.approx(67.3)


def test_dataset_dump_load(tmp_path):
    path = str(tmp_path / "toy.ds")
    ds = Dataset(dataset_name="toy",
                 reqs=[TestRequest(prompt="hi", prompt_len=2, output_len=4)])
    ds.dump(path)
    loaded = Dataset.load(path)
    assert loaded.dataset_name == "toy"
    assert loaded.reqs[0].output_len == 4


def test_make_prompts_hits_target_len():
    enc = lambda s: len(s.split())
    prompts = make_prompts(enc, 8, 2)
    assert len(prompts) == 2
    assert enc(prompts[0]) == 8


def test_profile_sweep_protocol_and_compose(tmp_path):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script = os.path.join(root, "new-motivations", "scripts",
                          "run_profile_sweep.py")
    out = str(tmp_path / "proto")
    subprocess.check_call(
        [sys.executable, script, "--protocol-only", "--outdir", out],
        cwd=root)
    cells = json.load(open(os.path.join(out, "cells.json"), encoding="utf-8"))
    assert cells["n_cells"] == 2 * 3 * 3 * 2
    assert os.path.isfile(os.path.join(out, "README.md"))
    sys.path.insert(0, os.path.dirname(script))
    import run_profile_sweep as sweep
    mixed = []
    for freq, e1, en, ttft, tpot in (
            (2520, 10.0, 40.0, 0.05, 0.02),
            (1500, 12.0, 30.0, 0.08, 0.03),
            (900, 16.0, 28.0, 0.12, 0.05)):
        mixed.append(dict(
            model="14b", layout="mixed", tp=2, freq_p=freq, freq_d=freq,
            freq_mhz=freq, size_bucket="medium", batch=1,
            prompt_len=512, gen_len=200,
            ttft_s=ttft, tpot_s=tpot, energy_j=en, e1_j=e1, en_j=en,
            e2e_s=ttft + tpot * 199, mean_w=100.0, repeat=0,
        ))
    composed = sweep.compose_spatial(mixed)
    layouts = {r["layout"] for r in composed}
    assert layouts == {"spatial_pd"}
    pairs = {(int(r["freq_p"]), int(r["freq_d"])) for r in composed
             if r.get("source") == "composed"}
    assert (2520, 1500) in pairs
    assert (2520, 900) in pairs


def test_switch_cost_protocol_only(tmp_path):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script = os.path.join(root, "new-motivations", "scripts",
                          "run_switch_cost.py")
    out = str(tmp_path / "switch")
    subprocess.check_call(
        [sys.executable, script, "--protocol-only", "--outdir", out],
        cwd=root)
    payload = json.load(open(os.path.join(out, "switch_cost.json"),
                             encoding="utf-8"))
    sc = payload["switch_cost"]
    assert sc["freq_transient_j"] == 20.0
    assert sc["role_switch_cost_j"] is None
    assert sc["instance_start_s"] is None
    assert "layout" in payload["unmeasured"] or "role_switch_cost_j" in payload["unmeasured"]
    assert os.path.isfile(os.path.join(out, "README.md"))
    assert os.path.isfile(os.path.join(out, "protocol.json"))


def test_build_lookup_script(tmp_path):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    meas = tmp_path / "measurements.csv"
    with meas.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=("model", "layout", "tp", "freq_p", "freq_d",
                           "freq_mhz", "size_bucket", "batch",
                           "ttft_s", "tpot_s", "energy_j"))
        w.writeheader()
        w.writerow(dict(model="14b", layout="mixed", tp=2, freq_p=2520,
                        freq_d=2520, freq_mhz=2520, size_bucket="short",
                        batch=1, ttft_s=0.2, tpot_s=0.04, energy_j=80))
        w.writerow(dict(model="14b", layout="spatial_pd", tp=2, freq_p=2520,
                        freq_d=1500, freq_mhz=1500, size_bucket="short",
                        batch=1, ttft_s=0.2, tpot_s=0.05, energy_j=60))
    script = os.path.join(root, "new-motivations", "scripts",
                          "build_lookup.py")
    out = tmp_path / "lookup.json"
    subprocess.check_call(
        [sys.executable, script, "--meas", str(meas), "--out", str(out)])
    table = json.loads(out.read_text(encoding="utf-8"))
    cfg = table["keys"]["short|14b|alpaca"]
    assert cfg["layout"] == "spatial_pd"
    assert cfg["freq_d"] == 1500


def test_scripts_common_reexports():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(root, "new-motivations", "scripts"))
    from common.backends import FakeBackend as FB
    from common.gpu_utils import nearest_freq as nf
    assert FB().current_freq(0) == 2520
    assert nf(1500, (2520, 1500, 900)) == 1500
