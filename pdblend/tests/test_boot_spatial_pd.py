# -*- coding: utf-8 -*-
"""CPU tests for static spatial-PD boot (no inherit / regime lookup)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecopadg.planner import spatial_pd_partition  # noqa: E402


def _pin_place(tmp_path, **kw):
    place = dict(pp_cross=2, tp_prefill=2, tp_decode=2,
                 pp_prefill=1, pp_decode=1, n_segments=2, freq=2520)
    place.update(kw)
    path = tmp_path / "ds_placement.json"
    path.write_text(json.dumps(place), encoding="utf-8")
    return path


def _run_choose(tmp_path, mode, rate, place=None):
    # ShareGPT-like lengths: n=200 r=12 上 predictor 对 PD/A9 会炸 energy。
    trace = {
        "requests": [
            {"arrival_s": 0.0, "prompt_len": 720, "output_len": 190},
            {"arrival_s": 1.0, "prompt_len": 720, "output_len": 190},
        ]
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace), encoding="utf-8")
    pin = place or _pin_place(tmp_path)
    env = os.environ.copy()
    env["PDBLEND_WS"] = str(ROOT.parent)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "script" / "bench" / "choose_partition.py"),
         "--model-key", "14b", "--trace", str(path),
         "--mode", mode, "--rate-override", str(rate),
         "--slo-ttft", "5", "--slo-tpot", "0.15",
         "--ds-placement", str(pin)],
        check=False, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_spatial_pd_partition_14b_is_two_tp2_pairs():
    part = spatial_pd_partition(8, 2)
    assert part.n_mixed == 0
    assert part.pairs() == 2
    assert part.tp_prefill == part.tp_decode == 2
    assert part.total_gpus() == 8


def test_spatial_pd_partition_72b_is_one_tp4_pair():
    part = spatial_pd_partition(8, 4)
    assert part.n_mixed == 0
    assert part.pairs() == 1
    assert part.tp_prefill == part.tp_decode == 4


def test_spatial_pd_partition_rejects_too_few_gpus():
    with pytest.raises(ValueError, match="2\\*tp"):
        spatial_pd_partition(2, 2)


def test_choose_partition_boot_spatial_pd_cli(tmp_path):
    trace = {
        "requests": [
            {"arrival_s": 0.0, "prompt_len": 128, "output_len": 32},
            {"arrival_s": 1.0, "prompt_len": 128, "output_len": 32},
        ]
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace), encoding="utf-8")
    env = os.environ.copy()
    env["PDBLEND_WS"] = str(ROOT.parent)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    pin = _pin_place(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "script" / "bench" / "choose_partition.py"),
         "--model-key", "14b", "--trace", str(path),
         "--mode", "boot-spatial-pd", "--slo-ttft", "5", "--slo-tpot", "0.15",
         "--ds-placement", str(pin)],
        check=False, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["n_mixed"] == 0
    assert payload["n_segments"] == 2
    assert payload["layout_decision"] == "boot-spatial-pd"
    assert payload["layout_reason"] == "profiler-default-spatial-pd"
    assert "regime" not in json.dumps(payload).lower()
    assert "inherit" not in json.dumps(payload).lower()
    assert payload["calib"]["note"] == "profiler-default-spatial-pd-no-oracle"


def test_choose_partition_boot_slo_layout_mixed_lengths_can_pick_a9(tmp_path):
    trace = {
        "requests": [
            {"arrival_s": 0.0, "prompt_len": 200, "output_len": 190},
            {"arrival_s": 1.0, "prompt_len": 900, "output_len": 190},
        ]
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace), encoding="utf-8")
    env = os.environ.copy()
    env["PDBLEND_WS"] = str(ROOT.parent)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    pin = _pin_place(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "script" / "bench" / "choose_partition.py"),
         "--model-key", "14b", "--trace", str(path),
         "--mode", "boot-slo-layout", "--rate-override", "2",
         "--slo-ttft", "5", "--slo-tpot", "0.15",
         "--ds-placement", str(pin)],
        check=False, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["layout_decision"] in (
        "boot-slo-mixed", "boot-slo-spatial-pd", "boot-slo-hybrid-a9")
    assert int(payload["used_gpus"]) == 8
    assert int(payload["unused_gpus"]) == 0
    assert int(payload["freq0"]) in (900, 2520)
    assert payload["hybrid_guard"]["ok"] is True
    assert payload["pd_threshold"] == 550
    assert payload["predicted_attainment_used"] is False


def test_choose_partition_boot_slo_layout_rate12_picks_mixed(tmp_path):
    payload = _run_choose(tmp_path, "boot-slo-layout", 12)
    assert payload["mode"] == "boot-slo-layout"
    assert int(payload["n_mixed"]) >= 1
    assert payload["n_segments"] == 0
    assert payload["layout_decision"] == "boot-slo-mixed"
    assert payload["layout_reason"] in (
        "capacity-rho-guard-mixed", "capacity-decode-rho-guard-mixed",
        "boot-mixed-fallback", "untrusted-min-j-mixed")
    assert int(payload["used_gpus"]) == 8
    assert int(payload["n_mixed"]) == 1
    assert int(payload.get("tp_mixed") or payload.get("tp") or 8) == 8
    assert int(payload["freq0"]) == 900
    assert payload["capacity_guard"]["pd_ok"] is False
    assert int(payload["freq0"]) in (900, 1500, 2100, 2520)
    assert payload["calib"]["note"] == "boot-slo-layout-no-oracle"
    assert payload["capacity_trusted"] is False
    assert payload["trusted"] is False
    assert payload["predicted_attainment"] is None
    assert payload["predicted_attainment_used"] is False
    assert "regime" not in json.dumps(payload).lower()
    assert "inherit" not in json.dumps(payload).lower()


def test_load_lookup_table_empty_path_does_not_autodiscover():
    from ecopadg.profiler import load_lookup_table
    table = load_lookup_table("", model="14b")
    assert table.get("keys") == {}


def test_choose_boot_lookup_respects_capacity_guard():
    from ecopadg.planner import choose_boot_lookup

    class Pred:
        def stage_rate_prefill(self, n, freq=None):
            del freq
            return 20.0 if int(n) > 0 else 0.0

        def stage_rate_decode(self, n, freq=None):
            del freq
            return 20.0 if int(n) > 0 else 0.0

        def predict(self, *_a, **_k):
            return type("R", (), {
                "attainment": 0.99, "energy_j_per_s": 100.0,
                "energy_j_per_req": 10.0, "power_w": 400.0,
                "attainable": True})()

    pred = Pred()
    low = choose_boot_lookup(
        pred, 2.0, 0.99, dict(layout="spatial_pd", freq_d=1500, freq_p=2520),
        gpu_count=8, tp=2, trusted=False)
    assert low.fallback == "boot-lookup-spatial-pd"
    assert low.chosen.pairs() == 2
    assert low.chosen.freq_decode == 2520
    assert low.chosen.freq_prefill == 2520
    high = choose_boot_lookup(
        pred, 12.0, 0.99, dict(layout="spatial_pd", freq_d=1500, freq_p=2520),
        gpu_count=8, tp=2, trusted=False)
    assert high.fallback == "boot-lookup-capacity-mixed"
    assert high.chosen.n_mixed == 4
    assert high.chosen.freq_mixed == 2520
    assert high.chosen.freq_decode == 2520


def test_choose_partition_boot_lookup_uses_table(tmp_path):
    lookup = tmp_path / "lookup.json"
    lookup.write_text(json.dumps({
        "model": "14b",
        "keys": {
            "medium|14b|sharegpt": {
                "layout": "spatial_pd", "tp": 2,
                "freq_p": 2520, "freq_d": 1500,
            }
        },
    }), encoding="utf-8")
    trace = {
        "requests": [
            {"arrival_s": 0.0, "prompt_len": 512, "output_len": 200},
            {"arrival_s": 1.0, "prompt_len": 512, "output_len": 200},
        ]
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace), encoding="utf-8")
    env = os.environ.copy()
    env["PDBLEND_WS"] = str(ROOT.parent)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "script" / "bench" / "choose_partition.py"),
         "--model-key", "14b", "--trace", str(path),
         "--mode", "boot-lookup", "--lookup", str(lookup),
         "--dataset", "sharegpt", "--rate-override", "2",
         "--slo-ttft", "5", "--slo-tpot", "0.15"],
        check=False, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["layout_decision"] == "boot-lookup-spatial-pd"
    assert payload["n_segments"] == 2
    assert int(payload["freq_decode"]) == 2520
    assert int(payload["freq_prefill"]) == 2520


def test_choose_partition_boot_lookup_fallback_without_table(tmp_path):
    payload = _run_choose(tmp_path, "boot-lookup", 12)
    assert payload["mode"] == "boot-lookup"
    assert int(payload["n_mixed"]) >= 1
    assert payload["n_segments"] == 0
    assert payload["layout_decision"].startswith("boot-lookup")


def test_choose_partition_boot_slo_layout_low_rate_fullnode(tmp_path):
    payload = _run_choose(tmp_path, "boot-slo-layout", 2)
    assert int(payload["used_gpus"]) == 8
    assert int(payload["unused_gpus"]) == 0
    assert payload["layout_decision"] in (
        "boot-slo-spatial-pd", "boot-slo-hybrid-a9", "boot-slo-mixed")
    assert payload["layout_reason"] != "untrusted-scaleinst-mixed"
    assert int(payload["freq0"]) in (900, 2520)
    assert payload["capacity_trusted"] is False
    assert payload["trusted"] is False
    assert payload["predicted_attainment_used"] is False


def test_choose_partition_honors_ds_placement_tp1(tmp_path):
    pin = _pin_place(tmp_path, pp_cross=4, tp_prefill=1, tp_decode=1,
                     n_segments=4)
    payload = _run_choose(tmp_path, "boot-slo-layout", 2, place=pin)
    assert payload["layout_decision"] in (
        "boot-slo-mixed", "boot-slo-spatial-pd")
    if payload["layout_decision"] == "boot-slo-spatial-pd":
        assert payload["n_segments"] == 4
        assert int(payload["tp_prefill"]) == 1
        assert int(payload["tp_decode"]) == 1
    else:
        assert int(payload["unused_gpus"]) == 0
