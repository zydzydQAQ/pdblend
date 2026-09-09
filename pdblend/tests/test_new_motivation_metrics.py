# -*- coding: utf-8 -*-
"""new-motivations iso-load / 联合 SLO 判定。"""
from __future__ import annotations

import os
import sys

import pytest

_SCRIPTS = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "new-motivations", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import metrics as M  # noqa: E402


def test_dataset_slo_table():
    assert M.DATASET_SLO["sharegpt"].ttft_ms == 5000.0
    assert M.DATASET_SLO["longbench"].ttft_ms == 15000.0
    assert M.DATASET_SLO["alpaca"].ttft_ms == 1000.0
    assert M.DATASET_SLO["sharegpt"].tpot_ms == 150.0
    assert M.DATASET_SLO["longbench"].tpot_ms == 200.0
    assert M.DATASET_SLO["alpaca"].tpot_ms == 100.0


def test_joint_window_ok():
    # 608 ms TTFT：旧 500 ms 会绑，三套正式 SLO 都不绑
    assert M.joint_window_ok(608.0, 23.0, "alpaca")
    assert M.joint_window_ok(608.0, 23.0, "sharegpt")
    # 1223 ms：只破 alpaca
    assert not M.joint_window_ok(1223.0, 24.0, "alpaca")
    assert M.joint_window_ok(1223.0, 24.0, "sharegpt")
    assert M.joint_window_ok(1223.0, 24.0, "longbench")
    # TPOT：ShareGPT 150 ms，Alpaca 仍 100 ms
    assert not M.joint_window_ok(10.0, 150.0, "sharegpt")
    assert M.joint_window_ok(10.0, 149.9, "sharegpt")
    assert not M.joint_window_ok(10.0, 100.0, "alpaca")
    assert M.joint_window_ok(10.0, 99.9, "alpaca")


def test_iso_load_window_x_when_ref_infeasible():
    # 对照满频自身破 TTFT → 不比能耗
    assert M.classify_iso_load_window(
        100.0, 200.0, 200.0, 20.0, 2000.0, 20.0, "alpaca") == "X"


def test_iso_load_window_b_when_j_down_and_slo_ok():
    assert M.classify_iso_load_window(
        100.0, 120.0, 50.0, 20.0, 40.0, 18.0, "sharegpt") == "B"


def test_iso_load_window_a_when_cand_breaks_slo():
    assert M.classify_iso_load_window(
        80.0, 120.0, 2000.0, 20.0, 40.0, 18.0, "alpaca") == "A"


def test_iso_load_window_a_when_energy_not_lower():
    assert M.classify_iso_load_window(
        120.0, 120.0, 50.0, 20.0, 40.0, 18.0, "sharegpt") == "A"


def test_min_energy_feasible_skips_left_arm():
    energy = {300: 10.0, 1500: 8.0, 2520: 12.0}
    ttft = {300: 90.0, 1500: 20.0, 2520: 10.0}
    tpot = {300: 160.0, 1500: 21.0, 2520: 20.0}  # 300 破 ShareGPT 150 ms
    assert M.min_energy_feasible_freq(energy, ttft, tpot, "sharegpt") == 1500


def test_prefill_branch_slo_x_when_max_breaks_ttft():
    jpt = {2100: 0.04, 2520: 0.05}
    ttft = {2100: 1200.0, 2520: 1100.0}
    assert M.classify_prefill_branch_slo(jpt, ttft, "alpaca") == "X"


def test_prefill_branch_slo_b_when_mid_saves_and_ttft_ok():
    jpt = {2100: 0.04, 2520: 0.05}
    ttft = {2100: 200.0, 2520: 180.0}
    assert M.classify_prefill_branch_slo(jpt, ttft, "sharegpt") == "B"


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        M.resolve_slo("unknown")


def test_model_tag_32b():
    assert M.model_tag("/root/workspace/models/Qwen2.5-32B-Instruct") == "32b"
    assert M.model_tag("Qwen2.5-14B-Instruct") == "14b"
