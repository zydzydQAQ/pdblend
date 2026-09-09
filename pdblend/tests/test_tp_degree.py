# -*- coding: utf-8 -*-
"""C5：同 GPU 预算 TP 度数门禁与协议。"""
from __future__ import annotations

import os
import sys

_SCRIPTS = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "new-motivations", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import metrics as M  # noqa: E402
import p0_common as P0  # noqa: E402
import plot_tp_degree as PT  # noqa: E402
import run_tp_degree as RT  # noqa: E402


def test_same_gpu_budget():
    assert P0.same_gpu_budget(8, 1, 8)
    assert P0.same_gpu_budget(8, 2, 4)
    assert P0.same_gpu_budget(8, 4, 2)
    assert not P0.same_gpu_budget(8, 2, 2)
    assert not P0.same_gpu_budget(8, 3, 2)


def test_default_replicas_and_label():
    assert P0.default_replicas(8, 1) == 8
    assert P0.default_replicas(8, 2) == 4
    assert P0.default_replicas(8, 4) == 2
    assert P0.layout_label(1, 8) == "8×TP1"
    assert P0.layout_label(4, 2) == "2×TP4"


def test_iso_action_tp_degree():
    assert M.classify_iso_load_action(80.0, 100.0, 0.97, 0.97) == "B"
    assert M.classify_iso_load_action(80.0, 100.0, 0.90, 0.97) == "A"
    assert M.classify_iso_load_action(110.0, 100.0, 0.99, 0.97) == "A"


def test_protocol_and_plot(tmp_path):
    out = str(tmp_path / "tp-degree")
    os.makedirs(out, exist_ok=True)
    RT.write_protocol(out, "Qwen2.5-32B-Instruct", "4,5,6,7", ran_table=False)
    text = open(os.path.join(out, "README.md"), encoding="utf-8").read()
    assert "排除的是 **PP**" in text
    assert "8×TP1" in text
    assert "2×TP4" in text
    assert "4×TP2" in text
    matrix = open(os.path.join(out, "matrix.json"), encoding="utf-8").read()
    assert "14b" in matrix and "32b" in matrix
    saved = sys.argv
    try:
        sys.argv = ["plot_tp_degree.py", "--outdir", out]
        assert PT.main() == 0
    finally:
        sys.argv = saved
    assert os.path.isfile(os.path.join(out, "compare.md"))
    fig = os.path.join(out, "figures", "32b_iter_tp2_vs_tp4.png")
    assert os.path.isfile(fig)
    cmp_txt = open(os.path.join(out, "compare.md"), encoding="utf-8").read()
    assert "32B-TP4 表" in cmp_txt
    assert "缺" in cmp_txt
