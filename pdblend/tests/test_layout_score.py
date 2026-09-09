# -*- coding: utf-8 -*-
"""离线 2-候选打分器：不可信默认 1P1D；不接入启动路径。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ecopadg.global_scheduler import Prediction
from ecopadg.layout_score import (
    CHOSEN_MIXED, CHOSEN_SPATIAL, REASON_MIXED_WINS, REASON_UNTRUSTED,
    score_boot_layout, score_from_tables,
)
from ecopadg.predictor import WorkloadStats
from ecopadg.types import SloSpec

ROOT = Path(__file__).resolve().parents[1]


class _OM:
    def prefill_time_ms(self, tokens, freq=2520):
        return 80.0


class _Pred:
    def __init__(self, mixed_j=100.0, spatial_j=200.0,
                 mixed_att=0.98, spatial_att=0.97):
        self.opmodel = _OM()
        self.slo = SloSpec.from_dataset("sharegpt")
        self.stats = WorkloadStats(prompt_len=128, output_len=32)
        self.mixed_j = mixed_j
        self.spatial_j = spatial_j
        self.mixed_att = mixed_att
        self.spatial_att = spatial_att

    def predict(self, p, lam, state=None):
        if p.n_mixed:
            return Prediction(attainable=True, attainment=self.mixed_att,
                              energy_j_per_s=self.mixed_j, power_w=self.mixed_j)
        return Prediction(attainable=True, attainment=self.spatial_att,
                          energy_j_per_s=self.spatial_j, power_w=self.spatial_j)


def test_untrusted_defaults_to_spatial_pd():
    result = score_boot_layout(
        _Pred(), 2.0, synthetic=True, freq_trusted=True,
        capacity_trusted=True)
    assert result.chosen == CHOSEN_SPATIAL
    assert result.reason == REASON_UNTRUSTED
    assert result.report_only is True
    assert result.online_system_result is False


def test_untrusted_if_capacity_not_trusted():
    result = score_boot_layout(
        _Pred(), 2.0, synthetic=False, freq_trusted=True,
        capacity_trusted=False)
    assert result.chosen == CHOSEN_SPATIAL
    assert result.reason == REASON_UNTRUSTED


def test_trusted_mixed_wins_when_noninferior_and_cheaper():
    result = score_boot_layout(
        _Pred(mixed_j=80.0, spatial_j=120.0), 2.0,
        synthetic=False, freq_trusted=True, capacity_trusted=True)
    assert result.trusted is True
    assert result.chosen == CHOSEN_MIXED
    assert result.reason == REASON_MIXED_WINS


def test_trusted_keeps_spatial_when_mixed_more_expensive():
    result = score_boot_layout(
        _Pred(mixed_j=200.0, spatial_j=90.0), 2.0,
        synthetic=False, freq_trusted=True, capacity_trusted=True)
    assert result.chosen == CHOSEN_SPATIAL


def test_score_from_14b_tables_fail_closed_without_capacity():
    result = score_from_tables(
        "14b", WorkloadStats(), 2.0, dataset="sharegpt")
    assert result.chosen == CHOSEN_SPATIAL
    assert result.reason == REASON_UNTRUSTED
    assert result.capacity_trusted is False


def test_boot_path_does_not_import_layout_score():
    choose = (ROOT / "script" / "bench" / "choose_partition.py").read_text(
        encoding="utf-8")
    runner = (ROOT / "script" / "bench" / "run_cell.sh").read_text(
        encoding="utf-8")
    assert "layout_score" not in choose
    assert "score_boot_layout" not in choose
    assert "layout_score" not in runner
    assert "score_boot_layout" not in runner
    assert "--mode boot-slo-layout" in runner


def test_cli_refuses_energy_summary(tmp_path):
    script = ROOT / "new-results" / "scripts" / "score_boot_layout.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get(
        "PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(script), "--rate", "2",
         "--out", str(tmp_path / "energy_summary.csv")],
        check=False, capture_output=True, text=True, env=env)
    assert proc.returncode != 0
    assert "energy_summary" in (proc.stderr + proc.stdout)


def test_cli_writes_report_json(tmp_path):
    script = ROOT / "new-results" / "scripts" / "score_boot_layout.py"
    out = tmp_path / "boot_layout_score.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get(
        "PYTHONPATH", "")
    env["PDBLEND_WS"] = str(ROOT.parent)
    proc = subprocess.run(
        [sys.executable, str(script), "--model-key", "14b",
         "--dataset", "sharegpt", "--rate", "2", "--out", str(out)],
        check=False, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["report_only"] is True
    assert payload["online_system_result"] is False
    assert payload["chosen"] == CHOSEN_SPATIAL
