# -*- coding: utf-8 -*-
"""CPU dry-run: three-dataset boot cliffs under the prefill ρ guard."""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "new-results", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dry_run_boot_layout import dry_run_rows  # noqa: E402
from ecopadg.opmodel import default_table_dir  # noqa: E402


def test_default_table_dir_14b_prefers_new_motivations():
    ws = os.path.abspath(os.path.join(ROOT, ".."))
    path = default_table_dir("14b", ws)
    norm = path.replace("\\", "/")
    assert "new-motivations" in norm
    assert "e1b_14b_tp2" in norm
    assert os.path.isfile(os.path.join(path, "p1b_latency.csv"))
    assert "motivations/results/E1b_14B_tp2" not in norm


def test_estimate_decode_s_from_e1b_batch1():
    from ecopadg.opmodel import load_opmodel
    from ecopadg.planner import estimate_decode_s, estimate_mu_decode_single
    from ecopadg.predictor import WorkloadStats

    ws = os.path.abspath(os.path.join(ROOT, ".."))
    om = load_opmodel(default_table_dir("14b", ws))
    assert om is not None
    pred = type("P", (), {
        "opmodel": om,
        "stats": WorkloadStats(prompt_len=720.0, output_len=190.0),
    })()
    dec_s = estimate_decode_s(pred, 2520)
    assert dec_s is not None and dec_s > 0.5
    mu = estimate_mu_decode_single(pred, 2, 2520)
    assert mu is not None and mu > 12.0
    # n_D × LOAD_CAP / T_dec, enough that ShareGPT r=6 stays PD.
    assert mu < 40.0


def test_dry_run_cliffs_sharegpt_alpaca_longbench():
    rows = dry_run_rows("14b")
    by = {(r["dataset"], r["rate"]): r for r in rows}
    # OpModel taxed J: 14B 对 1×TP8 不买第二份权重。容量门仍在 r≥8 / 16 / 3 关上 PD。
    assert by[("sharegpt", "2")]["layout"] == "mixed"
    assert by[("sharegpt", "2")]["reason"] == "untrusted-min-j-mixed"
    assert by[("sharegpt", "4")]["layout"] == "mixed"
    assert by[("sharegpt", "6")]["layout"] == "mixed"
    assert by[("sharegpt", "8")]["layout"] == "mixed"
    assert by[("sharegpt", "8")]["reason"] == "capacity-rho-guard-mixed"
    assert by[("sharegpt", "10")]["layout"] == "mixed"
    assert by[("alpaca", "12")]["layout"] == "mixed"
    assert by[("alpaca", "12")]["reason"] == "untrusted-min-j-mixed"
    assert by[("alpaca", "16")]["layout"] == "mixed"
    assert by[("alpaca", "20")]["layout"] == "mixed"
    assert by[("alpaca", "32")]["layout"] == "mixed"
    assert by[("longbench", "2")]["layout"] == "mixed"
    assert by[("longbench", "2")]["reason"] == "untrusted-min-j-mixed"
    assert by[("longbench", "3")]["layout"] == "mixed"
    assert by[("longbench", "4")]["layout"] == "mixed"
    assert all(r["layout"] != "hybrid-a9"
               for r in rows if r["dataset"] in (
                   "sharegpt", "alpaca", "longbench"))
    tables = rows[0]["tables"].replace("\\", "/")
    assert "new-motivations" in tables


def test_dir_verdict_marks_holes_when_dest_empty(tmp_path):
    from write_dir_fullnode_verdict import evaluate, write_verdict
    payload = write_verdict(tmp_path, tmp_path / "DIR_FULLNODE_VERDICT.json")
    assert payload["dir_green"] is False
    assert payload["holes"]
    assert evaluate(payload["rows"])


def test_dir_verdict_j_win_is_saturation_total_j(tmp_path):
    from write_dir_fullnode_verdict import evaluate

    def row(rate, *, sat, pb_j, mix_j, dvfs_j, pd_j, eco_j, att=0.5):
        return dict(
            dataset="sharegpt", rate=rate, kind="mixed", missing=False,
            saturated=sat, j_win=pb_j < min(mix_j, dvfs_j, pd_j, eco_j),
            att_pdblend=att, j_pdblend=pb_j,
            att_mixed=0.4 if sat else 0.97,
            j_mixed=mix_j, j_mixed_dvfs=dvfs_j, j_strict_pd=pd_j,
            j_ecoserve=eco_j, completed=200, n_requests=200, valid_run=True)

    ok = [
        row("2", sat=False, pb_j=70e3, mix_j=60e3, dvfs_j=58e3,
            pd_j=80e3, eco_j=90e3, att=0.97),
        row("12", sat=True, pb_j=50e3, mix_j=61e3, dvfs_j=59e3,
            pd_j=80e3, eco_j=90e3, att=0.40),
    ]
    assert evaluate(ok) == []
    lose = [dict(ok[0]), dict(ok[1], j_pdblend=60e3, j_win=False)]
    holes = evaluate(lose)
    assert any("sat J not below" in h for h in holes)


def test_aggregate_joins_slo_and_marks_incomplete_run(tmp_path):
    from aggregate_iso_load import load_rows

    cell = tmp_path / "sharegpt_14b_pdblend_poisson_r12_n200_seed0"
    cell.mkdir()
    (cell / "energy_summary.csv").write_text(
        "system,model,dataset,process,rate,n,seed,gpu_count,"
        "slo_ttft_s,slo_tpot_s,total_j,slo_attainment,gpu_util,validity\n"
        "pdblend,14b,sharegpt,poisson,12,200,0,8,5.0,0.15,50000,0.4,0.8,ok\n",
        encoding="utf-8")
    (cell / "slo_summary.csv").write_text(
        "completed,ttft_avg_s,tpot_avg_s,req_throughput,output_tok_throughput\n"
        "120,1.2,0.2,1.5,300\n",
        encoding="utf-8")
    rows = load_rows([str(tmp_path)])
    assert len(rows) == 1
    assert rows[0]["ttft_avg_s"] == 1.2
    assert rows[0]["req_throughput"] == 1.5
    assert rows[0]["gpu_util"] == 0.8
    assert rows[0]["validity"] == "invalid_run"


def test_pub_bounds_ignore_invalid_rows(tmp_path):
    from write_pub_figab_bounds import load_rows, select, abstract_lines
    (tmp_path / "paired_aggregate.csv").write_text(
        "system,dataset,rate,publication_valid,energy_saving_ci_lower_pct,"
        "attainment_delta_ci_lower_pp,baseline_system\n"
        "pdblend,sharegpt,2,false,9.0,-0.2,mixed_dvfs\n"
        "pdblend,sharegpt,2,true,8.0,-0.4,mixed_dvfs\n"
        "pdblend,sharegpt,12,true,1.0,-0.2,strict_pd\n",
        encoding="utf-8")
    selected = select(load_rows(tmp_path))
    text = "\n".join(abstract_lines(selected))
    assert "r=2" in text
    assert "8.0" in text
    assert "Fig A" in text
