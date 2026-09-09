# -*- coding: utf-8 -*-
"""profiler:统一 profile JSON 导出(调频点/延迟/功率/切换成本)。

测试先行(M1):export_unified_profile 是三池调度器与 DVFS 的共享输入 ——
  - 频率决策点:decode 饱和频率 f*、逐 batch 能量最优频率、slack 选频表;
  - 延迟表:prefill_ms(f, plen)、decode_iter_ms(f, b);
  - 功率/能耗:decode 动态 mJ/tok、prefill J/tok、驻留功率;
  - 切换成本:切频 settle/瞬态(E5)+ drain/角色切换(PE2 实测后填充)。
"""
from __future__ import annotations

import json
import os

import pytest

from ecopadg.profiler import (
    DEFAULT_SWITCH_COST, LAYER_L0, LAYER_L2, attach_lookup,
    build_instance_sweep_cells, build_lookup_table, choose_reconfig,
    classify_size_bucket, conservative_kv_ttft_s, estimate_max_num_tokens,
    export_three_way_profile, export_unified_profile, interp_lookup_freq,
    layer_enabled, load_switch_cost, load_unified_profile, lookup_config,
    lookup_reconfig, reconfig_layer, rematerialization_open,
    MIXED_TPS, PD_TP_PAIRS, shape_name, summarize_pd_table_i,
    summarize_pd_table_i_tp, switch_cost_j, synthesize_lookup_rows,
    synthesize_pd_table_i_rows,
)
from ecopadg.types import SLO_TIER_USER, USER_DATASET_SLO


@pytest.fixture
def profile_synth(synth_opmodel, tmp_path):
    out = str(tmp_path / "profile_synth.json")
    prof = export_unified_profile(
        synth_opmodel, model_name="synth-model", model_key="synth", tp=2,
        out_path=out, freq_candidates=(2520, 1800, 1500, 1200, 900, 600),
        batches=(1, 4, 8, 16), prompt_lens=(128, 1024, 2048),
        ctx_buckets=(272, 2048), slo_tpot_s=0.1, tpot_margin=0.15)
    return prof, out


def test_export_writes_json_and_roundtrips(profile_synth):
    prof, out = profile_synth
    assert os.path.exists(out)
    loaded = load_unified_profile(out)
    assert loaded == prof
    for key in ("model", "model_key", "tp", "freq_candidates",
                "decode_iter_ms", "prefill_ms", "decision", "switch_cost"):
        assert key in prof, key
    assert prof["model_key"] == "synth"
    assert prof["tp"] == 2


def test_decode_tables_cover_grid(profile_synth):
    prof, _ = profile_synth
    it = prof["decode_iter_ms"]
    assert set(it.keys()) == {"2520", "1800", "1500", "1200", "900", "600"}
    assert set(it["2520"].keys()) == {"1", "4", "8", "16"}
    # 低频更慢(Synth 线性变慢)
    assert it["600"]["8"] > it["2520"]["8"]


def test_prefill_table_monotonic_tokens(profile_synth):
    prof, _ = profile_synth
    pm = prof["prefill_ms"]["2520"]
    assert pm["128"] < pm["1024"] < pm["2048"]


def test_energy_opt_freq_is_u_shape_minimum(profile_synth):
    # SynthOpModel 能耗 U 型,1500 最优
    prof, _ = profile_synth
    eo = prof["decision"]["energy_opt_decode_freq"]
    assert eo["1"] == 1500
    assert eo["8"] == 1500


def test_slack_freq_table_prefers_energy_optimum(profile_synth):
    # 100ms SLO × margin 0.15 → 预算 85ms;Synth b=1 全档可行 → 取能量最优 1500
    prof, _ = profile_synth
    table = prof["decision"]["decode_freq_slack"]
    assert table["1"]["272"] == 1500


def test_f_star_saturation_freq(profile_synth):
    # Synth iter 随频率单调线性变化,无饱和平台 → f* = 最高频
    prof, _ = profile_synth
    assert prof["decision"]["f_star_mhz"] == 2520


def test_switch_cost_defaults_and_override(synth_opmodel, tmp_path):
    out = str(tmp_path / "p.json")
    prof = export_unified_profile(
        synth_opmodel, model_name="m", model_key="k", tp=1, out_path=out,
        switch_cost=dict(drain_s=4.2, role_switch_s=5.0))
    sc = prof["switch_cost"]
    # E5 缺省值保留
    assert sc["freq_settle_s"] == DEFAULT_SWITCH_COST["freq_settle_s"]
    assert sc["freq_transient_j"] == DEFAULT_SWITCH_COST["freq_transient_j"]
    # PE2 实测覆盖
    assert sc["drain_s"] == 4.2
    assert sc["role_switch_s"] == 5.0


def test_residency_and_prefill_energy_present(profile_synth):
    prof, _ = profile_synth
    assert prof["residency_w"] == 140.0
    assert prof["prefill_dyn_j_per_tok"] == pytest.approx(2.0e-3)


# ---------------------------------------------------------------------------
# 真表(E1b_32B)回归:32B-TP2 decode 能量最优在低频段(报告:900MHz)
# ---------------------------------------------------------------------------

def test_real_32b_profile(e1b_tables, tmp_path):
    from ecopadg.opmodel import OpModel
    lat, pw = e1b_tables
    om = OpModel.from_tables(lat, pw)
    out = str(tmp_path / "profile_32b.json")
    prof = export_unified_profile(
        om, model_name="Qwen2.5-32B-Instruct", model_key="32b", tp=2,
        out_path=out)
    eo = prof["decision"]["energy_opt_decode_freq"]
    # E1b_32B 报告:能量最优 900MHz,f*≈1200 —— 允许插值误差,断言在低频段
    assert int(eo["8"]) <= 1500
    assert prof["decision"]["f_star_mhz"] <= 1800
    assert prof["residency_w"] > 0.0
    # slack 表全部落在候选档内
    for b, row in prof["decision"]["decode_freq_slack"].items():
        for ctx, f in row.items():
            assert int(f) in prof["freq_candidates"]


def test_default_slack_is_sharegpt_user_tier(synth_opmodel, tmp_path):
    out = str(tmp_path / "profile_user.json")
    prof = export_unified_profile(
        synth_opmodel, model_name="synth-model", model_key="synth", tp=2,
        out_path=out, freq_candidates=(2520, 1500, 900),
        batches=(1, 8), prompt_lens=(128,), ctx_buckets=(272,))
    assert prof["slo_tier"] == SLO_TIER_USER
    assert prof["slo_tpot_s"] == USER_DATASET_SLO["sharegpt"][1]
    by_ds = prof["decision"]["decode_freq_slack_by_dataset"]
    assert set(by_ds) == set(USER_DATASET_SLO)
    assert prof["decision"]["decode_freq_slack"] == by_ds["sharegpt"]


def test_conservative_kv_ttft_constraint(synth_opmodel, tmp_path):
    assert conservative_kv_ttft_s(512) == (0.0, "untrusted")
    assert conservative_kv_ttft_s(2048) == (0.0, "untrusted")
    same_s, same_trust = conservative_kv_ttft_s(4096)
    cross_s, cross_trust = conservative_kv_ttft_s(4096, cross_numa=True)
    assert same_trust == cross_trust == "b5-4096"
    assert same_s == 0.0123
    assert cross_s == 0.0283
    scaled, _ = conservative_kv_ttft_s(8192)
    assert scaled == pytest.approx(0.0246)
    out = str(tmp_path / "profile_kv.json")
    prof = export_unified_profile(
        synth_opmodel, model_name="m", model_key="k", tp=2, out_path=out,
        freq_candidates=(2520,), batches=(1,), prompt_lens=(128,),
        ctx_buckets=(272,))
    kv = prof["constraints"]["kv_ttft"]
    assert kv["energy_objective"] is False
    assert kv["by_prompt_len"]["512"]["trust"] == "untrusted"
    assert kv["by_prompt_len"]["4096"]["ttft_s"] == 0.0123


def test_three_way_and_max_tokens(synth_opmodel, tmp_path):
    assert estimate_max_num_tokens(46.0, 20.0, 160.0 * 1024) > 0
    cells = build_instance_sweep_cells("prefill", freqs=(2520,),
                                       batches=(1,), prompt_lens=(128,),
                                       ctxs=(256,))
    assert cells[0]["kind"] == "prefill"
    out = export_three_way_profile(
        synth_opmodel, "synth", "synth", tp=2, out_dir=str(tmp_path),
        max_num_tokens=4096)
    assert set(out) == {"mixed", "prefill", "decode"}
    assert out["prefill"]["max_num_tokens"] == 4096
    assert out["decode"]["batching"]["decode_max_batch"] >= 1


def _lookup_row(**kw):
    row = dict(model="14b", layout="mixed", tp=2, freq_p=2520, freq_d=2520,
               freq_mhz=2520, size_bucket="medium", batch=1,
               ttft_s=0.4, tpot_s=0.05, energy_j=100.0)
    row.update(kw)
    return row


def test_classify_size_bucket_nearest():
    assert classify_size_bucket(128, 64) == "short"
    assert classify_size_bucket(512, 200) == "medium"
    assert classify_size_bucket(2048, 200) == "long"
    assert classify_size_bucket(140, 70) == "short"


def test_lookup_picks_min_energy_feasible():
    rows = [
        _lookup_row(energy_j=100, freq_p=2520, freq_d=2520),
        _lookup_row(energy_j=70, freq_p=1500, freq_d=1500),
        _lookup_row(energy_j=50, freq_p=900, freq_d=900, tpot_s=0.20),
        _lookup_row(layout="spatial_pd", energy_j=60,
                    freq_p=2520, freq_d=1500),
    ]
    cfg = lookup_config(rows, "medium", "14b", "sharegpt")
    assert cfg is not None
    assert cfg["layout"] == "spatial_pd"
    assert cfg["freq_p"] == 2520
    assert cfg["freq_d"] == 1500
    assert cfg["energy_j"] == pytest.approx(60.0)


def test_lookup_infeasible_returns_none():
    rows = [_lookup_row(ttft_s=2.0, tpot_s=0.05)]
    assert lookup_config(rows, "medium", "14b", "alpaca") is None


def test_lookup_defaults_to_batch_one():
    rows = [
        _lookup_row(batch=1, energy_j=100, freq_p=2520, freq_d=2520),
        _lookup_row(batch=8, energy_j=10, freq_p=900, freq_d=900),
    ]
    cfg = lookup_config(rows, "medium", "14b", "sharegpt")
    assert cfg["freq_p"] == 2520
    cfg8 = lookup_config(rows, "medium", "14b", "sharegpt", batch=8)
    assert cfg8["freq_p"] == 900


def test_lookup_averages_repeats():
    rows = [
        _lookup_row(energy_j=80, freq_p=1500, freq_d=1500, repeat=0),
        _lookup_row(energy_j=100, freq_p=1500, freq_d=1500, repeat=1),
        _lookup_row(energy_j=90, freq_p=2520, freq_d=2520, repeat=0),
    ]
    cfg = lookup_config(rows, "medium", "14b", "sharegpt")
    assert cfg["freq_p"] == 1500
    assert cfg["energy_j"] == pytest.approx(90.0)


def test_build_and_attach_lookup():
    rows = [
        _lookup_row(size_bucket="short", energy_j=40),
        _lookup_row(size_bucket="medium", energy_j=80),
        _lookup_row(size_bucket="long", energy_j=120, ttft_s=2.0),
    ]
    table = build_lookup_table(rows, model="14b")
    assert "short|14b|alpaca" in table["keys"]
    assert "medium|14b|sharegpt" in table["keys"]
    assert table["keys"]["short|14b|alpaca"]["layout"] == "mixed"
    assert table["keys"]["short|14b|alpaca"]["freq_d_b1"] == 2520


def test_interp_lookup_freq_snaps_between_batches():
    table = dict(keys={
        "medium|14b|sharegpt": dict(
            freq_d=1500, freq_d_b1=900, freq_d_b8=1800),
    })
    assert interp_lookup_freq(table, "medium", "14b", "sharegpt", 1) == 900
    assert interp_lookup_freq(table, "medium", "14b", "sharegpt", 8) == 1800
    mid = interp_lookup_freq(table, "medium", "14b", "sharegpt", 4.5)
    assert mid in (900, 1800)


def test_synthesize_lookup_rows_from_synth(synth_opmodel):
    rows = synthesize_lookup_rows(synth_opmodel, model="14b")
    assert rows
    assert {r["layout"] for r in rows} == {"mixed", "spatial_pd"}
    table = build_lookup_table(rows, model="14b")
    assert table["keys"]
    cfg = table["keys"].get("medium|14b|sharegpt")
    assert cfg is not None
    assert int(cfg["freq_d_b1"]) > 0
    prof = attach_lookup({"model_key": "14b"}, rows)
    assert "lookup" in prof
    assert prof["lookup"]["model"] == "14b"
    assert "switch_cost" in prof
    assert "role_switch_cost_j" in prof["switch_cost"]
    assert "freq_transient_j" in prof["switch_cost"]


def test_shape_name_crossed_not_diagonal():
    assert shape_name(128, 64) == "SS"
    assert shape_name(128, 200) == "SM"
    assert shape_name(2048, 64) == "LS"
    assert classify_size_bucket(128, 64) == "short"
    assert classify_size_bucket(2048, 200) == "long"


def test_pd_table_i_unsets_diagonal_and_splits_energy(synth_opmodel):
    rows = synthesize_pd_table_i_rows(synth_opmodel, model="14b")
    assert {r["shape"] for r in rows} == {"SS", "SM", "MS", "MM", "LS", "LM"}
    lookup = synthesize_lookup_rows(synth_opmodel, model="14b")
    assert {r["size_bucket"] for r in lookup} == {"short", "medium", "long"}
    pd = [r for r in rows if r["layout"] == "spatial_pd"]
    assert pd
    assert all(int(r["freq_p"]) == 2520 for r in pd)
    sm = next(r for r in rows if r["shape"] == "SM" and r["batch"] == 8
              and r["layout"] == "spatial_pd" and r["freq_d"] == 1500
              and int(r["tp_p"]) == 2 and int(r["tp_d"]) == 2)
    ls = next(r for r in rows if r["shape"] == "LS" and r["batch"] == 8
              and r["layout"] == "spatial_pd" and r["freq_d"] == 1500
              and int(r["tp_p"]) == 2 and int(r["tp_d"]) == 2)
    assert sm["e_pre_j"] < ls["e_pre_j"]
    assert sm["e_dec_j"] > ls["e_dec_j"]
    assert ls["e_pre_frac"] > sm["e_pre_frac"]
    assert sm["kv_trust"] == "untrusted"
    assert ls["energy_j_taxed"] > ls["energy_j"]
    summary = summarize_pd_table_i(rows, slo_name="sharegpt", batch=8)
    by = {r["shape"]: r for r in summary}
    assert set(by) == {"SS", "SM", "MS", "MM", "LS", "LM"}
    assert by["SM"]["pd_e_pre_frac"] < by["LS"]["pd_e_pre_frac"]


def test_pd_table_i_tp_same_tp_moves_freq(synth_opmodel):
    rows = synthesize_pd_table_i_rows(
        synth_opmodel, model="14b", mixed_tps=MIXED_TPS,
        pd_tp_pairs=PD_TP_PAIRS)
    assert {(int(r["tp_p"]), int(r["tp_d"])) for r in rows
            if r["layout"] == "spatial_pd"} == set(PD_TP_PAIRS)
    tp2 = [r for r in rows if r["batch"] == 8 and r["shape"] == "SM"
           and int(r["tp_p"]) == 2 and int(r["tp_d"]) == 2]
    mixed = [r for r in tp2 if r["layout"] == "mixed"]
    pd = [r for r in tp2 if r["layout"] == "spatial_pd"]
    assert mixed and pd
    assert all(int(r["freq_p"]) == int(r["freq_d"]) for r in mixed)
    assert all(int(r["freq_p"]) == 2520 for r in pd)
    summary = summarize_pd_table_i_tp(rows, slo_name="sharegpt", batch=8)
    sm = next(r for r in summary if r["shape"] == "SM")
    assert "2" in sm["same_tp"]
    slice2 = sm["same_tp"]["2"]
    assert slice2["pd_freq_p"] == 2520
    assert int(slice2["pd_freq_d"]) != int(slice2["pd_freq_p"])
    ls = next(r for r in summary if r["shape"] == "LS")
    sm_pd = [r for r in rows if r["shape"] == "SM" and r["batch"] == 8
             and r["layout"] == "spatial_pd" and r["freq_d"] == 1500]
    ls_pd = [r for r in rows if r["shape"] == "LS" and r["batch"] == 8
             and r["layout"] == "spatial_pd" and r["freq_d"] == 1500]
    sm21 = next(r for r in sm_pd if (r["tp_p"], r["tp_d"]) == (2, 1))
    sm22 = next(r for r in sm_pd if (r["tp_p"], r["tp_d"]) == (2, 2))
    ls21 = next(r for r in ls_pd if (r["tp_p"], r["tp_d"]) == (2, 1))
    ls22 = next(r for r in ls_pd if (r["tp_p"], r["tp_d"]) == (2, 2))
    assert sm21["energy_j"] < sm22["energy_j"]
    assert ls21["energy_j"] < ls22["energy_j"]
    assert ls.get("pd_tp_d") is not None
    assert int(ls["pd_tp_d"]) <= int(ls["pd_tp_p"] or 0)


def test_layer_unknown_blocks_l1_l2():
    assert layer_enabled(DEFAULT_SWITCH_COST, "freq")
    assert not layer_enabled(DEFAULT_SWITCH_COST, "tp")
    assert not layer_enabled(DEFAULT_SWITCH_COST, "layout")
    assert not layer_enabled(dict(role_switch_cost_j=float("nan")), "layout")
    assert rematerialization_open(True, DEFAULT_SWITCH_COST) is False
    assert rematerialization_open(True, dict(role_switch_cost_j=1e5)) is True
    assert rematerialization_open(False, dict(role_switch_cost_j=1e5)) is False


def test_reconfig_layer_and_unknown_cost():
    cur = dict(layout="mixed", tp=2, freq_p=2520, freq_d=2520)
    assert reconfig_layer(cur, dict(layout="mixed", tp=2,
                                    freq_p=1500, freq_d=1500)) == "freq"
    assert reconfig_layer(cur, dict(layout="mixed", tp=1,
                                    freq_p=2520, freq_d=2520)) == "tp"
    assert reconfig_layer(cur, dict(layout="spatial_pd", tp=2,
                                    freq_p=2520, freq_d=1500)) == "layout"
    assert switch_cost_j(cur, cur) == 0.0
    assert switch_cost_j(cur, dict(layout="spatial_pd", tp=2,
                                   freq_p=2520, freq_d=1500)) is None


def test_choose_reconfig_stays_l0_when_l2_unmeasured():
    current = dict(layout="mixed", tp=2, freq_p=2520, freq_d=2520,
                   energy_j=100)
    cands = [
        current,
        dict(layout="mixed", tp=2, freq_p=1500, freq_d=1500, energy_j=80),
        dict(layout="spatial_pd", tp=2, freq_p=2520, freq_d=1500,
             energy_j=20),
        dict(layout="mixed", tp=1, freq_p=1500, freq_d=1500, energy_j=10),
    ]
    out = choose_reconfig(current, cands)
    assert out is not None
    assert out["layout"] == "mixed"
    assert out["tp"] == 2
    assert out["freq_d"] == 1500
    assert out["reconfig_layer"] == LAYER_L0


def test_choose_reconfig_l2_when_measured_and_amortized():
    costs = dict(DEFAULT_SWITCH_COST)
    costs["role_switch_cost_j"] = 50.0
    current = dict(layout="mixed", tp=2, freq_p=2520, freq_d=2520,
                   energy_j=100)
    cands = [
        current,
        dict(layout="mixed", tp=2, freq_p=1500, freq_d=1500, energy_j=95),
        dict(layout="spatial_pd", tp=2, freq_p=2520, freq_d=1500,
             energy_j=40),
    ]
    out = choose_reconfig(current, cands, costs)
    assert out["layout"] == "spatial_pd"
    assert out["reconfig_layer"] == LAYER_L2


def test_choose_reconfig_hysteresis_holds():
    current = dict(layout="mixed", tp=2, freq_p=2520, freq_d=2520,
                   energy_j=100)
    cands = [
        current,
        dict(layout="mixed", tp=2, freq_p=1500, freq_d=1500, energy_j=82),
    ]
    out = choose_reconfig(current, cands)
    assert out["freq_d"] == 2520
    assert out["reconfig_reason"] == "hysteresis"


def test_lookup_reconfig_prefers_freq():
    rows = [
        _lookup_row(energy_j=100, freq_p=2520, freq_d=2520),
        _lookup_row(energy_j=70, freq_p=1500, freq_d=1500),
        _lookup_row(layout="spatial_pd", energy_j=40,
                    freq_p=2520, freq_d=1500),
    ]
    current = dict(layout="mixed", tp=2, freq_p=2520, freq_d=2520,
                   energy_j=100)
    out = lookup_reconfig(rows, current, "medium", "14b", "sharegpt")
    assert out["layout"] == "mixed"
    assert out["freq_d"] == 1500


def test_load_switch_cost_explicit_json(tmp_path):
    path = tmp_path / "switch_cost.json"
    path.write_text(json.dumps(dict(
        switch_cost=dict(role_switch_cost_j=1234.0, instance_start_s=80.0),
    )), encoding="utf-8")
    sc = load_switch_cost(str(path), ws=str(tmp_path))
    assert sc["role_switch_cost_j"] == 1234.0
    assert sc["instance_start_s"] == 80.0
    assert sc["freq_transient_j"] == DEFAULT_SWITCH_COST["freq_transient_j"]
    empty = load_switch_cost(str(tmp_path / "missing.json"), ws=str(tmp_path))
    assert empty["role_switch_cost_j"] is None


def test_phase_mu_matches_single_request_prefill(synth_opmodel):
    from ecopadg.profiler import mu_prefill, node_energy_j, prefill_s
    t = prefill_s(synth_opmodel, 2048, 2520, 2)
    assert abs(t - (50.0 + 0.05 * 2048) / 1000.0) < 1e-9
    assert abs(mu_prefill(synth_opmodel, 2048, 2, 2520, 2) - 2.0 / t) < 1e-9
    short = prefill_s(synth_opmodel, 128, 2520, 2)
    assert short < t
    prof = dict(
        freqs={"2520": {"gpu_w": 100.0}},
        idle_empty_w=35.0, loaded_idle_w=76.0)
    mixed = node_energy_j(8, 2520, 2.0, 200, prof, layout="mixed")
    pd = node_energy_j(8, 2520, 2.0, 200, prof, layout="spatial_pd")
    assert pd > mixed
    four = node_energy_j(4, 2520, 2.0, 200, prof, layout="mixed")
    assert four < mixed
