# -*- coding: utf-8 -*-
"""NUMA 放置:注入 numa 图,不碰真卡。"""
from __future__ import annotations

import sys
import os
_SCRIPTS = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "new-motivations", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from run_kv_xfer_pynccl import evaluate_gate  # noqa: E402

from ecopadg.gpu_topo import (
    REASON_INTERLEAVE, REASON_NUMA, _cli, interleave_place, is_p_then_d,
    parse_gpu_list, pick_pd_gpus, place_partition,
)
from ecopadg.runner import assign_gpus, assign_gpus_placement
from ecopadg.types import Partition


LINEAR = list(range(8))
NUMA_4_4 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}
# 偶数一岛、奇数一岛:线性下标会把 TP 拆开
NUMA_EVEN_ODD = {i: (0 if i % 2 == 0 else 1) for i in range(8)}


def _pd2() -> Partition:
    return Partition(n_prefill=2, n_decode=2, tp_prefill=2, tp_decode=2,
                     gpus_total=8)


def _same_island(gpus, numa):
    nodes = {numa[int(g)] for g in gpus}
    return len(nodes) == 1


def test_parse_gpu_list_range_and_csv():
    assert parse_gpu_list("0-7") == list(range(8))
    assert parse_gpu_list("0,1,2,3") == [0, 1, 2, 3]


def test_two_by_1p1d_same_island():
    p = place_partition(_pd2(), LINEAR, NUMA_4_4)
    assert p.reason == REASON_NUMA
    assert p.numa_trusted
    assert len(p.segments) == 2
    for seg in p.segments:
        assert not seg.kv_cross_numa
        assert _same_island(seg.pgpus, NUMA_4_4)
        assert _same_island(seg.dgpus, NUMA_4_4)
        assert _same_island(seg.pgpus + seg.dgpus, NUMA_4_4)
        assert len(seg.pgpus) == 2 and len(seg.dgpus) == 2
    used = [g for s in p.segments for g in s.pgpus + s.dgpus]
    assert sorted(used) == LINEAR


def test_scattered_ids_still_same_island():
    p = place_partition(_pd2(), LINEAR, NUMA_EVEN_ODD)
    assert p.reason == REASON_NUMA
    for seg in p.segments:
        assert not seg.kv_cross_numa
        assert _same_island(seg.pgpus + seg.dgpus, NUMA_EVEN_ODD)
        # TP 组不得跨岛
        assert _same_island(seg.pgpus, NUMA_EVEN_ODD)
        assert _same_island(seg.dgpus, NUMA_EVEN_ODD)
    # 不能退化成线性 0,1 / 2,3(那是跨岛 TP)
    first_p = p.segments[0].pgpus
    assert not (set(first_p) == {0, 1})


def test_reject_p_then_d_order():
    placed = assign_gpus(_pd2(), LINEAR, NUMA_4_4)
    assert not is_p_then_d(placed, LINEAR, n_segments=2, tp=2)
    # 旧顺序必须被识别为失败形态
    legacy = {
        "mixed": [],
        "prefill": [[0, 1], [2, 3]],
        "decode": [[4, 5], [6, 7]],
    }
    assert is_p_then_d(legacy, LINEAR, n_segments=2, tp=2)
    # 新放置的 P0 与 D0 必须同岛,旧顺序 D0 在另一岛
    p0 = placed["prefill"][0]
    d0 = placed["decode"][0]
    assert {NUMA_4_4[g] for g in p0 + d0} == {0}


def test_tp4_pair_cross_numa_but_tp_local():
    part = Partition(n_prefill=1, n_decode=1, tp_prefill=4, tp_decode=4,
                     gpus_total=8)
    p = place_partition(part, LINEAR, NUMA_4_4)
    assert len(p.segments) == 1
    seg = p.segments[0]
    assert seg.kv_cross_numa
    assert _same_island(seg.pgpus, NUMA_4_4)
    assert _same_island(seg.dgpus, NUMA_4_4)
    assert set(seg.pgpus + seg.dgpus) == set(LINEAR)


def test_force_interleave_and_untrusted_fallback():
    forced = place_partition(_pd2(), LINEAR, NUMA_4_4, force_interleave=True)
    assert forced.reason == REASON_INTERLEAVE
    assert forced.segments[0].pgpus == [0, 1]
    assert forced.segments[0].dgpus == [2, 3]
    assert forced.segments[1].pgpus == [4, 5]
    assert forced.segments[1].dgpus == [6, 7]
    # interleave 本身也不是 P-then-D
    assert not is_p_then_d(forced.as_assign(), LINEAR, 2, 2)
    raw = interleave_place(_pd2(), LINEAR, NUMA_4_4, trusted=False)
    assert raw.reason == REASON_INTERLEAVE
    assert not raw.numa_trusted


def test_assign_gpus_matches_place_partition():
    part = _pd2()
    a = assign_gpus(part, LINEAR, NUMA_4_4)
    b = place_partition(part, LINEAR, NUMA_4_4).as_assign()
    assert a == b
    plc = assign_gpus_placement(part, LINEAR, NUMA_4_4)
    assert plc.as_assign() == a
    assert all(not s.kv_cross_numa for s in plc.segments)


def test_layer_pp_world_size_is_tp_times_pp():
    part = Partition(n_prefill=1, n_decode=1, tp_prefill=2, tp_decode=2,
                     pp_prefill=2, pp_decode=2, gpus_total=8)
    p = place_partition(part, LINEAR, NUMA_4_4, force_interleave=True)
    assert len(p.segments) == 1
    assert len(p.segments[0].pgpus) == 4
    assert len(p.segments[0].dgpus) == 4
    used = p.segments[0].pgpus + p.segments[0].dgpus
    assert sorted(used) == LINEAR


def test_hybrid_mixed_then_pd_same_island():
    part = Partition(n_mixed=2, n_prefill=1, n_decode=1,
                     tp_mixed=2, tp_prefill=2, tp_decode=2, gpus_total=8)
    p = place_partition(part, LINEAR, NUMA_4_4)
    assert len(p.mixed) == 2
    for m in p.mixed:
        assert not m.tp_cross_numa
    assert len(p.segments) == 1
    assert not p.segments[0].kv_cross_numa


def test_assign_gpus_no_overlap_still_holds():
    part = Partition(n_mixed=1, n_prefill=1, n_decode=1, tp_mixed=2,
                     tp_prefill=2, tp_decode=2, gpus_total=8)
    a = assign_gpus(part, LINEAR, NUMA_4_4)
    flat = [g for grp in a.values() for pair in grp for g in pair]
    assert len(flat) == 6
    assert len(set(flat)) == 6


def test_cli_json(tmp_path, capsys):
    out = tmp_path / "place.json"
    rc = _cli(["--gpus", "0-7", "--n-segments", "2", "--tp-p", "2",
               "--tp-d", "2", "--force-interleave", "--out", str(out)])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "interleave-fallback" in text
    assert '"n_segments": 2' in text


def test_pick_pd_gpus_same_and_cross():
    same_p, same_d, cross_flag = pick_pd_gpus(
        LINEAR, NUMA_4_4, cross=False, tp=2)
    assert not cross_flag
    assert _same_island(same_p + same_d, NUMA_4_4)
    xp, xd, xf = pick_pd_gpus(LINEAR, NUMA_4_4, cross=True, tp=2)
    assert xf
    assert _same_island(xp, NUMA_4_4)
    assert _same_island(xd, NUMA_4_4)
    assert NUMA_4_4[xp[0]] != NUMA_4_4[xd[0]]


def _gate_row(protocol, plen, ttft, pre, energy, **kw):
    row = dict(protocol=protocol, placement="same_numa",
               prompt_tokens=plen, ttft_proxy_ms=ttft,
               prefill_ms=pre, energy_j=energy, timing_untrusted=False)
    row.update(kw)
    return row


def test_evaluate_gate_pass_on_4096():
    rows = [
        _gate_row("serial", 2048, 100.0, 80.0, 10.0),
        _gate_row("layer_async_tp", 2048, 99.0, 80.0, 10.0),
        _gate_row("serial", 4096, 200.0, 160.0, 20.0),
        _gate_row("layer_async_tp", 4096, 180.0, 161.0, 19.0),
    ]
    gate = evaluate_gate(rows)
    assert gate["passed"]
    assert gate["patch_v0"] is False
    assert any(c["status"] == "pass" for c in gate["checks"])


def test_evaluate_gate_fail_prefill_slowdown():
    rows = [
        _gate_row("serial", 4096, 200.0, 100.0, 20.0),
        _gate_row("layer_async_tp", 4096, 170.0, 120.0, 19.0),
    ]
    gate = evaluate_gate(rows)
    assert not gate["passed"]
    assert "do not patch V0" in gate["reason"]
