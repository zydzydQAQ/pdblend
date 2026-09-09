# -*- coding: utf-8 -*-
"""hybrid iteration LUT:统一 (prefill, n_decode, kv, freq) 口径。"""
from __future__ import annotations

import csv

from ecopadg.opmodel import HybridIterLUT, OpModel


def _write_lut(path):
    fields = ["prefill_tokens", "n_decode", "kv_tokens", "freq_mhz",
              "iter_p50_ms", "iter_p90_ms", "board_w"]
    rows = [
        dict(prefill_tokens=0, n_decode=1, kv_tokens=16, freq_mhz=2520,
             iter_p50_ms=10.0, iter_p90_ms=12.0, board_w=200.0),
        dict(prefill_tokens=0, n_decode=8, kv_tokens=272, freq_mhz=1500,
             iter_p50_ms=40.0, iter_p90_ms=48.0, board_w=180.0),
        dict(prefill_tokens=512, n_decode=1, kv_tokens=512, freq_mhz=2520,
             iter_p50_ms=80.0, iter_p90_ms=90.0, board_w=220.0),
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_hybrid_lut_nearest(tmp_path):
    p = str(tmp_path / "hybrid_iter.csv")
    _write_lut(p)
    lut = HybridIterLUT.from_csv(p)
    p50, p90, w = lut.query(0, 8, 272, 1500)
    assert p50 == 40.0 and p90 == 48.0 and w == 180.0
    p50, _, _ = lut.query(0, 7, 270, 1480)
    assert p50 == 40.0


def test_opmodel_uses_lut_for_iter(tmp_path, e1b_tables):
    lat, pw = e1b_tables
    hy = str(tmp_path / "hybrid_iter.csv")
    _write_lut(hy)
    om = OpModel.from_tables(lat, pw, hybrid_csv=hy)
    assert om.iter_time_ms(8, 1500, 272.0) == 40.0
    assert om.hybrid_iter_ms(512, 1, 512, 2520) == 80.0
    assert om.hybrid_iter_ms(512, 1, 512, 2520, quantile="p90") == 90.0


def test_hybrid_fallback_without_lut(e1b_tables):
    lat, pw = e1b_tables
    om = OpModel.from_tables(lat, pw)
    assert om.hybrid_lut is None
    v = om.hybrid_iter_ms(0, 8, 272, 2520)
    assert v == om.iter_time_ms(8, 2520, 272.0)
