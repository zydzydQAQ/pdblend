# -*- coding: utf-8 -*-
"""两层 SLO：发表主表冻死用户档；硬件档按 (model, tp, dataset)。"""
from __future__ import annotations

import os
import sys

import pytest

_SCRIPTS = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "new-motivations", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from ecopadg.types import (  # noqa: E402
    PUBLICATION_SLO_TIER, SLO_TIER_USER, SloSpec, USER_DATASET_SLO,
)
import metrics as M  # noqa: E402


def test_publication_tier_is_user():
    assert PUBLICATION_SLO_TIER == SLO_TIER_USER == "user"
    assert M.PUBLICATION_SLO_TIER == "user"


def test_user_tier_per_dataset_tpot():
    assert USER_DATASET_SLO["sharegpt"] == (5.0, 0.15)
    assert USER_DATASET_SLO["longbench"] == (15.0, 0.2)
    assert USER_DATASET_SLO["alpaca"] == (1.0, 0.1)
    assert SloSpec.from_dataset("sharegpt") == SloSpec(5.0, 0.15)
    assert SloSpec.from_dataset("alpaca-gpt4").ttft_s == 1.0
    assert SloSpec.from_dataset("alpaca-gpt4").tpot_s == 0.1


def test_from_hw_keeps_dataset_ttft_axis():
    alp = SloSpec.from_hw("14b", 2, "alpaca")
    lb = SloSpec.from_hw("14b", 2, "longbench")
    assert alp.tpot_s == pytest.approx(lb.tpot_s)
    assert lb.ttft_s > alp.ttft_s
    assert alp.tpot_s == pytest.approx(0.105, abs=0.01)


def test_hw_tpot_scales_with_model():
    t14 = SloSpec.from_hw("14b", 2, "sharegpt").tpot_s
    t32 = SloSpec.from_hw("32b", 2, "sharegpt").tpot_s
    t72 = SloSpec.from_hw("72b", 4, "sharegpt").tpot_s
    assert t14 < t32 < t72
    assert t14 == pytest.approx(0.105, abs=0.01)
    assert t32 == pytest.approx(0.230, abs=0.01)
    assert t72 == pytest.approx(0.275, abs=0.02)


def test_from_tier_default_is_user():
    assert SloSpec.from_tier("user", "longbench").ttft_s == 15.0
    hw = SloSpec.from_tier("hw", "sharegpt", model="32b", tp=2)
    assert hw.tpot_s > 0.2


def test_metrics_user_per_dataset_tpot():
    assert M.resolve_slo("sharegpt").tpot_ms == 150.0
    assert M.resolve_slo("longbench").tpot_ms == 200.0
    assert M.resolve_slo("alpaca").tpot_ms == 100.0
    assert M.resolve_slo("alpaca").ttft_ms == 1000.0
    assert M.joint_window_ok(1223.0, 24.0, "sharegpt")
    assert not M.joint_window_ok(1223.0, 24.0, "alpaca")


def test_metrics_hw_lookup():
    slo = M.resolve_slo("sharegpt", tier="hw", model="72b", tp=4)
    assert slo.tpot_ms == pytest.approx(275.0, abs=20.0)
    assert slo.ttft_ms > 4000.0
