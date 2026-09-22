"""G5 replay harness must agree with the frozen baselines on the synthetic model."""
from pathlib import Path

import pytest

from pdblend.bench.gate_g5 import gate_g5
from synthetic import synthetic_model

pytest.importorskip("pdblend_baselines")


def test_gate_g5_agrees_on_synthetic_model(tmp_path: Path):
    prof = tmp_path / "p.json"
    synthetic_model().save(prof)
    r = gate_g5(prof, tmp_path / "g5.json", trials=40, seed=3)
    assert r["eco_admission"]["agreement"] == 1.0
    assert r["eco_scaling"]["agreement"] == 1.0
    assert r["dynamo_scale_freq"]["agreement"] == 1.0
    assert all(d["port_capacity_rps"] >= d["upstream_unit_capacity_rps"] - 1e-9 for d in r["distserve"])
    assert (tmp_path / "g5.json").exists()
