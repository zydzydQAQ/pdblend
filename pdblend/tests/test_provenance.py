# -*- coding: utf-8 -*-
"""不可变 benchmark provenance 与 fail-closed reuse。"""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from script.bench.provenance import (
    _profile_dir,
    build_cell_record,
    compatible,
    main as provenance_main,
    source_tree_digest,
    write_json,
)


def _args(root: Path, output: Path):
    trace = root / "trace.json"
    trace.write_text('[{"t": 0.0}]', encoding="utf-8")
    profile = root / "profile"
    profile.mkdir(exist_ok=True)
    (profile / "p1b_latency.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    model = root / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text('{"model":"tiny"}', encoding="utf-8")
    return SimpleNamespace(
        output=str(output), source_root=str(root), trace=str(trace),
        profile_dir=str(profile), model_dir=str(model), image="missing:image",
        system="pdblend", model_key="14b", dataset="sharegpt",
        process="poisson", rate=2.0, n=80, seed="0", gpu_count=8,
        slo_ttft=5.0, slo_tpot=0.15, slo_tier="user",
        baseline_att=0.975, baseline_source="same-campaign",
        start_record="",
    )


def test_source_tree_digest_is_deterministic_and_ignores_results(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("x = 1\n", encoding="utf-8")
    ignored = tmp_path / "script" / "bench" / "results"
    ignored.mkdir(parents=True)
    (ignored / "large.csv").write_text("old\n", encoding="utf-8")
    first = source_tree_digest(str(tmp_path))
    (ignored / "large.csv").write_text("new\n", encoding="utf-8")
    second = source_tree_digest(str(tmp_path))
    assert first["sha256"] == second["sha256"]
    (src / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert source_tree_digest(str(tmp_path))["sha256"] != first["sha256"]


def test_cell_identity_round_trip_and_change_detection(tmp_path):
    args = _args(tmp_path, tmp_path / "provenance.json")
    first = build_cell_record(args)
    second = build_cell_record(args)
    assert first["record_id"] == second["record_id"]
    assert first["profile"]["files"] == 1
    assert first["identity"]["profile_sha256"] != (
        "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855")
    assert compatible(first, second)
    write_json(args.output, first)
    loaded = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert loaded["record_id"] == first["record_id"]
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src" / "new.py").write_text("changed = True\n", encoding="utf-8")
    changed = build_cell_record(args)
    assert not compatible(first, changed)


def test_profile_dir_finds_14b_tables_from_pdblend_root():
    root = Path(__file__).resolve().parents[1]
    found = _profile_dir(str(root), "14b")
    assert (Path(found) / "p1b_latency.csv").is_file()
    assert "pdblend/pdblend/" not in found.replace("\\", "/")


def test_missing_or_nan_baseline_serializes_as_null(tmp_path):
    args = _args(tmp_path, tmp_path / "provenance.json")
    args.baseline_att = float("nan")
    record = build_cell_record(args)
    assert record["cell"]["baseline_att"] is None
    write_json(args.output, record)
    assert json.loads(Path(args.output).read_text())["cell"]["baseline_att"] is None


def test_reuse_defaults_to_off(monkeypatch, tmp_path):
    monkeypatch.delenv("REUSE_POLICY", raising=False)
    path = (
        Path(__file__).resolve().parents[1]
        / "new-results" / "scripts" / "asplos_matrix.py"
    )
    spec = importlib.util.spec_from_file_location("asplos_matrix_prov_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.find_reuse(
        "sharegpt", "14b", "mixed", "poisson", "2", 80,
        root=str(tmp_path)) == ""


def test_campaign_manifest_preserves_mutable_cell_statuses(
        monkeypatch, tmp_path):
    plan = tmp_path / "plan.json"
    output = tmp_path / "manifest.json"
    plan.write_text(json.dumps({
        "protocol": {"wave": "smoke"},
        "cells": [{"tag": "cell-0", "status": "pending"}],
    }), encoding="utf-8")
    monkeypatch.setattr("sys.argv", [
        "provenance.py", "campaign",
        "--output", str(output),
        "--source-root", str(tmp_path),
        "--plan", str(plan),
        "--images", "missing:image",
    ])
    assert provenance_main() == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["cells"] == [{"tag": "cell-0", "status": "pending"}]
    assert manifest["status_counts"]["pending"] == 1

