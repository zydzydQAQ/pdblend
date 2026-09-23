"""The active workload schedule must not be confused with profiler repeats."""
import json
from pathlib import Path
import runpy
import sys

import pytest

from pdblend.bench.campaign import audit_campaign
from pdblend.bench.dominance import aggregate, compare_seed
from pdblend.bench.matrix import _write_evidence, run_matrix
from pdblend.seed_config import SEEDS, SEED_POLICY, has_active_seeds, seed_metadata


def test_active_schedule_requires_one_701_result_without_duplicates():
    assert has_active_seeds([701])
    assert not has_active_seeds([])
    assert not has_active_seeds([701, 701])
    assert not has_active_seeds([701, 1701, 2701])
    assert seed_metadata() == dict(seeds=[701], single_seed=True, seed_policy=SEED_POLICY)
    assert seed_metadata([1701])["seed_policy"] == "custom"


def test_matrix_evidence_binds_actual_seed_marker(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "prompts.json").write_text("[]")
    out = tmp_path / "result"
    out.mkdir()
    (out / "summary.json").write_text("{}")
    args = dict(profile=profile, dataset="sharegpt", rate=1, seed=701,
                duration=300, model="m", tp=1, gpus="0")
    result = dict(window_s=300, requests=0, policy={"name": "mixed"})
    _write_evidence(out, args, result, [], corpus)
    evidence = json.loads((out / "evidence.json").read_text())
    assert evidence["seed_policy"] == evidence["identity"]["seed_policy"] == SEED_POLICY
    assert evidence["single_seed"] is evidence["identity"]["single_seed"] is True
    assert evidence["identity"]["seed"] == 701
    args["seed"] = 1701
    _write_evidence(out, args, result, [], corpus)
    assert json.loads((out / "evidence.json").read_text())["seed_policy"] == "custom"


def test_active_matrix_refuses_old_seed_before_skipping_existing_summary(tmp_path):
    out = tmp_path / "result"
    out.mkdir()
    (out / "summary.json").write_text("{}")
    (out / "evidence.json").write_text(json.dumps(dict(status="complete", returncode=0)))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(dict(root=str(tmp_path), **seed_metadata(),
                                   points=[dict(name="result", seed=1701)])))
    with pytest.raises(ValueError, match=SEED_POLICY):
        run_matrix(spec, dry=True)
    spec.write_text(json.dumps(dict(root=str(tmp_path), **seed_metadata(),
                                   points=[dict(name="result", seed=701)])))
    (out / "evidence.json").write_text(json.dumps(dict(status="complete", returncode=0,
                                                      identity={"seed": 1701})))
    with pytest.raises(RuntimeError, match="evidence violates"):
        run_matrix(spec, dry=True)


def test_campaign_header_does_not_hide_wrong_seed_point(tmp_path):
    spec = tmp_path / "campaign.json"
    payload = dict(schema=1, **seed_metadata(),
                   points=[dict(status="ready", seed=1701, **seed_metadata())])
    spec.write_text(json.dumps(payload))
    assert "invalid point seed policy" in audit_campaign(spec)["reasons"]
    payload["points"] = []
    spec.write_text(json.dumps(payload))
    assert "missing matrix points" in audit_campaign(spec)["reasons"]


def test_dominance_accepts_one_seed_but_rejects_historical_and_duplicate_rows():
    row = dict(seed=SEEDS[0], status="screen_pass", metrics={})
    result = aggregate([row])
    assert result["complete"] and result["status"] == "win"
    assert not aggregate([row, row])["complete"]
    historical = dict(identity={"seed": 1701}, metrics={})
    assert compare_seed(historical, {})["status"] == "inconclusive"


def test_adaptive_collector_uses_one_seed_and_does_not_invent_variance(tmp_path, monkeypatch):
    from pdblend.profile import validation

    requested = []

    def benchmark(folder, profile, seed, *args):
        requested.append(seed)
        return (dict(slo=dict(joint_slo_rate=1, ttft_p90=1, ttft_p99=1, tpot_p99=.1),
                     controller={"events": {}}, mean_power_w=100, window_mean_power_w=100,
                     j_per_token=1), None, {})

    monkeypatch.setattr(validation, "benchmark_evidence", benchmark)
    monkeypatch.setattr(validation, "verify_audit_binding", lambda _: "profile-digest")
    out = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["collector", str(tmp_path), str(tmp_path / "profile.json"), str(out)])
    script = Path(__file__).resolve().parents[2] / "scripts/collect_decode900_adaptive.py"
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(script), run_name="__main__")
    assert exited.value.code == 0
    result = json.loads(out.read_text())
    assert requested == [701]
    assert result["gate"]["passed"]
    assert result["seed_policy"] == SEED_POLICY
    assert all(stats["std"] is None and stats["n"] == 1 for stats in result["stats"].values())
