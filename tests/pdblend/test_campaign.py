from pathlib import Path

import pytest

from pdblend.bench.campaign import SEEDS, SEED_POLICY, audit_campaign, build_campaign, write_campaign
from pdblend.model_registry import ModelRegistry
from pdblend.profile.identity import ProfileKey, require_profile_identity


def test_registry_geometry_and_memory_filters(tmp_path):
    registry = ModelRegistry(tmp_path)
    seven = registry.get("7b")
    assert seven.legal_tp(require_memory=False) == (1, 2, 4)
    assert 8 not in seven.legal_tp(require_memory=False)
    thirty_two = registry.get("32b")
    assert 1 not in thirty_two.legal_tp()
    assert (2, 1) in thirty_two.legal_topologies()


def test_campaign_is_three_model_and_fail_closed(tmp_path):
    payload = build_campaign(models_dir=tmp_path, corpus_root=tmp_path, formal=True, include_pp=True)
    assert {m["model_id"] for m in payload["models"].values()} == {
        "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct"
    }
    assert payload["seeds"] == list(SEEDS)
    assert payload["seeds"] == [701]
    assert payload["single_seed"] is True
    assert payload["seed_policy"] == SEED_POLICY
    assert all(p["seed"] == 701 and p["single_seed"] for p in payload["points"])
    assert payload["summary"]["points"] > 0
    assert all(p["status"] in {"missing_corpus", "unsupported_engine"} for p in payload["points"])
    out = write_campaign(tmp_path / "spec.json", models_dir=tmp_path, corpus_root=tmp_path)
    audit = audit_campaign(tmp_path / "spec.json")
    assert not audit["formal_eligible"]
    assert "incomplete profile/corpus/engine coverage" in audit["reasons"]


def test_campaign_only_profiles_pdblend_tp_with_symmetric_pair_capacity(tmp_path):
    payload = build_campaign(models_dir=tmp_path, corpus_root=tmp_path, include_pp=True)
    expected = {
        "Qwen2.5-7B-Instruct": {(1, 1), (2, 1), (4, 1)},
        "Qwen2.5-14B-Instruct": {(1, 1), (2, 1), (4, 1)},
        "Qwen2.5-32B-Instruct": {(2, 1), (4, 1)},
    }
    for model, topologies in expected.items():
        assert {(p["tp"], p["pp"]) for p in payload["points"]
                if p["model"] == model and p["system"] == "pdblend"} == topologies
        profiles = [p for p in payload["profiles"] if p["model_id"] == model and p["system"] == "pdblend"]
        assert {(p["tp"], p["pp"]) for p in profiles} == topologies
        assert len(profiles) == len(topologies) * 3
    # DistServe retains offline PP search; the TP-only restriction is not
    # propagated into the independent baseline's search space.
    assert any(p["system"] == "distserve" and p["pp"] > 1 for p in payload["points"])


def test_no_pp_filters_existing_topologies_instead_of_rewriting_them(tmp_path):
    payload = build_campaign(models_dir=tmp_path, corpus_root=tmp_path, include_pp=False)
    assert all(p["pp"] == 1 for p in payload["points"] + payload["profiles"])
    assert len({p["name"] for p in payload["points"]}) == len(payload["points"])
    assert not any(p["model"] == "Qwen2.5-32B-Instruct" and p["tp"] == 1 for p in payload["points"])
    assert not any(p["model_id"] == "Qwen2.5-32B-Instruct" and p["tp"] == 1 for p in payload["profiles"])


def test_campaign_rejects_historical_seed_schedule(tmp_path):
    with pytest.raises(ValueError, match="single_seed_701"):
        build_campaign(models_dir=tmp_path, corpus_root=tmp_path, seeds=(701, 1701, 2701))


def test_profile_identity_is_strict():
    key = ProfileKey("pdblend", "Qwen2.5-7B-Instruct", "vllm-0.10.1.1", "l20-test", 2)
    raw = {"profile_key": key.as_dict()}
    require_profile_identity(raw, expected=key)
    raw["profile_key"]["tp"] = 1
    try:
        require_profile_identity(raw, expected=key)
    except ValueError as exc:
        assert "tp" in str(exc)
    else:
        raise AssertionError("identity mismatch was accepted")
