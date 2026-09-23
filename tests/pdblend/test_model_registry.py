from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pdblend.model_registry import ModelRegistry


GEOMETRY = {
    "7b": (28, 28, 4, 3584),
    "14b": (48, 40, 8, 5120),
    "32b": (64, 40, 8, 5120),
}


def _make_models(root: Path) -> None:
    for key, (layers, heads, kv, hidden) in GEOMETRY.items():
        name = {"7b": "Qwen2.5-7B-Instruct", "14b": "Qwen2.5-14B-Instruct", "32b": "Qwen2.5-32B-Instruct"}[key]
        path = root / name
        path.mkdir(parents=True)
        (path / "config.json").write_text(json.dumps({"num_hidden_layers": layers, "num_attention_heads": heads, "num_key_value_heads": kv, "hidden_size": hidden, "torch_dtype": "bfloat16"}))


def test_config_validation_and_model_specific_predictors(tmp_path):
    _make_models(tmp_path)
    registry = ModelRegistry(tmp_path)
    assert registry.get("14b").validate_config()["hidden_size"] == 5120
    assert registry.get("14b").profile_namespace == "qwen2.5-14b"
    assert "7b" not in registry.get("14b").predictor_paths[0]
    with pytest.raises(FileNotFoundError):
        registry.get("14b").require_predictor()


def test_verification_receipt_binds_hashes_without_rehashing_weights(tmp_path):
    _make_models(tmp_path)
    files = {}
    models = {}
    for key, (layers, heads, kv, hidden) in GEOMETRY.items():
        name = {"7b": "Qwen2.5-7B-Instruct", "14b": "Qwen2.5-14B-Instruct", "32b": "Qwen2.5-32B-Instruct"}[key]
        root = tmp_path / name
        weight = {"path": "model-00001.safetensors", "bytes": 10, "sha256": "a" * 64, "kind": "weight"}
        tokenizer = {"path": "tokenizer.json", "bytes": 20, "sha256": "b" * 64, "kind": "tokenizer"}
        models[key] = {"verified": True, "model_path": str(root), "files": [weight, tokenizer]}
    receipt = {"schema": 1, "all_pass": True, "models": models}
    registry = ModelRegistry(tmp_path, verification_receipt=receipt)
    assert registry.get("7b").model_hash
    assert registry.get("7b").tokenizer_hash
    assert registry.get("32b").verification_receipt == "<mapping>"
    manifest = registry.manifest()
    assert manifest["14b"]["profile_namespace"] == "qwen2.5-14b"
    assert manifest["14b"]["predictor_paths"]


def test_receipt_path_mismatch_fails_closed(tmp_path):
    _make_models(tmp_path)
    receipt = {"schema": 1, "all_pass": True, "models": {"7b": {"verified": True, "model_path": str(tmp_path / "wrong"), "files": []}}}
    with pytest.raises(ValueError, match="path mismatch"):
        ModelRegistry(tmp_path, verification_receipt=receipt)
