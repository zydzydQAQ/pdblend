from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "2026-09-22_prepare_three_model_corpora.py"
spec = importlib.util.spec_from_file_location("prepare_three_model_corpora", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


@pytest.mark.historical
def test_frozen_source_splits_are_exact_and_disjoint():
    base = Path("/home/pdblend/datasets/prepared/2026-09-13-7b-v1")
    if not base.exists():
        pytest.skip("frozen source corpus is an external mounted input")
    for dataset in ("alpaca", "sharegpt", "longbench"):
        splits = module.load_split_sources(base, dataset)
        assert {name: len(rows) for name, rows in splits.items()} == module.EXPECTED_SIZES
        assert len(set().union(*(set(rows) for rows in splits.values()))) == 2012


class _Tokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return list(range(len(messages[0]["content"]) + 12))

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text) + 2))


def test_encode_has_no_reference_and_middle_truncates():
    record = module.encode_workload(_Tokenizer(), [{"role": "user", "content": "x" * 40}], "secret answer", max_input=8, max_output=5)
    assert "secret answer" not in repr(record)
    assert record["input_tokens"] == 8
    assert record["output_tokens"] == 5
    assert record["truncated"] is True
