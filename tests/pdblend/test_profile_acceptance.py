import json
from pathlib import Path

import pytest

from pdblend.profile.acceptance import m2_gate, relative_error, validate_parallel_layout
from pdblend.profile.merge import merge_raw


def test_relative_error_uses_observed_denominator():
    assert relative_error(11, 10) == pytest.approx(.1)
    assert relative_error(9, 10) == pytest.approx(.1)


def test_m2_gate_requires_all_three_seeds_and_tails():
    rows = [dict(seed=s, complete=True, joint_slo_rate=1.0, ttft_p99=1.0,
                 tpot_p99=.1, power_error=.01) for s in (701, 1701, 2701)]
    assert m2_gate(rows)["passed"]
    rows[-1]["tpot_p99"] = .151
    assert not m2_gate(rows)["passed"]


def test_merge_rejects_overlapping_frequency_shards(tmp_path):
    base = dict(schema=2, model="m", tp=1, freqs=[2100], kv_capacity_tokens=1,
                kv_bytes_per_token=1, config=dict(decode_repeats=3, decode_settle_s=2,
                decode_measure_s=5, decode_batches=[1]), environment=dict(
                image_digest="i", source_hash="s", vllm="v", torch="t", cuda="c",
                python="p", gpu_uuids=["GPU-a"]), prefill=[], decode=[], mixed=[],
                static={}, transfer=[], freq_switch_s=[])
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(base)); base["environment"]["gpu_uuids"] = ["GPU-b"]; b.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="overlapping"):
        merge_raw([a, b], tmp_path / "out")


def test_parallel_layout_validation_checks_ownership_and_concurrency():
    layout = {
        "instances": [{"instance_id": "i0", "gpus": [2, 3], "tp": 2, "pp": 1}],
        "gpus": [2, 3],
    }
    raw = {
        "parallel_layout": layout,
        "concurrency": {"decode_max_batch": 4, "mixed_background_max_batch": 4,
                         "prefill_inflight": 1, "transfer_inflight": 1},
        "prefill": [{"concurrency": 1, "parallel_layout": layout}],
        "decode": [{"batch": 4, "concurrency": 4, "parallel_layout": layout}],
        "mixed": [{"batch": 4, "concurrency": 4, "parallel_layout": layout}],
        "transfer": [{"concurrency": 1, "parallel_layout": layout}],
    }
    assert validate_parallel_layout(raw)["passed"]
    raw["decode"][0]["concurrency"] = 1
    result = validate_parallel_layout(raw)
    assert not result["passed"]
    assert any("decode[0]" in failure for failure in result["failures"])


def test_legacy_profile_without_parallel_metadata_is_skipped():
    result = validate_parallel_layout({"decode": [{"batch": 8}]})
    assert result == {"passed": True, "skipped": True, "failures": []}
