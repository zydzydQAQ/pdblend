import json
from pathlib import Path

import pytest

from pdblend.profile.acceptance import m2_gate, relative_error
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
