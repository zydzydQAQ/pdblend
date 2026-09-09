import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

import protocol as p
import trace_source as t


@pytest.fixture(scope="module")
def source():
    return t.load_source()


def test_existing_actual_sharegpt_work_and_arrivals_are_unchanged(source):
    # Independently compare with a historical, frozen 100s trace: exact original
    # prompts, lengths, split indices and arrivals, not just summary statistics.
    manifest = json.loads((t.CAMPAIGN / "five-system-fixed-window-v1/sources/A14B/manifest.json").read_text())
    row = next(row for row in manifest["workloads"]
               if row["dataset"] == "sharegpt" and row["rate_rps"] == 0.4)
    original_path = t.frozen(dict(path=row["trace"], sha256=row["trace_sha256"]))
    original = json.loads(original_path.read_text())
    current = t.build_trace("0.4", source)
    for field in ("requests", "prompts", "source_shapes", "source_pool_indices", "content_pairing_sha256"):
        assert current[field] == original[field]
    assert current["n_requests"] == original["n_requests"]
    assert current["protocol_id"] != original["protocol_id"]
    assert current["allowed_slo_scales"] == [0.5, 2.0]


def test_arbitrary_extension_rate_and_full_prescribed_outputs(source):
    current = t.build_trace("20.25", source)
    assert current["rate_rps_decimal"] == "20.25"
    assert current["within_trace_resampling"] is True
    assert current["requests"][0]["arrival_s"] == 0
    assert current["requests"][-1]["arrival_s"] < 100
    for request, prompt, index in zip(current["requests"], current["prompts"], current["source_pool_indices"]):
        original = source["records"][index]
        assert prompt == original["prompt"]
        assert request["prompt_len"] == original["input_tokens"]
        assert request["output_len"] == original["output_tokens"]
    assert current["unique_selected_pool_records"] == len(source["records"])


def test_rate_and_system_pairing_has_one_exact_trace_and_no_scale_one(tmp_path, source):
    result = t.materialize("0.40", tmp_path / "point", source)
    assert len(result["cells"]) == 10
    assert {row["system"] for row in result["cells"]} == set(p.SYSTEMS)
    assert {row["slo_scale"] for row in result["cells"]} == set(p.SCALES)
    assert {row["trace_sha256"] for row in result["cells"]} == {result["trace"]["sha256"]}
    assert len({row["content_pairing_sha256"] for row in result["cells"]}) == 1
    assert all(row["reuse_main_cell_id"] is None for row in result["cells"])
    assert result["all_rows_execution_required"] is False
    before = Path(result["trace"]["path"]).read_bytes()
    with pytest.raises(FileExistsError):
        t.materialize("0.4", tmp_path / "point", source)
    assert Path(result["trace"]["path"]).read_bytes() == before


def test_loaded_source_mutation_and_materialized_trace_mutation_are_rejected(tmp_path, source):
    changed = copy.deepcopy(source)
    changed["records"][0]["output_tokens"] -= 1
    with pytest.raises(ValueError, match="content/order changed"):
        t.build_trace("0.4", changed)
    result = t.materialize("0.4", tmp_path / "point", source)
    path = Path(result["trace"]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="frozen source changed"):
        t.execution_row(result["trace"], "pdblend", 0.5)


def test_arrival_resource_limit_raises_instead_of_silently_truncating(monkeypatch):
    monkeypatch.setattr(t, "MAX_REQUESTS", 2)
    with pytest.raises(ValueError, match="no workload truncation"):
        t.window_arrivals("20.25")
    for rate in (0, -1, "Infinity", "NaN", True):
        with pytest.raises(ValueError):
            t.window_arrivals(rate)


def test_lower_rate_keeps_the_same_ordered_work_prefix(source):
    lower = t.build_trace("0.4", source)
    upper = t.build_trace("0.8", source)
    n = lower["n_requests"]
    for field in ("prompts", "source_shapes", "source_pool_indices"):
        assert lower[field] == upper[field][:n]
    assert [r["arrival_s"] / 2 for r in lower["requests"]] == [r["arrival_s"] for r in upper["requests"][:n]]
    assert all(r["output_len"] == q["output_len"] for r, q in zip(lower["requests"], upper["requests"]))


@pytest.mark.parametrize("parent_kind", ["pdb", "baseline"])
def test_real_adapted_cell_accepts_generated_trace_both_scales(tmp_path, source, parent_kind):
    import runtime_adapter as adapter
    parent = adapter.PDB_PARENT if parent_kind == "pdb" else adapter.BASELINE_PARENT
    host = tmp_path / "host"
    adapter.prepare_host(parent, host)
    trace_path = tmp_path / "trace.json"
    trace_path.write_bytes(t.encode(t.build_trace("20.25", source)))
    # Import the real prepared Cell and its dependencies in a fresh process.
    # Only its pure configure function runs: no Controller, GPU or HTTP starts.
    script = """
import json, sys
from pathlib import Path
from types import SimpleNamespace
host, trace_path = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(host/'src'), str(host), '/root/workspace/pdblend/.runtime-deps']
from ecopadg.serving.cell import configure_fixed_window
trace = json.loads(trace_path.read_text())
assert trace['rate_rps'] == trace['rate'] == 20.25
for scale in (0.5, 2.0):
    args = SimpleNamespace(split='development', dataset='sharegpt', seed=701,
        slo_scale=scale, slo_ttft_s=5*scale, slo_tpot_s=.15*scale)
    config = dict(measurement_window_protocol=trace['protocol_id'],
        evaluation_protocol='evaluation-v3', strategy='pdblend-joint', arrival_window_s=100)
    result = configure_fixed_window(args, config, trace)
    assert result['effective_slo_s'] == dict(ttft=5*scale, tpot=.15*scale)
    assert result['n_expected'] == len(trace['requests'])
print('both scales accepted by the actual imported Cell')
"""
    result = subprocess.run([sys.executable, "-B", "-c", script, str(host), str(trace_path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
