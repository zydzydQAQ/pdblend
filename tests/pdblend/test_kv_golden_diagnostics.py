import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "diagnose_kv_golden", Path(__file__).resolve().parents[2] / "scripts/diagnose_kv_golden.py")
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def test_native_first_difference_and_prefill_first_token():
    row = dict(references=[dict(token_ids=[100, 101, 102]), dict(token_ids=[100, 101, 102])],
               prefill=dict(outputs=[dict(token_ids=[100])]), decode=dict(token_ids=[100, 201, 202]))
    result = diagnostics.inspect_native_row(row)
    assert result["token_comparison"]["first_difference_index"] == 1
    assert result["token_comparison"]["reference_token_id"] == 101
    assert result["token_comparison"]["candidate_token_id"] == 201
    assert result["prefill_first_vs_reference"]["exact_match"]
    assert result["within_reference_path"]["stable"]
    assert not result["formal_eligible"]


def test_reported_match_and_text_are_not_token_evidence():
    result = diagnostics.inspect_native_row(dict(
        reference_stable=True, tokens_match=True,
        decode=dict(text="same"), references=[dict(text="same"), dict(text="same")]))
    assert result["token_comparison"]["status"] == "missing_token_ids"
    assert result["within_reference_path"]["stable"] is None
    assert diagnostics.first_difference([10, 20], [10])["first_difference_index"] == 1


def test_historical_export_does_not_retokenize_text(tmp_path):
    artifact = tmp_path / "completion.json"
    artifact.write_text(json.dumps(dict(model_id="Qwen2.5-32B-Instruct", tp=2, kv=dict(rows=[
        dict(input_tokens=7168, repeat=0, mixed_text="y", pd_text="asignatured", text_match=False)]))))
    report, plan = diagnostics.audit([tmp_path])
    assert report["artifacts"][0]["rows"][0]["token_golden_status"] == "missing_token_ids"
    case = plan["cases"][0]
    from pdblend.bench.gates import random_prompt
    assert case["prompt"] == random_prompt(7168, 716800)
    assert case["same_prompt_path_repeats"] == 3
    assert case["historical_failures"][0]["tp"] == 2


def test_same_prompt_repeats_grouped_and_progress_not_double_counted(tmp_path):
    rows = [dict(length=2, prompt=[1, 2], repeat=i,
                 references=[dict(token_ids=[10, 11]), dict(token_ids=[10, 11])],
                 decode=dict(token_ids=[10, 11 if i == 0 else 12]),
                 prefill=dict(outputs=[dict(token_ids=[10])])) for i in range(3)]
    data = dict(model="m", tp=2, kv=rows)
    for name in ("completion.json", "progress.json"):
        (tmp_path / name).write_text(json.dumps(data))
    report, _ = diagnostics.audit([tmp_path])
    assert report["artifact_count"] == 1
    assert report["same_prompt_path_stability"][0]["decode"]["stable"] is False
    assert report["same_prompt_path_stability"][0]["prefill_first"]["stable"] is True


def test_prompt_checksum_mismatch_cannot_claim_same_path_repeats(tmp_path):
    row = dict(length=2, prompt=[1, 2], prompt_sha256="0" * 64,
               references=[dict(token_ids=[1]), dict(token_ids=[1])], decode=dict(token_ids=[1]))
    (tmp_path / "completion.json").write_text(json.dumps(dict(model="m", tp=1, kv=[row, row])))
    report, _ = diagnostics.audit([tmp_path])
    assert report["same_prompt_path_stability"] == []
    assert "prompt_identity_error" in report["artifacts"][0]["rows"][0]
