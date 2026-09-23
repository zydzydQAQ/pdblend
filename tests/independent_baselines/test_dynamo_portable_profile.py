import copy
import hashlib
import json
from pathlib import Path

import pytest

from pdblend_baselines.dynamollm.prepare_v1 import merge_own_profiles
from pdblend_baselines.dynamollm.portable_profile import PortableProfileError, relocate_profile, validate_profile
from pdblend_baselines.dynamollm.profiles import PaperProfiles


FREQS = (900, 1200, 1500, 1800, 2100, 2520)
IDENTITY = {
    "system": "dynamollm", "model_id": "Qwen2.5-7B-Instruct", "tp": 1, "pp": 1,
    "engine_revision": "vllm-0.10.1.1", "image_digest": "sha256:image",
    "source_sha256": "a" * 64,
    "model_identity": {"model": "Qwen2.5-7B-Instruct", "tokenizer_sha256": "b" * 64},
    "gpu_uuids": {"0": "GPU-test0"},
}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    from pdblend_baselines.dynamollm.profile_v1 import fit_cell, key_for
    root = tmp_path / "attempt"
    prof = root / "dynamollm-profile"
    cells, raw = prof / "cells", prof / "raw"
    cells.mkdir(parents=True)
    raw.mkdir()
    capability = prof / "capabilities" / "cap.json"
    capability.parent.mkdir()
    capability.write_text(json.dumps(dict(
        model_id=IDENTITY["model_id"], model_hash="1"*64, tokenizer_hash="2"*64,
        verification_receipt_sha256="3"*64, engine_revision=IDENTITY["engine_revision"],
        source_revision=IDENTITY["source_sha256"], image_digest=IDENTITY["image_digest"],
        tp=1, pp=1, gpu_uuids=["GPU-test0"], native_evidence_complete=True,
        state=dict(native_evidence_complete=True, healthy=True, generation=2,
                   ranks=[dict(rank=0, generation=2, healthy=True, native_evidence_complete=True)]))))
    points = []
    for freq in FREQS:
        point = dict(frequency_mhz=freq, input_tokens=512, output_tokens=64, batch=1)
        key = key_for(IDENTITY, point)
        artifacts, windows = {}, []
        for repeat in range(4):
            start = 100 + repeat * 10
            rid = f"request-{freq}-{repeat}"
            samples = [dict(role="prefill" if step == 0 else "decode", batch=1,
                            max_input_tokens=512, max_context_tokens=512+step,
                            rank=0, tp=1, pp=1, system="dynamollm", measurement_scope="runner",
                            gpu_elapsed_ms=20., request_ids=[rid], at_s=start+.1+step*.02)
                       for step in range(9)]
            window = dict(point=point, repeat=repeat, settle_s=2, started_s=start, finished_s=start+5,
                          requests=[dict(request_id=rid, submitted_s=start, first_token_s=start+.12,
                                         finished_s=start+1.38, tokens=64, ok=True)],
                          native=dict(ranks=[dict(rank=0, samples=samples)]), gpu_ids=[0],
                          power=[dict(gpu=0, gpu_uuid="GPU-test0", timestamp=start+delta, frequency_mhz=freq, power_w=100.,
                                      source="nvml:field:186:scope:0:mW") for delta in (.1, 4.)])
            path = raw / f"{key}-{'repeat'+str(repeat) if repeat < 3 else 'holdout'}.json"
            path.write_text(json.dumps(window))
            artifacts[f"raw/{path.name}"] = _sha(path)
            windows.append(window)
            if repeat == 2:
                frozen = raw / f"{key}-fit-inputs.json"
                frozen.write_text(json.dumps(dict(point=point, frozen_at_s=start+5.1,
                                                 artifacts=copy.deepcopy(artifacts))))
                artifacts[f"raw/{frozen.name}"] = _sha(frozen)
        fitted = fit_cell(windows[:3], windows[3], tp=1, point=point)
        cell = dict(identity=copy.deepcopy(IDENTITY), point=point, artifacts=artifacts,
                    capability_path="capabilities/cap.json", capability_sha256=_sha(capability), fit=fitted)
        cell_file = cells / f"{key}.json"
        cell_file.write_text(json.dumps(cell, sort_keys=True))
        points.append(dict(role="mixed", tp=1, pp=1, frequency_mhz=freq, input_tokens=512,
                           context_tokens=576, batch=1, samples=3,
                           **{field:fitted[field] for field in ('prefill_s','iteration_s','power_w')},
                           source_sha256=_sha(cell_file),
                           source_profile_path=f"/output/dynamollm-profile/cells/{cell_file.name}"))
    original = dict(schema=2, measurement="hardware", coordinate_system="input_output_batch",
                    independent_profile=True, **IDENTITY, model=IDENTITY['model_id'], points=points,
                    formal_eligible=False, hardware_qualified=False)
    profile = prof / "profile.json"
    profile.write_text(json.dumps(original, sort_keys=True))
    return root, profile


def _first_cell(root, profile):
    point = json.loads(profile.read_text())["points"][0]
    return root / Path(point["source_profile_path"]).relative_to("/output")


def _save_cell(profile, cell_path, cell):
    cell_path.write_text(json.dumps(cell))
    value = json.loads(profile.read_text())
    value["points"][0]["source_sha256"] = _sha(cell_path)
    profile.write_text(json.dumps(value))


def test_relocation_is_mergeable_and_queries_all_six_frequencies(tmp_path):
    root, profile = _fixture(tmp_path)
    out = tmp_path / "derived" / "profile.json"
    value = relocate_profile(artifact_root=root, recorded_root="/output", profile=profile, out=out)
    assert value["portable_profile_receipt"]["cell_count"] == 6
    assert value["formal_eligible"] is False
    merged = merge_own_profiles([out], model_id="Qwen2.5-7B-Instruct")
    loaded = PaperProfiles.load(out)
    assert loaded.frequencies(1) == FREQS
    for frequency in FREQS:
        assert loaded.query(1, frequency, 512, 576, 1).decode_s == pytest.approx(.02)
    assert len(merged["points"]) == 6


@pytest.mark.parametrize("change", ["checksum", "traversal", "system", "holdout"])
def test_relocation_rejects_changed_or_untrusted_cell(tmp_path, change):
    root, profile = _fixture(tmp_path)
    data = json.loads(profile.read_text())
    cell_path = _first_cell(root, profile)
    cell = json.loads(cell_path.read_text())
    if change == "checksum":
        cell["fit"]["power_w"] = 999
        cell_path.write_text(json.dumps(cell))
    elif change == "system":
        cell["identity"]["system"] = "other"
        cell_path.write_text(json.dumps(cell))
    elif change == "holdout":
        cell["fit"]["holdout_passed"] = False
        cell_path.write_text(json.dumps(cell))
    else:
        data["points"][0]["source_profile_path"] = "/output/../outside.json"
        profile.write_text(json.dumps(data))
    with pytest.raises(PortableProfileError):
        relocate_profile(artifact_root=root, recorded_root="/output", profile=profile,
                         out=tmp_path / "derived.json")


def test_relocation_rejects_raw_checksum_change(tmp_path):
    root, profile = _fixture(tmp_path)
    raw = root / "dynamollm-profile" / "raw" / (_first_cell(root, profile).stem + "-repeat0.json")
    raw.write_text("changed")
    with pytest.raises(PortableProfileError, match="raw artifact checksum"):
        relocate_profile(artifact_root=root, recorded_root="/output", profile=profile,
                         out=tmp_path / "derived.json")


@pytest.mark.parametrize("strict", [True, False, None])
def test_no_strict_flag_can_bypass_missing_evidence(tmp_path, strict):
    root, profile = _fixture(tmp_path)
    value = json.loads(profile.read_text())
    if strict is not None:
        value["strict_provenance"] = strict
    profile.write_text(json.dumps(value))
    cell_path = _first_cell(root, profile)
    cell = json.loads(cell_path.read_text())
    cell["artifacts"] = {}
    _save_cell(profile, cell_path, cell)
    with pytest.raises(PortableProfileError, match="three repeats"):
        validate_profile(artifact_root=root, recorded_root="/output", profile=profile)


@pytest.mark.parametrize("kind", ["point", "capability", "double_source"])
def test_relocation_rejects_provenance_or_point_mutation(tmp_path, kind):
    root, profile = _fixture(tmp_path)
    data = json.loads(profile.read_text())
    if kind == "point":
        data["points"][0]["prefill_s"] = .123
    elif kind == "double_source":
        data["points"][0]["source_profile_sha256"] = "f" * 64
    else:
        cap = root / "dynamollm-profile" / "capabilities" / "cap.json"
        cap.write_text("changed")
    profile.write_text(json.dumps(data))
    with pytest.raises(PortableProfileError):
        relocate_profile(artifact_root=root, recorded_root="/output", profile=profile,
                         out=tmp_path / "derived.json")


@pytest.mark.parametrize("model,tp", [("7b",1), ("14b",1), ("32b",2)])
@pytest.mark.historical
def test_completed_artifact_roundtrip_when_present(tmp_path, model, tp):
    matches = list(Path("results/2026-09-22/three-model/queue-attempts").glob(
        f"native-probe-{model}-*/attempt-*/dynamollm-profile/profile.json"))
    if not matches:
        pytest.skip("completed 7B Dynamo artifact is not in this checkout")
    profile = matches[0]
    root = profile.parents[1]
    out = tmp_path / "derived-7b.json"
    relocate_profile(artifact_root=root, recorded_root="/output", profile=profile, out=out)
    model_id = f"Qwen2.5-{model.upper()}-Instruct"
    merged = merge_own_profiles([out], model_id=model_id)
    assert len(merged["points"]) == 6
    assert PaperProfiles.load(out).query(tp, 900, 512, 576, 1).prefill_s > 0


@pytest.mark.parametrize("kind", ["empty_capability", "wrong_uuid", "rank_missing", "fit",
                                 "wrong_output", "wrong_input", "rank_shape", "frozen_holdout",
                                 "overlap", "duplicate_request", "frozen_hash", "power_uuid"])
def test_rehashed_incomplete_or_misbound_raw_is_rejected(tmp_path, kind):
    root, profile = _fixture(tmp_path)
    cell_path = _first_cell(root, profile)
    cell = json.loads(cell_path.read_text())
    prof = cell_path.parent.parent
    if kind in ("empty_capability", "wrong_uuid", "rank_missing"):
        path = prof / cell["capability_path"]
        cap = json.loads(path.read_text())
        if kind == "empty_capability": cap = {}
        if kind == "wrong_uuid": cap["gpu_uuids"] = ["GPU-other"]
        if kind == "rank_missing": cap["state"]["ranks"] = []
        path.write_text(json.dumps(cap))
        cell["capability_sha256"] = _sha(path)
    elif kind == "fit":
        cell["fit"]["power_w"] = 110.
    else:
        suffix = "fit-inputs" if kind in ("frozen_holdout", "frozen_hash") else "holdout"
        name = f"raw/{cell_path.stem}-{suffix}.json"
        path = prof / name
        raw = json.loads(path.read_text())
        if kind == "wrong_output": raw["requests"][0]["tokens"] = 16
        if kind == "wrong_input": raw["point"]["input_tokens"] = 128
        if kind == "rank_shape": raw["native"]["ranks"][0]["samples"][0]["max_input_tokens"] = 128
        if kind == "power_uuid": raw["power"][0]["gpu_uuid"] = "GPU-another-physical-card"
        if kind == "frozen_holdout": raw["frozen_at_s"] = 1000
        if kind == "overlap": raw["started_s"] = 119
        if kind == "duplicate_request": raw["requests"][0]["request_id"] = "request-900-0"
        if kind == "frozen_hash": raw["artifacts"][next(iter(raw["artifacts"]))] = "f" * 64
        path.write_text(json.dumps(raw))
        cell["artifacts"][name] = _sha(path)
    _save_cell(profile, cell_path, cell)
    with pytest.raises(PortableProfileError):
        validate_profile(artifact_root=root, recorded_root="/output", profile=profile)


def test_validate_only_does_not_write_in_artifact_tree(tmp_path):
    root, profile = _fixture(tmp_path)
    before = {str(path.relative_to(root)): _sha(path) for path in root.rglob("*") if path.is_file()}
    assert len(validate_profile(artifact_root=root, recorded_root="/output", profile=profile)["points"]) == 6
    after = {str(path.relative_to(root)): _sha(path) for path in root.rglob("*") if path.is_file()}
    assert before == after
