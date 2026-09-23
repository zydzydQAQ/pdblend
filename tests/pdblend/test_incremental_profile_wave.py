import importlib.util
import json
from pathlib import Path

import pytest


def _load_builder():
    path = Path(__file__).parents[2] / "scripts/2026-09-23_prepare_incremental_profile_wave.py"
    spec = importlib.util.spec_from_file_location("incremental_wave_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _fixture(tmp_path):
    b = _load_builder()
    root = tmp_path / "root"; (root / "src/pdblend/profile").mkdir(parents=True)
    impl = root / "src/pdblend/profile/local_power_job.py"; impl.write_text("stable\n")
    source = tmp_path / "frozen-source"; (source / "pdblend/profile").mkdir(parents=True)
    (source / "pdblend/profile/local_power_job.py").write_text(impl.read_text())
    verify = root / "verification.json"; verify.write_text("{}\n")
    local_package = root / "local-package"; local_package.mkdir()
    local_manifest = {"model_id": "Qwen2.5-14B-Instruct", "tp": 4, "pp": 1,
                      "inputs": {}, "implementation_sha256": {
                          "profile/local_power_job.py": b.digest(source / "pdblend/profile/local_power_job.py")}}
    (local_package / "manifest.json").write_text(json.dumps(local_manifest))
    local_review = root / "local-review.json"
    local_review.write_text(json.dumps({"package": str(local_package),
        "source_snapshot": str(source), "exact_path_readonly_roots": [str(local_package)]}))
    def package_for(model_id, tp):
        package = root / (model_id.split('-')[-1].lower() + f"-tp{tp}-package"); package.mkdir()
        (package / "manifest.json").write_text(json.dumps({"model_id": model_id, "tp": tp, "pp": 1,
            "inputs": {}, "implementation_sha256": {"profile/local_power_job.py": b.digest(source / "pdblend/profile/local_power_job.py")}}))
        return package
    members = {
        "32b-tp2-longctx": {"model_id": "Qwen2.5-32B-Instruct", "tp": 2, "kind": "followup",
                            "package": str(package_for("Qwen2.5-32B-Instruct", 2)), "readonly_roots": [str(local_package)]},
        "7b-tp1-longctx": {"model_id": "Qwen2.5-7B-Instruct", "tp": 1, "kind": "followup",
                           "package": str(package_for("Qwen2.5-7B-Instruct", 1)), "readonly_roots": [str(local_package)]},
        "14b-tp1-longctx": {"model_id": "Qwen2.5-14B-Instruct", "tp": 1, "kind": "training",
                             "plan": str(root / "training.json"), "readonly_roots": [str(local_package)]},
    }
    for member in members.values():
        member['readonly_roots'] = [str(root)]
        if member["kind"] == "training":
            training = root / "training.json"; training.write_text(json.dumps({
                "model_id": member["model_id"], "tp": 1, "pp": 1,
                "fit_existing_holdout": False, "training_source": str(verify),
                "training_source_sha256": b.digest(verify)}))
    review = root / "long-review.json"; review.write_text(json.dumps({"members": members}))
    b.ROOT = root; b.LOCAL_REVIEW = local_review; b.VERIFY = verify
    local_data = json.loads(local_review.read_text())
    local_data['exact_path_readonly_roots'] = [str(root)]
    local_review.write_text(json.dumps(local_data))
    b.freeze_source = lambda base, files: (source, "source-sha")
    # Every package has the same immutable input binding for this contract test.
    return b, review, root


def test_builds_four_jobs_with_common_dependencies_and_receipts(tmp_path):
    b, review, root = _fixture(tmp_path)
    # The local package manifest is rewritten per member above; use the actual
    # source file as each package input so check_member validates its SHA.
    package = root / "local-package"
    manifest = json.loads((package / "manifest.json").read_text())
    manifest["inputs"] = {"source": {"path": str(root / "src/pdblend/profile/local_power_job.py"),
                                      "sha256": b.digest(root / "src/pdblend/profile/local_power_job.py")}}
    (package / "manifest.json").write_text(json.dumps(manifest))
    out = tmp_path / "prepared"
    path = b.build(review=review, out=out, dependencies=["dep-a", "dep-b", "dep-c"], overlay_files=[])
    jobs = json.loads(path.read_text())
    assert len(jobs) == 4
    assert [job["payload"]["gpu_count"] for job in jobs] == [4, 2, 1, 1]
    for job in jobs:
        payload = job["payload"]
        assert payload["depends_on"] == ["dep-a", "dep-b", "dep-c"]
        assert payload["required_receipts"]
        assert payload["formal_eligible"] is False
        gpu_flag = max(i for i, value in enumerate(payload["argv"]) if value == "--gpus")
        assert payload["argv"][gpu_flag + 1:gpu_flag + 1 + payload["tp"]] == [str(i) for i in range(payload["tp"])]
        assert "{lease_port}" in payload["argv"]
        assert any("PDBLEND_GPU_UUIDS={lease_gpu_uuids}" == value for value in payload["argv"])


def test_changed_immutable_input_is_rejected(tmp_path):
    b, review, root = _fixture(tmp_path)
    package = root / "local-package"; manifest = json.loads((package / "manifest.json").read_text())
    source_file = root / "src/pdblend/profile/local_power_job.py"
    manifest["inputs"] = {"source": {"path": str(source_file), "sha256": b.digest(source_file)}}
    (package / "manifest.json").write_text(json.dumps(manifest)); source_file.write_text("tampered\n")
    with pytest.raises(ValueError, match="input changed"):
        b.build(review=review, out=tmp_path / "rejected", dependencies=["dep"], overlay_files=[])


def test_missing_package_mount_rejected_before_job_creation(tmp_path):
    b, review, root = _fixture(tmp_path)
    value = json.loads(review.read_text())
    value['members']['32b-tp2-longctx']['readonly_roots'] = [str(root/'verification.json')]
    review.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='missing read-only mount'):
        b.build(review=review, out=tmp_path/'bad', dependencies=['dep'], overlay_files=[])
    assert not (tmp_path/'bad').exists()


def test_overlay_cannot_change_package_bound_implementation(tmp_path):
    b, review, root = _fixture(tmp_path)
    source = Path(json.loads(b.LOCAL_REVIEW.read_text())['source_snapshot'])
    (source/'pdblend/profile/local_power_job.py').write_text('different implementation')
    with pytest.raises(ValueError, match='changes local power package implementation'):
        b.build(review=review, out=tmp_path/'bad', dependencies=['dep'], overlay_files=[])
    assert not (tmp_path/'bad').exists()
