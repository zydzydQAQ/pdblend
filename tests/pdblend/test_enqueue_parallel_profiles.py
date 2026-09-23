from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from pdblend.experimentation.lease import GPULeaseQueue


def load_script():
    path = Path(__file__).parents[2] / "scripts/2026-09-22_enqueue_parallel_profiles.py"
    spec = importlib.util.spec_from_file_location("enqueue_parallel_profiles", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli():
    return load_script()


def specs(cli, root, **kwargs):
    return cli.build_specs(cli.TOPOLOGIES, snapshot=root / "snapshot", source_sha256="source",
                           image_digest="sha256:" + "a" * 64, receipt_path=root / "receipt.json",
                           receipt_sha256="receipt", out=root, **kwargs)


def test_import_does_not_inspect_docker(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("import must not invoke docker")
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    load_script()


def test_source_is_content_addressed_and_existing_bytes_verified(cli, tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "model.py").write_text("version = 1\n")
    snapshot, digest = cli.freeze_source(source, tmp_path / "snapshots")
    assert snapshot.name == digest
    assert cli.freeze_source(source, tmp_path / "snapshots") == (snapshot, digest)
    (snapshot / "model.py").write_text("tampered\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        cli.freeze_source(source, tmp_path / "snapshots")
    (source / "model.py").write_text("version = 2\n")
    newer, new_digest = cli.freeze_source(source, tmp_path / "snapshots")
    assert newer != snapshot and new_digest != digest


def test_snapshot_rejects_tampered_manifest_and_extra_file(cli, tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "model.py").write_text("pass\n")
    snapshot, _ = cli.freeze_source(source, tmp_path / "snapshots")
    manifest = snapshot / "manifest.json"
    data = json.loads(manifest.read_text())
    original = dict(data)
    data["source_sha256"] = "wrong"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum mismatch"):
        cli.freeze_source(source, tmp_path / "snapshots", dry_run=True)
    manifest.write_text(json.dumps(original))
    (snapshot / "__pycache__").mkdir()
    (snapshot / "__pycache__/model.pyc").write_bytes(b"unexpected bytecode")
    with pytest.raises(ValueError, match="checksum mismatch"):
        cli.freeze_source(source, tmp_path / "snapshots", dry_run=True)


def test_specs_only_legal_pp1_and_resumable_group_leases(cli, tmp_path):
    rows = specs(cli, tmp_path)
    assert len(rows) == 8
    assert {cli.job_topology(row) for row in rows} == set(cli.TOPOLOGIES)
    for row in rows:
        payload = row["payload"]
        assert payload["gpu_count"] == 2 * payload["tp"]
        assert payload["pp"] == 1 and row["max_attempts"] == 2
        assert not payload["exclusive"] and not payload["global_lock"]
        assert payload["depends_on"] == []
        assert payload["resume_profile"] and "--resume" in payload["argv"]
        assert payload["formal_eligible"] is False and payload["completion_is_formal_qualification"] is False
        assert "PDBLEND_CONCURRENCY_ENVIRONMENT" in payload["argv"]
    whole = specs(cli, tmp_path, full_host=True)
    assert all(row["payload"]["gpu_count"] == 8 and not row["payload"]["exclusive"] for row in whole)


def test_only_rejects_impossible_or_duplicate_topologies(cli):
    assert cli.parse_only("7b:4,14b:4,32b:2,32b:4") == (("7b", 4), ("14b", 4), ("32b", 2), ("32b", 4))
    for value in ("7b:8", "32b:1", "14b:3", "7b:1,7b:1", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parse_only(value)


def test_active_or_succeeded_legacy_job_prevents_duplicate(cli, tmp_path):
    row = specs(cli, tmp_path)[0]
    legacy = dict(job_id="profile-parallel-v2-7b-tp1-mixed", payload={"argv": [
        "profile", "--model", "/models/Qwen2.5-7B-Instruct", "--tp", "1", "--system", "pdblend"]})
    for status in ("running", "succeeded"):
        legacy["status"] = status
        plan = cli.plan_jobs({legacy["job_id"]: legacy}, [row], replace_queued=True)
        assert plan[0]["action"] == "skip"


def test_enqueue_is_idempotent_and_queued_replacement_is_explicit(cli, tmp_path):
    queue = GPULeaseQueue(tmp_path / "queue.json", gpu_probe=lambda: [])
    row = specs(cli, tmp_path)[0]
    queue.enqueue("profile-old-7b", dict(row["payload"], source_sha256="old"))
    assert cli.enqueue_specs(queue, [row])[0]["reason"] == "preserved_queued_topology"
    assert len(queue.list_jobs()) == 1
    assert cli.enqueue_specs(queue, [row], replace_queued=True)[0]["action"] == "enqueue"
    assert {job.job_id: job.status for job in queue.list_jobs()} == {
        "profile-old-7b": "blocked", row["job_id"]: "queued"}
    assert cli.enqueue_specs(queue, [row], replace_queued=True)[0]["action"] == "keep"
    assert queue.snapshot()["jobs"][row["job_id"]]["status"] == "queued"


def test_canonical_identity_conflict_fails_without_queue_mutation(cli, tmp_path):
    queue = GPULeaseQueue(tmp_path / "queue.json", gpu_probe=lambda: [])
    row = specs(cli, tmp_path)[0]
    queue.enqueue(row["job_id"], dict(row["payload"], timeout_s=1), max_attempts=2)
    before = queue.path.read_bytes()
    with pytest.raises(ValueError, match="immutable spec"):
        cli.enqueue_specs(queue, [row])
    assert queue.path.read_bytes() == before


def test_dry_run_has_no_filesystem_writes_and_prepare_exports_fixed_specs(cli, tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "model.py").write_text("pass\n")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"models": {"m": {"model_path": "/home/models/m", "files": []}}}))
    out = tmp_path / "output"
    args = ["--out", str(out), "--source-dir", str(source), "--model-receipt", str(receipt),
            "--image-digest", "sha256:" + "a" * 64, "--only", "32b:2"]
    result = cli.main(args + ["--dry-run"])
    assert result["dry_run"] and not out.exists()
    export = tmp_path / "jobs.json"
    result = cli.main(args + ["--prepare-only", "--spec-out", str(export)])
    assert Path(result["source_snapshot"]).is_dir()
    assert not (out / "queue.json").exists()
    exported = json.loads(export.read_text())
    assert len(exported) == 1 and exported[0]["payload"]["tp"] == 2
    assert exported[0]["payload"]["model_id"] == "Qwen2.5-32B-Instruct"
    original = export.read_bytes()
    cli.main(args + ["--prepare-only", "--spec-out", str(export)])
    assert export.read_bytes() == original
