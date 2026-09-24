"""Deletion safety checks use isolated files; no tests touch the real repository."""
import importlib.util
import json
import os
from pathlib import Path
from zipfile import ZipFile

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "scripts/results/cleanup_workspace.py"
spec = importlib.util.spec_from_file_location("workspace_cleanup", SOURCE)
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


@pytest.fixture
def environment(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / ".git/objects/pack").mkdir(parents=True)
    state = {"head": "test", "refs": {"refs/heads/main": "test"}, "formal_packs": {}}
    monkeypatch.setattr(cleanup, "git_snapshot", lambda _: state)
    monkeypatch.setattr(cleanup, "connectivity", lambda *_: {"returncode": 0})
    monkeypatch.setattr(cleanup, "processes_and_locks", lambda _: {})
    monkeypatch.setattr(cleanup, "open_target_fds", lambda _: [])
    monkeypatch.setattr(cleanup.time, "time", lambda: 200000.0)
    return root, tmp_path / "receipt"


def candidate(root, name="tmp_pack_old"):
    p = root / ".git/objects/pack" / name
    p.write_bytes(b"temporary data")
    os.utime(p, (1, 1))
    return p


def test_allowlist_excludes_formal_objects_and_traversal():
    assert cleanup.kind_allowed(".git/objects/pack/tmp_pack_abcd", "git_temporary")
    assert cleanup.kind_allowed(".git/objects/ab/tmp_obj_abcd", "git_temporary")
    assert not cleanup.kind_allowed(".git/objects/pack/pack-abcd.pack", "git_temporary")
    assert not cleanup.kind_allowed(".git/objects/pack/tmp_obj_abcd", "git_temporary")
    assert not cleanup.kind_allowed("results/tmp_pack_abcd", "git_temporary")


def test_two_observations_required(environment):
    root, output = environment
    p = candidate(root)
    cleanup.prepare(root, output)
    with pytest.raises(RuntimeError, match="60 seconds"):
        cleanup.apply(root, output / "allowlist.json", now=200059)
    assert p.exists()


def test_exact_delete_keeps_new_temporary_and_formal_pack(environment):
    root, output = environment
    old = candidate(root)
    new = candidate(root, "tmp_pack_new")
    os.utime(new, (199999, 199999))
    formal = candidate(root, "pack-abc.pack")
    receipt = cleanup.prepare(root, output)
    assert [e["path"] for e in receipt["entries"]] == [str(old.relative_to(root))]
    result = cleanup.apply(root, output / "allowlist.json", now=200061)
    assert not old.exists() and new.exists() and formal.exists()
    assert result["deleted_files"] == 1 and result["formal_packs_unchanged"]
    assert not result["entries"][0]["fingerprint"]["full_checksum"]


def test_changed_file_is_skipped(environment):
    root, output = environment
    p = candidate(root)
    cleanup.prepare(root, output)
    p.write_bytes(b"changed")
    result = cleanup.apply(root, output / "allowlist.json", now=200061)
    assert p.exists() and result["deleted_files"] == 0
    assert result["entries"][0]["skip_reason"] == "metadata_changed"


def test_open_file_is_skipped(environment, monkeypatch):
    root, output = environment
    p = candidate(root)
    cleanup.prepare(root, output)
    monkeypatch.setattr(cleanup, "open_target_fds", lambda _: [{"path": str(p.relative_to(root))}])
    result = cleanup.apply(root, output / "allowlist.json", now=200061)
    assert p.exists() and result["entries"][0]["skip_reason"] == "open_fd"


def test_tampered_manifest_cannot_delete_formal_pack(environment):
    root, output = environment
    p = candidate(root)
    formal = candidate(root, "pack-abc.pack")
    cleanup.prepare(root, output)
    manifest = output / "allowlist.json"
    receipt = json.loads(manifest.read_text())
    receipt["entries"][0]["path"] = str(formal.relative_to(root))
    manifest.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="not in cleanup scope"):
        cleanup.apply(root, manifest, now=200061)
    assert p.exists() and formal.exists()


def test_symlink_cannot_escape_cleanup_root(environment):
    root, _ = environment
    external = root.parent / "external"
    external.write_bytes(b"keep")
    (root / "link").symlink_to(external)
    with pytest.raises(ValueError, match="symlink"):
        cleanup.regular_beneath(root, "link")
    with pytest.raises(ValueError, match="traversal"):
        cleanup.regular_beneath(root, "../external")


def test_archive_requires_all_sources_to_match(environment):
    root, _ = environment
    folder = root / "docs/pdblend-method"
    folder.mkdir(parents=True)
    source = folder / "source.txt"
    source.write_text("retain me")
    archive = folder / "pdblend-method-source.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.write(source, "pdblend-method/source.txt")
    assert len(cleanup.verify_archive(root, archive)) == 1
    source.write_text("different")
    with pytest.raises(RuntimeError, match="not duplicate"):
        cleanup.verify_archive(root, archive)
