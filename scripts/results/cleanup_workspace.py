#!/usr/bin/env python3
"""Audited, two-phase removal of exact stale Git temporaries and generated docs.

This tool never prunes Git objects, rewrites history, or deletes an experiment.
Prepare writes a reviewable allowlist. Apply requires 60 seconds of stability and
revalidates every inode immediately before unlinking that one file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import time
from zipfile import ZipFile

SCHEMA = "workspace-generated-cleanup-v1"
SAMPLE_BYTES = 65536
MIN_AGE = 24 * 60 * 60
MIN_OBSERVATION = 60
DOC_ARCHIVES = (
    "docs/pdblend-method/pdblend-method-source.zip",
    "docs/pdblend-method-latex/pdblend-method-latex-source.zip",
)
DOC_GENERATED = (
    "docs/pdblend-method/preview",
    "docs/pdblend-method/__pycache__",
    "docs/pdblend-method-latex/preview",
    "docs/pdblend-method-latex/build",
    "docs/pdblend-method-latex/figures/build",
)


def dump(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def identity(path: Path) -> dict:
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"not a regular file: {path}")
    return {"device": st.st_dev, "inode": st.st_ino, "bytes": st.st_size,
            "blocks": st.st_blocks, "mode": st.st_mode, "links": st.st_nlink,
            "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns}


def regular_beneath(root: Path, relative: str) -> Path:
    candidate = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("absolute/traversal allowlist path")
    for parent in (candidate, *candidate.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"symlink in cleanup path: {parent}")
    candidate.relative_to(root)
    identity(candidate)
    return candidate


def kind_allowed(relative: str, kind: str) -> bool:
    p = Path(relative)
    if kind == "git_temporary":
        return ((p.parent == Path(".git/objects/pack") and p.name.startswith("tmp_pack_"))
                or (len(p.parts) == 4 and p.parts[:2] == (".git", "objects")
                    and len(p.parts[2]) == 2 and all(c in "0123456789abcdef" for c in p.parts[2])
                    and p.name.startswith("tmp_obj_")))
    if kind == "duplicate_archive":
        return relative in DOC_ARCHIVES
    return kind == "regenerable_document" and any(p.is_relative_to(d) for d in DOC_GENERATED)


def fingerprint(path: Path, sampled: bool) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        if sampled:
            size = path.stat().st_size
            h.update(str(size).encode() + b"\0")
            h.update(stream.read(SAMPLE_BYTES))
            stream.seek(max(0, size - SAMPLE_BYTES))
            h.update(stream.read(SAMPLE_BYTES))
        else:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
    return {"algorithm": "sha256", "scope": "size+first64KiB+last64KiB" if sampled else "full-file",
            "value": h.hexdigest(), "full_checksum": not sampled}


def processes_and_locks(root: Path) -> dict:
    processes = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            comm = (proc / "comm").read_text().strip()
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        if (comm == "git" or comm.startswith("git-") or comm in {"pack-objects", "index-pack"}
                or argv and Path(os.fsdecode(argv[0])).name in {"git", "git-pack-objects", "git-index-pack"}):
            processes.append({"pid": int(proc.name), "comm": comm})
    locks = sorted(str(p.relative_to(root)) for p in (root / ".git").rglob("*.lock"))
    for extra in ("gc.pid", "maintenance.pid"):
        if (root / ".git" / extra).exists():
            locks.append(".git/" + extra)
    result = {"git_processes": processes, "git_locks": locks}
    if processes or locks:
        raise RuntimeError("Git is busy: " + json.dumps(result))
    return result


def open_target_fds(entries: list[dict]) -> list[dict]:
    targets = {(e["identity"]["device"], e["identity"]["inode"]): e["path"] for e in entries}
    hits = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            fds = list((proc / "fd").iterdir())
        except (FileNotFoundError, ProcessLookupError):
            continue
        for fd in fds:
            try:
                st = fd.stat()
            except (FileNotFoundError, ProcessLookupError):
                continue
            target = targets.get((st.st_dev, st.st_ino))
            if target:
                hits.append({"pid": int(proc.name), "fd": fd.name, "path": target})
    return hits


def git_snapshot(root: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "--no-optional-locks", *args], cwd=root, text=True).strip()
    formal = {}
    for path in sorted((root / ".git/objects/pack").glob("pack-*")):
        info = identity(path)
        if path.suffix == ".pack":
            with path.open("rb") as stream:
                stream.seek(-20, 2)
                info["git_sha1_trailer"] = stream.read(20).hex()
            info["identity_scope"] = "metadata+git-pack-trailer; not a full-file digest"
        else:
            info["fingerprint"] = fingerprint(path, False)
        formal[str(path.relative_to(root))] = info
    refs = {line.split(" ", 1)[1]: line.split(" ", 1)[0] for line in git("show-ref").splitlines()}
    return {"head": git("rev-parse", "HEAD"), "refs": refs, "formal_packs": formal}


def connectivity(root: Path, output: Path) -> dict:
    with output.open("w") as stream:
        run = subprocess.run(["git", "--no-optional-locks", "fsck", "--connectivity-only", "--no-dangling"],
                             cwd=root, stdout=stream, stderr=subprocess.STDOUT)
    result = {"returncode": run.returncode, "output": str(output), "fingerprint": fingerprint(output, False)}
    if run.returncode:
        raise RuntimeError(f"Git connectivity failed; see {output}")
    return result


def verify_archive(root: Path, path: Path) -> list[dict]:
    members = []
    with ZipFile(path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            name = Path(item.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("archive path traversal")
            target = root / "docs" / name
            if not target.exists():
                target = path.parent / name
            target = regular_beneath(root, str(target.relative_to(root)))
            archived = hashlib.sha256(archive.read(item)).hexdigest()
            if fingerprint(target, False)["value"] != archived:
                raise RuntimeError(f"archive is not duplicate: {item.filename}")
            members.append({"archive_member": item.filename, "retained": str(target.relative_to(root)),
                            "sha256": archived})
    return members


def prepare(root: Path, output: Path, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    output.mkdir(parents=True, exist_ok=False)
    guard = processes_and_locks(root)
    before = git_snapshot(root)
    checked = connectivity(root, output / "git-connectivity-before.txt")
    paths = [(p, "git_temporary") for p in (root / ".git/objects").rglob("tmp_*")
             if kind_allowed(str(p.relative_to(root)), "git_temporary") and not p.is_symlink()
             and p.is_file() and now - p.stat().st_mtime >= MIN_AGE]
    paths += [(root / p, "duplicate_archive") for p in DOC_ARCHIVES if (root / p).is_file()]
    paths += [(p, "regenerable_document") for d in DOC_GENERATED for p in (root / d).rglob("*")
              if p.is_file()]
    entries = []
    for path, kind in sorted(paths):
        rel = str(path.relative_to(root))
        regular_beneath(root, rel)
        row = {"path": rel, "kind": kind, "identity": identity(path),
               "fingerprint": fingerprint(path, kind == "git_temporary"),
               "reference_check": "only abandoned Git temporary names" if kind == "git_temporary"
               else "generated output; editable source and final PDF retained"}
        if kind == "duplicate_archive":
            row["retained_members"] = verify_archive(root, path)
        entries.append(row)
    opened = open_target_fds(entries)
    if opened:
        raise RuntimeError("candidate has open FD: " + json.dumps(opened))
    result = {"schema": SCHEMA, "root": str(root), "prepared_at": time.time(), "observed_at": now,
              "min_observation_seconds": MIN_OBSERVATION, "minimum_git_age_seconds": MIN_AGE,
              "git_before": before, "connectivity_before": checked, "guard_before": guard,
              "open_fds_before": opened, "entries": entries, "status": "prepared",
              "candidate_bytes": sum(e["identity"]["bytes"] for e in entries)}
    dump(output / "allowlist.json", result)
    return result


def apply(root: Path, manifest: Path, now: float | None = None) -> dict:
    result = json.loads(manifest.read_text())
    now = time.time() if now is None else now
    if result["schema"] != SCHEMA or result["root"] != str(root) or result["status"] != "prepared":
        raise ValueError("invalid, foreign or already applied manifest")
    if now - result["prepared_at"] < MIN_OBSERVATION:
        raise RuntimeError("two observations must be at least 60 seconds apart")
    result["guard_apply"] = processes_and_locks(root)
    result["second_observed_at"] = now
    before_apply = git_snapshot(root)
    for key, value in result["git_before"]["formal_packs"].items():
        if before_apply["formal_packs"].get(key) != value:
            raise RuntimeError("formal pack changed since first observation: " + key)
    entries = result["entries"]
    for entry in entries:
        if not kind_allowed(entry["path"], entry["kind"]):
            raise ValueError("not in cleanup scope: " + entry["path"])
        path = regular_beneath(root, entry["path"])
        if identity(path) != entry["identity"]:
            entry["skip_reason"] = "metadata_changed"
            continue
        if entry["kind"] == "git_temporary" and now - path.stat().st_mtime < MIN_AGE:
            entry["skip_reason"] = "not_old_enough"
            continue
        if fingerprint(path, entry["kind"] == "git_temporary") != entry["fingerprint"]:
            entry["skip_reason"] = "fingerprint_changed"
            continue
        if entry["kind"] == "duplicate_archive" and verify_archive(root, path) != entry["retained_members"]:
            entry["skip_reason"] = "retained_source_changed"
    open_fds = open_target_fds(entries)
    result["open_fds_apply"] = open_fds
    for entry in entries:
        if any(fd["path"] == entry["path"] for fd in open_fds):
            entry["skip_reason"] = "open_fd"
    result["status"] = "applying"
    dump(manifest, result)  # durable exact tombstones before the first unlink
    for entry in entries:
        if entry.get("skip_reason"):
            continue
        processes_and_locks(root)
        path = regular_beneath(root, entry["path"])
        if identity(path) != entry["identity"] or open_target_fds([entry]):
            entry["skip_reason"] = "changed_or_open_immediately_before_delete"
        else:
            path.unlink()  # never recursive; parent directories and all other files remain
            entry["deleted_at"] = time.time()
        dump(manifest, result)
    result["git_after"] = git_snapshot(root)
    result["connectivity_after"] = connectivity(root, manifest.parent / "git-connectivity-after.txt")
    result["formal_packs_unchanged"] = all(result["git_after"]["formal_packs"].get(k) == v
        for k, v in result["git_before"]["formal_packs"].items())
    result["head_unchanged"] = result["git_after"]["head"] == result["git_before"]["head"]
    before_refs, after_refs = result["git_before"]["refs"], result["git_after"]["refs"]
    result["ref_changes"] = {k: {"before": before_refs.get(k), "after": after_refs.get(k)}
                             for k in sorted(before_refs.keys() | after_refs.keys())
                             if before_refs.get(k) != after_refs.get(k)}
    result["deleted_files"] = sum("deleted_at" in e for e in entries)
    result["deleted_bytes"] = sum(e["identity"]["bytes"] for e in entries if "deleted_at" in e)
    result["deleted_allocated_bytes"] = sum(e["identity"]["blocks"] * 512 for e in entries if "deleted_at" in e)
    result["status"] = "complete"
    dump(manifest, result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "apply"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    result = prepare(root, args.output) if args.action == "prepare" else apply(root, args.output / "allowlist.json")
    print(json.dumps({k: result[k] for k in ("status", "candidate_bytes", "deleted_files", "deleted_bytes",
                                          "deleted_allocated_bytes") if k in result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
