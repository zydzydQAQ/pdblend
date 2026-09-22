#!/usr/bin/env python3
"""Audit the PDblend workspace after each migration/cleanup stage.

The audit is deliberately local and deterministic: it checks package naming,
result completeness, duplicate metric aliases, transient files, and reports a
size breakdown.  It does not inspect or modify the independent ``/home/pdblend``
checkout.
"""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "v2" / "eval-7b-v2"
SOURCE_ROOTS = (ROOT / "src", ROOT / "tests", ROOT / "scripts")
TEXT_SUFFIXES = {".py", ".sh", ".toml", ".yaml", ".yml", ".md"}
TEMP_NAMES = {".pytest_cache", "__pycache__"}
TEMP_SUFFIXES = (".tmp", ".bak", ".orig", "~")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_workspace_files():
    for base in SOURCE_ROOTS:
        if base.exists():
            yield from (p for p in base.rglob("*") if p.is_file())
    for path in (ROOT / "pyproject.toml", ROOT / "MIGRATION.md", ROOT / "RESTART.md"):
        if path.exists():
            yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="return non-zero on any finding")
    args = parser.parse_args()
    findings: list[str] = []

    # The compatibility namespace is intentionally allowed for the transition.
    stale_names: list[str] = []
    for path in iter_workspace_files():
        if path.suffix not in TEXT_SUFFIXES:
            continue
        text = path.read_text(errors="replace")
        rel = path.relative_to(ROOT).as_posix()
        if "pdblend2" in text and path.name != "audit_workspace.py" and not rel.startswith("src/pdblend2/") and rel != "pyproject.toml":
            stale_names.append(rel)
        if "pdblend4" in text and path.suffix in {".py", ".sh", ".toml"} and path.name != "audit_workspace.py":
            stale_names.append(rel + " (pdblend4 in executable source)")
    if stale_names:
        findings.append("stale executable naming/path: " + ", ".join(sorted(set(stale_names))))

    if RESULTS.exists():
        for d in sorted(RESULTS.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            if (d / "error.txt").exists():
                findings.append(f"incomplete result has error.txt: {d.relative_to(ROOT)}")
            elif not (d / "summary.json").exists():
                findings.append(f"incomplete result has no summary.json: {d.relative_to(ROOT)}")
            for short, long in (("util.jsonl", "utilization.jsonl"), ("freq.jsonl", "frequency.jsonl")):
                a, b = d / short, d / long
                if a.exists() and b.exists():
                    if sha256(a) == sha256(b):
                        findings.append(f"duplicate metric aliases: {d.name}/{short},{long}")
                    else:
                        findings.append(f"conflicting metric aliases: {d.name}/{short},{long}")

    transient: list[str] = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.name in TEMP_NAMES or any(path.name.endswith(s) for s in TEMP_SUFFIXES):
            transient.append(path.relative_to(ROOT).as_posix())
    if transient:
        findings.append("transient files/directories: " + ", ".join(sorted(transient)))

    by_top: dict[str, int] = defaultdict(int)
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        try:
            top = path.relative_to(ROOT).parts[0]
        except ValueError:
            continue
        by_top[top] += path.stat().st_size

    complete = sum(1 for d in RESULTS.iterdir() if d.is_dir() and (d / "summary.json").exists()) if RESULTS.exists() else 0
    print(f"workspace={ROOT}")
    print(f"complete_result_dirs={complete}")
    print("size_by_top_level=" + ", ".join(f"{k}:{v}" for k, v in sorted(by_top.items())))
    if findings:
        print("findings:")
        for finding in findings:
            print(f"- {finding}")
    else:
        print("findings: none")
    return 1 if args.strict and findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
