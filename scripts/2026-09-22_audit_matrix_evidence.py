#!/usr/bin/env python3
"""Fail-closed audit for an active PDblend matrix execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--profile-sha", required=True)
    ap.add_argument("--source-sha", required=True)
    ap.add_argument("--corpus-sha", required=True)
    ap.add_argument("--expected", type=int, default=27)
    args = ap.parse_args()
    execution = json.loads((args.root / "execution.json").read_text())
    failures: list[dict] = []
    if execution.get("status") != "complete" or execution.get("returncode") != 0:
        failures.append({"kind": "execution", "detail": execution.get("status")})
    for key, expected in (("profile_sha256", args.profile_sha), ("source_sha256", args.source_sha),
                          ("corpus_sha256", args.corpus_sha)):
        if execution.get(key) != expected:
            failures.append({"kind": "execution_input", "key": key})
    spec = json.loads((args.root / "spec.json").read_text())
    rows = []
    for point in spec["points"]:
        name = point["name"]
        folder = args.root / name
        try:
            evidence = json.loads((folder / "evidence.json").read_text())
            ident = evidence["identity"]
            if evidence.get("status") != "complete" or evidence.get("returncode") != 0:
                raise ValueError("incomplete")
            if not evidence.get("inputs_unchanged") or evidence.get("identity_sha256") != digest(ident):
                raise ValueError("identity attestation mismatch")
            for key, expected in (("profile_sha256", args.profile_sha), ("source_sha256", args.source_sha),
                                  ("corpus_sha256", args.corpus_sha)):
                if ident.get(key) != expected:
                    raise ValueError(f"{key} mismatch")
            for filename, checksum in evidence["artifacts"].items():
                path = folder / filename
                if not path.is_file() or sha(path) != checksum:
                    raise ValueError(f"artifact {filename} mismatch")
            required = {"summary.json", "outcomes.jsonl", "power.jsonl", "controller.jsonl", "freq.jsonl"}
            if not required.issubset(evidence["artifacts"]):
                raise ValueError("required artifact missing")
            summary = json.loads((folder / "summary.json").read_text())
            if summary.get("policy", {}).get("name") != ident.get("policy"):
                raise ValueError("policy mismatch")
            rows.append({"name": name, "status": "complete", "seed": ident.get("seed")})
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            failures.append({"kind": "point", "name": name, "detail": str(exc)})
            rows.append({"name": name, "status": "invalid", "detail": str(exc)})
    report = {"status": "pass" if not failures and len(rows) == args.expected else "fail",
              "expected": args.expected, "complete": sum(r["status"] == "complete" for r in rows),
              "rows": rows, "failures": failures}
    (args.root / "evidence-audit.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: report[k] for k in ("status", "expected", "complete", "failures")}, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
