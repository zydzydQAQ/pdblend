#!/usr/bin/env python3
"""CPU-only KV artifact audit and exact historical prompt reproducer export.

This command never connects to a service or changes an experiment.  Decoded
legacy text cannot recover output token IDs and is never promoted to a token
golden.  Native rows should retain references, prefill.outputs, decode.token_ids,
the actual prompt (or its canonical SHA256), and per-layer KV receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def prompt_hash(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def historical_prompt(length: int, repeat: int = 0) -> list[int]:
    # Identical to frozen smoke-source/src/pdblend/bench/gates.py, rather than
    # loading mutable runtime code or trying to retokenize a decoded string.
    rng = random.Random(100 * length + repeat)
    return [rng.randint(1000, 60000) for _ in range(length)]


def token_ids(value):
    """Return explicit generated token IDs only; None means missing evidence."""
    if not isinstance(value, dict):
        return None
    if "token_ids" in value:
        ids = value["token_ids"]
        if isinstance(ids, list) and all(type(x) is int and x >= 0 for x in ids):
            return list(ids)
        return None
    events = value.get("events", value.get("outputs"))
    if not isinstance(events, list) or not events:
        return None
    pieces = [token_ids(event) for event in events]
    return None if any(x is None for x in pieces) else [t for piece in pieces for t in piece]


def first_difference(reference, candidate):
    if reference is None or candidate is None:
        return dict(status="missing_token_ids", exact_match=None,
                    first_difference_index=None, reference_token_id=None, candidate_token_id=None)
    common = min(len(reference), len(candidate))
    index = next((i for i in range(common) if reference[i] != candidate[i]), common)
    same = index == len(reference) == len(candidate)
    return dict(status="match" if same else "mismatch", exact_match=same,
                first_difference_index=None if same else index,
                reference_token_id=reference[index] if index < len(reference) else None,
                candidate_token_id=candidate[index] if index < len(candidate) else None,
                reference_tokens=len(reference), candidate_tokens=len(candidate))


def repeat_evidence(values):
    known = [x for x in values if x is not None]
    complete = len(known) == len(values) and len(known) >= 2 and all(known)
    return dict(observed=len(values), explicit_token_runs=len(known),
                stable=all(x == known[0] for x in known) if complete else None,
                missing_token_evidence=not complete)


def inspect_native_row(row):
    references = row.get("references", [])
    if not isinstance(references, list):
        references = []
    ids = [token_ids(x) for x in references]
    reference = ids[0] if ids else None
    decoded = token_ids(row.get("decode"))
    prefill = token_ids(row.get("prefill"))
    return dict(
        token_comparison=first_difference(reference, decoded),
        within_reference_path=repeat_evidence(ids),
        prefill_first_vs_reference=first_difference(
            None if reference is None else reference[:1],
            None if prefill is None else prefill[:1]),
        prefill_first_vs_decode=first_difference(
            None if prefill is None else prefill[:1],
            None if decoded is None else decoded[:1]),
        reference_token_ids=reference, decode_token_ids=decoded, prefill_token_ids=prefill,
        reported_reference_stable=row.get("reference_stable"),
        reported_tokens_match=row.get("tokens_match"),
        formal_eligible=False)


def input_files(paths):
    found = set()
    for path in paths:
        if path.is_file():
            found.add(path.resolve())
        elif path.is_dir():
            for name in ("completion.json", "progress.json"):
                found.update(x.resolve() for x in path.rglob(name))
        else:
            raise FileNotFoundError(path)
    # A final artifact supersedes the progress snapshot from the same attempt.
    return sorted(p for p in found if p.name != "progress.json" or p.with_name("completion.json") not in found)


def audit(paths):
    artifacts, reproductions, repeat_groups = [], {}, defaultdict(list)
    for path in input_files(paths):
        raw = path.read_bytes()
        data = json.loads(raw)
        if not isinstance(data, dict) or "kv" not in data:
            continue
        model = data.get("model_id", data.get("model"))
        common = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                      model=model, tp=data.get("tp"), pp=data.get("pp", 1),
                      image_digest=data.get("image_digest"), source_sha256=data.get("source_sha256"))
        kv = data["kv"]
        if isinstance(kv, dict):
            rows = []
            for row in kv.get("rows", []):
                length, repeat = row["input_tokens"], row.get("repeat", 0)
                mixed, pd = row.get("mixed_text"), row.get("pd_text")
                prefix = 0
                if isinstance(mixed, str) and isinstance(pd, str):
                    while prefix < min(len(mixed), len(pd)) and mixed[prefix] == pd[prefix]:
                        prefix += 1
                rows.append(dict(length=length, repeat=repeat, text_match=row.get("text_match"),
                                 common_prefix_chars=prefix, mixed_text=mixed, pd_text=pd,
                                 error=row.get("error"), token_golden_status="missing_token_ids"))
                if row.get("text_match") is False:
                    key = (length, repeat)
                    prompt = historical_prompt(*key)
                    case = reproductions.setdefault(key, dict(
                        case_id=f"legacy-random-{length}-{repeat}", input_tokens=length,
                        generator="random.Random(100*length+repeat).randint(1000,60000)",
                        generator_seed=100 * length + repeat, generator_repeat=repeat,
                        prompt=prompt, prompt_sha256=prompt_hash(prompt), decode_tokens=16,
                        temperature=0.0, seed=701, same_prompt_path_repeats=3,
                        historical_failures=[], formal_eligible=False))
                    case["historical_failures"].append(dict(model=model, tp=data.get("tp"), artifact=str(path)))
            artifacts.append(dict(**common, evidence="historical_text_only", rows=rows))
        elif isinstance(kv, list):
            rows = []
            for row in kv:
                evidence = inspect_native_row(row)
                prompt = row.get("prompt")
                fingerprint = prompt_hash(prompt) if isinstance(prompt, list) else row.get("prompt_sha256")
                if prompt is not None and row.get("prompt_sha256") not in (None, fingerprint):
                    evidence["prompt_identity_error"] = "recorded prompt checksum mismatch"
                    fingerprint = None
                item = dict(length=row.get("length"), repeat=row.get("repeat"),
                            prompt_sha256=fingerprint, **evidence)
                rows.append(item)
                if fingerprint:
                    repeat_groups[(str(path), model, data.get("tp"), fingerprint)].append(item)
            artifacts.append(dict(**common, evidence="native_token_artifact", rows=rows))
    stability = []
    for (path, model, tp, fingerprint), rows in repeat_groups.items():
        stability.append(dict(path=path, model=model, tp=tp, prompt_sha256=fingerprint,
                              decode=repeat_evidence([r["decode_token_ids"] for r in rows]),
                              prefill_first=repeat_evidence([r["prefill_token_ids"] for r in rows])))
    report = dict(schema=1, artifact_count=len(artifacts), artifacts=artifacts,
                  same_prompt_path_stability=stability, formal_eligible=False,
                  conclusion="Diagnostic evidence only; exact output golden remains required.")
    plan = dict(schema=1, cases=[reproductions[k] for k in sorted(reproductions)],
                collection_order=["D reference 16 tokens (3 identical-prompt repeats)",
                                  "P retained prefill 1 token and D PD 16 tokens (3 identical-prompt repeats)",
                                  "compare exact token IDs, first difference, and source/received/injected KV digests"],
                formal_eligible=False)
    return report, plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+", help="artifact files or attempt directories")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report, plan = audit(args.inputs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (args.out / "reproducer-prompts.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(dict(artifacts=report["artifact_count"], historical_prompt_cases=len(plan["cases"]),
                          out=str(args.out.resolve()), formal_eligible=False)))


if __name__ == "__main__":
    main()
