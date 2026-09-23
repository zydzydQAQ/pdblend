#!/usr/bin/env python3
"""Prepare identical source splits independently for Qwen2.5 7B/14B/32B.

The frozen 7B corpus is used only as a split index (source file and source
row). Raw records are re-read and tokenized for each model. References are
used only to determine output length and are never written to the artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterable

MODELS = (("7b", "Qwen2.5-7B-Instruct"), ("14b", "Qwen2.5-14B-Instruct"), ("32b", "Qwen2.5-32B-Instruct"))
SPLITS = ("calibration", "tuning", "evaluation")
EXPECTED_SIZES = {"calibration": 256, "tuning": 256, "evaluation": 1500}
DATASETS = ("alpaca", "sharegpt", "longbench")
TOKENIZER_FILES = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model",
    "sentencepiece.bpe.model", "chat_template.jinja",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def raw_examples(dataset: str, raw_root: Path, templates: dict[str, str]) -> Iterable[tuple[str, int, list[dict[str, str]], str]]:
    """Copy of the raw parser, with no dependence on the old preparation command."""
    if dataset == "alpaca":
        path = raw_root / "alpaca_gpt4.json"
        for i, item in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            text = str(item.get("instruction", ""))
            if item.get("input"): text += "\n\n" + str(item["input"])
            yield str(path), i, [{"role": "user", "content": text}], str(item.get("output", ""))
    elif dataset == "sharegpt":
        path = raw_root / "ShareGPT_V3_unfiltered_cleaned_split.json"
        for i, item in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            messages: list[dict[str, str]] = []; choices = []
            for turn in item.get("conversations", []):
                role = {"human": "user", "gpt": "assistant", "system": "system"}.get(turn.get("from"))
                if role is None: continue
                if role == "assistant" and messages and messages[-1]["role"] == "user": choices.append((list(messages), str(turn.get("value", ""))))
                messages.append({"role": role, "content": str(turn.get("value", ""))})
            if choices:
                prompt, answer = choices[-1]; yield str(path), i, prompt, answer
    elif dataset == "longbench":
        for path in sorted((raw_root / "longbench").glob("*.jsonl")):
            if path.stem.endswith("_e"): continue
            if path.stem not in templates: raise ValueError(f"missing LongBench template: {path.stem}")
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if not line.strip(): continue
                item = json.loads(line); answers = item.get("answers", [])
                if answers: yield str(path), i, [{"role": "user", "content": templates[path.stem].format(**item)}], str(answers[0])
    else: raise ValueError(f"unknown dataset: {dataset}")


def load_split_sources(base: Path, dataset: str) -> dict[str, list[tuple[str, int]]]:
    payload = json.loads((base / f"{dataset}.json").read_text(encoding="utf-8")); result = {}
    seen = set()
    for split in SPLITS:
        rows = payload.get(split)
        if not isinstance(rows, list) or len(rows) != EXPECTED_SIZES[split]: raise ValueError(f"{dataset}/{split}: frozen split size mismatch")
        pairs = [(str(Path(r["source_file"]).resolve()), int(r["source_index"])) for r in rows]
        if len(set(pairs)) != len(pairs): raise ValueError(f"{dataset}/{split}: duplicate source identity")
        if seen.intersection(pairs): raise ValueError(f"{dataset}/{split}: source identity overlaps another split")
        if any(index < 0 for _, index in pairs): raise ValueError(f"{dataset}/{split}: negative source index")
        seen.update(pairs)
        result[split] = pairs
    return result


def encode_workload(tokenizer: Any, messages: list[dict[str, str]], reference: str, *, max_input: int = 7168, max_output: int = 512) -> dict:
    if max_input < 2 or max_output < 2 or max_input + max_output > 8192:
        raise ValueError("input/output caps must be >= 2 and fit 8192 tokens")
    ids = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)); ref_ids = list(tokenizer.encode(reference, add_special_tokens=False))
    if not ids or len(ref_ids) < 2: raise ValueError("selected source has no usable reference")
    original = len(ids)
    if original > max_input:
        left = max_input // 2; ids = ids[:left] + ids[-(max_input - left):]
    content_hash = sha256_json(messages); output_len = min(max_output, len(ref_ids))
    return {"prompt": ids, "input_tokens": len(ids), "output_tokens": output_len, "original_input_tokens": original, "truncated": original > max_input, "content_sha256": content_hash, "request_shape_sha256": sha256_json({"input_tokens": len(ids), "output_tokens": output_len, "content_sha256": content_hash})}


def input_binding(*, model_key: str, model_name: str, model_path: Path,
                  raw_root: Path, templates_path: Path, base_prepared: Path,
                  max_input: int, max_output: int, split_sources: dict) -> dict:
    """Hash the actual inputs on every invocation, including resume."""
    tokenizer_files = {str(p.relative_to(model_path)): sha256_file(p)
                       for p in sorted(model_path.rglob("*"))
                       if p.is_file() and (p.name in TOKENIZER_FILES or p.suffix == ".jinja")}
    if not tokenizer_files:
        raise ValueError(f"no tokenizer files found in {model_path}")
    sources = sorted({source for ds in split_sources.values() for rows in ds.values()
                      for source, _ in rows})
    for source in sources:
        if not Path(source).is_relative_to(raw_root.resolve()):
            raise ValueError(f"frozen source lies outside raw root: {source}")
    return {
        "model": model_key, "model_name": model_name, "model_path": str(model_path.resolve()),
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "tokenizer_files_sha256": tokenizer_files, "tokenizer_sha256": sha256_json(tokenizer_files),
        "raw_root": str(raw_root.resolve()),
        "raw_files_sha256": {source: sha256_file(Path(source)) for source in sources},
        "template_sha256": sha256_file(templates_path),
        "base_prepared": str(base_prepared.resolve()),
        "base_manifest_sha256": sha256_file(base_prepared / "manifest.json"),
        "split_source_sha256": sha256_json(split_sources),
        "input_cap": max_input, "output_cap": max_output,
        "preparation_script_sha256": sha256_file(Path(__file__)),
    }


def _verified(path: Path, binding: dict, split_sources: dict) -> bool:
    """Never accept a directory solely because it exists or has a manifest."""
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("complete") is not True or any(manifest.get(k) != v for k, v in binding.items()):
            return False
        for dataset in DATASETS:
            file = path / f"{dataset}.json"
            if sha256_file(file) != manifest.get("dataset_sha256", {}).get(dataset):
                return False
            data = json.loads(file.read_text(encoding="utf-8"))
            if data.get("model") != binding["model"] or data.get("dataset") != dataset:
                return False
            for split, count in EXPECTED_SIZES.items():
                rows = data[split]
                if len(rows) != count:
                    return False
                if [(r["source_file"], r["source_index"]) for r in rows] != split_sources[dataset][split]:
                    return False
                for row in rows:
                    if any(k in row for k in ("reference", "answer", "answers", "messages")):
                        return False
                    if not (0 < row["input_tokens"] == len(row["prompt"]) <= binding["input_cap"]
                            and 2 <= row["output_tokens"] <= binding["output_cap"]):
                        return False
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def write_json(path: Path, value: Any, *, indent: int | None = None) -> None:
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, sort_keys=True, indent=indent,
                  separators=(",", ":") if indent is None else None)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())


def prepare_model(*, model_key: str, model_name: str, model_path: Path, raw_root: Path, templates_path: Path, base_prepared: Path, output: Path, max_input: int = 7168, max_output: int = 512, force: bool = False) -> dict:
    if max_input < 2 or max_output < 2 or max_input + max_output > 8192:
        raise ValueError("input/output caps must be >= 2 and fit 8192 tokens")
    output.parent.mkdir(parents=True, exist_ok=True)
    split_sources = {ds: load_split_sources(base_prepared, ds) for ds in DATASETS}
    binding = input_binding(model_key=model_key, model_name=model_name, model_path=model_path,
                            raw_root=raw_root, templates_path=templates_path, base_prepared=base_prepared,
                            max_input=max_input, max_output=max_output, split_sources=split_sources)
    if output.exists():
        if _verified(output, binding, split_sources):
            return json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if not force: raise FileExistsError(f"output exists but is not verified complete: {output}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    templates = json.loads(templates_path.read_text(encoding="utf-8")); raw_root = raw_root.resolve(); examples = {}
    for dataset in DATASETS:
        selected = {pair for rows in split_sources[dataset].values() for pair in rows}
        examples[dataset] = {}
        for source, index, messages, reference in raw_examples(dataset, raw_root, templates):
            pair = (str(Path(source).resolve()), index)
            if pair in selected:
                examples[dataset][pair] = (messages, reference)
        missing = [pair for split in SPLITS for pair in split_sources[dataset][split] if pair not in examples[dataset]]
        if missing: raise ValueError(f"{dataset}: {len(missing)} frozen rows missing from raw files")
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)); dataset_sha = {}; summaries = {}
    try:
        for dataset in ("alpaca", "sharegpt", "longbench"):
            payload = {"schema": 2, "model": model_key, "model_name": model_name, "dataset": dataset, "derived_from": str(base_prepared.resolve()), "input_cap": max_input, "output_cap": max_output}
            for split in SPLITS:
                records = []
                for source, index in split_sources[dataset][split]:
                    messages, reference = examples[dataset][(source, index)]
                    try:
                        record = encode_workload(tokenizer, messages, reference, max_input=max_input, max_output=max_output)
                    except ValueError as exc:
                        raise ValueError(f"{model_key}/{dataset}/{split} source {source}:{index}: {exc}") from exc
                    record.update(source_file=source, source_index=index); records.append(record)
                if len(records) != EXPECTED_SIZES[split]: raise AssertionError(f"{dataset}/{split}: wrong output count")
                payload[split] = records
            out_file = temp / f"{dataset}.json"; write_json(out_file, payload); dataset_sha[dataset] = sha256_file(out_file)
            all_records = [r for split in SPLITS for r in payload[split]]; lengths = sorted(r["input_tokens"] for r in all_records)
            summaries[dataset] = {"counts": {split: len(payload[split]) for split in SPLITS}, "input_p50": lengths[len(lengths) // 2], "input_max": max(lengths), "truncated": sum(bool(r["truncated"]) for r in all_records), "sha256": dataset_sha[dataset]}
        manifest = {"schema": 3, "complete": True, **binding, "dataset_sha256": dataset_sha, "datasets": summaries,
                    "truncation": "middle; preserve prompt head and final question/chat suffix",
                    "output_work": "min(model-tokenized reference length, output cap); reference text omitted"}
        write_json(temp / "manifest.json", manifest, indent=2)
        if not _verified(temp, binding, split_sources):
            raise RuntimeError("new corpus failed its own integrity verification")
        backup = output.with_name(f".{output.name}.superseded-{uuid.uuid4().hex}") if output.exists() else None
        if backup is not None:
            output.rename(backup)
        try:
            temp.rename(output)
        except Exception:
            if backup is not None:
                backup.rename(output)
            raise
        return manifest
    except Exception:
        shutil.rmtree(temp, ignore_errors=True); raise


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--raw-root", type=Path, default=Path("/home/pdblend/datasets/raw")); p.add_argument("--base-prepared", type=Path, default=Path("/home/pdblend/datasets/prepared/2026-09-13-7b-v1")); p.add_argument("--templates", type=Path, default=Path("/home/pdblend/datasets/raw/dataset2prompt.json")); p.add_argument("--models-dir", type=Path, default=Path("/home/models")); p.add_argument("--out-root", type=Path, default=Path("datasets/prepared")); p.add_argument("--max-input", type=int, default=7168); p.add_argument("--max-output", type=int, default=512); p.add_argument("--force", action="store_true"); args = p.parse_args(argv)
    if args.max_input < 2 or args.max_output < 2 or args.max_input + args.max_output > 8192:
        p.error("input/output caps must be >= 2 and fit the 8192-token engine limit")
    args.out_root.mkdir(parents=True, exist_ok=True)
    for key, name in MODELS:
        output = args.out_root / f"2026-09-22-{key}-v1"; manifest = prepare_model(model_key=key, model_name=name, model_path=args.models_dir / name, raw_root=args.raw_root, templates_path=args.templates, base_prepared=args.base_prepared, output=output, max_input=args.max_input, max_output=args.max_output, force=args.force); print(json.dumps({"model": key, "status": "complete", "path": str(output), "datasets": manifest["datasets"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
