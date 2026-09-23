"""Relocate a recorded Dynamo profile without changing measured artifacts.

Profile runs happen in a container where cells are recorded below ``/output``.
The host cannot use those paths directly.  This module creates a derived JSON
whose cell paths point at the corresponding files in a completed attempt,
while checking every digest and identity field before writing anything.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


class PortableProfileError(ValueError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _recorded_path(value: str, recorded_root: Path, artifact_root: Path) -> tuple[Path, str]:
    """Resolve a path from the recorded namespace and reject traversal."""
    raw = Path(value)
    rec = (recorded_root / raw.relative_to(recorded_root)) if raw.is_absolute() and _inside(raw, recorded_root) else None
    if rec is None:
        # A relative path is allowed only as a path relative to recorded_root.
        if raw.is_absolute():
            raise PortableProfileError(f"path outside recorded root: {value}")
        rec = recorded_root / raw
    rec = rec.resolve()
    if not _inside(rec, recorded_root):
        raise PortableProfileError(f"path traversal/outside recorded root: {value}")
    rel = rec.relative_to(recorded_root)
    actual = (artifact_root / rel).resolve()
    if not _inside(actual, artifact_root) or not actual.is_file():
        raise PortableProfileError(f"recorded artifact is absent in attempt: {value}")
    return actual, str(actual)


def _map_raw_artifacts(cell: dict[str, Any], actual_cell: Path, recorded_root: Path,
                       artifact_root: Path, mappings: list[dict[str, str]]) -> None:
    """Verify the explicit raw artifact table emitted by profile_v1."""
    artifacts = cell.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise PortableProfileError("cell artifacts must be a mapping")
    # profile_v1 writes paths relative to dynamollm-profile, not the cell dir.
    profile_dir = actual_cell.parent.parent
    for name, expected in artifacts.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise PortableProfileError("invalid raw artifact reference")
        ref = Path(name)
        if ref.is_absolute() or ".." in ref.parts:
            raise PortableProfileError(f"raw artifact escapes profile: {name}")
        path = (profile_dir / ref).resolve()
        if not _inside(path, artifact_root) or not path.is_file():
            raise PortableProfileError(f"raw artifact is absent: {name}")
        digest = _sha(path)
        if digest != expected:
            raise PortableProfileError(f"raw artifact checksum differs: {name}")
        # Convert the host path back to the recorded namespace for an auditable
        # receipt.  The raw JSON itself is deliberately never rewritten.
        recorded = str(recorded_root / path.relative_to(artifact_root))
        mappings.append({"recorded": recorded, "actual": str(path), "sha256": digest,
                         "kind": "raw"})


def _validate_capability(capability: dict[str, Any], identity: dict[str, Any]) -> None:
    """Bind the native all-rank receipt to this collector's physical instance."""
    required = ("model_hash", "tokenizer_hash", "verification_receipt_sha256")
    if any(not isinstance(capability.get(key), str) or len(capability[key]) != 64
           for key in required):
        raise PortableProfileError("native model/tokenizer verification identity incomplete")
    for key, native_key in (("model_id", "model_id"), ("engine_revision", "engine_revision"),
                            ("image_digest", "image_digest"), ("source_sha256", "source_revision"),
                            ("tp", "tp"), ("pp", "pp")):
        if capability.get(native_key) != identity.get(key):
            raise PortableProfileError("native capability identity differs: " + key)
    uuids = identity.get("gpu_uuids")
    if (not isinstance(uuids, dict) or len(uuids) != identity["tp"]
            or len(set(uuids.values())) != identity["tp"]
            or sorted(capability.get("gpu_uuids", [])) != sorted(uuids.values())):
        raise PortableProfileError("native capability physical GPU identity differs")
    state = capability.get("state", {})
    ranks = state.get("ranks", [])
    if (capability.get("native_evidence_complete") is not True
            or state.get("native_evidence_complete") is not True
            or state.get("healthy") is not True
            or len(ranks) != identity["tp"]
            or {row.get("rank") for row in ranks} != set(range(identity["tp"]))
            or any(row.get("native_evidence_complete") is not True or row.get("healthy") is not True
                   or row.get("generation") != state.get("generation") for row in ranks)):
        raise PortableProfileError("native capability all-rank evidence incomplete")
    # Real jobs mount the independently verified model inventory. The native
    # hash format differs from predictor.model_identity, so compare inventories
    # rather than pretending those differently constructed hashes are equal.
    receipt_path = os.environ.get("PDBLEND_MODEL_VERIFICATION_RECEIPT")
    if receipt_path:
        receipt_file = Path(receipt_path)
        receipt = json.loads(receipt_file.read_text())
        if _sha(receipt_file) != capability["verification_receipt_sha256"] or receipt.get("all_pass") is not True:
            raise PortableProfileError("native model verification receipt differs")
        model = next((row for row in receipt.get("models", {}).values()
                      if row.get("model_id") == identity["model_id"]), None)
        if not model or model.get("verified") is not True:
            raise PortableProfileError("native model lacks verified inventory")
        for kind, key in (("weight", "model_hash"), ("tokenizer", "tokenizer_hash")):
            files = [row for row in model["files"] if row["kind"] == kind]
            inventory = [(row["path"], row["bytes"], row["sha256"]) for row in files]
            digest = hashlib.sha256(json.dumps(inventory, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
            if not files or digest != capability[key]:
                raise PortableProfileError("native verified inventory differs: " + key)
        files = {row["path"]: row["sha256"] for row in model["files"]}
        for name, digest in identity.get("model_identity", {}).get("tokenizer_files", {}).items():
            if files.get(name) != digest:
                raise PortableProfileError("predictor and native tokenizer inventories differ")


def _validate_raw_cell(cell: dict[str, Any], actual: Path) -> None:
    """Recompute the fit from exactly three windows and a later unseen window."""
    from .profile_v1 import fit_cell, key_for
    identity, point = cell["identity"], cell["point"]
    if actual.stem != key_for(identity, point):
        raise PortableProfileError("cell filename does not bind its identity and shape")
    if (any(type(point.get(key)) is not int or point[key] <= 0
            for key in ("batch", "input_tokens", "output_tokens", "frequency_mhz"))
            or point["input_tokens"] + point["output_tokens"] > 8192):
        raise PortableProfileError("invalid measured workload geometry")
    names = {f"raw/{actual.stem}-{suffix}.json" for suffix in
             ("repeat0", "repeat1", "repeat2", "fit-inputs", "holdout")}
    artifacts = cell.get("artifacts", {})
    if set(artifacts) != names:
        raise PortableProfileError("three repeats, frozen fit-inputs, and independent holdout artifacts required")
    def load(suffix: str) -> dict[str, Any]:
        return json.loads((actual.parent.parent / f"raw/{actual.stem}-{suffix}.json").read_text())
    try:
        windows = [load(f"repeat{i}") for i in range(3)]
        holdout, frozen = load("holdout"), load("fit-inputs")
        repeat_names = {f"raw/{actual.stem}-repeat{i}.json": artifacts[f"raw/{actual.stem}-repeat{i}.json"]
                        for i in range(3)}
        if frozen.get("artifacts") != repeat_names or frozen.get("point") != point:
            raise PortableProfileError("frozen fit inputs differ from the three training repeats")
        previous_end = -math.inf
        request_ids: set[str] = set()
        for repeat, window in enumerate([*windows, holdout]):
            from .profile_epochs import validate_window_qualification
            qualification = validate_window_qualification(window, actual.parent.parent)
            if identity.get('sampling_protocol') == 'cohort-epochs-v1' and qualification is None:
                raise PortableProfileError('epoch-qualified raw repeat lacks its bound qualification')
            if window.get("point") != point or window.get("repeat") != repeat:
                raise PortableProfileError("raw window shape or independent repeat identity differs")
            start, end = window["started_s"], window["finished_s"]
            if not all(math.isfinite(x) for x in (start, end)) or start < previous_end or end <= start:
                raise PortableProfileError("raw independent windows overlap or have invalid timestamps")
            previous_end = end
            if set(map(str, window["gpu_ids"])) != set(identity["gpu_uuids"]):
                raise PortableProfileError("raw power GPU group differs from physical identity")
            if any(not math.isfinite(row.get("power_w", float("nan")))
                   or not math.isfinite(row.get("timestamp", float("nan")))
                   for row in window["power"]):
                raise PortableProfileError("raw power samples must be finite measurements")
            if any(row.get("gpu_uuid") != identity["gpu_uuids"].get(str(row.get("gpu")))
                   for row in window["power"]):
                raise PortableProfileError("raw power sample physical GPU UUID differs")
            local_ids = set()
            for request in window["requests"]:
                rid = request.get("request_id")
                if not isinstance(rid, str) or rid in request_ids or rid in local_ids:
                    raise PortableProfileError("raw repeats do not contain independent requests")
                local_ids.add(rid)
                if (request.get("tokens") != point["output_tokens"] or request.get("ok") is not True
                        or not start <= request["submitted_s"] < request["first_token_s"]
                        <= request["finished_s"] <= end):
                    raise PortableProfileError("raw SSE output geometry or timestamps differ")
            request_ids.update(local_ids)
            for rank in window["native"]["ranks"]:
                seen = set()
                for sample in rank["samples"]:
                    ids = sample.get("request_ids", [])
                    seen.update(ids)
                    if (not ids or not set(ids) <= local_ids or sample.get("rank") != rank["rank"]
                            or sample.get("max_input_tokens") != point["input_tokens"]
                            or not point["input_tokens"] <= sample.get("max_context_tokens", -1)
                            <= point["input_tokens"] + point["output_tokens"]
                            or not start <= sample.get("at_s", -1) <= end):
                        raise PortableProfileError("native sample does not bind the requested input/output/rank")
                if seen != local_ids:
                    raise PortableProfileError("native rank lacks measured SSE requests")
        if not windows[-1]["finished_s"] <= frozen.get("frozen_at_s", -1) <= holdout["started_s"]:
            raise PortableProfileError("holdout did not follow frozen training inputs")
        fitted = fit_cell(windows, holdout, tp=identity["tp"], point=point)
        if fitted != cell["fit"] or fitted.get("holdout_passed") is not True:
            raise PortableProfileError("fit or holdout errors differ from complete raw evidence")
    except PortableProfileError:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise PortableProfileError("complete raw fit evidence cannot be recomputed") from exc


def validate_profile(*, artifact_root: str | Path, recorded_root: str | Path,
                     profile: str | Path,
                     allow_profile_outside: bool = False) -> dict[str, Any]:
    """Pure-read validation; return a portable value without writing any file.

    ``profile`` may be absolute or relative to ``artifact_root``.  It must be
    within the completed attempt. No caller-controlled strictness flag bypasses
    raw, native, or independent-holdout verification.
    """
    root = Path(artifact_root).resolve()
    recroot = Path(recorded_root).resolve()
    original = Path(profile)
    if not original.is_absolute():
        # Accept either a path relative to the attempt or the usual cwd
        # relative path printed by queue artifacts.
        candidate = original.resolve()
        original = candidate if _inside(candidate, root) else root / original
    original = original.resolve()
    if (not allow_profile_outside and not _inside(original, root)) or not original.is_file():
        raise PortableProfileError("original profile must be inside artifact root")
    value = json.loads(original.read_text())
    if value.get("schema") != 2 or value.get("measurement") != "hardware":
        raise PortableProfileError("only measured schema-2 profiles are portable")
    expected = {
        "system": "dynamollm", "model_id": value.get("model_id"),
        "tp": value.get("tp"), "pp": value.get("pp", 1),
        "engine_revision": value.get("engine_revision"),
        "image_digest": value.get("image_digest"),
        "source_sha256": value.get("source_sha256"),
    }
    if expected["model_id"] is None or expected["tp"] is None:
        raise PortableProfileError("profile lacks model/topology identity")
    if expected["engine_revision"] != "vllm-0.10.1.1":
        raise PortableProfileError("unsupported engine revision")
    if not expected["image_digest"]:
        raise PortableProfileError("profile lacks source image identity")
    if not expected["source_sha256"]:
        raise PortableProfileError("profile lacks source revision identity")
    if value.get("system") != "dynamollm" or value.get("independent_profile") is not True:
        raise PortableProfileError("profile is not an independent Dynamo profile")
    mappings: list[dict[str, str]] = []
    out_value = copy.deepcopy(value)
    identities: list[dict[str, Any]] = []
    native_models: set[tuple[str, str, str]] = set()
    points = out_value.get("points")
    if not isinstance(points, list) or not points:
        raise PortableProfileError("profile has no measured points")
    for point in points:
        source = point.get("source_profile_path")
        if not isinstance(source, str):
            raise PortableProfileError("point lacks source_profile_path")
        actual, actual_name = _recorded_path(source, recroot, root)
        digest = _sha(actual)
        first_sha = point.get("source_sha256")
        second_sha = point.get("source_profile_sha256")
        if first_sha and second_sha and first_sha != second_sha:
            raise PortableProfileError("point source checksum fields disagree")
        wanted = first_sha or second_sha
        if digest != wanted:
            raise PortableProfileError(f"cell checksum differs: {source}")
        cell = json.loads(actual.read_text())
        ident = cell.get("identity")
        if not isinstance(ident, dict):
            raise PortableProfileError("cell lacks identity")
        for key in ("system", "model_id", "tp", "pp", "engine_revision", "image_digest"):
            expected_value = expected[key]
            if ident.get(key, 1 if key == "pp" else None) != expected_value:
                raise PortableProfileError(f"cell {key} identity differs")
        if ident.get("source_sha256") != expected["source_sha256"]:
            raise PortableProfileError("cell source revision differs")
        if ident.get("model_identity") != value.get("model_identity"):
            raise PortableProfileError("cell model/tokenizer identity differs")
        if cell.get("fit", {}).get("holdout_passed") is not True:
            raise PortableProfileError("cell holdout_passed is not true")
        fit = cell.get('fit', {})
        for key in ("tp", "pp"):
            if point.get(key, 1 if key == "pp" else None) != expected[key]:
                raise PortableProfileError(f"point {key} differs from profile")
        cell_point = cell.get("point")
        if not isinstance(cell_point, dict):
            raise PortableProfileError("cell lacks point geometry")
        point_pairs = (("frequency_mhz", "frequency_mhz"), ("input_tokens", "input_tokens"),
                       ("batch", "batch"))
        for point_key, cell_key in point_pairs:
            if point.get(point_key) != cell_point.get(cell_key):
                raise PortableProfileError(f"point geometry differs: {point_key}")
        if point.get("role") != "mixed":
            raise PortableProfileError("portable Dynamo points must be mixed role")
        output_tokens = cell_point.get("output_tokens")
        if output_tokens is None or point.get("context_tokens") != point.get("input_tokens", 0) + output_tokens:
            raise PortableProfileError("point output/context geometry differs")
        for key in ("prefill_s", "iteration_s", "power_w"):
            if point.get(key) != fit.get(key):
                raise PortableProfileError(f"point measurement differs: {key}")
        capability = cell.get("capability_path")
        capability_sha = cell.get("capability_sha256")
        if not isinstance(capability, str) or not isinstance(capability_sha, str):
            raise PortableProfileError("cell capability provenance is incomplete")
        capability_ref = Path(capability)
        if capability_ref.is_absolute() or ".." in capability_ref.parts:
            raise PortableProfileError("capability path escapes profile")
        capability_file = (actual.parent.parent / capability_ref).resolve()
        if not _inside(capability_file, root) or not capability_file.is_file():
            raise PortableProfileError("capability artifact is absent")
        capability_digest = _sha(capability_file)
        if capability_digest != capability_sha:
            raise PortableProfileError("capability checksum differs")
        capability_value = json.loads(capability_file.read_text())
        _validate_capability(capability_value, ident)
        native_models.add(tuple(capability_value[key] for key in
                                ("model_hash", "tokenizer_hash", "verification_receipt_sha256")))
        mappings.append({"recorded": str(recorded_root / capability_ref),
                         "actual": str(capability_file), "sha256": capability_digest,
                         "kind": "capability"})
        _map_raw_artifacts(cell, actual, recroot, root, mappings)
        _validate_raw_cell(cell, actual)
        point["source_profile_path"] = actual_name
        mappings.append({"recorded": source, "actual": actual_name, "sha256": digest,
                         "kind": "cell"})
        identities.append(ident)
    # All cells must carry the same complete identity, including model receipt.
    first = identities[0]
    if len(native_models) != 1:
        raise PortableProfileError("native model/tokenizer identities differ across measured cells")
    for ident in identities[1:]:
        if ident != first:
            raise PortableProfileError("cell identities are not identical")
    original_bytes = original.read_bytes()
    out_value["formal_eligible"] = False
    out_value["hardware_qualified"] = False
    out_value["portable_derived"] = True
    out_value["portable_profile_receipt"] = {
        "original_profile": str(original),
        "original_profile_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "recorded_root": str(recroot), "artifact_root": str(root),
        "cell_count": len(points), "mappings": mappings,
        "holdout_passed": True, "formal_eligible": False,
    }
    return out_value


def relocate_profile(*, artifact_root: str | Path, recorded_root: str | Path,
                     profile: str | Path, out: str | Path,
                     allow_profile_outside: bool = False) -> dict[str, Any]:
    """Validate without modifying inputs, then write a derived portable profile."""
    destination = Path(out)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite derived profile: {destination}")
    value = validate_profile(artifact_root=artifact_root, recorded_root=recorded_root,
                             profile=profile, allow_profile_outside=allow_profile_outside)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value
