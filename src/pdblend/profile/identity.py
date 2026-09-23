"""Independent profile identity and provenance checks."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..model_registry import ModelSpec


@dataclass(frozen=True)
class ProfileKey:
    system: str
    model_id: str
    engine_revision: str
    hardware_id: str
    tp: int
    pp: int = 1
    role: str = "mixed"
    workload_shape: str = "default"
    frequency_mhz: int | None = None

    def __post_init__(self) -> None:
        for name in ("system", "model_id", "engine_revision", "hardware_id", "role", "workload_shape"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"profile identity requires {name}")
        if int(self.tp) < 1 or int(self.pp) < 1:
            raise ValueError("profile TP and PP must be positive")
        if self.frequency_mhz is not None and int(self.frequency_mhz) <= 0:
            raise ValueError("profile frequency must be positive")

    def as_dict(self) -> dict:
        return asdict(self)

    def namespace(self) -> str:
        return "/".join(str(x).replace("/", "_") for x in (
            self.system, self.model_id, self.engine_revision, self.hardware_id,
            f"tp{self.tp}", f"pp{self.pp}", self.role, self.workload_shape))


def source_sha256(root: Path | str) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def gpu_identity() -> dict:
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=5)
        rows = [line.strip() for line in text.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        rows = []
    # nvidia-smi/NVML reports board power and identity; it is not a whole-host
    # energy meter.  Keep this scope explicit in every provenance record.
    return {"hostname": platform.node(), "gpus": rows, "meter_scope": "gpu_board"}


def provenance(*, model: ModelSpec, engine_revision: str, root: Path | str | None = None,
               hardware_id: str | None = None) -> dict:
    source = os.environ.get("PDBLEND_SOURCE_HASH") or os.environ.get("PDBLEND_SOURCE_SHA256")
    if root is not None and not source:
        source = source_sha256(root)
    return {
        "model_id": model.model_id,
        "model_manifest_sha256": model.manifest_sha256,
        "tokenizer_path": model.model_path,
        "engine_revision": engine_revision,
        "environment": {
            "image_digest": os.environ.get("PDBLEND_IMAGE_DIGEST") or os.environ.get("PDBLEND_IMAGE_ID"),
            "source_hash": source,
            "vllm": os.environ.get("PDBLEND_VLLM_VERSION"),
            "torch": os.environ.get("PDBLEND_TORCH_VERSION"),
            "cuda": os.environ.get("CUDA_VERSION"),
        },
        "hardware": dict(gpu_identity(), **({"hardware_id": hardware_id} if hardware_id else {})),
    }


def write_profile_identity(path: Path | str, key: ProfileKey, evidence: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"profile_key": key.as_dict(), "profile_namespace": key.namespace(), "provenance": evidence}
    if isinstance(evidence.get("evidence_bindings"), Mapping):
        payload["evidence_bindings"] = evidence["evidence_bindings"]
    # Keep the writer and strict validator on the same canonical byte stream.
    payload["identity_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def require_profile_identity(raw: dict, *, expected: ProfileKey, strict: bool = False,
                             require_provenance: bool | None = None) -> None:
    observed = raw.get("profile_key")
    if not isinstance(observed, dict):
        raise ValueError("profile is missing independent profile_key")
    for field in ("system", "model_id", "engine_revision", "hardware_id", "tp", "pp", "role", "workload_shape"):
        actual = observed.get(field)
        wanted = getattr(expected, field)
        if actual != wanted:
            raise ValueError(f"profile identity mismatch for {field}: {actual!r} != {wanted!r}")
    if require_provenance is not None:
        strict = require_provenance
    if strict:
        require_profile_provenance(raw, expected=expected)


# ---- strict provenance -----------------------------------------------------

MISSING = {None, "", "unknown", "UNKNOWN", "missing", "null"}
ENVIRONMENT_FIELDS = ("image_digest", "source_hash", "vllm", "torch", "cuda", "gpu_uuids")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_value(value: object) -> str:
    return sha256_bytes(canonical_json(value).encode())


def _known(value: object) -> bool:
    if value is None or (isinstance(value, str) and value in MISSING):
        return False
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _uuid_values(value: object) -> list[str]:
    if isinstance(value, str):
        return sorted(x.strip() for x in value.split(",") if x.strip())
    if isinstance(value, (list, tuple)):
        return sorted(str(x).strip() for x in value if str(x).strip())
    return []


def _environment(raw: Mapping) -> dict:
    direct = raw.get("environment")
    env = dict(direct) if isinstance(direct, Mapping) else {}
    prov = raw.get("provenance")
    if isinstance(prov, Mapping):
        penv = prov.get("environment")
        if isinstance(penv, Mapping):
            env = {**dict(penv), **env}
        hardware = prov.get("hardware")
        if isinstance(hardware, Mapping):
            env.setdefault("hardware_id", hardware.get("hardware_id"))
            if not env.get("gpu_uuids"):
                env["gpu_uuids"] = hardware.get("gpu_uuids") or hardware.get("gpus")
            env.setdefault("meter_scope", hardware.get("meter_scope"))
    for field in ENVIRONMENT_FIELDS + ("hardware_id", "meter_scope"):
        if field not in env and field in raw:
            env[field] = raw[field]
    env["gpu_uuids"] = _uuid_values(env.get("gpu_uuids"))
    return env


def profile_identity(raw: Mapping) -> dict:
    key = raw.get("profile_key")
    if not isinstance(key, Mapping):
        raise ValueError("profile is missing independent profile_key")
    fields = ("system", "model_id", "engine_revision", "hardware_id", "tp", "pp", "role", "workload_shape")
    result = {field: key.get(field) for field in fields}
    result["frequency_mhz"] = key.get("frequency_mhz")
    result["environment"] = _environment(raw)
    return result


def require_profile_provenance(raw: Mapping, *, expected: ProfileKey | None = None,
                               require_evidence: bool = True) -> dict:
    """Strictly validate profile key, runtime environment and evidence hashes."""
    observed = raw.get("profile_key")
    if not isinstance(observed, Mapping):
        raise ValueError("profile is missing independent profile_key")
    fields = ("system", "model_id", "engine_revision", "hardware_id", "tp", "pp", "role", "workload_shape")
    for field in fields:
        if not _known(observed.get(field)):
            raise ValueError(f"profile identity missing {field}")
        if expected is not None and observed.get(field) != getattr(expected, field):
            raise ValueError(f"profile identity mismatch for {field}: {observed.get(field)!r} != {getattr(expected, field)!r}")
    for field in ("system", "model_id", "tp", "pp", "role", "workload_shape"):
        if field in raw and raw.get(field) != observed.get(field):
            raise ValueError(f"profile top-level identity mismatch for {field}")
    # A formal profile must bind the exact weights and tokenizer used by the
    # engine.  These values are populated only when the model verification
    # receipt has been supplied; legacy/screening raw files therefore fail the
    # strict gate instead of silently reusing another model's calibration.
    for field in ("model_hash", "tokenizer_hash", "verification_receipt"):
        if not _known(raw.get(field)):
            raise ValueError(f"profile provenance missing {field}")
    env = _environment(raw)
    missing = [field for field in ENVIRONMENT_FIELDS if not _known(env.get(field))]
    if missing:
        raise ValueError("profile provenance missing " + ",".join(missing))
    if not _known(env.get("hardware_id")) and not _known(observed.get("hardware_id")):
        raise ValueError("profile provenance missing hardware_id")
    if env.get("meter_scope") in ("whole_host", "host"):
        raise ValueError("GPU board evidence cannot claim whole-host power")
    if require_evidence:
        bindings = raw.get("evidence_bindings")
        if not isinstance(bindings, Mapping):
            raise ValueError("profile is missing separate sample/holdout evidence bindings")
        for kind in ("sample", "holdout"):
            item = bindings.get(kind)
            if not isinstance(item, Mapping) or item.get("kind") != kind or not item.get("sha256"):
                raise ValueError(f"profile is missing {kind} evidence binding")
            files = item.get("files")
            if not isinstance(files, Mapping) or not files:
                raise ValueError(f"profile {kind} evidence files are empty")
            if item.get("sha256") != sha256_value({"kind": kind, "files": dict(files)}):
                raise ValueError(f"profile {kind} evidence digest mismatch")
        if bindings["sample"].get("sha256") == bindings["holdout"].get("sha256"):
            raise ValueError("sample and holdout evidence must be independently bound")
    digest = raw.get("identity_sha256")
    if not digest:
        raise ValueError("profile identity digest missing")
    payload = {key: value for key, value in raw.items() if key != "identity_sha256"}
    if digest != sha256_value(payload):
        raise ValueError("profile identity digest mismatch")
    return profile_identity(raw)


def evidence_binding(paths: Sequence[Path | str] | Mapping[str, str], *, kind: str) -> dict:
    """Hash a sample or holdout set under a distinct, typed binding."""
    if kind not in ("sample", "holdout"):
        raise ValueError("evidence kind must be sample or holdout")
    if isinstance(paths, Mapping):
        files = {str(key): str(value) for key, value in sorted(paths.items())}
    else:
        files = {}
        for value in sorted(str(p) for p in paths):
            path = Path(value)
            if not path.is_file():
                raise ValueError(f"missing {kind} evidence: {value}")
            files[value] = sha256_bytes(path.read_bytes())
    return {"kind": kind, "files": files, "sha256": sha256_value({"kind": kind, "files": files})}


strict_profile_identity = require_profile_provenance
