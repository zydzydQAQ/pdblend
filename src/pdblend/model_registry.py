"""Model and topology metadata shared by profiling, planning and campaigns.

The registry deliberately keeps model geometry separate from any system's
performance model.  A profile is only usable when its model, TP/PP layout and
engine provenance match this registry.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_MODELS_DIR = Path("/models")


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_path: str
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    dtype: str = "bfloat16"
    max_model_len: int = 8192
    weight_bytes: int = 0
    # A small reserve for CUDA/NCCL/KV buffers on 46 GiB L20 cards.
    gpu_memory_bytes: int = 46_068 * 1024 * 1024
    # These are identity fields, populated from a verified receipt when one is
    # supplied.  They are deliberately not computed by registry properties:
    # hashing 100+ GiB of weights during every planner invocation is unsafe.
    profile_namespace: str = ""
    predictor_paths: tuple[str, ...] = ()
    model_hash: str | None = None
    tokenizer_hash: str | None = None
    verification_receipt: str | None = None

    @property
    def manifest_path(self) -> Path:
        return Path(self.model_path) / "config.json"

    @property
    def manifest_sha256(self) -> str:
        path = self.manifest_path
        if not path.is_file():
            return "missing"
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @property
    def config_path(self) -> Path:
        return self.manifest_path

    def validate_config(self, *, strict_dtype: bool = True) -> dict:
        """Read and validate cheap model geometry/configuration metadata.

        This intentionally reads only ``config.json``.  Full weight/tokenizer
        SHA-256 validation belongs to the offline model-verification receipt.
        """
        if not self.config_path.is_file():
            raise FileNotFoundError(self.config_path)
        try:
            config = json.loads(self.config_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid model config: {self.config_path}") from exc
        expected = {
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "hidden_size": self.hidden_size,
        }
        for field, value in expected.items():
            if config.get(field) != value:
                raise ValueError(f"{self.model_id} config mismatch for {field}: {config.get(field)!r} != {value!r}")
        if strict_dtype and config.get("torch_dtype", self.dtype) != self.dtype:
            raise ValueError(f"{self.model_id} config dtype mismatch: {config.get('torch_dtype')!r} != {self.dtype!r}")
        return config

    def require_predictor(self, *, system: str = "dynamollm") -> str:
        """Return this model's predictor path; never fall back to 7B."""
        candidates = tuple(self.predictor_paths)
        if not candidates:
            raise FileNotFoundError(f"no predictor registered for {self.model_id}")
        for candidate in candidates:
            path = Path(candidate)
            if path.is_file():
                return str(path)
        raise FileNotFoundError(f"no predictor for {self.model_id} ({system}): {candidates}")

    def legal_tp(self, *, available_gpus: int = 8, require_memory: bool = True) -> tuple[int, ...]:
        values = []
        for tp in range(1, min(available_gpus, self.num_attention_heads) + 1):
            # vLLM supports evenly sharded KV heads or an integer replication
            # factor.  This is the same geometry check used by DistServe.
            if self.num_attention_heads % tp:
                continue
            if not (self.num_key_value_heads % tp == 0 or tp % self.num_key_value_heads == 0):
                continue
            if require_memory and self.weight_bytes and self.weight_bytes / tp > self.gpu_memory_bytes * .90:
                continue
            values.append(tp)
        return tuple(values)

    def legal_pp(self, *, available_gpus: int = 8) -> tuple[int, ...]:
        return tuple(p for p in range(1, available_gpus + 1) if self.num_hidden_layers % p == 0)

    def legal_topologies(self, *, available_gpus: int = 8, require_memory: bool = True) -> tuple[tuple[int, int], ...]:
        out = []
        for tp in self.legal_tp(available_gpus=available_gpus, require_memory=False):
            for pp in self.legal_pp(available_gpus=available_gpus):
                if tp * pp > available_gpus:
                    continue
                if require_memory and self.weight_bytes and self.weight_bytes / (tp * pp) > self.gpu_memory_bytes * .90:
                    continue
                out.append((tp, pp))
        return tuple(out)

    def validate_topology(self, tp: int, pp: int = 1, *, available_gpus: int = 8,
                          require_memory: bool = True) -> None:
        if (tp, pp) not in self.legal_topologies(available_gpus=available_gpus, require_memory=require_memory):
            raise ValueError(f"unsupported topology for {self.model_id}: TP{tp} PP{pp}")


def _default_specs(models_dir: Path) -> dict[str, ModelSpec]:
    return {
        "7b": ModelSpec("Qwen2.5-7B-Instruct", str(models_dir / "Qwen2.5-7B-Instruct"), 28, 28, 4, 3584,
                         weight_bytes=15_231_233_024,
                         profile_namespace="qwen2.5-7b",
                         predictor_paths=(str(models_dir / "predictors" / "7b" / "predictor.json"),)),
        "14b": ModelSpec("Qwen2.5-14B-Instruct", str(models_dir / "Qwen2.5-14B-Instruct"), 48, 40, 8, 5120,
                          weight_bytes=29_540_067_328,
                          profile_namespace="qwen2.5-14b",
                          predictor_paths=(str(models_dir / "predictors" / "14b" / "predictor.json"),)),
        "32b": ModelSpec("Qwen2.5-32B-Instruct", str(models_dir / "Qwen2.5-32B-Instruct"), 64, 40, 8, 5120,
                          weight_bytes=65_527_752_704,
                          profile_namespace="qwen2.5-32b",
                          predictor_paths=(str(models_dir / "predictors" / "32b" / "predictor.json"),)),
    }


def _normalise_key(key: str) -> str:
    value = Path(key).name.lower().replace("qwen2.5-", "").replace("-instruct", "")
    return {"7": "7b", "7b": "7b", "14": "14b", "14b": "14b", "32": "32b", "32b": "32b"}.get(value, key)


class ModelRegistry:
    def __init__(self, models_dir: Path | str = DEFAULT_MODELS_DIR,
                 overrides: Mapping[str, Mapping] | None = None,
                 verification_receipt: Path | str | Mapping | None = None):
        self.models_dir = Path(models_dir)
        self._specs = _default_specs(self.models_dir)
        for key, values in (overrides or {}).items():
            norm = _normalise_key(key)
            if norm not in self._specs:
                raise KeyError(f"unknown model key: {key}")
            base = self._specs[norm]
            self._specs[norm] = ModelSpec(**{**base.__dict__, **dict(values)})
        if verification_receipt is not None:
            self._apply_verification_receipt(verification_receipt)

    def _apply_verification_receipt(self, receipt: Path | str | Mapping) -> None:
        if isinstance(receipt, Mapping):
            payload = dict(receipt)
            receipt_name = "<mapping>"
        else:
            receipt_path = Path(receipt)
            payload = json.loads(receipt_path.read_text())
            receipt_name = str(receipt_path)
        if payload.get("schema") != 1 or not isinstance(payload.get("models"), Mapping):
            raise ValueError(f"invalid model verification receipt: {receipt_name}")
        if payload.get("all_pass") is not True:
            raise ValueError(f"model verification receipt is not fully verified: {receipt_name}")
        for key, item in payload["models"].items():
            norm = _normalise_key(str(key))
            if norm not in self._specs:
                continue
            if item.get("verified") is not True:
                raise ValueError(f"model {norm} is not verified in {receipt_name}")
            expected_path = Path(self._specs[norm].model_path).resolve()
            actual_path = Path(str(item.get("model_path", ""))).resolve()
            if expected_path != actual_path:
                raise ValueError(f"model path mismatch for {norm}: {actual_path} != {expected_path}")
            files = item.get("files", [])
            model_hash = hashlib.sha256(
                json.dumps([(f.get("path"), f.get("bytes"), f.get("sha256")) for f in files if f.get("kind") == "weight"], separators=(",", ":"), sort_keys=True).encode()
            ).hexdigest()
            tokenizer_hash = hashlib.sha256(
                json.dumps([(f.get("path"), f.get("bytes"), f.get("sha256")) for f in files if f.get("kind") == "tokenizer"], separators=(",", ":"), sort_keys=True).encode()
            ).hexdigest()
            base = self._specs[norm]
            self._specs[norm] = ModelSpec(**{**base.__dict__,
                                             "weight_bytes": int(item.get("weight_bytes", base.weight_bytes)),
                                             "model_hash": model_hash,
                                             "tokenizer_hash": tokenizer_hash,
                                             "verification_receipt": receipt_name})

    def get(self, key: str) -> ModelSpec:
        norm = _normalise_key(key)
        try:
            return self._specs[norm]
        except KeyError as exc:
            raise KeyError(f"unknown model key: {key}; expected 7b, 14b or 32b") from exc

    def all(self) -> tuple[ModelSpec, ...]:
        return tuple(self._specs[k] for k in ("7b", "14b", "32b"))

    def manifest(self) -> dict:
        return {k: {**spec.__dict__, "predictor_paths": list(spec.predictor_paths),
                    "manifest_sha256": spec.manifest_sha256,
                    "legal_tp": list(spec.legal_tp()), "legal_pp": list(spec.legal_pp()),
                    "legal_topologies": [list(x) for x in spec.legal_topologies()]}
                for k, spec in self._specs.items()}

    def write_manifest(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.manifest(), indent=2, sort_keys=True) + "\n")
        return path


def pd_configurations(spec: ModelSpec, *, available_gpus: int = 8,
                      require_memory: bool = True) -> tuple[dict, ...]:
    """Enumerate ordered P/D layouts without claiming profile qualification."""
    topologies = spec.legal_topologies(available_gpus=available_gpus, require_memory=require_memory)
    rows = []
    for tp_p, pp_p in topologies:
        for tp_d, pp_d in topologies:
            if tp_p * pp_p + tp_d * pp_d <= available_gpus:
                rows.append({"tp_p": tp_p, "pp_p": pp_p, "tp_d": tp_d, "pp_d": pp_d,
                             "gpu_count": tp_p * pp_p + tp_d * pp_d,
                             "status": "missing_profile"})
    return tuple(rows)


def registry_from_env() -> ModelRegistry:
    import os
    return ModelRegistry(os.environ.get("PDBLEND_MODELS_DIR", str(DEFAULT_MODELS_DIR)))
