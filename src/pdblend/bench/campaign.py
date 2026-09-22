"""Three-model campaign specification and fail-closed coverage accounting.

This module only creates immutable work descriptions.  GPU execution is done by
the lease queue and the existing matrix runner; a point whose profile or
corpus is absent remains explicitly pending instead of silently using TP1/7B
data.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from ..model_registry import ModelRegistry, ModelSpec, pd_configurations

SYSTEMS = ("mixed", "distserve", "dynamollm", "ecoserve", "pdblend")
TP_MODES = ("fixed_tp", "offline_tp", "resident_hetero_tp", "slow_reshard_tp")
# The current campaign is intentionally a single workload-seed comparison.
# Keep this in one place so matrix generation, manifests and audits cannot
# silently drift back to the historical three-seed schedule.
SINGLE_SEED = 701
SEEDS = (SINGLE_SEED,)
SEED_POLICY = "single_seed_701"
DATASETS = {
    "alpaca": {"ttft_s": 1.0, "tpot_s": 0.10},
    "sharegpt": {"ttft_s": 5.0, "tpot_s": 0.15},
    "longbench": {"ttft_s": 15.0, "tpot_s": 0.20},
}
SCREENING_SCALES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
FORMAL_SCALES = (0.25, 0.5, 0.75, 1.0)
# The lease queue binds this logical host identity to the eight physical GPU
# UUIDs in every attempt manifest.  Profiles with another host label cannot be
# paired into the formal matrix.
HARDWARE_ID = "8xL20-lease"


@dataclass(frozen=True)
class CampaignPoint:
    name: str
    model: str
    system: str
    tp_mode: str
    tp: int
    pp: int
    dataset: str
    scale: float
    seed: int
    duration_s: float = 300.0
    stages: str = ""
    status: str = "pending_profile"
    single_seed: bool = True
    seed_policy: str = SEED_POLICY

    def to_matrix_point(self) -> dict:
        # The existing matrix runner consumes these keys.  The extra fields
        # remain in the spec and are copied into evidence by campaign runners.
        return {"name": self.name, "model": self.model, "system": self.system,
                "policy": "pdblend" if self.system == "pdblend" else self.system,
                "tp_mode": self.tp_mode, "tp": self.tp, "pp": self.pp,
                "dataset": self.dataset, "scale": self.scale, "seed": self.seed,
                "single_seed": self.seed == SINGLE_SEED, "seed_policy": SEED_POLICY,
                "duration": self.duration_s, "stages": self.stages,
                "status": self.status}


def corpus_path(root: Path, model: ModelSpec) -> Path:
    key = "7b" if "7B" in model.model_id else "14b" if "14B" in model.model_id else "32b"
    return root / f"2026-09-22-{key}-v1"


def required_profile_keys(model: ModelSpec, *, systems: Iterable[str] = SYSTEMS,
                          require_memory: bool = True) -> list[dict]:
    rows = []
    for system in systems:
        for tp, pp in model.legal_topologies(require_memory=require_memory):
            roles = ("mixed",) if system in ("mixed", "ecoserve") else ("prefill", "decode", "mixed")
            for role in roles:
                rows.append({"system": system, "model_id": model.model_id,
                             "engine_revision": "vllm-0.10.1.1",
                             "hardware_id": HARDWARE_ID,
                             "tp": tp, "pp": pp, "role": role,
                             "workload_shape": "alpaca/sharegpt/longbench",
                             "frequency_mhz": [900, 1200, 1500, 1800, 2100, 2520],
                             "status": "missing_profile"})
    return rows


def build_campaign(*, models_dir: Path | str = "/models", corpus_root: Path | str = "datasets/prepared",
                   formal: bool = True, include_pp: bool = True,
                   verification_receipt: Path | str | None = None,
                   systems: Iterable[str] = SYSTEMS, seeds: Iterable[int] = SEEDS) -> dict:
    # Materialise iterables once (callers commonly pass generators), then
    # fail closed if an old multi-seed schedule is accidentally supplied.
    systems = tuple(systems)
    seeds = tuple(int(seed) for seed in seeds)
    if seeds != SEEDS:
        raise ValueError(f"campaign uses {SEED_POLICY}; expected seeds {list(SEEDS)}, got {list(seeds)}")
    registry = ModelRegistry(models_dir, verification_receipt=verification_receipt) if verification_receipt else ModelRegistry(models_dir)
    corpus_root = Path(corpus_root)
    scales = FORMAL_SCALES if formal else SCREENING_SCALES
    points: list[dict] = []
    profiles: list[dict] = []
    for model in registry.all():
        corpus = corpus_path(corpus_root, model)
        corpus_status = "ready" if (corpus / "manifest.json").is_file() else "missing_corpus"
        tops = model.legal_topologies(require_memory=True)
        # The current V1 launcher can execute PP1.  PP candidates are retained
        # for DistServe qualification and explicitly marked unsupported until
        # the stage adapter is installed.
        if not include_pp:
            tops = tuple((tp, 1) for tp, _ in tops)
        for row in required_profile_keys(model, systems=systems, require_memory=True):
            row = dict(row, corpus=str(corpus), corpus_status=corpus_status)
            if row["pp"] != 1:
                row["status"] = "unsupported_engine"
            elif corpus_status != "ready":
                row["status"] = "missing_corpus"
            profiles.append(row)
        for system in systems:
            allowed_tops = tops
            if system == "dynamollm":
                allowed_tops = tuple(x for x in tops if x[1] == 1)
            if system == "ecoserve":
                # EcoServe keeps TP fixed, but fixed does not mean TP1. Use
                # the smallest memory-feasible TP for each model (7B=1,
                # 14B=1, 32B=2) and never reuse another system's profile.
                min_tp = min((x[0] for x in tops if x[1] == 1), default=None)
                allowed_tops = tuple(x for x in tops if x[1] == 1 and x[0] == min_tp)
            for tp, pp in allowed_tops:
                for mode in TP_MODES if system == "pdblend" else ("fixed_tp",):
                    for dataset in DATASETS:
                        for scale in scales:
                            for seed in seeds:
                                status = "missing_profile" if corpus_status == "ready" else corpus_status
                                if pp != 1:
                                    status = "unsupported_engine"
                                name = f"{model.model_id}-tp{tp}-pp{pp}-{system}-{mode}-{dataset}-x{scale:g}-seed{seed}"
                                points.append(asdict(CampaignPoint(name, model.model_id, system, mode, tp, pp,
                                                                    dataset, scale, seed, status=status)))
    payload = {
        "schema": 1, "campaign": "three-model-unified-vllm-0.10.1.1",
        "models": registry.manifest(), "systems": list(systems), "seeds": list(seeds),
        "single_seed": True, "seed_policy": SEED_POLICY,
        "datasets": DATASETS, "scales": list(scales), "formal": formal,
        "profiles": profiles, "points": points,
        "summary": {"models": 3, "profile_requirements": len(profiles), "points": len(points),
                    "ready_points": sum(p["status"] == "ready" for p in points),
                    "unsupported_points": sum(p["status"] == "unsupported_engine" for p in points)},
    }
    payload["campaign_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def write_campaign(path: Path | str, **kwargs) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_campaign(**kwargs)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def audit_campaign(path: Path | str) -> dict:
    payload = json.loads(Path(path).read_text())
    points = payload.get("points", [])
    reasons = []
    if payload.get("schema") != 1:
        reasons.append("schema")
    if tuple(payload.get("seeds", [])) != SEEDS:
        reasons.append(f"requires seed policy {SEED_POLICY}")
    if payload.get("single_seed") is not True or payload.get("seed_policy") != SEED_POLICY:
        reasons.append("missing single-seed marker")
    if any(p.get("status") != "ready" for p in points):
        reasons.append("incomplete profile/corpus/engine coverage")
    return {"formal_eligible": not reasons, "reasons": reasons,
            "points": len(points), "ready": sum(p.get("status") == "ready" for p in points),
            "single_seed": payload.get("single_seed") is True,
            "seed_policy": payload.get("seed_policy")}
