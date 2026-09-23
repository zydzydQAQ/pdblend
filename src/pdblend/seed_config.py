"""Workload seed policy for the active campaign (independent of sample repeats)."""
from __future__ import annotations

from collections.abc import Iterable

SINGLE_SEED = 701
SEEDS = (SINGLE_SEED,)
SEED_POLICY = "single_seed_701"


def has_active_seeds(seeds: Iterable[int]) -> bool:
    """Require exactly one result per active seed; duplicates are incomplete."""
    return tuple(seeds) == SEEDS


def seed_metadata(seeds: Iterable[int] = SEEDS) -> dict:
    """Describe the actual schedule without labelling old runs as seed 701."""
    values = tuple(seeds)
    return {"seeds": list(values), "single_seed": len(values) == 1,
            "seed_policy": SEED_POLICY if values == SEEDS else "custom"}
