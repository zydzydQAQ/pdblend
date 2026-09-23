"""Deterministic short/medium/long development smoke request trace."""
from __future__ import annotations

from .client import Request, poisson_trace
from .gates import random_prompt


def build_smoke_trace(duration_s: float = 100.0, seed: int = 701,
                      rate_rps: float = 0.2) -> list[Request]:
    records = [{"prompt": random_prompt(n, seed + n), "output_tokens": 16}
               for n in (128, 512, 2048)]
    return poisson_trace(records, rate_rps=rate_rps, duration_s=duration_s,
                         seed=seed, source=f"smoke-seed{seed}")
