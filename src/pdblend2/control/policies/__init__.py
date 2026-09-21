"""Policy table: every system is the same engine/proxy stack with a different planner configuration."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from ..planner import PlannerConfig


@dataclass
class Policy:
    name: str
    fixed_mixed: bool = False
    allow_dvfs: bool = True
    allow_pd: bool = True
    allow_park: tuple = ("L1", "off")
    shield: bool = True
    freeze: bool = False              # plan once from the offline trace statistics, never adapt
    ported: bool = False              # decision logic lives in policies/baselines.py
    description: str = ""

    def planner_config(self, base: PlannerConfig) -> PlannerConfig:
        return replace(base, fixed_mixed=self.fixed_mixed, allow_dvfs=self.allow_dvfs,
                       allow_pd=self.allow_pd, allow_park=self.allow_park)


POLICIES = {
    "mixed": Policy("mixed", fixed_mixed=True, allow_dvfs=False, allow_pd=False, allow_park=(), shield=False,
                    freeze=True, description="all slots mixed at max clock"),
    "mixed_dvfs": Policy("mixed_dvfs", fixed_mixed=True, allow_pd=False, allow_park=(),
                         description="all slots mixed, periodic clock selection"),
    "mixed_dvfs_park": Policy("mixed_dvfs_park", allow_pd=False,
                              description="mixed pools with DVFS and multi-level parking (no PD)"),
    "static_best": Policy("static_best", freeze=True, shield=False,
                          description="best static configuration in the planner space from offline trace statistics"),
    "pdblend": Policy("pdblend", description="full: PD/M pools, DVFS, parking, shield"),
    "pdblend_no_park": Policy("pdblend_no_park", allow_park=(), description="ablation: no parking"),
    "pdblend_no_pd": Policy("pdblend_no_pd", allow_pd=False, description="ablation: no PD pools"),
    "pdblend_no_shield": Policy("pdblend_no_shield", shield=False, description="ablation: planner only"),
    "manual": Policy("manual", freeze=True, shield=False, description="fixed layout given on the command line"),
    "manual_shield": Policy("manual_shield", freeze=True, description="fixed layout, shield may raise clocks/wake"),
    "pdblend_fixed_pools": Policy("pdblend_fixed_pools", freeze=True, description="ablation: pools frozen from offline stats, shield on"),
    "distserve_static": Policy("distserve_static", allow_dvfs=False, allow_park=(), shield=False, freeze=True, ported=True,
                               description="DistServe: goodput-maximising static P/D split, max clock"),
    "dynamollm": Policy("dynamollm", allow_pd=False, allow_park=("off",), shield=False, ported=True,
                        description="DynamoLLM: ScaleInst 1800 s (oracle epoch peak) + ScaleFreq 5 s, spare GPUs off"),
    "ecoserve": Policy("ecoserve", allow_dvfs=False, allow_pd=False, allow_park=("idle",), shield=False, ported=True,
                       description="EcoServe: rotating-prefill macros, TTFT-driven instance scaling, reset-clock parking"),
}


def get_policy(name: str) -> Policy:
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name}; known: {sorted(POLICIES)}")
    return POLICIES[name]
