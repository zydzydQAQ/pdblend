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
    freeze: bool = False              # plan once from the offline trace statistics, never replan
    warm_start: bool = False          # start from the offline plan and hold it until the forecaster is informed
    margin: Optional[float] = None    # hysteresis override: min fractional saving to switch plans
    ported: bool = False              # decision logic lives in policies/baselines.py
    history_s: float = 0.0            # unmeasured pre-window history replay, for history-driven policies
    description: str = ""
    bootstrap_forecast: bool = False
    plan_hold_s: float = 0.0
    down_plan_votes: int = 1
    home_margin: float = 0.0          # >0: warm-start home anchor; pull back to the offline plan when feasible and this much cheaper
    min_m_instances: int = 0          # temporary empirical floor for PDblend-only candidates

    def planner_config(self, base: PlannerConfig) -> PlannerConfig:
        cfg = replace(base, fixed_mixed=self.fixed_mixed, allow_dvfs=self.allow_dvfs,
                      allow_pd=self.allow_pd, allow_park=self.allow_park)
        if self.margin is not None:
            cfg = replace(cfg, margin=self.margin)
        if self.min_m_instances:
            cfg = replace(cfg, min_m_instances=self.min_m_instances)
        return cfg


POLICIES = {
    "mixed": Policy("mixed", fixed_mixed=True, allow_dvfs=False, allow_pd=False, allow_park=(), shield=False,
                    freeze=True, description="all slots mixed at max clock"),
    "mixed_dvfs": Policy("mixed_dvfs", fixed_mixed=True, allow_pd=False, allow_park=(),
                         description="all slots mixed, periodic clock selection"),
    "mixed_dvfs_park": Policy("mixed_dvfs_park", allow_pd=False,
                              description="mixed pools with DVFS and multi-level parking (no PD)"),
    "static_best": Policy("static_best", freeze=True, shield=False,
                          description="best static configuration in the planner space from offline trace statistics"),
    "pdblend": Policy("pdblend", warm_start=True, margin=0.08, bootstrap_forecast=True,
                      plan_hold_s=30.0, down_plan_votes=2, home_margin=0.01, min_m_instances=4,
                      description="full: PD/M pools, DVFS, parking, shield"),
    "pdblend_no_park": Policy("pdblend_no_park", allow_park=(), description="ablation: no parking"),
    "pdblend_no_pd": Policy("pdblend_no_pd", allow_pd=False, description="ablation: no PD pools"),
    "pdblend_no_shield": Policy("pdblend_no_shield", shield=False, description="ablation: planner only"),
    "manual": Policy("manual", freeze=True, shield=False, description="fixed layout given on the command line"),
    "manual_shield": Policy("manual_shield", freeze=True, description="fixed layout, shield may raise clocks/wake"),
    "pdblend_fixed_pools": Policy("pdblend_fixed_pools", freeze=True, description="ablation: pools frozen from offline stats, shield on"),
    "distserve_static": Policy("distserve_static", allow_dvfs=False, allow_park=(), shield=False, freeze=True, ported=True,
                               description="DistServe: goodput-maximising static P/D split, max clock"),
    "dynamollm": Policy("dynamollm", allow_pd=False, allow_park=("off",), shield=False, ported=True, history_s=300.0,
                        description="DynamoLLM: ScaleInst 1800 s (load template = peak observed 60 s bin from an "
                                    "unmeasured history replay; fail-open until first bin) + ScaleFreq 5 s"),
    "ecoserve": Policy("ecoserve", allow_dvfs=False, allow_pd=False, allow_park=("idle",), shield=False, ported=True,
                       description="EcoServe: rotating-prefill macros, TTFT-driven instance scaling, reset-clock parking"),
}


def get_policy(name: str) -> Policy:
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name}; known: {sorted(POLICIES)}")
    return POLICIES[name]
