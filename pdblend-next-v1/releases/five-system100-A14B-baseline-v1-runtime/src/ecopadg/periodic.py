# -*- coding: utf-8 -*-
"""periodic:周期调度闭环(M4)。

每周期 T_g:
  1. LoadMonitor(或调用方)给出 λ̂ 与负载画像;
  2. GlobalScheduler 在分区空间选能耗最低可行分区(dwell/迟滞/成本三关);
  3. PoolManager.plan_transition + apply_transition 执行迁移
     (park/unpark inplace;跨角色 restart;未排空实例下周期续迁);
  4. 迁移全部落地(计数吻合)才更新 current_partition 与 last_switch。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from ecopadg.global_scheduler import Action, GlobalScheduler, GlobalState
from ecopadg.pool_manager import (
    MODE_RESTART,
    PoolManager,
    TransitionStep,
)
from ecopadg.predictor import LoadMonitor
from ecopadg.topology import TopologyCoordinator, TopologyState
from ecopadg.types import (
    ROLE_DECODE, ROLE_MIXED, ROLE_PREFILL, Partition, SystemConfig,
)


@dataclass
class CoordDecision:
    """一次周期决策的产出(遥测用)。"""
    lam_hat: float
    target: Partition
    migrated: bool
    action_reasons: List[str] = field(default_factory=list)
    planned_steps: List[TransitionStep] = field(default_factory=list)
    executed_steps: List[TransitionStep] = field(default_factory=list)
    topology_state: str = TopologyState.STEADY.value


class PeriodicCoordinator:
    """周期调度器闭环:观测 → 决策 → 迁移 → 状态推进。"""

    def __init__(self, pool_manager: PoolManager, scheduler: GlobalScheduler,
                 config: SystemConfig, initial_partition: Partition,
                 monitor: Optional[LoadMonitor] = None, clock=time.time,
                 topology_coordinator: Optional[TopologyCoordinator] = None):
        self.pm = pool_manager
        self.scheduler = scheduler
        self.cfg = config
        self.monitor = monitor
        self.current_partition = initial_partition
        self.last_switch: float = -1e9
        self._clock = clock
        self._pending_target: Optional[Partition] = None
        self._pending_requires_health = False
        self.topology = topology_coordinator

    # ------------------------------------------------------------------
    def on_arrival(self, t: float, prompt_len: int, output_len: int) -> None:
        if self.monitor is not None:
            self.monitor.on_arrival(t, prompt_len, output_len)

    def _counts_match(self, target: Partition) -> bool:
        c = self.pm._counts()
        return (c[ROLE_MIXED] == target.n_mixed
                and c[ROLE_PREFILL] == target.n_prefill
                and c[ROLE_DECODE] == target.n_decode)

    def _target_healthy(self, target: Partition) -> bool:
        if not self._counts_match(target):
            return False
        return all(
            instance.state in ("ready", "parked")
            for instance in self.pm._live()
        )

    # ------------------------------------------------------------------
    def step(self, now: Optional[float] = None,
             lam_hat: Optional[float] = None) -> CoordDecision:
        now = float(now if now is not None else self._clock())
        if lam_hat is None:
            lam_hat = self.monitor.rate(now) if self.monitor else 0.0

        # Physical restart has its own slow state machine.  No GlobalScheduler
        # or inplace action may race it, and current_partition changes only
        # after TopologyCoordinator reports a validated commit.
        if (
            self.topology is not None
            and self.topology.state != TopologyState.STEADY
        ):
            topology_decision = self.topology.step(now)
            status = self.topology.status()
            if topology_decision.committed or topology_decision.rolled_back:
                self.current_partition = status.current_partition
                self.last_switch = status.last_switch
            target = (
                topology_decision.target
                or status.target_partition
                or status.current_partition
            )
            return CoordDecision(
                lam_hat=lam_hat,
                target=target,
                migrated=topology_decision.committed,
                action_reasons=[
                    "topology:%s" % topology_decision.reason
                ],
                topology_state=status.state.value,
            )

        # 上周期未完成的迁移优先续迁(幂等:plan 基于当前实况重算)
        if self._pending_target is not None:
            steps = self.pm.plan_transition(self._pending_target)
            executed = self.pm.apply_transition(steps)
            if self._pending_requires_health:
                self.pm.poll_health()
            complete = (
                self._target_healthy(self._pending_target)
                if self._pending_requires_health
                else self._counts_match(self._pending_target)
            )
            if complete:
                self.current_partition = self._pending_target
                self._pending_target = None
                self._pending_requires_health = False
                self.last_switch = now
                return CoordDecision(lam_hat=lam_hat,
                                     target=self.current_partition,
                                     migrated=True,
                                     action_reasons=["pending-completed"],
                                     planned_steps=steps,
                                     executed_steps=executed)
            return CoordDecision(lam_hat=lam_hat,
                                 target=self._pending_target, migrated=False,
                                 action_reasons=[
                                     "pending-health"
                                     if self._pending_requires_health
                                     and not steps
                                     else "pending-draining"
                                 ],
                                 planned_steps=steps,
                                 executed_steps=executed)

        raw = getattr(self.cfg, "baseline_att", float("nan"))
        try:
            base = float(raw)
        except (TypeError, ValueError):
            base = float("nan")
        if base != base:
            base = float(self.cfg.target_attainment)
        state = GlobalState(partition=self.current_partition,
                            lam_hat=float(lam_hat),
                            baseline_attainment=base,
                            last_switch=self.last_switch)
        action: Action = self.scheduler.step(state, now)
        if not action.migrate:
            return CoordDecision(lam_hat=lam_hat,
                                 target=self.current_partition,
                                 migrated=False,
                                 action_reasons=action.reasons)

        steps = self.pm.plan_transition(action.partition)
        if (
            self.topology is not None
            and any(step.mode == MODE_RESTART for step in steps)
        ):
            self.topology.reconcile_steady(self.current_partition)
            topology_decision = self.topology.request_transition(
                action.partition,
                projected_savings_w=float(
                    getattr(action, "projected_savings_w", float("nan"))
                ),
                now=now,
                reason="scheduler",
            )
            return CoordDecision(
                lam_hat=lam_hat,
                target=action.partition,
                migrated=False,
                action_reasons=(
                    list(action.reasons)
                    + ["topology:%s" % topology_decision.reason]
                ),
                planned_steps=steps,
                topology_state=topology_decision.state.value,
            )
        executed = self.pm.apply_transition(steps)
        requires_health = any(
            step.mode == MODE_RESTART for step in steps
        )
        if requires_health:
            self.pm.poll_health()
        complete = (
            self._target_healthy(action.partition)
            if requires_health else self._counts_match(action.partition)
        )
        if complete:
            self.current_partition = action.partition
            self.last_switch = now
            return CoordDecision(lam_hat=lam_hat, target=action.partition,
                                 migrated=True,
                                 action_reasons=action.reasons,
                                 planned_steps=steps,
                                 executed_steps=executed)
        # 部分实例等待排空:挂起目标,下周期续迁
        self._pending_target = action.partition
        self._pending_requires_health = requires_health
        return CoordDecision(lam_hat=lam_hat, target=action.partition,
                             migrated=False,
                             action_reasons=[
                                 "awaiting-health"
                                 if requires_health
                                 and self._counts_match(action.partition)
                                 else "awaiting-drain"
                             ],
                             planned_steps=steps, executed_steps=executed)
