# -*- coding: utf-8 -*-
"""pool_manager:实例池运行时(M2)。

管理三池(mixed / prefill / decode)成员的生命周期与迁移:
  - RuntimeInstance:一个 vLLM 实例的运行时状态(端口/角色/健康/inflight);
  - P/D 实例按段(segment)成对启动(PyNccl KV producer/consumer,启动期配置);
  - 迁移两级:
      inplace(秒级):park/unpark —— drain 后停驻(不再路由,DVFS 降至驻留),
        权重保留,随时激活;
      restart(分钟级):mixed ↔ prefill/decode 角色转换必须重启(KV 连接器
        只能启动期配置),由全局调度器的迁移成本关口定价;
  - 执行经注入 executor(真实现=docker CLI,测试=Fake)。
"""
from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from ecopadg.runner import assign_gpus
from ecopadg.types import (
    ROLE_DECODE, ROLE_MIXED, ROLE_PREFILL, InstanceSpec, Partition,
)

STATE_STARTING = "starting"
STATE_READY = "ready"
STATE_DRAINING = "draining"
STATE_PARKED = "parked"
STATE_STOPPED = "stopped"
STATE_UNHEALTHY = "unhealthy"

MODE_INPLACE = "inplace"
MODE_RESTART = "restart"

ROLE_PARKED = "parked"


@dataclass
class RuntimeInstance:
    """一个 vLLM 实例的运行时视图。"""
    name: str
    spec: InstanceSpec
    port: int
    handle: Optional[object] = None
    state: str = STATE_STARTING
    inflight: int = 0
    segment: Optional[int] = None          # P/D 段编号(成对)
    kv_pair: Optional[Tuple[int, ...]] = None  # rank,size[,unique kv_port]

    def endpoint(self) -> str:
        return "http://localhost:%d" % self.port


@dataclass
class TransitionStep:
    """一步迁移:实例 × 目标角色 × 执行模式。"""
    instance: str
    from_role: str
    to_role: str
    mode: str
    gpus: Tuple[int, ...] = ()


@dataclass(frozen=True)
class InstanceSnapshot:
    """Immutable pre-transaction description of one live engine."""

    name: str
    spec: InstanceSpec
    port: int
    handle: Optional[object]
    state: str
    inflight: int
    segment: Optional[int]
    kv_pair: Optional[Tuple[int, ...]]


@dataclass(frozen=True)
class PoolSnapshot:
    """Prior physical pool state used for restart rollback."""

    instances: Tuple[InstanceSnapshot, ...]


@dataclass
class RestartTransaction:
    """PoolManager-owned physical transaction.

    ``target`` is deliberately not made visible as the current partition until
    TopologyCoordinator completes health and canary validation.
    """

    target: Partition
    steps: Tuple[TransitionStep, ...]
    snapshot: PoolSnapshot
    restarted_names: Tuple[str, ...] = ()
    executed_steps: List[TransitionStep] = field(default_factory=list)
    executed: bool = False
    rollback_started: bool = False
    error: str = ""


class PoolManager:
    """三池成员管理 + 分区 diff 迁移计划/执行。"""

    def __init__(self, model: str, tp: int, gpus: Sequence[int], executor,
                 port_base: int = 8100, max_model_len: int = 8192,
                 gpu_mem_util: float = 0.85,
                 kv_port_base: Optional[int] = None,
                 kv_port_stride: int = 100):
        self.model = model
        self.tp = int(tp)
        self.gpus = tuple(int(g) for g in gpus)
        self.executor = executor
        self.port_base = int(port_base)
        self.max_model_len = int(max_model_len)
        self.gpu_mem_util = float(gpu_mem_util)
        self.kv_port_base = (
            None if kv_port_base is None else int(kv_port_base)
        )
        self.kv_port_stride = int(kv_port_stride)
        if self.kv_port_base is not None and self.kv_port_base <= 0:
            raise ValueError("kv_port_base must be positive")
        if self.kv_port_stride <= 0:
            raise ValueError("kv_port_stride must be positive")
        self._insts: Dict[str, RuntimeInstance] = {}
        self._port_iter = itertools.count(self.port_base)
        self._name_iter = itertools.count(0)
        self._lock = threading.RLock()
        self._transaction: Optional[RestartTransaction] = None

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------

    def _next_port(self) -> int:
        return next(self._port_iter)

    def _next_segment(self) -> int:
        used = {
            int(inst.segment) for inst in self._live()
            if inst.segment is not None
        }
        segment = 0
        while segment in used:
            segment += 1
        return segment

    def _kv_pair(self, rank: int, segment: int) -> Tuple[int, ...]:
        if self.kv_port_base is None:
            return (int(rank), 2)
        return (
            int(rank),
            2,
            self.kv_port_base + int(segment) * self.kv_port_stride,
        )

    def _new_name(self, role: str) -> str:
        while True:
            name = "%s-%d" % (role, next(self._name_iter))
            if name not in self._insts:
                return name

    def _start_one(self, role: str, gpu_list: Sequence[int],
                   segment: Optional[int] = None,
                   kv_pair: Optional[Tuple[int, ...]] = None,
                   name: Optional[str] = None,
                   port: Optional[int] = None,
                   max_model_len: Optional[int] = None,
                   gpu_mem_util: Optional[float] = None,
                   ) -> RuntimeInstance:
        spec = InstanceSpec(role=role, model=self.model, tp=len(gpu_list),
                            gpus=tuple(gpu_list),
                            max_model_len=int(
                                self.max_model_len if max_model_len is None
                                else max_model_len),
                            gpu_mem_util=float(
                                self.gpu_mem_util if gpu_mem_util is None
                                else gpu_mem_util))
        name = str(name) if name is not None else self._new_name(role)
        port = int(port) if port is not None else self._next_port()
        handle = self.executor.start(name, spec, port, kv_pair=kv_pair)
        inst = RuntimeInstance(name=name, spec=spec, port=port, handle=handle,
                               state=STATE_STARTING, segment=segment,
                               kv_pair=kv_pair)
        self._insts[name] = inst
        return inst

    def materialize(self, partition: Partition) -> List[RuntimeInstance]:
        """按分区启动全部实例(P/D 成段配对)。"""
        if partition.total_gpus() > len(self.gpus):
            raise ValueError("分区超出 GPU 预算: %s > %d 卡"
                             % (partition, len(self.gpus)))
        alloc = assign_gpus(partition, self.gpus)
        out: List[RuntimeInstance] = []
        for gpu_list in alloc[ROLE_MIXED]:
            out.append(self._start_one(ROLE_MIXED, gpu_list))
        # P/D 成段:第 i 个 prefill 与第 i 个 decode 配对
        for i, (pg, dg) in enumerate(zip(alloc[ROLE_PREFILL],
                                         alloc[ROLE_DECODE])):
            seg = self._next_segment()
            out.append(self._start_one(ROLE_PREFILL, pg, segment=seg,
                                       kv_pair=self._kv_pair(0, seg)))
            out.append(self._start_one(ROLE_DECODE, dg, segment=seg,
                                       kv_pair=self._kv_pair(1, seg)))
        return out

    def adopt_instance(
        self,
        name: str,
        spec: InstanceSpec,
        port: int,
        handle: Optional[object],
        *,
        state: str = STATE_READY,
        segment: Optional[int] = None,
        kv_pair: Optional[Tuple[int, ...]] = None,
    ) -> RuntimeInstance:
        """Register a supervisor-owned engine without starting it.

        This is the explicit bridge used when a controller attaches to engines
        that the host supervisor already owns.
        """
        if state not in (
            STATE_STARTING,
            STATE_READY,
            STATE_DRAINING,
            STATE_PARKED,
            STATE_UNHEALTHY,
        ):
            raise ValueError("cannot adopt instance in state %s" % state)
        with self._lock:
            if name in self._insts and self._insts[name].state != STATE_STOPPED:
                raise ValueError("instance already exists: %s" % name)
            for current in self._live():
                if current.port == int(port):
                    raise ValueError("port already registered: %s" % port)
                if set(current.spec.gpus).intersection(spec.gpus):
                    raise ValueError("GPU already registered")
            inst = RuntimeInstance(
                name=str(name),
                spec=replace(spec),
                port=int(port),
                handle=handle,
                state=state,
                segment=segment,
                kv_pair=kv_pair,
            )
            self._insts[inst.name] = inst
            return inst

    # ------------------------------------------------------------------
    # 视图 / 健康 / inflight
    # ------------------------------------------------------------------

    def instances(self) -> List[RuntimeInstance]:
        return list(self._insts.values())

    def get(self, name: str) -> RuntimeInstance:
        return self._insts[name]

    def poll_health(self) -> None:
        """STARTING → READY(executor 健康检查通过)。"""
        for inst in self._insts.values():
            if inst.state not in (STATE_STARTING, STATE_UNHEALTHY):
                continue
            try:
                healthy = self.executor.is_healthy(
                    inst.name, inst.handle, inst.port)
            except Exception:  # executor boundary: never infer readiness
                healthy = False
            if healthy:
                inst.state = STATE_READY

    def mark_unhealthy(self, name: str) -> None:
        inst = self._insts[name]
        if inst.state not in (STATE_STOPPED, STATE_PARKED):
            inst.state = STATE_UNHEALTHY

    def all_ready(self, names: Optional[Sequence[str]] = None) -> bool:
        selected = (
            [self._insts[name] for name in names]
            if names is not None else self._live()
        )
        return bool(selected) and all(
            inst.state in (STATE_READY, STATE_PARKED) for inst in selected
        )

    def queues_empty(self) -> bool:
        return all(inst.inflight == 0 for inst in self._live())

    def ready_instances(self, role: str) -> List[RuntimeInstance]:
        return [i for i in self._insts.values()
                if i.state == STATE_READY and i.spec.role == role]

    def acquire(self, name: str) -> None:
        inst = self._insts[name]
        if inst.state != STATE_READY:
            raise RuntimeError("实例 %s 状态 %s 不接受新请求"
                               % (name, inst.state))
        inst.inflight += 1

    def release(self, name: str) -> None:
        inst = self._insts[name]
        inst.inflight = max(0, inst.inflight - 1)

    # ------------------------------------------------------------------
    # drain / park
    # ------------------------------------------------------------------

    def start_drain(self, name: str) -> None:
        inst = self._insts[name]
        if inst.state in (STATE_READY, STATE_STARTING):
            inst.state = STATE_DRAINING

    def is_drained(self, name: str) -> bool:
        inst = self._insts[name]
        return inst.state == STATE_DRAINING and inst.inflight == 0

    def park(self, name: str) -> None:
        """停驻:不停实例(保权重),路由摘除;调用方负责把该实例 DVFS 降频。"""
        inst = self._insts[name]
        if not (inst.state == STATE_DRAINING and inst.inflight == 0):
            raise RuntimeError("实例 %s 未排空,不能停驻" % name)
        inst.state = STATE_PARKED

    def unpark(self, name: str) -> None:
        inst = self._insts[name]
        if inst.state == STATE_PARKED:
            inst.state = STATE_READY

    # ------------------------------------------------------------------
    # 分区 diff → 迁移
    # ------------------------------------------------------------------

    def _live(self) -> List[RuntimeInstance]:
        return [i for i in self._insts.values() if i.state != STATE_STOPPED]

    def _counts(self) -> Dict[str, int]:
        c = {ROLE_MIXED: 0, ROLE_PREFILL: 0, ROLE_DECODE: 0, ROLE_PARKED: 0}
        for i in self._live():
            if i.state == STATE_PARKED:
                c[ROLE_PARKED] += 1
            else:
                c[i.spec.role] += 1
        return c

    def snapshot(self) -> PoolSnapshot:
        """Capture the live pool before a physical role switch."""
        with self._lock:
            return PoolSnapshot(instances=tuple(
                InstanceSnapshot(
                    name=inst.name,
                    spec=replace(inst.spec),
                    port=inst.port,
                    handle=inst.handle,
                    state=inst.state,
                    inflight=inst.inflight,
                    segment=inst.segment,
                    kv_pair=inst.kv_pair,
                )
                for inst in sorted(self._live(), key=lambda item: item.name)
            ))

    def plan_transition(self, target: Partition) -> List[TransitionStep]:
        """当前状态 → 目标分区的迁移步(优先 park 空闲实例;跨角色需重启)。

        规则:
          - 各角色多余(donor)/缺口(receiver)配对;
          - donor → parked:inplace(池收缩);
          - parked → 同 TP 同角色:inplace(unpark);
          - mixed ↔ prefill/decode 或 P↔D:restart。
        """
        cur = self._counts()
        want = {ROLE_MIXED: target.n_mixed, ROLE_PREFILL: target.n_prefill,
                ROLE_DECODE: target.n_decode}
        surplus: List[RuntimeInstance] = []
        deficit: List[str] = []
        for role in (ROLE_MIXED, ROLE_PREFILL, ROLE_DECODE):
            diff = cur[role] - want[role]
            if diff > 0:
                cands = [i for i in self._live()
                         if i.spec.role == role and i.state != STATE_PARKED]
                # 优先动空闲实例(inflight 少者先)
                cands.sort(key=lambda i: (i.inflight, i.name))
                surplus.extend(cands[:diff])
            elif diff < 0:
                deficit.extend([role] * (-diff))
        # parked 实例优先补同角色缺口(unpark,inplace)
        steps: List[TransitionStep] = []
        parked = [i for i in self._live() if i.state == STATE_PARKED]
        rest_deficit: List[str] = []
        for role in deficit:
            hit = next((p for p in parked if p.spec.role == role), None)
            if hit is not None:
                parked.remove(hit)
                steps.append(TransitionStep(instance=hit.name,
                                            from_role=ROLE_PARKED,
                                            to_role=role, mode=MODE_INPLACE,
                                            gpus=hit.spec.gpus))
            else:
                rest_deficit.append(role)
        # surplus → 缺口角色(restart)或 parked(inplace)
        for inst in surplus:
            if rest_deficit:
                to_role = rest_deficit.pop(0)
                steps.append(TransitionStep(instance=inst.name,
                                            from_role=inst.spec.role,
                                            to_role=to_role,
                                            mode=MODE_RESTART,
                                            gpus=inst.spec.gpus))
            else:
                steps.append(TransitionStep(instance=inst.name,
                                            from_role=inst.spec.role,
                                            to_role=ROLE_PARKED,
                                            mode=MODE_INPLACE,
                                            gpus=inst.spec.gpus))
        # 仍有缺口且有 parked 异角色实例 → restart parked
        for role in list(rest_deficit):
            if parked:
                p = parked.pop(0)
                steps.append(TransitionStep(instance=p.name,
                                            from_role=ROLE_PARKED,
                                            to_role=role, mode=MODE_RESTART,
                                            gpus=p.spec.gpus))
                rest_deficit.remove(role)
        return steps

    def apply_transition(self, steps: Sequence[TransitionStep]
                         ) -> List[TransitionStep]:
        """执行迁移步;未排空的实例先转 DRAINING,本轮跳过(幂等,下周期重试)。

        restart 步的 P/D 目标按顺序两两成段(先 prefill 后 decode)。
        返回本轮实际完成的步。
        """
        done: List[TransitionStep] = []
        pending_seg: Dict[str, List[TransitionStep]] = {}
        for s in steps:
            inst = self._insts[s.instance]
            if s.mode == MODE_INPLACE and s.from_role == ROLE_PARKED:
                self.unpark(s.instance)
                done.append(s)
                continue
            # 需要先排空
            if inst.state != STATE_PARKED:
                if inst.state != STATE_DRAINING:
                    self.start_drain(s.instance)
                if not self.is_drained(s.instance):
                    continue
            if s.mode == MODE_INPLACE:
                if inst.state != STATE_PARKED:
                    self.park(s.instance)
                done.append(s)
                continue
            pending_seg.setdefault(s.to_role, []).append(s)
        # restart:停旧起新;P/D 成对分段
        exec_steps = pending_seg.get(ROLE_PREFILL, []) \
            + pending_seg.get(ROLE_DECODE, []) \
            + pending_seg.get(ROLE_MIXED, [])
        seg_id: Optional[int] = None
        for s in exec_steps:
            inst = self._insts[s.instance]
            self.executor.stop(inst.name, inst.handle)
            inst.state = STATE_STOPPED
            if s.to_role == ROLE_PREFILL:
                seg_id = self._next_segment()
                self._start_one(ROLE_PREFILL, inst.spec.gpus, segment=seg_id,
                                kv_pair=self._kv_pair(0, seg_id))
            elif s.to_role == ROLE_DECODE:
                sid = (
                    seg_id if seg_id is not None else self._next_segment()
                )
                self._start_one(ROLE_DECODE, inst.spec.gpus, segment=sid,
                                kv_pair=self._kv_pair(1, sid))
                seg_id = None
            else:
                self._start_one(ROLE_MIXED, inst.spec.gpus)
            done.append(s)
        return done

    # ------------------------------------------------------------------
    # transactional restart / rollback (P2)
    # ------------------------------------------------------------------

    def begin_restart_transaction(
        self,
        target: Partition,
        steps: Optional[Sequence[TransitionStep]] = None,
    ) -> RestartTransaction:
        """Reserve a single restart transaction and snapshot prior engines."""
        with self._lock:
            if self._transaction is not None:
                raise RuntimeError("another restart transaction is active")
            planned = tuple(
                self.plan_transition(target) if steps is None else steps
            )
            if not any(step.mode == MODE_RESTART for step in planned):
                raise ValueError("transactional path requires a restart step")
            tx = RestartTransaction(
                target=target,
                steps=planned,
                snapshot=self.snapshot(),
            )
            self._transaction = tx
            return tx

    def _require_transaction(
        self, tx: RestartTransaction
    ) -> RestartTransaction:
        if tx is not self._transaction:
            raise RuntimeError("restart transaction is not active")
        return tx

    def drain_restart_transaction(self, tx: RestartTransaction) -> bool:
        """Start draining all donors and report whether they are empty."""
        with self._lock:
            self._require_transaction(tx)
            drained = True
            for step in tx.steps:
                if (
                    step.mode == MODE_INPLACE
                    and step.from_role == ROLE_PARKED
                ):
                    continue
                inst = self._insts[step.instance]
                if inst.state == STATE_PARKED:
                    continue
                if inst.state != STATE_DRAINING:
                    self.start_drain(inst.name)
                if not self.is_drained(inst.name):
                    drained = False
            return drained

    @staticmethod
    def _snapshot_by_name(
        snapshot: PoolSnapshot,
    ) -> Dict[str, InstanceSnapshot]:
        return {item.name: item for item in snapshot.instances}

    def execute_restart_transaction(
        self, tx: RestartTransaction
    ) -> List[TransitionStep]:
        """Stop all donors, then start replacements under the same names/ports."""
        with self._lock:
            self._require_transaction(tx)
            if tx.executed:
                return list(tx.executed_steps)
            if not self.drain_restart_transaction(tx):
                raise RuntimeError("restart transaction is not drained")
            restart_steps = [
                step for step in tx.steps if step.mode == MODE_RESTART
            ]
            prefills = [
                step for step in restart_steps
                if step.to_role == ROLE_PREFILL
            ]
            decodes = [
                step for step in restart_steps
                if step.to_role == ROLE_DECODE
            ]
            # Runtime P/D uses 1P1D PyNccl segments.  Refuse an incomplete
            # materialization instead of fabricating a usable topology.
            if len(prefills) != len(decodes):
                raise RuntimeError(
                    "transactional P/D restart requires paired P and D steps"
                )
            snapshots = self._snapshot_by_name(tx.snapshot)
            restarted = tuple(step.instance for step in restart_steps)
            tx.restarted_names = restarted
            try:
                # Free every GPU and port before starting any replacement.
                for step in restart_steps:
                    inst = self._insts[step.instance]
                    self.executor.stop(inst.name, inst.handle)
                    inst.state = STATE_STOPPED

                placement: Dict[
                    str, Tuple[Optional[int], Optional[Tuple[int, ...]]]
                ] = {}
                used_segments = {
                    int(inst.segment) for inst in self._live()
                    if inst.segment is not None
                }
                for prefill, decode in zip(prefills, decodes):
                    segment = 0
                    while segment in used_segments:
                        segment += 1
                    used_segments.add(segment)
                    placement[prefill.instance] = (
                        segment, self._kv_pair(0, segment)
                    )
                    placement[decode.instance] = (
                        segment, self._kv_pair(1, segment)
                    )
                for step in restart_steps:
                    old = snapshots[step.instance]
                    segment, kv_pair = placement.get(
                        step.instance, (None, None)
                    )
                    self._start_one(
                        step.to_role,
                        old.spec.gpus,
                        segment=segment,
                        kv_pair=kv_pair,
                        name=old.name,
                        port=old.port,
                        max_model_len=old.spec.max_model_len,
                        gpu_mem_util=old.spec.gpu_mem_util,
                    )
                for step in tx.steps:
                    if step.mode != MODE_INPLACE:
                        continue
                    inst = self._insts[step.instance]
                    if step.from_role == ROLE_PARKED:
                        self.unpark(step.instance)
                    else:
                        if inst.state != STATE_PARKED:
                            self.park(step.instance)
                    tx.executed_steps.append(step)
                tx.executed_steps.extend(restart_steps)
                tx.executed = True
                return list(tx.executed_steps)
            except Exception as exc:
                tx.error = "restart:%s" % type(exc).__name__
                raise

    def restart_transaction_healthy(
        self, tx: RestartTransaction
    ) -> bool:
        """Poll replacement health without committing the target partition."""
        with self._lock:
            self._require_transaction(tx)
            if not tx.executed:
                return False
            self.poll_health()
            if not all(
                self._insts.get(name) is not None
                and self._insts[name].state == STATE_READY
                for name in tx.restarted_names
            ):
                return False
            if not all(
                inst.state in (STATE_READY, STATE_PARKED)
                for inst in self._live()
            ):
                return False
            counts = self._counts()
            return (
                counts[ROLE_MIXED] == tx.target.n_mixed
                and counts[ROLE_PREFILL] == tx.target.n_prefill
                and counts[ROLE_DECODE] == tx.target.n_decode
            )

    def restart_transaction_canary(
        self, tx: RestartTransaction
    ) -> bool:
        """Run an executor-provided canary for each replacement if available."""
        with self._lock:
            self._require_transaction(tx)
            probe = getattr(self.executor, "canary", None)
            if probe is None:
                return self.restart_transaction_healthy(tx)
            for inst in self._live():
                if inst.state == STATE_PARKED:
                    continue
                try:
                    if not probe(inst.name, inst.handle, inst.port):
                        return False
                except Exception:
                    return False
            return True

    def commit_restart_transaction(self, tx: RestartTransaction) -> None:
        with self._lock:
            self._require_transaction(tx)
            if not self.restart_transaction_healthy(tx):
                raise RuntimeError("cannot commit unhealthy restart")
            self._transaction = None

    def cancel_restart_transaction(self, tx: RestartTransaction) -> None:
        """Cancel safely while still draining (before any engine was stopped)."""
        with self._lock:
            self._require_transaction(tx)
            if tx.executed or tx.restarted_names:
                raise RuntimeError("executed transaction must be rolled back")
            snapshots = self._snapshot_by_name(tx.snapshot)
            for name, old in snapshots.items():
                current = self._insts.get(name)
                if current is not None and current.state != STATE_STOPPED:
                    current.state = old.state
                    current.inflight = old.inflight
            self._transaction = None

    def rollback_restart_transaction(
        self, tx: RestartTransaction
    ) -> bool:
        """Restore prior specs under stable names; return True once healthy."""
        with self._lock:
            self._require_transaction(tx)
            snapshots = self._snapshot_by_name(tx.snapshot)
            changed = set(tx.restarted_names)
            if not changed:
                # A failure can occur before ``restarted_names`` is assigned
                # only on malformed setup.  Derive the donor set defensively.
                changed = {
                    step.instance for step in tx.steps
                    if step.mode == MODE_RESTART
                }
            if not tx.rollback_started:
                try:
                    for name in sorted(changed):
                        current = self._insts.get(name)
                        if (
                            current is not None
                            and current.state != STATE_STOPPED
                        ):
                            self.executor.stop(current.name, current.handle)
                            current.state = STATE_STOPPED
                    for name in sorted(changed):
                        old = snapshots[name]
                        self._start_one(
                            old.spec.role,
                            old.spec.gpus,
                            segment=old.segment,
                            kv_pair=old.kv_pair,
                            name=old.name,
                            port=old.port,
                            max_model_len=old.spec.max_model_len,
                            gpu_mem_util=old.spec.gpu_mem_util,
                        )
                    # Pure inplace steps never destroyed an engine.
                    for name, old in snapshots.items():
                        if name in changed:
                            continue
                        current = self._insts.get(name)
                        if current is not None:
                            current.state = old.state
                            current.inflight = old.inflight
                    tx.rollback_started = True
                except Exception as exc:
                    tx.error = "rollback:%s" % type(exc).__name__
                    raise
            self.poll_health()
            for name in changed:
                current = self._insts.get(name)
                if current is None or current.state != STATE_READY:
                    return False
            for name, old in snapshots.items():
                current = self._insts.get(name)
                if current is None:
                    return False
                current.state = (
                    STATE_PARKED if old.state == STATE_PARKED
                    else STATE_READY
                )
                current.inflight = old.inflight
            self._transaction = None
            return True

    # ------------------------------------------------------------------
    # DVFS 作用域
    # ------------------------------------------------------------------

    def pool_gpus(self) -> Dict[str, List[int]]:
        """各池占用的 GPU(DVFS/功率采样口径);parked 单列。"""
        out: Dict[str, List[int]] = {ROLE_MIXED: [], ROLE_PREFILL: [],
                                     ROLE_DECODE: [], ROLE_PARKED: []}
        for i in self._live():
            key = ROLE_PARKED if i.state == STATE_PARKED else i.spec.role
            out[key].extend(i.spec.gpus)
        return out
