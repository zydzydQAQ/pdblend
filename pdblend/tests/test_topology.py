"""Transactional mixed↔P/D rematerialization with an in-memory executor."""
from __future__ import annotations

from types import SimpleNamespace

from ecopadg.controller_topology import ControllerTopologyAdapter
from ecopadg.engine_supervisor import (
    EngineSupervisor,
    FakeBackend,
    SupervisorPolicy,
)
from ecopadg.global_scheduler import Action
from ecopadg.periodic import PeriodicCoordinator
from ecopadg.pool_manager import PoolManager, STATE_READY
from ecopadg.topology import TopologyCoordinator, TopologyState
from ecopadg.types import Partition, SystemConfig
from ecopadg.supervisor_client import InProcessSupervisorClient
from tests.conftest import FakeClock


MIXED = Partition(n_mixed=4, tp_mixed=2, gpus_total=8)
PD = Partition(
    n_prefill=2,
    n_decode=2,
    tp_prefill=2,
    tp_decode=2,
    gpus_total=8,
)


class FakeExecutor:
    def __init__(self):
        self.handles = {}
        self.started = []
        self.stopped = []
        self.healthy = True
        self.canary_ok = True
        self.fail_role_once = None
        self._counter = 0

    def start(self, name, spec, port, kv_pair=None):
        if self.fail_role_once == spec.role:
            self.fail_role_once = None
            raise RuntimeError("injected start failure")
        self._counter += 1
        handle = "h%d" % self._counter
        self.handles[name] = handle
        self.started.append((name, spec.role, port, kv_pair))
        return handle

    def stop(self, name, handle):
        self.stopped.append((name, handle))
        self.handles.pop(name, None)

    def is_healthy(self, name, handle, port):
        del port
        return self.healthy and self.handles.get(name) == handle

    def canary(self, name, handle, port):
        return self.canary_ok and self.is_healthy(name, handle, port)


def _setup(*, cost=100.0, dwell=10.0, health_timeout=5.0):
    clock = FakeClock(0.0)
    executor = FakeExecutor()
    pm = PoolManager(
        model="/models/m",
        tp=2,
        gpus=range(8),
        executor=executor,
        port_base=8300,
    )
    pm.materialize(MIXED)
    pm.poll_health()
    topology = TopologyCoordinator(
        pm,
        MIXED,
        min_dwell_s=dwell,
        role_switch_cost_j=cost,
        drain_timeout_s=3.0,
        health_timeout_s=health_timeout,
        clock=clock,
    )
    return clock, executor, pm, topology


def _complete(topology, clock):
    decisions = []
    for _ in range(4):
        clock.advance(1.0)
        decisions.append(topology.step())
    return decisions


def test_full_mixed_pd_mixed_fake_cycle_preserves_names_and_ports():
    clock, executor, pm, topology = _setup()
    before = {item.name: item.port for item in pm.instances()}
    decision = topology.request(PD, projected_savings_w=20.0)
    assert decision.accepted
    decisions = _complete(topology, clock)
    assert decisions[-1].committed
    assert topology.current_partition.n_prefill == 2
    assert sorted(item.spec.role for item in pm._live()) == [
        "decode", "decode", "prefill", "prefill"
    ]
    assert {item.name: item.port for item in pm._live()} == before

    clock.advance(10.0)
    assert topology.request(
        MIXED, projected_savings_w=20.0
    ).accepted
    _complete(topology, clock)
    assert topology.state == TopologyState.STEADY
    assert topology.current_partition.n_mixed == 4
    assert all(item.spec.role == "mixed" for item in pm._live())
    assert {item.name: item.port for item in pm._live()} == before


def test_queue_busy_blocks_before_drain():
    clock, executor, pm, topology = _setup()
    first = pm.instances()[0]
    pm.acquire(first.name)
    decision = topology.request(PD, projected_savings_w=20.0)
    assert not decision.accepted
    assert decision.reason == "queue-not-empty"
    assert topology.state == TopologyState.STEADY
    assert executor.stopped == []


def test_health_timeout_rolls_back_prior_partition():
    clock, executor, pm, topology = _setup(health_timeout=2.0)
    assert topology.request(PD, projected_savings_w=20.0).accepted
    topology.step()  # DRAINING -> RESTARTING
    topology.step()  # restart -> STARTING
    executor.healthy = False
    clock.advance(2.0)
    timeout = topology.step()
    assert timeout.state == TopologyState.ROLLBACK
    executor.healthy = True
    rolled_back = topology.step()
    assert rolled_back.rolled_back
    assert topology.current_partition.n_mixed == 4
    assert all(item.state == STATE_READY for item in pm._live())


def test_partial_start_failure_rolls_back():
    clock, executor, pm, topology = _setup()
    assert topology.request(PD, projected_savings_w=20.0).accepted
    topology.step()
    executor.fail_role_once = "decode"
    failed = topology.step()
    assert failed.state == TopologyState.ROLLBACK
    recovered = topology.step()
    assert recovered.rolled_back
    assert topology.state == TopologyState.STEADY
    assert topology.current_partition.n_mixed == 4
    assert all(item.spec.role == "mixed" for item in pm._live())


def test_cost_trust_slo_and_payback_gates():
    _, _, _, unknown = _setup(cost=float("nan"))
    assert unknown.request(PD, projected_savings_w=20).reason == (
        "restart-cost-unknown"
    )

    _, _, _, payback = _setup(cost=500.0, dwell=10.0)
    assert payback.request(PD, projected_savings_w=20).reason == "payback"

    _, _, _, untrusted = _setup()
    assert untrusted.request(
        PD, projected_savings_w=20, trusted=False
    ).reason == "untrusted"

    _, _, _, slo = _setup()
    assert slo.request(
        PD, projected_savings_w=20, slo_ok=False
    ).reason == "slo-unsafe"


def test_canary_failure_rolls_back_and_never_commits_target():
    clock, executor, pm, topology = _setup()
    assert topology.request(PD, projected_savings_w=20).accepted
    topology.step()
    topology.step()
    topology.step()
    assert topology.state == TopologyState.VALIDATING
    executor.canary_ok = False
    decision = topology.step()
    assert decision.state == TopologyState.ROLLBACK
    executor.canary_ok = True
    topology.step()
    assert topology.current_partition.n_mixed == 4


def test_abort_during_drain_restores_prior_state():
    clock, executor, pm, topology = _setup()
    assert topology.request(PD, projected_savings_w=20).accepted
    decision = topology.abort()
    assert decision.rolled_back
    assert topology.state == TopologyState.STEADY
    assert executor.stopped == []


def test_periodic_delegates_restart_and_commits_only_after_validation():
    clock, executor, pm, topology = _setup()

    class FixedScheduler:
        def step(self, state, now):
            return Action(
                migrate=True,
                partition=PD,
                reasons=["ok"],
                projected_savings_w=20.0,
                restart_required=True,
            )

    coordinator = PeriodicCoordinator(
        pool_manager=pm,
        scheduler=FixedScheduler(),
        config=SystemConfig(model="m"),
        initial_partition=MIXED,
        clock=clock,
        topology_coordinator=topology,
    )
    requested = coordinator.step(lam_hat=1.0)
    assert not requested.migrated
    assert coordinator.current_partition.n_mixed == 4
    for _ in range(3):
        clock.advance(1.0)
        decision = coordinator.step(lam_hat=1.0)
        assert not decision.migrated
        assert coordinator.current_partition.n_mixed == 4
    clock.advance(1.0)
    committed = coordinator.step(lam_hat=1.0)
    assert committed.migrated
    assert coordinator.current_partition.n_prefill == 2


def test_controller_adapter_uses_fake_supervisor_for_full_role_cycle():
    image = "vllm:test"
    model = "/models/test"
    policy = SupervisorPolicy(
        [image],
        [model],
        range(8),
        list(range(8100, 8400)) + list(range(14579, 15400)),
    )
    supervisor = EngineSupervisor(policy, FakeBackend())
    for index in range(4):
        supervisor.start({
            "name": "slot-%d" % index,
            "image": image,
            "model": model,
            "role": "mixed",
            "gpus": [index * 2, index * 2 + 1],
            "port": 8300 + index,
            "tp": 2,
        })

    class Controller:
        strict_padg = False
        mcfg = {"tp": 2}
        args = SimpleNamespace(
            supervisor_url="http://unused",
            gpu_count=8,
            topology_min_dwell=0.0,
            role_switch_cost_j=0.0,
            topology_health_timeout=5.0,
        )
        cfg = SimpleNamespace(drain_timeout_s=3.0)
        partition = MIXED
        last_switch = -1e30
        segments = []
        mixed = [
            SimpleNamespace(
                url="http://localhost:%d" % (8300 + index),
                gpus=[index * 2, index * 2 + 1],
                parked=False,
                draining=False,
                inflight=0,
            )
            for index in range(4)
        ]

        def _topology_queues_empty(self):
            return True

        def _topology_slo_safe(self):
            return True

        def _topology_trusted(self):
            return True

        def _rebuild_runtime_from_pool(self, pool_manager, partition):
            self.partition = partition
            self.published_roles = sorted(
                instance.spec.role for instance in pool_manager._live()
            )

    controller = Controller()
    adapter = ControllerTopologyAdapter(
        controller, client=InProcessSupervisorClient(supervisor)
    )
    topology = adapter.topology
    assert topology.request(PD, projected_savings_w=1.0).accepted
    for _ in range(4):
        topology.step()
    assert controller.published_roles == [
        "decode", "decode", "prefill", "prefill"
    ]
    assert topology.request(MIXED, projected_savings_w=1.0).accepted
    for _ in range(4):
        topology.step()
    assert controller.published_roles == ["mixed"] * 4
    assert all(
        record["healthy"]
        for record in InProcessSupervisorClient(
            supervisor
        ).list_engines()
    )

