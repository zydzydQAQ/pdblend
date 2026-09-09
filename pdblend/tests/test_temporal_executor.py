# -*- coding: utf-8 -*-
"""CPU-only coverage for atomic joint temporal actuation."""
from __future__ import annotations

import json
import threading
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from ecopadg.ecospd_controller import EcoSpdController, build_arg_parser
from ecopadg.engine_telemetry import FileTelemetryRegistry
from ecopadg.learned_controller import LearnedController
from ecopadg.mpc.types import ControlAction, FastAction, SlowAction
from ecopadg.strict_padg_executor import (
    EcoSpdTemporalExecutor,
    ExecutionResult,
    NoOpStrictPaDGExecutor,
)
from ecopadg.temporal_coordinator import (
    MODE_CONTINUOUS,
    MODE_TEMPORAL,
    GlobalTemporalCoordinator,
    TemporalConfig,
)


class _Clock:
    def __init__(self, now=100.0):
        self.now = float(now)
        self.on_sleep = None

    def __call__(self):
        return self.now

    def sleep(self, delay):
        self.now += float(delay)
        if self.on_sleep is not None:
            self.on_sleep()


class _Controller:
    def __init__(self):
        self.mixed = [
            SimpleNamespace(
                parked=False,
                draining=False,
                sched=SimpleNamespace(prefill_token_budget=8192),
            )
            for _ in range(4)
        ]
        self.topology_coordinator = SimpleNamespace(
            topology_lock=threading.RLock())
        self._force_max_freq = False
        self._last_mpc_fast_switch = -1e9

    def active_mixed(self):
        return [
            index for index, mixed in enumerate(self.mixed)
            if not mixed.parked and not mixed.draining
        ]

    def _unpark_all(self):
        for mixed in self.mixed:
            mixed.parked = False
            mixed.draining = False


def _action(
        *, mode=MODE_TEMPORAL, budget=4096, fP=2200, fD=1100,
        replicas=4):
    return ControlAction(
        fast=FastAction(
            mode=mode,
            frequency_mhz=fD,
            token_budget=budget,
            rolling_offset=0,
            n_prefill_active=1 if mode == MODE_TEMPORAL else 4,
            window_s=1.5,
            prefill_freq_mhz=fP,
            decode_freq_mhz=fD,
        ),
        slow=SlowAction(active_replicas=replicas),
    )


def _write_ack(directory, control_directory, instance_ids, now, *,
               error=""):
    for instance_id in instance_ids:
        control = json.loads(
            (control_directory / ("%s.json" % instance_id)).read_text(
                encoding="utf-8"))
        payload = {
            "schema_version": 1,
            "instance_id": instance_id,
            "ts": float(now),
            "phase": "idle",
            "n_scheduled_groups": 0,
            "overlap": False,
            "requested_generation": control["generation"],
            "applied_generation": (
                None if error else control["generation"]),
            "pending_generation": (
                control["generation"] if error else None),
            "requested_mode": control["mode"],
            "applied_mode": (
                MODE_CONTINUOUS if error else control["mode"]),
            "pending_mode": control["mode"] if error else None,
            "control_error": error or None,
        }
        (directory / ("%s.json" % instance_id)).write_text(
            json.dumps(payload), encoding="utf-8")


def _executor(tmp_path, clock, *, timeout=0.3):
    telemetry_dir = tmp_path / "telemetry"
    control_dir = tmp_path / "control"
    telemetry_dir.mkdir()
    control_dir.mkdir()
    instance_ids = tuple("mixed-%d" % index for index in range(4))
    registry = FileTelemetryRegistry(
        str(telemetry_dir), instance_ids, stale_after_s=2.0)
    controller = _Controller()
    coordinator = GlobalTemporalCoordinator(
        4,
        TemporalConfig(
            mode=MODE_CONTINUOUS,
            n_prefill_active=4,
            window_s=1.5,
            token_budget=8192,
            fP=2520,
            fD=2520,
        ),
        clock=clock,
    )
    executor = EcoSpdTemporalExecutor(
        controller,
        coordinator,
        registry,
        str(control_dir),
        ack_timeout_s=timeout,
        ack_poll_s=0.1,
        clock=clock,
        sleep=clock.sleep,
    )
    return (
        executor,
        controller,
        coordinator,
        telemetry_dir,
        control_dir,
        instance_ids,
    )


def test_all_ack_publishes_joint_action_only_after_ack(tmp_path):
    clock = _Clock()
    (executor, controller, coordinator, telemetry_dir, control_dir,
     instance_ids) = _executor(tmp_path, clock)
    state_before_ack = []

    def acknowledge():
        state_before_ack.append((
            coordinator.snapshot(),
            tuple(
                mixed.sched.prefill_token_budget
                for mixed in controller.mixed
            ),
            hasattr(controller, "_temporal_fP_target"),
        ))
        _write_ack(
            telemetry_dir, control_dir, instance_ids, clock.now)

    clock.on_sleep = acknowledge
    result = executor.apply(_action())

    assert result.applied is True
    assert state_before_ack
    assert state_before_ack[0][0].mode == MODE_CONTINUOUS
    assert state_before_ack[0][1] == (8192, 8192, 8192, 8192)
    assert state_before_ack[0][2] is False
    state = coordinator.snapshot()
    assert state.mode == MODE_TEMPORAL
    assert state.n_prefill_active == 1
    assert state.token_budget == 4096
    assert state.fP == 2200
    assert state.fD == 1100
    assert all(
        mixed.sched.prefill_token_budget == 4096
        for mixed in controller.mixed
    )
    assert controller._temporal_fP_target == 2200
    assert controller._temporal_fD_target == 1100
    assert controller._engine_control_generation == 1
    for instance_id in instance_ids:
        payload = json.loads(
            (control_dir / ("%s.json" % instance_id)).read_text())
        assert payload == {
            "schema_version": 1,
            "generation": 1,
            "mode": MODE_TEMPORAL,
        }


def test_stale_present_locally_idle_engine_uses_lazy_next_schedule_ack(
        tmp_path):
    clock = _Clock(now=100.0)
    (executor, _controller, coordinator, telemetry_dir, _control_dir,
     instance_ids) = _executor(tmp_path, clock)
    for instance_id in instance_ids:
        (telemetry_dir / ("%s.json" % instance_id)).write_text(
            json.dumps({
                "schema_version": 1,
                "instance_id": instance_id,
                "ts": 90.0,
                "phase": "idle",
                "n_scheduled_groups": 0,
                "overlap": False,
                "requested_generation": 0,
                "applied_generation": 0,
                "requested_mode": MODE_CONTINUOUS,
                "applied_mode": MODE_CONTINUOUS,
                "pending_generation": None,
                "control_error": None,
            }),
            encoding="utf-8",
        )

    result = executor.apply(_action())

    assert result.applied is True
    assert result.ack_latency_s == 0.0
    assert coordinator.snapshot().mode == MODE_TEMPORAL


def test_same_acknowledged_action_is_no_change_without_generation_churn(
        tmp_path):
    clock = _Clock()
    (executor, _controller, _coordinator, telemetry_dir, control_dir,
     instance_ids) = _executor(tmp_path, clock)
    action = _action()

    def acknowledge():
        _write_ack(
            telemetry_dir, control_dir, instance_ids, clock.now)

    clock.on_sleep = acknowledge
    first = executor.apply(action)
    controls_before = {
        instance_id: (
            control_dir / ("%s.json" % instance_id)
        ).read_text(encoding="utf-8")
        for instance_id in instance_ids
    }

    second = executor.apply(action)

    assert first.applied is True
    assert first.generation == 1
    assert first.ack_latency_s == pytest.approx(0.1)
    assert second.applied is True
    assert second.reason == "no-change"
    assert second.generation == 1
    assert second.ack_latency_s == pytest.approx(0.0)
    assert second.rolled_back is False
    assert executor.generation == 1
    assert executor.last_requested_generation == 1
    assert {
        instance_id: (
            control_dir / ("%s.json" % instance_id)
        ).read_text(encoding="utf-8")
        for instance_id in instance_ids
    } == controls_before


def test_lost_ack_for_same_action_forces_new_generation_revalidation(
        tmp_path):
    clock = _Clock()
    (executor, _controller, _coordinator, telemetry_dir, control_dir,
     instance_ids) = _executor(tmp_path, clock)
    action = _action()

    def acknowledge():
        _write_ack(
            telemetry_dir, control_dir, instance_ids, clock.now)

    clock.on_sleep = acknowledge
    assert executor.apply(action).generation == 1
    (telemetry_dir / ("%s.json" % instance_ids[0])).unlink()

    result = executor.apply(action)

    assert result.applied is True
    assert result.reason.startswith("applied:generation=2")
    assert result.generation == 2
    assert executor.generation == 2
    assert executor.last_applied_generation == 2


def test_timeout_writes_newer_rollback_and_keeps_prior_publish(tmp_path):
    clock = _Clock()
    (executor, controller, coordinator, _telemetry_dir, control_dir,
     instance_ids) = _executor(tmp_path, clock, timeout=0.2)
    previous = coordinator.snapshot()

    result = executor.apply(_action())

    assert result.applied is False
    assert "ack-timeout" in result.reason
    assert "rollback-generation=2" in result.reason
    assert coordinator.snapshot() == previous
    assert controller._force_max_freq is True
    assert controller._temporal_fail_closed_pending is True
    assert all(
        mixed.sched.prefill_token_budget == 8192
        for mixed in controller.mixed
    )
    for instance_id in instance_ids:
        payload = json.loads(
            (control_dir / ("%s.json" % instance_id)).read_text())
        assert payload["generation"] == 2
        assert payload["mode"] == MODE_CONTINUOUS


def test_partial_control_write_failure_restores_local_state(
        monkeypatch, tmp_path):
    clock = _Clock()
    (executor, controller, coordinator, telemetry_dir, control_dir,
     instance_ids) = _executor(tmp_path, clock)
    previous = coordinator.snapshot()
    previous_config = coordinator.snapshot_config()
    previous_switch = controller._last_mpc_fast_switch
    original_write = executor._atomic_write_control

    def fail_one_action_write(instance_id, generation, mode):
        if generation == 1 and instance_id == instance_ids[0]:
            raise OSError("write-test")
        original_write(instance_id, generation, mode)

    def acknowledge_rollback():
        _write_ack(
            telemetry_dir, control_dir, instance_ids, clock.now)

    monkeypatch.setattr(
        executor, "_atomic_write_control", fail_one_action_write)
    clock.on_sleep = acknowledge_rollback
    result = executor.apply(_action())

    assert result.applied is False
    assert result.generation == 1
    assert result.rolled_back is True
    assert result.rollback_generation == 2
    assert "control-write-failed" in result.reason
    assert coordinator.snapshot() == previous
    assert coordinator.snapshot_config() == previous_config
    assert controller._last_mpc_fast_switch == previous_switch
    assert all(
        mixed.sched.prefill_token_budget == 8192
        for mixed in controller.mixed
    )


def test_control_failure_reports_acknowledged_rollback(tmp_path):
    clock = _Clock()
    (executor, _controller, coordinator, telemetry_dir, control_dir,
     instance_ids) = _executor(tmp_path, clock)
    previous = coordinator.snapshot()

    def reject_then_ack_rollback():
        control = json.loads(
            (control_dir / ("%s.json" % instance_ids[0])).read_text(
                encoding="utf-8"))
        _write_ack(
            telemetry_dir,
            control_dir,
            instance_ids,
            clock.now,
            error=(
                "unable to apply control: test"
                if control["generation"] == 1
                else ""
            ),
        )

    clock.on_sleep = reject_then_ack_rollback
    result = executor.apply(_action())

    assert result.applied is False
    assert result.generation == 1
    assert result.ack_latency_s == pytest.approx(0.1)
    assert result.rolled_back is True
    assert result.rollback_generation == 2
    assert "acknowledged=true" in result.reason
    assert coordinator.snapshot() == previous
    status = executor.status()
    assert status["executor_generation"] == 1
    assert status["ack_latency_s"] == pytest.approx(0.1)
    assert status["rolled_back"] is True
    assert status["rollback_generation"] == 2


def test_control_error_rolls_back_without_coordinator_publish(tmp_path):
    clock = _Clock()
    (executor, _controller, coordinator, telemetry_dir, control_dir,
     instance_ids) = _executor(tmp_path, clock)
    previous = coordinator.snapshot()

    def reject():
        _write_ack(
            telemetry_dir,
            control_dir,
            instance_ids,
            clock.now,
            error="unable to apply control: test",
        )

    clock.on_sleep = reject
    result = executor.apply(_action())

    assert result.applied is False
    assert "engine-control-error" in result.reason
    assert coordinator.snapshot() == previous
    assert executor.last_rollback_generation == 2


def test_publish_failure_restores_full_local_snapshot_and_acks_rollback(
        monkeypatch, tmp_path):
    clock = _Clock()
    (executor, controller, coordinator, telemetry_dir, control_dir,
     instance_ids) = _executor(tmp_path, clock)

    def acknowledge():
        _write_ack(
            telemetry_dir, control_dir, instance_ids, clock.now)

    clock.on_sleep = acknowledge
    assert executor.apply(_action()).applied is True
    prior_state = coordinator.snapshot()
    prior_config = coordinator.snapshot_config()
    prior_budgets = tuple(
        mixed.sched.prefill_token_budget for mixed in controller.mixed)
    prior_targets = {
        name: (hasattr(controller, name), getattr(controller, name, None))
        for name in executor._LOCAL_TARGET_ATTRIBUTES
    }
    original_configure = coordinator.configure

    def configure_then_fail(*args, **kwargs):
        original_configure(*args, **kwargs)
        raise RuntimeError("publish-test")

    monkeypatch.setattr(coordinator, "configure", configure_then_fail)
    result = executor.apply(_action(
        mode=MODE_CONTINUOUS,
        budget=2048,
        fP=2400,
        fD=1400,
    ))

    assert result.applied is False
    assert result.generation == 2
    assert result.rolled_back is True
    assert result.rollback_generation == 3
    assert "publish-failed" in result.reason
    assert coordinator.snapshot() == prior_state
    assert coordinator.snapshot_config() == prior_config
    assert tuple(
        mixed.sched.prefill_token_budget for mixed in controller.mixed
    ) == prior_budgets
    assert {
        name: (hasattr(controller, name), getattr(controller, name, None))
        for name in executor._LOCAL_TARGET_ATTRIBUTES
    } == prior_targets


def test_execution_result_serialization_and_sched_fields(tmp_path):
    action = _action()
    legacy = ExecutionResult(True, action, "legacy")
    assert legacy.generation is None
    assert legacy.ack_latency_s is None
    assert legacy.rolled_back is False
    assert legacy.rollback_generation is None
    assert set(asdict(legacy)) == {
        "applied",
        "action",
        "reason",
        "generation",
        "ack_latency_s",
        "rolled_back",
        "rollback_generation",
    }
    assert json.loads(json.dumps(legacy.to_dict()))["action"] == (
        action.to_dict())

    clock = _Clock()
    (executor, controller, coordinator, _telemetry_dir, _control_dir,
     _instance_ids) = _executor(tmp_path, clock)
    execution = ExecutionResult(
        False,
        action,
        "test-rollback",
        generation=7,
        ack_latency_s=0.125,
        rolled_back=True,
        rollback_generation=8,
    )
    controller.joint_temporal = True
    controller.temporal_coordinator = coordinator
    controller.temporal_executor = executor
    controller._controller_execution_result = execution
    controller._telemetry_last = {}
    controller._temporal_fail_closed_pending = True

    fields = EcoSpdController._temporal_sched_fields(controller)

    assert fields["executor_generation"] == 7
    assert fields["ack_latency"] == pytest.approx(0.125)
    assert fields["rolled_back"] == 1
    assert fields["reason"] == "test-rollback"


def test_generation_recovers_from_existing_engine_controls(tmp_path):
    clock = _Clock()
    (executor, controller, coordinator, telemetry_dir, control_dir,
     instance_ids) = _executor(tmp_path, clock)
    for instance_id in instance_ids:
        (control_dir / ("%s.json" % instance_id)).write_text(
            json.dumps({
                "schema_version": 1,
                "generation": 9,
                "mode": MODE_CONTINUOUS,
            }),
            encoding="utf-8",
        )
    restarted = EcoSpdTemporalExecutor(
        controller,
        coordinator,
        executor.telemetry_registry,
        str(control_dir),
        ack_timeout_s=0.2,
        ack_poll_s=0.1,
        clock=clock,
        sleep=clock.sleep,
    )

    def acknowledge():
        _write_ack(
            telemetry_dir, control_dir, instance_ids, clock.now)

    clock.on_sleep = acknowledge
    result = restarted.apply(_action(mode=MODE_CONTINUOUS))

    assert result.applied is True
    assert restarted.last_applied_generation == 10


def test_executor_rejects_nonphysical_replica_action(tmp_path):
    clock = _Clock()
    executor, *_ = _executor(tmp_path, clock)

    result = executor.apply(_action(replicas=3))

    assert result.applied is False
    assert "physical-replicas-must-be-4" in result.reason
    assert executor.generation == 0


class _Dvfs:
    def __init__(self):
        self.desired = {}

    def apply(self, desired, now):
        self.desired = dict(desired)


def _dvfs_controller(tmp_path, phases, *, stale=()):
    controller = object.__new__(EcoSpdController)
    controller.mixed = [
        SimpleNamespace(
            parked=False,
            draining=False,
            freq=2520,
            gpus=[index],
            name="",
        )
        for index in range(4)
    ]
    controller.segments = []
    controller.dvfs = _Dvfs()
    controller._prefill_pin_mask = []
    controller._prefill_pin_reasons = []
    controller.rematerialization_enabled = False
    controller.topology_coordinator = None
    controller.args = SimpleNamespace(gpu_count=4)
    controller.mcfg = {"tp": 1}
    controller.partition = SimpleNamespace()
    controller.engine_telemetry = FileTelemetryRegistry(
        str(tmp_path),
        tuple("mixed-%d" % index for index in range(4)),
        stale_after_s=2.0,
    )
    for index, phase in enumerate(phases):
        ts = 90.0 if index in stale else 100.0
        (tmp_path / ("mixed-%d.json" % index)).write_text(json.dumps({
            "instance_id": "mixed-%d" % index,
            "ts": ts,
            "phase": phase,
            "n_scheduled_groups": int(phase != "idle"),
            "overlap": phase == "overlap",
        }), encoding="utf-8")
    controller.temporal_coordinator = GlobalTemporalCoordinator(
        4,
        TemporalConfig(
            mode=MODE_TEMPORAL,
            n_prefill_active=1,
            window_s=1.5,
            token_budget=4096,
            fP=2200,
            fD=1100,
        ),
        clock=lambda: 100.0,
    )
    controller._unpark_all = lambda: None
    return controller


def test_joint_phase_dvfs_uses_fP_fD_and_idle_min(tmp_path):
    controller = _dvfs_controller(
        tmp_path, ("prefill", "decode", "idle", "decode"))

    controller._joint_temporal_dvfs_step(
        now=100.5, idle_f=600, max_f=2520, force=False)

    assert controller.dvfs.desired == {
        0: 2200,
        1: 1100,
        2: 600,
        3: 1100,
    }


def test_joint_continuous_overlap_uses_max_phase_target_not_fmax(tmp_path):
    controller = _dvfs_controller(
        tmp_path, ("overlap", "decode", "idle", "prefill"))
    controller.temporal_coordinator.configure(
        mode=MODE_CONTINUOUS,
        n_prefill_active=4,
        window_s=1.5,
        token_budget=4096,
        fP=2200,
        fD=1100,
        active=[0, 1, 2, 3],
        now=100.0,
    )

    controller._joint_temporal_dvfs_step(
        now=100.5, idle_f=600, max_f=2520, force=False)

    assert controller.dvfs.desired[0] == 2200
    assert controller._prefill_pin_reasons[0] == (
        "continuous-overlap-max-phase-target")


def test_joint_continuous_stale_locally_idle_engine_uses_idle_min(tmp_path):
    controller = _dvfs_controller(
        tmp_path, ("decode", "decode", "idle", "prefill"), stale=(0,))
    controller.temporal_coordinator.configure(
        mode=MODE_CONTINUOUS,
        n_prefill_active=4,
        window_s=1.5,
        token_budget=4096,
        fP=2200,
        fD=1100,
        active=[0, 1, 2, 3],
        now=100.0,
    )

    controller._joint_temporal_dvfs_step(
        now=100.5, idle_f=600, max_f=2520, force=False)

    assert controller.dvfs.desired[0] == 600
    assert controller._prefill_pin_reasons[0] == (
        "controller-idle-stale-min")


def test_joint_active_set_stale_phase_fails_closed_to_fP(tmp_path):
    controller = _dvfs_controller(
        tmp_path, ("prefill", "decode", "idle", "decode"), stale=(0,))

    controller._joint_temporal_dvfs_step(
        now=100.5, idle_f=600, max_f=2520, force=False)

    assert controller.dvfs.desired[0] == 2200
    assert "active-set-stale-fP-fail-closed" in (
        controller._prefill_pin_reasons[0])


def test_joint_global_override_pins_every_engine_to_fmax(tmp_path):
    controller = _dvfs_controller(
        tmp_path, ("prefill", "decode", "idle", "decode"))

    controller._joint_temporal_dvfs_step(
        now=100.5, idle_f=600, max_f=2520, force=True)

    assert controller.dvfs.desired == {
        0: 2520,
        1: 2520,
        2: 2520,
        3: 2520,
    }


def test_joint_mpc_fast_step_is_rate_limited_and_no_change_is_not_a_switch():
    controller = object.__new__(EcoSpdController)
    controller.joint_temporal = True
    controller.enable_temporal_actuation = True
    controller.controller_mode = "mpc"
    controller.temporal_decision_period_s = 3.0
    controller._last_joint_mpc_step = -1e9
    controller._last_mpc_fast_switch = -1e9
    controller._cliff_lock = False
    controller._force_max_until = 0.0
    controller._force_max_freq = False
    controller._slo_high_windows = {"ttft": 0, "tpot": 0}
    controller._slo_trip_reason = ""
    controller.mixed = [SimpleNamespace() for _ in range(4)]
    controller.engine_telemetry = SimpleNamespace(
        aggregate=lambda now=None: {
            "available": 4, "ack_complete": True})
    controller.monitor = SimpleNamespace(
        rate=lambda now: 1.0,
        rate_fast=lambda now: 1.0,
        stats=lambda now, default=None: default,
    )
    controller.predictor = SimpleNamespace(stats=SimpleNamespace())
    controller._learned_control_state = (
        lambda now, lam, lam_fast: SimpleNamespace())
    controller._unpark_all = lambda: None
    action = _action(mode=MODE_CONTINUOUS)

    class _Learned:
        def __init__(self):
            self.calls = 0
            self.applies = 0

        def decide(self, state):
            self.calls += 1
            return SimpleNamespace(chosen=action, fallback=False)

        def log_decision(self, state, decision, shadow):
            pass

        def apply(self, state, decision):
            self.applies += 1
            return ExecutionResult(
                True,
                decision.chosen,
                "test-applied" if self.applies == 1 else "no-change",
                generation=1,
            )

    controller.learned_controller = _Learned()

    assert controller._joint_mpc_step(0.0).applied is True
    assert controller._joint_mpc_step(1.0) is None
    assert controller._joint_mpc_step(2.999) is None
    assert controller._joint_mpc_step(3.0).applied is True
    assert controller.learned_controller.calls == 2
    assert controller.learned_controller.applies == 2
    assert controller._last_mpc_fast_switch == 0.0


def test_one_second_dvfs_loop_dispatches_mpc_only_each_three_seconds():
    controller = object.__new__(EcoSpdController)
    controller.enable_temporal_actuation = True
    controller.joint_temporal = True
    controller.temporal_decision_period_s = 3.0
    controller._last_joint_mpc_dispatch = -1e9
    controller._cliff_lock = False
    controller._control_plane = False
    controller.no_dvfs = True
    controller.mixed = []
    controller.segments = []
    controller.dvfs = _Dvfs()
    controller.monitor = SimpleNamespace(
        arrival_cv=lambda now: 0.0,
        rate_fast=lambda now: 0.0,
        rate=lambda now: 0.0,
    )
    controller._want_max_freq = lambda now=None: False
    controller._temporal_pressure = lambda *args, **kwargs: None
    controller._emergency_unpark = lambda: None
    calls = []
    controller._joint_mpc_step = lambda now: calls.append(now)

    for now in (0.0, 1.0, 2.0, 3.0):
        controller._dvfs_step(now, idle_f=600, max_f=2520)

    assert calls == [0.0, 3.0]


def test_default_executor_and_temporal_actuation_cli_gate(tmp_path):
    assert isinstance(
        LearnedController().executor, NoOpStrictPaDGExecutor)
    parser = build_arg_parser()
    default = parser.parse_args(["--out", str(tmp_path)])
    assert default.enable_temporal_actuation is False
    assert default.engine_control_dir == ""
    assert default.temporal_ack_timeout == 5.0
    assert default.temporal_decision_period == 3.0

    with pytest.raises(SystemExit):
        parser.parse_args([
            "--out", str(tmp_path),
            "--enable-temporal-actuation",
        ])

    mixed = ";".join(
        "http://127.0.0.1:%d@%d" % (8100 + index, index)
        for index in range(4)
    )
    selected = parser.parse_args([
        "--out", str(tmp_path),
        "--controller", "mpc",
        "--joint-temporal",
        "--enable-temporal-actuation",
        "--mixed", mixed,
        "--telemetry-dir", str(tmp_path / "telemetry"),
        "--engine-control-dir", str(tmp_path / "control"),
        "--joint-capacity-per-replica", "1.5",
    ])
    assert selected.enable_temporal_actuation is True


def test_controller_injects_executor_only_when_actuation_enabled(
        monkeypatch, tmp_path, synth_opmodel):
    import ecopadg.ecospd_controller as controller_module

    class _Backend:
        def current_freq(self, gpu):
            return 2520

        def set_clock(self, gpu, freq):
            pass

        def reset_clock(self, gpu):
            pass

    monkeypatch.setattr(controller_module, "WS_ROOT", str(tmp_path))
    monkeypatch.setattr(controller_module, "get_backend",
                        lambda name: _Backend())
    monkeypatch.setattr(controller_module, "load_opmodel",
                        lambda path: synth_opmodel)
    monkeypatch.setattr(controller_module, "freq_model_trustworthy",
                        lambda model, freqs: True)
    monkeypatch.setattr(controller_module, "capacity_model_trustworthy",
                        lambda *args, **kwargs: True)
    mixed = ";".join(
        "http://127.0.0.1:%d@%d,%d" % (
            8100 + index, 2 * index, 2 * index + 1)
        for index in range(4)
    )
    parser = build_arg_parser()
    common = [
        "--out", str(tmp_path / "out"),
        "--controller", "mpc",
        "--mixed", mixed,
        "--gpu-count", "8",
        "--baseline-att", "0.98",
    ]
    disabled = EcoSpdController(
        parser.parse_args(common),
        {"tables": "unused", "tp": 2, "capacity_req_s": 10.0},
    )
    assert disabled.temporal_executor is None
    assert isinstance(
        disabled.learned_controller.executor, NoOpStrictPaDGExecutor)

    enabled_args = parser.parse_args(common + [
        "--joint-temporal",
        "--enable-temporal-actuation",
        "--telemetry-dir", str(tmp_path / "telemetry"),
        "--engine-control-dir", str(tmp_path / "control"),
        "--joint-capacity-per-replica", "1.5",
    ])
    enabled = EcoSpdController(
        enabled_args,
        {"tables": "unused", "tp": 2, "capacity_req_s": 10.0},
    )
    assert isinstance(enabled.temporal_executor, EcoSpdTemporalExecutor)
    assert enabled.learned_controller.executor is enabled.temporal_executor
    assert enabled.temporal_coordinator.snapshot().mode == MODE_CONTINUOUS
    assert len(enabled.active_mixed()) == 4
