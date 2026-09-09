# -*- coding: utf-8 -*-
"""生产 EcoSpdController 的 P0 安全不变量（不启动 Docker/GPU）。"""
from __future__ import annotations

import asyncio
import csv
import importlib.util
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
import json
import pytest

if importlib.util.find_spec("aiohttp") is None:
    _aiohttp = types.ModuleType("aiohttp")
    _web = types.ModuleType("aiohttp.web")
    _aiohttp.web = _web
    sys.modules["aiohttp"] = _aiohttp
    sys.modules["aiohttp.web"] = _web

from ecopadg.ecospd_controller import (
    EcoSpdController,
    _capacity_untrusted_continuous_fallback,
    _normalize_baseline_att,
    build_arg_parser,
)
from ecopadg.engine_telemetry import FileTelemetryRegistry
from ecopadg.global_scheduler import Action
from ecopadg.online_scheduler import (
    MODE_DECODE_WINDOW,
    MODE_PREFILL_WINDOW,
)
from ecopadg.predictor import WorkloadStats
from ecopadg.temporal_coordinator import (
    MODE_CONTINUOUS as TEMPORAL_CONTINUOUS,
    MODE_TEMPORAL,
    GlobalTemporalCoordinator,
    TemporalConfig,
)
from ecopadg.types import Partition, SloSpec, SystemConfig


class _Mixed:
    def __init__(self, parked=False, gpu=0, mode=MODE_DECODE_WINDOW,
                 freq=1200):
        self.parked = bool(parked)
        self.draining = False
        self.inflight = 0
        self.freq = 2520
        self.gpus = [int(gpu)]
        self.sched = SimpleNamespace(
            buffer=[], active={}, mode=mode, freq=int(freq))


class _Monitor:
    def rate(self, now):
        return 1.0

    def rate_fast(self, now):
        return 1.0

    def stats(self, now, default=None):
        return default or WorkloadStats()

    def arrival_cv(self, now):
        return 0.8


class _Predictor:
    def __init__(self):
        self.stats = WorkloadStats()
        self.mixed_scale = 1.0

    def rate_mixed(self, n):
        return 10.0


def _dummy_psched():
    return SimpleNamespace(
        busy=lambda: False,
        sample=lambda: None,
        policy="fcfs",
        stats=SimpleNamespace(snapshot=lambda: {
            "dispatched": 0, "hol_skips": 0, "mean_p_tokens": 0.0,
            "p_busy_frac": 0.0,
        }),
    )


class _MustNotRunScheduler:
    def step(self, state, now):  # pragma: no cover - failure path
        raise AssertionError("untrusted capacity must not call GlobalScheduler")


class _DvfsCapture:
    def __init__(self):
        self.desired = {}

    def apply(self, desired, now):
        self.desired = dict(desired)


def _controller_stub(*, capacity_trusted=False, baseline=0.98):
    c = object.__new__(EcoSpdController)
    c.mixed = [_Mixed(False), _Mixed(False), _Mixed(True), _Mixed(True)]
    c.segments = []
    c.mcfg = {"tp": 2}
    c.args = SimpleNamespace(gpu_count=8, model_key="14b", slo_tpot=0.1)
    c.slo = SloSpec(ttft_s=5.0, tpot_s=0.1)
    c.freq_trusted = True
    c.plan_freq_decode = 0
    c._slo_outcomes = []
    c._overload_rejects = 0
    c._ttft_samples = []
    c._tpot_samples = []
    c.partition = Partition(n_mixed=2, tp_mixed=2, gpus_total=8)
    c.monitor = _Monitor()
    c.predictor = _Predictor()
    c.gsched = _MustNotRunScheduler()
    c.psched = _dummy_psched()
    c._pd_sched_kwargs = dict(policy="fcfs")
    c.capacity_trusted = bool(capacity_trusted)
    c._capacity_untrusted_continuous = not c.capacity_trusted
    c.baseline_att = float(baseline)
    c._baseline_trusted = baseline == baseline and 0.0 <= baseline <= 1.0
    c._cliff_lock = False
    c.strict_padg = False
    c.engine_telemetry = FileTelemetryRegistry("", [])
    c._engine_phase_last = {}
    c._telemetry_last = c.engine_telemetry.aggregate()
    c._force_max_freq = not c._baseline_trusted
    c._force_max_until = 0.0
    c._slo_high_windows = {"ttft": 0, "tpot": 0}
    c._slo_trip_reason = ""
    c.slo_trip_hold_s = 5.0
    c.last_switch = -1e9
    c.cfg = SimpleNamespace(
        force_continuous=c._capacity_untrusted_continuous,
        global_period_s=30.0,
        freq_candidates=(2520, 1500, 900),
    )
    c._control_plane = not c.cfg.force_continuous
    c._ring_min = 2
    c._prefill_role = 0
    c._prefill_pin_mask = [0 for _ in c.mixed]
    c._prefill_pin_reasons = ["not-evaluated" for _ in c.mixed]
    c.no_park = False
    c.no_dvfs = False
    c.goodput_gate = False
    c.no_roll = False
    c.joint_temporal = False
    c.temporal_coordinator = None
    c.sched_rows = []
    c.temporal_rows = []
    return c


def _prepare_dvfs(c, mixed):
    c.mixed = list(mixed)
    c.segments = []
    c.dvfs = _DvfsCapture()
    c.psched = SimpleNamespace(busy=lambda: False)
    c._control_plane = False
    c._cliff_lock = False
    c._force_max_freq = False
    c._force_max_until = 0.0
    c._capacity_untrusted_continuous = False
    c.no_dvfs = False
    c._emergency_unpark = lambda: None
    return c


def test_normalize_baseline_accepts_only_measured_range():
    assert _normalize_baseline_att("0.975") == (0.975, True)
    assert _normalize_baseline_att(-0.1)[1] is False
    assert _normalize_baseline_att(1.1)[1] is False
    assert _normalize_baseline_att("nan")[1] is False
    assert _normalize_baseline_att(None)[1] is False


def test_capacity_untrusted_skips_global_scheduler_and_shrink():
    c = _controller_stub(capacity_trusted=False, baseline=0.98)
    action = c._periodic_step(now=100.0)
    assert action.migrate is False
    assert action.reasons == ["capacity-untrusted-continuous-mixed"]
    assert len(c.active_mixed()) == 4
    assert c.partition.n_mixed == 4
    assert c.cfg.force_continuous is True
    assert c._control_plane is False
    assert c.sched_rows[-1]["n_mixed_active"] == 4
    assert c.sched_rows[-1]["controller"] == "rule"
    assert c.sched_rows[-1]["shadow"] == "off"
    assert c.sched_rows[-1]["capacity_trusted"] == 0
    assert c.sched_rows[-1]["force_continuous"] == 1
    assert c.sched_rows[-1]["execution_mode"] == (
        "capacity-untrusted-continuous-mixed")
    assert "mixed_modes" in c.sched_rows[-1]
    assert "prefill_pin_mask" in c.sched_rows[-1]
    assert "prefill_pin_reasons" in c.sched_rows[-1]
    assert c.sched_rows[-1]["temporal_generation"] == ""
    assert c.sched_rows[-1]["temporal_mode"] == ""


def test_capacity_fallback_excludes_explicit_and_learned_modes():
    common = dict(capacity_trusted=False, controller_mode="rule",
                  strict_padg=False, explicit_force_continuous=False,
                  no_roll=False)
    assert _capacity_untrusted_continuous_fallback(**common) is True
    for override in (
            {"strict_padg": True},
            {"explicit_force_continuous": True},
            {"no_roll": True},
            {"controller_mode": "mpc"}):
        case = dict(common)
        case.update(override)
        assert _capacity_untrusted_continuous_fallback(**case) is False


def test_untrusted_rule_init_mutates_shared_cfg_and_pins_full_space(
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
                        lambda model, freqs: False)
    monkeypatch.setattr(controller_module, "capacity_model_trustworthy",
                        lambda *args, **kwargs: False)
    endpoints = ";".join(
        "http://127.0.0.1:%d@%d,%d" % (8100 + index, 2 * index,
                                       2 * index + 1)
        for index in range(4))
    args = build_arg_parser().parse_args([
        "--out", str(tmp_path / "out"),
        "--mixed", endpoints,
        "--gpu-count", "8",
        "--baseline-att", "0.98",
    ])
    controller = EcoSpdController(
        args, {"tables": "unused", "tp": 2, "capacity_req_s": 10.0})

    assert controller._capacity_untrusted_continuous is True
    assert controller.cfg.force_continuous is True
    assert controller._control_plane is False
    assert controller._ring_min == 4
    assert len(controller.active_mixed()) == 4
    assert all(mixed.sched.cfg is controller.cfg
               for mixed in controller.mixed)
    assert {
        (part.n_mixed, part.n_prefill, part.n_decode)
        for part in controller.gsched.space
    } == {(4, 0, 0)}


def test_joint_init_keeps_full_ring_and_local_admission_continuous(
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
    endpoints = ";".join(
        "http://127.0.0.1:%d@%d,%d" % (
            8100 + index, 2 * index, 2 * index + 1)
        for index in range(4))
    args = build_arg_parser().parse_args([
        "--out", str(tmp_path / "out"),
        "--mixed", endpoints,
        "--gpu-count", "8",
        "--baseline-att", "0.98",
        "--joint-temporal",
    ])
    controller = EcoSpdController(
        args, {"tables": "unused", "tp": 2, "capacity_req_s": 10.0})

    assert controller.cfg.force_continuous is False
    assert controller._control_plane is False
    assert controller._ring_min == 4
    assert len(controller.active_mixed()) == 4
    assert all(mixed.sched.cfg.force_continuous
               for mixed in controller.mixed)
    assert all(mixed.sched.mode == "continuous"
               for mixed in controller.mixed)
    state = controller.temporal_coordinator.snapshot()
    assert state.mode == MODE_TEMPORAL
    assert state.active_set == (0,)


def test_missing_baseline_forces_full_ring_and_never_schedules():
    c = _controller_stub(capacity_trusted=True, baseline=float("nan"))
    action = c._periodic_step(now=100.0)
    assert action.migrate is False
    assert "baseline-missing" in action.reasons
    assert c._want_max_freq() is True
    assert c.partition.n_mixed == 4


def test_no_park_ablation_rejects_scheduler_shrink():
    class _Shrink:
        def step(self, state, now):
            return Action(
                migrate=True,
                partition=Partition(
                    n_mixed=1, tp_mixed=2, gpus_total=8),
                reasons=["test-shrink"],
            )

    c = _controller_stub(capacity_trusted=True, baseline=0.98)
    c.no_park = True
    c._ring_min = 4
    c.gsched = _Shrink()
    action = c._periodic_step(now=100.0)
    assert action.migrate is False
    assert action.reasons[0] == "no-park-ablation"
    assert len(c.active_mixed()) == 4
    assert c.partition.n_mixed == 4


def test_goodput_gate_off_does_not_pin_when_baseline_below_90():
    c = _controller_stub(capacity_trusted=True, baseline=0.84)
    assert c.goodput_gate is False
    assert c._goodput_gate_blocks_dvfs() is False
    assert c._pd_busy_floor() == 1200
    c = _prepare_dvfs(c, [_Mixed(gpu=0, mode=MODE_DECODE_WINDOW, freq=1200)])
    c.mixed[0].sched.buffer.append(object())
    c._dvfs_step(now=100.0, idle_f=600, max_f=2520)
    assert c.dvfs.desired == {0: 1200}


def test_goodput_gate_recovers_when_baseline_below_90():
    c = _controller_stub(capacity_trusted=True, baseline=0.84)
    c.goodput_gate = True
    assert c._goodput_gate_blocks_dvfs() is True
    assert c._want_max_freq() is True
    assert c._pd_busy_floor() == 1200
    c = _prepare_dvfs(c, [_Mixed(gpu=0, mode=MODE_DECODE_WINDOW, freq=1200)])
    c.mixed[0].sched.buffer.append(object())
    c._dvfs_step(now=100.0, idle_f=600, max_f=2520)
    assert c.dvfs.desired == {0: 2520}


def test_goodput_gate_recovers_when_window_att_below_90():
    c = _controller_stub(capacity_trusted=True, baseline=0.98)
    c.goodput_gate = True
    for _ in range(7):
        c._note_slo_outcome(True)
    c._note_slo_outcome(False)
    assert c._window_slo_att() == 0.875
    assert c._goodput_gate_blocks_dvfs() is True
    assert c._want_max_freq() is True
    assert c._pd_busy_floor() == 1200


def test_goodput_gate_counts_overload_as_miss_recovery():
    c = _controller_stub(capacity_trusted=True, baseline=0.98)
    c.goodput_gate = True
    for _ in range(8):
        c._note_slo_outcome(True)
    c._note_slo_outcome(False)
    c._note_slo_outcome(False)
    assert c._window_slo_att() == 0.8
    assert c._goodput_gate_blocks_dvfs() is True
    assert c._want_max_freq() is True


def test_goodput_gate_allows_slack_when_baseline_and_window_ok():
    c = _controller_stub(capacity_trusted=True, baseline=0.98)
    c.goodput_gate = True
    for _ in range(8):
        c._note_slo_outcome(True)
    assert c._window_slo_att() == 1.0
    assert c._goodput_gate_blocks_dvfs() is False
    assert c._pd_busy_floor() == 1200
    c = _prepare_dvfs(c, [_Mixed(gpu=0, mode=MODE_DECODE_WINDOW, freq=1200)])
    c.mixed[0].sched.buffer.append(object())
    c._dvfs_step(now=100.0, idle_f=600, max_f=2520)
    assert c.dvfs.desired == {0: 1200}


def test_goodput_gate_recovers_when_baseline_in_knee_band():
    """r=32: mixed att 0.905 ≥ τ but inside τ+2pp → 不钉满,走 J/token。"""
    c = _controller_stub(capacity_trusted=True, baseline=0.905)
    c.goodput_gate = True
    for _ in range(8):
        c._note_slo_outcome(True)
    assert c._window_slo_att() == 1.0
    assert c._goodput_gate_blocks_dvfs() is True
    assert c._want_max_freq() is True
    assert c._pd_busy_floor() == 1200


def test_goodput_gate_allows_slack_when_baseline_above_knee():
    """r=12: mixed att 0.975 远离膝点,窗口满则仍允许 slack。"""
    c = _controller_stub(capacity_trusted=True, baseline=0.975)
    c.goodput_gate = True
    for _ in range(8):
        c._note_slo_outcome(True)
    assert c._goodput_gate_blocks_dvfs() is False
    assert c._pd_busy_floor() == 1200


def test_goodput_gate_recovers_when_window_below_baseline_minus_1pp():
    """相对盾:窗口 att < baseline − 1pp → 不钉 2520,走 J/token。"""
    c = _controller_stub(capacity_trusted=True, baseline=0.970)
    c.goodput_gate = True
    for _ in range(19):
        c._note_slo_outcome(True)
    c._note_slo_outcome(False)
    assert c._window_slo_att() == 0.95
    assert c._goodput_gate_blocks_dvfs() is True
    assert c._want_max_freq() is True
    assert c._pd_busy_floor() == 1200


def test_goodput_gate_allows_window_within_1pp_of_baseline():
    c = _controller_stub(capacity_trusted=True, baseline=0.970)
    c.goodput_gate = True
    c._note_slo_outcome(False)
    for _ in range(31):
        c._note_slo_outcome(True)
    assert c._window_slo_att() == 31 / 32
    assert c._goodput_gate_blocks_dvfs() is False


def test_goodput_gate_short_window_recovers_before_full_window():
    """8 样本相对盾:最后 8 里 1 个失败即可禁止钉满,不必等满 64。"""
    c = _controller_stub(capacity_trusted=True, baseline=0.970)
    c.goodput_gate = True
    for _ in range(7):
        c._note_slo_outcome(True)
    c._note_slo_outcome(False)
    assert c._window_slo_att() == 0.875
    assert c._goodput_gate_blocks_dvfs() is True
    assert c._want_max_freq() is True
    assert c._pd_busy_floor() == 1200


def test_goodput_gate_att_038_recovers_not_pin():
    """饱和 att=0.38:禁止钉 2520,选频跟 scheduler J/token。"""
    c = _controller_stub(capacity_trusted=True, baseline=0.38)
    c.goodput_gate = True
    assert c._slo_risk_detected() is True
    assert c._want_max_freq() is True
    c = _prepare_dvfs(c, [_Mixed(gpu=0, mode=MODE_DECODE_WINDOW, freq=1500)])
    c.mixed[0].sched.buffer.append(object())
    c._dvfs_step(now=100.0, idle_f=600, max_f=2520)
    assert c.dvfs.desired == {0: 2520}


def test_goodput_gate_pins_when_decode_mu_saturated():
    c = _controller_stub(capacity_trusted=True, baseline=0.975)
    c.goodput_gate = True
    for _ in range(8):
        c._note_slo_outcome(True)
    c.predictor.stage_rate_decode = lambda n, freq=None: (
        2.0 if int(n) > 0 else 0.0)
    c.monitor.rate = lambda now: 2.0
    assert c._decode_mu_blocks_dvfs() is True
    assert c._goodput_gate_blocks_dvfs() is True
    assert c._want_max_freq() is True
    assert c._pd_busy_floor() == 1200


def test_no_dvfs_ablation_pins_all_started_gpus_to_fmax():
    c = _controller_stub(capacity_trusted=True, baseline=0.98)
    for index, mixed in enumerate(c.mixed):
        mixed.gpus = [index]
    c.no_dvfs = True
    c.dvfs = _DvfsCapture()
    c._control_plane = False
    c._emergency_unpark = lambda: None
    c._dvfs_step(now=100.0, idle_f=600, max_f=2520)
    assert c.dvfs.desired == {0: 2520, 1: 2520, 2: 2520, 3: 2520}


def test_one_slo_outlier_does_not_trip_or_unpark():
    c = _controller_stub(capacity_trusted=True, baseline=0.98)
    c.slo = SloSpec(ttft_s=5.0, tpot_s=0.1)
    c.psched = SimpleNamespace(prefill_freq=0)
    c._tpot_samples = []
    for index in range(7):
        c._note_tpot(0.05, now=float(index))
    c._note_tpot(0.11, now=7.0)
    c._note_tpot(0.05, now=8.0)
    assert c._slo_trip_active(now=8.0) is False
    assert c._force_max_until == 0.0
    assert len(c.active_mixed()) == 2
    assert c.partition.n_mixed == 2
    assert c.psched.prefill_freq == 0


def test_consecutive_high_p90_windows_trip_for_five_seconds():
    c = _controller_stub(capacity_trusted=True, baseline=0.98)
    c.slo = SloSpec(ttft_s=5.0, tpot_s=0.1)
    c.psched = SimpleNamespace(prefill_freq=0)
    c._tpot_samples = []
    for index in range(7):
        c._note_tpot(0.05, now=float(index))
    c._note_tpot(0.095, now=8.0)
    assert c._slo_trip_active(now=8.0) is False
    c._note_tpot(0.096, now=9.0)
    assert c._slo_trip_active(now=13.999) is True
    assert c._force_max_until == 14.0
    assert c._slo_trip_reason.startswith("tpot-p90")
    assert len(c.active_mixed()) == 4
    assert c.partition.n_mixed == 4
    assert c.psched.prefill_freq == 2520
    assert c._slo_trip_active(now=14.0) is False
    assert c._slo_trip_reason == ""


def test_buffered_decode_mode_uses_scheduler_frequency():
    c = _prepare_dvfs(
        _controller_stub(capacity_trusted=True),
        [_Mixed(gpu=0, mode=MODE_DECODE_WINDOW, freq=1200)],
    )
    c.mixed[0].sched.buffer.append(object())
    c._dvfs_step(now=100.0, idle_f=600, max_f=2520)
    assert c.dvfs.desired == {0: 1200}
    assert c._prefill_pin_mask == [0]
    assert c._prefill_pin_reasons == ["scheduler-decode_window"]


def test_v1_prefill_window_is_pinned():
    c = _prepare_dvfs(
        _controller_stub(capacity_trusted=True),
        [_Mixed(gpu=0, mode=MODE_PREFILL_WINDOW, freq=1200)],
    )
    c._dvfs_step(now=100.0, idle_f=600, max_f=2520)
    assert c.dvfs.desired == {0: 2520}
    assert c._prefill_pin_mask == [1]
    assert c._prefill_pin_reasons == ["scheduler-prefill-window"]


def test_strict_telemetry_pins_only_fresh_exact_prefill(tmp_path):
    c = _prepare_dvfs(
        _controller_stub(capacity_trusted=True),
        [_Mixed(gpu=0, freq=1200), _Mixed(gpu=1, freq=1350)],
    )
    c.strict_padg = True
    c.engine_telemetry = FileTelemetryRegistry(
        str(tmp_path), ["mixed-0", "mixed-1"], stale_after_s=2.0)
    for index, phase in enumerate(("prefill", "idle")):
        (tmp_path / ("mixed-%d.json" % index)).write_text(json.dumps({
            "instance_id": "mixed-%d" % index,
            "ts": 100.0,
            "phase": phase,
            "n_prefill_groups": int(phase == "prefill"),
            "n_decode_groups": 0,
            "n_scheduled_groups": int(phase == "prefill"),
            "overlap": False,
        }), encoding="utf-8")
    c._dvfs_step(now=100.5, idle_f=600, max_f=2520)
    assert c.dvfs.desired == {0: 2520, 1: 1350}
    assert c._prefill_pin_mask == [1, 0]
    assert c._prefill_pin_reasons == [
        "strict-fresh-prefill", "strict-fresh-idle"]


@pytest.mark.parametrize("role_state", ["missing", "stale"])
def test_strict_missing_or_stale_fails_closed_only_for_role(
        tmp_path, role_state):
    c = _prepare_dvfs(
        _controller_stub(capacity_trusted=True),
        [_Mixed(gpu=0, freq=1200), _Mixed(gpu=1, freq=1350)],
    )
    c.strict_padg = True
    c._prefill_role = 0
    c.engine_telemetry = FileTelemetryRegistry(
        str(tmp_path), ["mixed-0", "mixed-1"], stale_after_s=2.0)
    if role_state == "stale":
        (tmp_path / "mixed-0.json").write_text(json.dumps({
            "instance_id": "mixed-0", "ts": 90.0, "phase": "prefill",
            "n_scheduled_groups": 1,
        }), encoding="utf-8")
    (tmp_path / "mixed-1.json").write_text(json.dumps({
        "instance_id": "mixed-1", "ts": 90.0, "phase": "prefill",
        "n_scheduled_groups": 1,
    }), encoding="utf-8")
    c.mixed[1].sched.buffer.append(object())
    c._dvfs_step(now=100.0, idle_f=600, max_f=2520)
    assert c.dvfs.desired == {0: 2520, 1: 1350}
    assert c._prefill_pin_mask == [1, 0]
    assert c._prefill_pin_reasons[0] == (
        "strict-role-%s-fail-closed" % role_state)
    assert c._prefill_pin_reasons[1] == "strict-nonrole-stale"


def test_benchmark_plumbs_measured_baseline_into_controller():
    root = Path(__file__).resolve().parents[1]
    cell = (root / "script" / "bench" / "run_cell.sh").read_text()
    matrix = (
        root / "new-results" / "scripts" / "run_iso_load_asplos_matrix.sh"
    ).read_text()
    assert cell.count('--baseline-att "$BASELINE_ATT"') >= 3
    assert 'export BASELINE_ATT="$BASE"' in matrix
    assert "baseline_source=bsrc" in cell


def test_strict_role_rotates_on_real_prefill_to_decode_edge(tmp_path):
    c = _controller_stub(capacity_trusted=True, baseline=0.98)
    c.strict_padg = True
    c.engine_telemetry = FileTelemetryRegistry(
        str(tmp_path), ["mixed-0", "mixed-1"])
    path = tmp_path / "mixed-0.json"
    path.write_text(json.dumps({
        "instance_id": "mixed-0", "ts": 100.0, "phase": "prefill",
        "n_prefill_groups": 1, "n_decode_groups": 0,
        "n_scheduled_groups": 1, "overlap": False,
    }), encoding="utf-8")
    c._refresh_strict_phase_roles(now=100.5)
    assert c._prefill_role == 0
    path.write_text(json.dumps({
        "instance_id": "mixed-0", "ts": 101.0, "phase": "decode",
        "n_prefill_groups": 0, "n_decode_groups": 1,
        "n_scheduled_groups": 1, "overlap": False,
    }), encoding="utf-8")
    c._refresh_strict_phase_roles(now=101.5)
    assert c._prefill_role == 1


def test_l2_gate_requires_measured_role_switch_cost():
    from ecopadg.profiler import rematerialization_open
    assert rematerialization_open(True, dict(role_switch_cost_j=None)) is False
    assert rematerialization_open(True, dict(role_switch_cost_j=float("nan"))) is False
    assert rematerialization_open(True, dict(role_switch_cost_j=8000.0)) is True


def test_refresh_lookup_pref_from_table():
    controller = object.__new__(EcoSpdController)
    controller.args = SimpleNamespace(model_key="14b", dataset="sharegpt")
    controller.lookup_table = {
        "keys": {
            "medium|14b|sharegpt": dict(freq_d=1500, layout="mixed", tp=2),
        }
    }
    controller.cfg = SystemConfig(model="14b")
    controller.predictor = SimpleNamespace(
        stats=SimpleNamespace(prompt_len=512, output_len=200))
    controller.segments = []
    assert controller._refresh_lookup_pref() == 1500
    assert controller.cfg.lookup_decode_freq == 1500


def test_controller_cli_defaults_and_mpc_flags(tmp_path):
    parser = build_arg_parser()
    default = parser.parse_args(["--out", str(tmp_path)])
    assert default.controller == "rule"
    assert default.shadow == "off"
    assert default.model_bundle == ""
    assert default.enable_rematerialization is False
    assert default.topology_period == 300.0
    assert default.role_switch_cost_j != default.role_switch_cost_j
    assert default.lookup == ""
    assert default.switch_cost == ""
    assert default.no_park is False
    assert default.no_dvfs is False
    assert default.no_roll is False
    assert default.joint_temporal is False
    assert default.slo_trip_hold_s == 5.0
    selected = parser.parse_args([
        "--out", str(tmp_path),
        "--controller", "mpc",
        "--shadow", "mpc",
        "--model-bundle", "bundle.json",
        "--no-park", "--no-dvfs", "--no-roll", "--joint-temporal",
    ])
    assert selected.controller == "mpc"
    assert selected.shadow == "mpc"
    assert selected.model_bundle == "bundle.json"
    assert selected.no_park is True
    assert selected.no_dvfs is True
    assert selected.no_roll is True
    assert selected.joint_temporal is True
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--out", str(tmp_path), "--topology-period", "299",
        ])


def test_controller_returns_503_during_topology_transition(monkeypatch):
    import ecopadg.ecospd_controller as controller_module

    class _Topology:
        topology_lock = threading.RLock()
        blocks_requests = True

        def status(self):
            return SimpleNamespace(
                state=SimpleNamespace(value="RESTARTING")
            )

    monkeypatch.setattr(
        controller_module,
        "web",
        SimpleNamespace(json_response=lambda body, status=200:
                        SimpleNamespace(status=status, body=body)),
    )
    controller = object.__new__(EcoSpdController)
    controller.rematerialization_enabled = True
    controller.topology_coordinator = _Topology()
    controller._lock = threading.Lock()
    controller._topology_admissions = 0
    response = asyncio.run(controller.handle_completion(object()))
    assert response.status == 503
    assert response.body["error"] == "topology-transition"


def test_disabled_topology_never_blocks_existing_request_path():
    controller = object.__new__(EcoSpdController)
    controller.rematerialization_enabled = False
    controller.topology_coordinator = None
    assert controller._topology_blocks_requests() is False
    status = controller.topology_status_payload()
    assert status["enabled"] is False
    assert status["blocks_requests"] is False


def test_joint_strict_telemetry_does_not_rotate_old_prefill_role(tmp_path):
    c = _controller_stub(capacity_trusted=True, baseline=0.98)
    c.joint_temporal = True
    c.strict_padg = True
    c._prefill_role = 1
    c.temporal_coordinator = GlobalTemporalCoordinator(
        4,
        TemporalConfig(
            mode=MODE_TEMPORAL,
            n_prefill_active=1,
            window_s=1.5,
        ),
        active=[0, 1],
        clock=lambda: 100.0,
    )
    c.engine_telemetry = FileTelemetryRegistry(
        str(tmp_path), ["mixed-0", "mixed-1"])
    path = tmp_path / "mixed-1.json"
    path.write_text(json.dumps({
        "instance_id": "mixed-1", "ts": 100.0, "phase": "prefill",
        "n_prefill_groups": 1, "n_decode_groups": 0,
        "n_scheduled_groups": 1, "overlap": False,
    }), encoding="utf-8")
    c._refresh_strict_phase_roles(now=100.5)
    path.write_text(json.dumps({
        "instance_id": "mixed-1", "ts": 101.0, "phase": "decode",
        "n_prefill_groups": 0, "n_decode_groups": 1,
        "n_scheduled_groups": 1, "overlap": False,
    }), encoding="utf-8")
    c._refresh_strict_phase_roles(now=101.5)
    assert c._prefill_role == 1


def test_joint_sustained_slo_trip_opens_all_mixed_engines():
    c = _controller_stub(capacity_trusted=True, baseline=0.98)
    c.joint_temporal = True
    c.slo = SloSpec(ttft_s=5.0, tpot_s=0.1)
    c.psched = SimpleNamespace(prefill_freq=0)
    c._tpot_samples = []
    c.temporal_coordinator = GlobalTemporalCoordinator(
        4,
        TemporalConfig(
            mode=MODE_TEMPORAL,
            n_prefill_active=1,
            window_s=1.5,
        ),
        active=[0, 1],
        clock=lambda: 0.0,
    )
    for index in range(7):
        c._note_tpot(0.05, now=float(index))
    c._note_tpot(0.095, now=8.0)
    c._note_tpot(0.096, now=9.0)

    state = c.temporal_coordinator.snapshot()
    assert state.mode == TEMPORAL_CONTINUOUS
    assert state.active_set == (0, 1, 2, 3)
    assert state.fallback_reason.startswith("slo-trip:tpot-")
    assert c.psched.prefill_freq == 2520


def test_joint_burst_expands_eligibility_without_physical_change():
    c = _controller_stub(capacity_trusted=True, baseline=0.98)
    for mixed in c.mixed:
        mixed.parked = False
    c.joint_temporal = True
    c.temporal_coordinator = GlobalTemporalCoordinator(
        4,
        TemporalConfig(
            mode=MODE_TEMPORAL,
            n_prefill_active=1,
            window_s=1.5,
        ),
        active=[0, 1, 2, 3],
        clock=lambda: 0.0,
    )
    physical_before = [
        (mixed.parked, mixed.draining) for mixed in c.mixed]
    c._temporal_pressure(1.0, burst=True, slo_trip=False)

    assert c.temporal_coordinator.snapshot().active_set == (0, 1, 2, 3)
    assert [
        (mixed.parked, mixed.draining) for mixed in c.mixed
    ] == physical_before


def test_joint_pump_rotates_from_fresh_engine_telemetry(monkeypatch):
    import ecopadg.ecospd_controller as controller_module

    c = _controller_stub(capacity_trusted=True, baseline=0.98)
    c.joint_temporal = True
    for index, mixed in enumerate(c.mixed):
        mixed.parked = index >= 2
        mixed.sched.step = lambda now: SimpleNamespace(released=[])
    c.temporal_coordinator = GlobalTemporalCoordinator(
        4,
        TemporalConfig(
            mode=MODE_TEMPORAL,
            n_prefill_active=1,
            window_s=1.5,
        ),
        active=[0, 1],
        clock=lambda: 0.0,
    )

    class _Telemetry:
        enabled = True
        instance_ids = ("mixed-0", "mixed-1", "mixed-2", "mixed-3")

        def read_all(self, now=None):
            return {
                "mixed-0": SimpleNamespace(
                    phase="decode", stale=False, n_prefill_groups=0),
                "mixed-1": SimpleNamespace(
                    phase="idle", stale=False, n_prefill_groups=0),
            }

    c.engine_telemetry = _Telemetry()
    c._refresh_strict_phase_roles = lambda now=None: {}
    c._events = {}
    c._lock = threading.Lock()
    c._stop = threading.Event()

    async def _one_tick(_delay):
        c._stop.set()

    monkeypatch.setattr(controller_module.time, "time", lambda: 2.0)
    monkeypatch.setattr(controller_module.asyncio, "sleep", _one_tick)
    asyncio.run(c._pump())

    state = c.temporal_coordinator.snapshot()
    assert state.epoch == 1
    assert state.active_set == (1,)


def test_joint_serve_mixed_selects_once_and_keeps_kv_local(monkeypatch):
    import ecopadg.ecospd_controller as controller_module

    class _Response:
        def __init__(self):
            self.headers = {}

        async def prepare(self, request):
            return None

        async def write_eof(self):
            return None

    monkeypatch.setattr(
        controller_module, "web",
        SimpleNamespace(StreamResponse=_Response))

    c = object.__new__(EcoSpdController)
    c.joint_temporal = True
    c._control_plane = True
    c._prefill_role = 3
    c.strict_padg = False
    c.engine_telemetry = SimpleNamespace(enabled=False)
    c._events = {}
    c._lock = threading.Lock()
    selected_urls = []

    class _Scheduler:
        def __init__(self):
            self.submitted = []
            self.completed = []

        def submit(self, rid, arrival, plen, olen):
            self.submitted.append(rid)
            c._events[rid].set()

        def complete(self, rid):
            self.completed.append(rid)

        def on_token(self, rid, now):
            pass

    c.mixed = []
    for index, inflight in enumerate((5, 1, 0, 0)):
        c.mixed.append(SimpleNamespace(
            parked=False,
            draining=False,
            inflight=inflight,
            url="http://mixed-%d" % index,
            sched=_Scheduler(),
        ))
    c.temporal_coordinator = GlobalTemporalCoordinator(
        4,
        TemporalConfig(
            mode=MODE_TEMPORAL,
            n_prefill_active=2,
            window_s=1.5,
        ),
        active=[0, 1, 2, 3],
        clock=lambda: 0.0,
    )

    async def _stream(resp, url, body, row, on_token=None):
        selected_urls.append(url)

    c._stream_from = _stream
    row = {"error": ""}
    asyncio.run(c._serve_mixed(
        object(), {"prompt": "x"}, 7, 32, 8, row))

    assert selected_urls == ["http://mixed-1"]
    assert c.mixed[1].sched.submitted == [7]
    assert c.mixed[1].sched.completed == [7]
    assert all(
        not mixed.sched.submitted
        for index, mixed in enumerate(c.mixed) if index != 1)
    assert c._prefill_role == 3


def test_temporal_event_logging_and_sched_fields(tmp_path):
    c = object.__new__(EcoSpdController)
    c.args = SimpleNamespace(out=str(tmp_path))
    c.rows = []
    c.sched_rows = []
    c.temporal_rows = []
    c.joint_temporal = True
    c.segments = []
    c.psched = _dummy_psched()
    c._pd_sched_kwargs = dict(policy="fcfs")
    c.temporal_coordinator = GlobalTemporalCoordinator(
        4,
        TemporalConfig(
            mode=MODE_TEMPORAL,
            n_prefill_active=1,
            window_s=1.5,
            token_budget=4096,
            fP=2520,
            fD=1350,
        ),
        clock=lambda: 0.0,
    )
    c.temporal_coordinator.step(
        1.5,
        {0: SimpleNamespace(
            phase="decode", stale=False, n_prefill_groups=0)},
    )

    fields = c._temporal_sched_fields()
    assert fields["temporal_generation"] == 1
    assert fields["temporal_mode"] == MODE_TEMPORAL
    assert fields["temporal_n_prefill_active"] == 1
    assert fields["temporal_active_set"] == "1"
    assert fields["temporal_fP_target"] == 2520
    assert fields["temporal_fD_target"] == 1350

    c._flush()
    with (tmp_path / "temporal_windows.csv").open(
            newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["event"] for row in rows] == [
        "configuration", "rotation"]
    assert rows[-1]["active_set"] == "1"


def test_joint_actuation_periodic_loop_never_applies_competing_mpc():
    c = _controller_stub(capacity_trusted=True, baseline=0.98)
    for mixed in c.mixed:
        mixed.parked = False
    c.partition = Partition(n_mixed=4, tp_mixed=2, gpus_total=8)
    c.joint_temporal = True
    c.enable_temporal_actuation = True
    c.controller_mode = "mpc"
    c.shadow_mode = "off"
    c._controller_execution_result = SimpleNamespace(applied=True)
    c._controller_decision = None
    c._shadow_decision = None
    c.shadow_runner = None
    c.temporal_executor = None
    c.temporal_coordinator = GlobalTemporalCoordinator(
        4,
        TemporalConfig(
            mode=TEMPORAL_CONTINUOUS,
            n_prefill_active=4,
            window_s=3.0,
        ),
        clock=lambda: 100.0,
    )
    c._learned_init_error = ""
    c.freq_trusted = True
    c._learned_control_state = (
        lambda now, lam, lam_fast: SimpleNamespace())
    c.psched = SimpleNamespace(queued=lambda: 0)

    class _MustNotRun:
        model_version = "test"

        def run(self, *args, **kwargs):  # pragma: no cover - failure path
            raise AssertionError("30s loop must not apply MPC")

    c.learned_controller = _MustNotRun()

    action = c._periodic_step(now=100.0)

    assert action.migrate is False
    assert action.reasons == ["mpc-joint-fast-loop"]
