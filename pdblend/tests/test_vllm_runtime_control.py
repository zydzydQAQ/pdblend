# -*- coding: utf-8 -*-
"""CPU-only tests for the patched vLLM runtime mode control helper."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


WORKSPACE = Path(__file__).resolve().parents[2]
FORK = WORKSPACE / "vllm-pd-fork"
HELPER_PATH = FORK / "vllm" / "pdblend_telemetry.py"
SCHEDULER_PATH = FORK / "vllm" / "core" / "scheduler.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "pdblend_test_vllm_telemetry", HELPER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def helper():
    return _load_helper()


class DummyGroup:
    def __init__(self, prefill: bool):
        self.prefill = prefill

    def is_prefill(self) -> bool:
        return self.prefill


class DummyBlockManager:
    def get_num_free_gpu_blocks(self) -> int:
        return 17


class DummyScheduler:
    def __init__(self, *, chunked: bool, running=()):
        self.scheduler_config = SimpleNamespace(
            chunked_prefill_enabled=chunked)
        self.cache_config = SimpleNamespace(num_gpu_blocks=32)
        self.block_manager = DummyBlockManager()
        self.running = list(running)
        self.waiting = []
        self.swapped = []


def _write_control(path: Path, generation: int, mode: str) -> None:
    _atomic_write(path, json.dumps({
        "schema_version": 1,
        "generation": generation,
        "mode": mode,
    }))


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _state(scheduler: DummyScheduler) -> dict:
    return scheduler._pdblend_control_state


def test_valid_temporal_control_applies_and_acks(
        helper, monkeypatch, tmp_path):
    control_path = tmp_path / "control.json"
    monkeypatch.setenv("PDBLEND_CONTROL_PATH", str(control_path))
    scheduler = DummyScheduler(chunked=True)
    _write_control(control_path, 1, "temporal")

    helper.apply_scheduler_control(scheduler)

    assert scheduler.scheduler_config.chunked_prefill_enabled is False
    assert _state(scheduler)["requested_generation"] == 1
    assert _state(scheduler)["applied_generation"] == 1
    assert _state(scheduler)["applied_mode"] == "temporal"
    assert _state(scheduler)["pending_generation"] is None
    assert _state(scheduler)["control_error"] is None


@pytest.mark.parametrize("chunked", [False, True])
def test_absent_control_path_preserves_default_mode(
        helper, monkeypatch, chunked):
    monkeypatch.delenv("PDBLEND_CONTROL_PATH", raising=False)
    scheduler = DummyScheduler(chunked=chunked)

    helper.apply_scheduler_control(scheduler)

    assert scheduler.scheduler_config.chunked_prefill_enabled is chunked
    assert not hasattr(scheduler, "_pdblend_control_state")


def test_generation_is_monotonic(helper, monkeypatch, tmp_path):
    control_path = tmp_path / "control.json"
    monkeypatch.setenv("PDBLEND_CONTROL_PATH", str(control_path))
    scheduler = DummyScheduler(chunked=False)
    _write_control(control_path, 2, "continuous")
    helper.apply_scheduler_control(scheduler)

    _write_control(control_path, 1, "temporal")
    helper.apply_scheduler_control(scheduler)

    assert scheduler.scheduler_config.chunked_prefill_enabled is True
    assert _state(scheduler)["applied_generation"] == 2
    assert _state(scheduler)["applied_mode"] == "continuous"
    assert "stale control generation" in _state(scheduler)["control_error"]


def test_invalid_json_and_mode_fail_closed(
        helper, monkeypatch, tmp_path):
    control_path = tmp_path / "control.json"
    monkeypatch.setenv("PDBLEND_CONTROL_PATH", str(control_path))
    scheduler = DummyScheduler(chunked=True)

    _atomic_write(control_path, "{not-json")
    helper.apply_scheduler_control(scheduler)
    assert scheduler.scheduler_config.chunked_prefill_enabled is True
    assert _state(scheduler)["applied_generation"] is None
    assert "invalid control JSON" in _state(scheduler)["control_error"]

    _write_control(control_path, 1, "unsupported")
    helper.apply_scheduler_control(scheduler)
    assert scheduler.scheduler_config.chunked_prefill_enabled is True
    assert _state(scheduler)["applied_generation"] is None
    assert _state(scheduler)["control_error"] == "invalid control mode"


def test_temporal_control_defers_then_acks_without_reparse(
        helper, monkeypatch, tmp_path):
    control_path = tmp_path / "control.json"
    telemetry_path = tmp_path / "telemetry.json"
    monkeypatch.setenv("PDBLEND_CONTROL_PATH", str(control_path))
    monkeypatch.setenv("PDBLEND_TELEMETRY_PATH", str(telemetry_path))
    scheduler = DummyScheduler(
        chunked=True, running=[DummyGroup(prefill=True)])
    _write_control(control_path, 7, "temporal")

    helper.apply_scheduler_control(scheduler)

    assert scheduler.scheduler_config.chunked_prefill_enabled is True
    assert _state(scheduler)["applied_generation"] is None
    assert _state(scheduler)["pending_generation"] == 7
    assert _state(scheduler)["pending_mode"] == "temporal"
    outputs = SimpleNamespace(
        scheduled_seq_groups=[], num_prefill_groups=0, num_batched_tokens=0)
    helper.emit_scheduler_snapshot(scheduler, outputs)
    pending_payload = json.loads(
        telemetry_path.read_text(encoding="utf-8"))
    assert pending_payload["applied_generation"] is None
    assert pending_payload["pending_generation"] == 7
    assert pending_payload["requested_mode"] == "temporal"
    assert pending_payload["chunked_prefill_enabled"] is True

    scheduler.running.clear()
    helper.apply_scheduler_control(scheduler)

    assert scheduler.scheduler_config.chunked_prefill_enabled is False
    assert _state(scheduler)["applied_generation"] == 7
    assert _state(scheduler)["applied_mode"] == "temporal"
    assert _state(scheduler)["pending_generation"] is None
    helper.emit_scheduler_snapshot(scheduler, outputs)
    applied_payload = json.loads(
        telemetry_path.read_text(encoding="utf-8"))
    assert applied_payload["applied_generation"] == 7
    assert applied_payload["pending_generation"] is None
    assert applied_payload["chunked_prefill_enabled"] is False


def test_continuous_control_applies_with_running_prefill(
        helper, monkeypatch, tmp_path):
    control_path = tmp_path / "control.json"
    monkeypatch.setenv("PDBLEND_CONTROL_PATH", str(control_path))
    scheduler = DummyScheduler(
        chunked=False, running=[DummyGroup(prefill=True)])
    _write_control(control_path, 8, "continuous")

    helper.apply_scheduler_control(scheduler)

    assert scheduler.scheduler_config.chunked_prefill_enabled is True
    assert _state(scheduler)["applied_generation"] == 8
    assert _state(scheduler)["applied_mode"] == "continuous"


def test_unchanged_control_uses_cached_parse(
        helper, monkeypatch, tmp_path):
    control_path = tmp_path / "control.json"
    monkeypatch.setenv("PDBLEND_CONTROL_PATH", str(control_path))
    scheduler = DummyScheduler(chunked=False)
    _write_control(control_path, 3, "continuous")
    real_load = helper.json.load
    calls = []

    def tracking_load(*args, **kwargs):
        calls.append(1)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(helper.json, "load", tracking_load)
    helper.apply_scheduler_control(scheduler)
    helper.apply_scheduler_control(scheduler)

    assert len(calls) == 1


def test_telemetry_includes_control_ack_and_existing_fields(
        helper, monkeypatch, tmp_path):
    control_path = tmp_path / "control.json"
    telemetry_path = tmp_path / "telemetry.json"
    monkeypatch.setenv("PDBLEND_CONTROL_PATH", str(control_path))
    monkeypatch.setenv("PDBLEND_TELEMETRY_PATH", str(telemetry_path))
    scheduler = DummyScheduler(chunked=False)
    _write_control(control_path, 4, "continuous")
    helper.apply_scheduler_control(scheduler)
    outputs = SimpleNamespace(
        scheduled_seq_groups=[object(), object()],
        num_prefill_groups=1,
        num_batched_tokens=12,
    )

    helper.emit_scheduler_snapshot(scheduler, outputs)
    payload = json.loads(telemetry_path.read_text(encoding="utf-8"))

    assert payload["phase"] == "overlap"
    assert payload["free_gpu_blocks"] == 17
    assert payload["requested_generation"] == 4
    assert payload["applied_generation"] == 4
    assert payload["pending_generation"] is None
    assert payload["requested_mode"] == "continuous"
    assert payload["applied_mode"] == "continuous"
    assert payload["pending_mode"] is None
    assert payload["control_error"] is None
    assert payload["chunked_prefill_enabled"] is True


def test_scheduler_applies_control_immediately_before_policy_branch():
    source = SCHEDULER_PATH.read_text(encoding="utf-8")
    method_start = source.index("    def _schedule(self)")
    method_end = source.index("\n    def ", method_start + 1)
    method = source[method_start:method_end]

    apply_at = method.index("apply_scheduler_control(self)")
    branch_at = method.index(
        "if self.scheduler_config.chunked_prefill_enabled")
    assert apply_at < branch_at
