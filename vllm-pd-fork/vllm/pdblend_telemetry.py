# SPDX-License-Identifier: Apache-2.0
"""Opt-in PDBlend scheduler control and telemetry for vLLM 0.9.2.

The implementation is dependency-free so it can run in the scheduler hot
path. Control and telemetry are both opt-in through environment variables.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional


_CONTROL_STATE_ATTR = "_pdblend_control_state"
_CONTROL_MODES = ("continuous", "temporal")
_UNCHANGED = object()


def _current_mode(scheduler: Any) -> str:
    scheduler_config = getattr(scheduler, "scheduler_config", None)
    enabled = bool(
        getattr(scheduler_config, "chunked_prefill_enabled", False))
    return "continuous" if enabled else "temporal"


def _control_state(scheduler: Any) -> dict[str, Any]:
    state = getattr(scheduler, _CONTROL_STATE_ATTR, None)
    if state is None:
        state = {
            "requested_generation": None,
            "requested_mode": None,
            "applied_generation": None,
            "applied_mode": _current_mode(scheduler),
            "pending_generation": None,
            "pending_mode": None,
            "control_error": None,
            "cache_signature": None,
        }
        setattr(scheduler, _CONTROL_STATE_ATTR, state)
    return state


def _cache_signature(path: str,
                     stat_result: os.stat_result) -> tuple[Any, ...]:
    return (
        path,
        getattr(stat_result, "st_mtime_ns",
                int(stat_result.st_mtime * 1_000_000_000)),
        stat_result.st_size,
        getattr(stat_result, "st_ino", None),
    )


def _read_control(path: str, state: dict[str, Any]) -> Any:
    try:
        stat_result = os.stat(path)
    except OSError as exc:
        state["control_error"] = "control file unavailable: %s" % exc
        return None

    signature = _cache_signature(path, stat_result)
    if signature == state["cache_signature"]:
        return _UNCHANGED
    state["cache_signature"] = signature

    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        state["control_error"] = "invalid control JSON: %s" % exc
        return None

    if not isinstance(payload, dict):
        state["control_error"] = "invalid control: expected JSON object"
        return None
    schema_version = payload.get("schema_version")
    if (not isinstance(schema_version, int)
            or isinstance(schema_version, bool)):
        state["control_error"] = "invalid control schema_version"
        return None
    if schema_version != 1:
        state["control_error"] = "unsupported control schema_version"
        return None

    generation = payload.get("generation")
    if not isinstance(generation, int) or isinstance(generation, bool):
        state["control_error"] = "invalid control generation"
        return None

    mode = payload.get("mode")
    if mode not in _CONTROL_MODES:
        state["control_error"] = "invalid control mode"
        return None
    return generation, mode


def _has_running_prefill(scheduler: Any) -> bool:
    """Fail closed if a running group cannot be classified safely."""
    try:
        running = getattr(scheduler, "running", ()) or ()
        for seq_group in running:
            is_prefill = getattr(seq_group, "is_prefill", None)
            if not callable(is_prefill) or bool(is_prefill()):
                return True
    except Exception:
        return True
    return False


def _set_mode(scheduler: Any, mode: str,
              state: dict[str, Any]) -> bool:
    try:
        scheduler.scheduler_config.chunked_prefill_enabled = (
            mode == "continuous")
    except Exception as exc:
        state["control_error"] = "unable to apply control: %s" % exc
        return False
    return True


def _highest_generation(state: dict[str, Any]) -> Optional[int]:
    generations = (
        state["applied_generation"],
        state["pending_generation"],
    )
    present = [value for value in generations if value is not None]
    return max(present) if present else None


def _accept_control(scheduler: Any, state: dict[str, Any],
                    generation: int, mode: str) -> None:
    state["requested_generation"] = generation
    state["requested_mode"] = mode

    highest = _highest_generation(state)
    if highest is not None and generation < highest:
        state["control_error"] = (
            "stale control generation %d; latest is %d" %
            (generation, highest))
        return

    if generation == state["pending_generation"]:
        if mode != state["pending_mode"]:
            state["control_error"] = (
                "conflicting mode for control generation %d" % generation)
            return
        state["control_error"] = None
        return

    if generation == state["applied_generation"]:
        if mode != state["applied_mode"]:
            state["control_error"] = (
                "conflicting mode for control generation %d" % generation)
            return
        state["control_error"] = None
        return

    state["control_error"] = None
    if (mode == "temporal" and _current_mode(scheduler) == "continuous"
            and _has_running_prefill(scheduler)):
        state["pending_generation"] = generation
        state["pending_mode"] = mode
        return

    if _set_mode(scheduler, mode, state):
        state["applied_generation"] = generation
        state["applied_mode"] = mode
        state["pending_generation"] = None
        state["pending_mode"] = None
    else:
        state["pending_generation"] = generation
        state["pending_mode"] = mode


def _apply_pending_control(scheduler: Any, state: dict[str, Any]) -> None:
    generation = state["pending_generation"]
    mode = state["pending_mode"]
    if generation is None or mode is None:
        return
    if mode == "temporal" and _has_running_prefill(scheduler):
        return
    if _set_mode(scheduler, mode, state):
        state["applied_generation"] = generation
        state["applied_mode"] = mode
        state["pending_generation"] = None
        state["pending_mode"] = None


def apply_scheduler_control(scheduler: Any) -> None:
    """Apply a newer atomic control request, if runtime control is enabled."""
    path = os.environ.get("PDBLEND_CONTROL_PATH", "")
    if not path:
        return

    state = _control_state(scheduler)
    control = _read_control(path, state)
    if control is not _UNCHANGED and control is not None:
        generation, mode = control
        _accept_control(scheduler, state, generation, mode)
    _apply_pending_control(scheduler, state)


def _control_telemetry(scheduler: Any) -> dict[str, Any]:
    state = _control_state(scheduler)
    return {
        "requested_generation": state["requested_generation"],
        "applied_generation": state["applied_generation"],
        "pending_generation": state["pending_generation"],
        "requested_mode": state["requested_mode"],
        "applied_mode": state["applied_mode"],
        "pending_mode": state["pending_mode"],
        "control_error": state["control_error"],
        "chunked_prefill_enabled": bool(getattr(
            getattr(scheduler, "scheduler_config", None),
            "chunked_prefill_enabled", False)),
    }


def emit_scheduler_snapshot(scheduler: Any, scheduler_outputs: Any) -> None:
    path = os.environ.get("PDBLEND_TELEMETRY_PATH", "")
    if not path:
        return
    groups = list(
        getattr(scheduler_outputs, "scheduled_seq_groups", ()) or ())
    n_prefill = int(
        getattr(scheduler_outputs, "num_prefill_groups", 0) or 0)
    n_decode = max(len(groups) - n_prefill, 0)
    free_gpu_blocks = None
    try:
        free_gpu_blocks = int(
            scheduler.block_manager.get_num_free_gpu_blocks())
    except (AttributeError, TypeError, ValueError):
        pass
    total_gpu_blocks = getattr(
        getattr(scheduler, "cache_config", None), "num_gpu_blocks", None)
    payload = {
        "schema_version": 1,
        "ts": time.time(),
        "pid": os.getpid(),
        "instance_id": os.environ.get("PDBLEND_INSTANCE_ID", ""),
        "phase": (
            "overlap" if n_prefill and n_decode
            else "prefill" if n_prefill
            else "decode" if n_decode
            else "idle"),
        "n_prefill_groups": n_prefill,
        "n_decode_groups": n_decode,
        "n_scheduled_groups": len(groups),
        "num_batched_tokens": int(
            getattr(scheduler_outputs, "num_batched_tokens", 0) or 0),
        "running": len(getattr(scheduler, "running", ()) or ()),
        "waiting": len(getattr(scheduler, "waiting", ()) or ()),
        "swapped": len(getattr(scheduler, "swapped", ()) or ()),
        "free_gpu_blocks": free_gpu_blocks,
        "total_gpu_blocks": (
            int(total_gpu_blocks) if total_gpu_blocks is not None else None),
        "overlap": bool(n_prefill and n_decode),
    }
    payload.update(_control_telemetry(scheduler))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = "%s.tmp.%d" % (path, os.getpid())
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"), sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
