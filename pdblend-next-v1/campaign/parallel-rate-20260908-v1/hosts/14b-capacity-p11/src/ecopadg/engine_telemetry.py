# -*- coding: utf-8 -*-
"""Read atomic scheduler snapshots emitted by the patched vLLM engine."""
from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class EngineTelemetry:
    instance_id: str
    ts: float
    phase: str
    n_prefill_groups: int = 0
    n_decode_groups: int = 0
    n_scheduled_groups: int = 0
    num_batched_tokens: int = 0
    running: int = 0
    waiting: int = 0
    swapped: int = 0
    free_gpu_blocks: Optional[int] = None
    total_gpu_blocks: Optional[int] = None
    overlap: bool = False
    stale: bool = False
    requested_generation: Optional[int] = None
    applied_generation: Optional[int] = None
    pending_generation: Optional[int] = None
    requested_mode: Optional[str] = None
    applied_mode: Optional[str] = None
    pending_mode: Optional[str] = None
    control_error: Optional[str] = None
    chunked_prefill_enabled: Optional[bool] = None

    @property
    def omega(self) -> float:
        return 1.0 if self.overlap else 0.0

    @property
    def fresh(self) -> bool:
        return not self.stale

    @classmethod
    def from_dict(cls, data: dict, *, now: Optional[float] = None,
                  stale_after_s: float = 2.0) -> "EngineTelemetry":
        ts = float(data.get("ts") or 0.0)
        current = time.time() if now is None else float(now)
        return cls(
            instance_id=str(data.get("instance_id") or ""),
            ts=ts,
            phase=str(data.get("phase") or "unknown"),
            n_prefill_groups=int(data.get("n_prefill_groups") or 0),
            n_decode_groups=int(data.get("n_decode_groups") or 0),
            n_scheduled_groups=int(data.get("n_scheduled_groups") or 0),
            num_batched_tokens=int(data.get("num_batched_tokens") or 0),
            running=int(data.get("running") or 0),
            waiting=int(data.get("waiting") or 0),
            swapped=int(data.get("swapped") or 0),
            free_gpu_blocks=_optional_int(data.get("free_gpu_blocks")),
            total_gpu_blocks=_optional_int(data.get("total_gpu_blocks")),
            overlap=bool(data.get("overlap")),
            stale=(
                not math.isfinite(ts)
                or current - ts > float(stale_after_s)
            ),
            requested_generation=_optional_int(
                data.get("requested_generation")),
            applied_generation=_optional_int(data.get("applied_generation")),
            pending_generation=_optional_int(data.get("pending_generation")),
            requested_mode=_optional_str(data.get("requested_mode")),
            applied_mode=_optional_str(data.get("applied_mode")),
            pending_mode=_optional_str(data.get("pending_mode")),
            control_error=_optional_str(data.get("control_error")),
            chunked_prefill_enabled=_optional_bool(
                data.get("chunked_prefill_enabled")),
        )


@dataclass(frozen=True)
class ControlAckStatus(Mapping[str, Any]):
    """Fresh, all-instance acknowledgement state for one generation."""

    expected_generation: int
    expected_mode: Optional[str]
    complete: bool
    all_fresh: bool
    has_error: bool
    pending: bool
    acked_instances: Tuple[str, ...]
    pending_instances: Tuple[str, ...]
    missing_instances: Tuple[str, ...]
    stale_instances: Tuple[str, ...]
    mismatched_instances: Tuple[str, ...]
    control_errors: Dict[str, str]
    per_instance: Dict[str, Dict[str, Any]]
    reason: str

    @property
    def acknowledged(self) -> bool:
        return self.complete

    @property
    def pending_generation(self) -> Optional[int]:
        return self.expected_generation if self.pending else None

    def to_dict(self) -> dict:
        return {
            "expected_generation": self.expected_generation,
            "expected_mode": self.expected_mode,
            "complete": self.complete,
            "acknowledged": self.complete,
            "all_fresh": self.all_fresh,
            "has_error": self.has_error,
            "pending": self.pending,
            "pending_generation": self.pending_generation,
            "acked_instances": list(self.acked_instances),
            "pending_instances": list(self.pending_instances),
            "missing_instances": list(self.missing_instances),
            "stale_instances": list(self.stale_instances),
            "mismatched_instances": list(self.mismatched_instances),
            "control_errors": dict(self.control_errors),
            "per_instance": {
                key: dict(value) for key, value in self.per_instance.items()
            },
            "reason": self.reason,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def _optional_int(value) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value) -> Optional[str]:
    return None if value is None else str(value)


def _optional_bool(value) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return None


class FileTelemetryRegistry:
    def __init__(self, directory: str, instance_ids: Iterable[str],
                 stale_after_s: float = 2.0):
        self.directory = str(directory or "")
        self.instance_ids = tuple(str(x) for x in instance_ids)
        self.stale_after_s = float(stale_after_s)

    @property
    def enabled(self) -> bool:
        return bool(self.directory)

    def path_for(self, instance_id: str) -> str:
        return os.path.join(self.directory, "%s.json" % instance_id)

    def read(self, instance_id: str,
             now: Optional[float] = None) -> Optional[EngineTelemetry]:
        if not self.enabled:
            return None
        path = self.path_for(instance_id)
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            view = EngineTelemetry.from_dict(
                data, now=now, stale_after_s=self.stale_after_s)
            if not view.instance_id:
                data["instance_id"] = instance_id
                view = EngineTelemetry.from_dict(
                    data, now=now, stale_after_s=self.stale_after_s)
        except (
            AttributeError,
            OSError,
            OverflowError,
            TypeError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return None
        return view

    def read_all(self, now: Optional[float] = None
                 ) -> Dict[str, EngineTelemetry]:
        out: Dict[str, EngineTelemetry] = {}
        for instance_id in self.instance_ids:
            view = self.read(instance_id, now=now)
            if view is not None:
                out[instance_id] = view
        return out

    @staticmethod
    def _view_control_status(
            view: Optional[EngineTelemetry]) -> Dict[str, Any]:
        if view is None:
            return {
                "fresh": False,
                "requested_generation": None,
                "applied_generation": None,
                "pending_generation": None,
                "requested_mode": None,
                "applied_mode": None,
                "pending_mode": None,
                "control_error": None,
                "chunked_prefill_enabled": None,
            }
        return {
            "fresh": view.fresh,
            "requested_generation": view.requested_generation,
            "applied_generation": view.applied_generation,
            "pending_generation": view.pending_generation,
            "requested_mode": view.requested_mode,
            "applied_mode": view.applied_mode,
            "pending_mode": view.pending_mode,
            "control_error": view.control_error,
            "chunked_prefill_enabled": view.chunked_prefill_enabled,
        }

    def _ack_status_from_views(
        self,
        views: Mapping[str, EngineTelemetry],
        expected_generation: int,
        expected_mode: Optional[str],
    ) -> ControlAckStatus:
        generation = int(expected_generation)
        mode = None if expected_mode is None else str(expected_mode)
        acked = []
        pending = []
        missing = []
        stale = []
        mismatched = []
        errors: Dict[str, str] = {}
        per_instance: Dict[str, Dict[str, Any]] = {}
        for instance_id in self.instance_ids:
            view = views.get(instance_id)
            status = self._view_control_status(view)
            per_instance[instance_id] = status
            if view is None:
                missing.append(instance_id)
                continue
            if view.stale:
                stale.append(instance_id)
                continue

            generation_associated = generation in (
                view.requested_generation,
                view.applied_generation,
                view.pending_generation,
            )
            if view.control_error and generation_associated:
                errors[instance_id] = view.control_error
                continue

            requested_matches = view.requested_generation == generation
            requested_mode_matches = (
                mode is None or view.requested_mode == mode
            )
            applied_matches = view.applied_generation == generation
            applied_mode_matches = (
                mode is None or view.applied_mode == mode
            )
            no_pending = view.pending_generation is None
            chunked_matches = (
                mode not in ("continuous", "temporal")
                or view.chunked_prefill_enabled is None
                or view.chunked_prefill_enabled == (mode == "continuous")
            )
            if (
                requested_matches
                and requested_mode_matches
                and applied_matches
                and applied_mode_matches
                and no_pending
                and not view.control_error
                and chunked_matches
            ):
                acked.append(instance_id)
            elif (
                view.pending_generation == generation
                and (mode is None or view.pending_mode == mode)
            ):
                pending.append(instance_id)
            else:
                mismatched.append(instance_id)

        all_fresh = not missing and not stale
        complete = bool(
            self.instance_ids
            and all_fresh
            and not errors
            and len(acked) == len(self.instance_ids)
        )
        if complete:
            reason = "all-engine-acks"
        else:
            details = []
            if errors:
                details.append(
                    "control-error="
                    + "|".join(
                        "%s:%s" % item for item in sorted(errors.items())
                    )
                )
            if pending:
                details.append("pending=" + ",".join(pending))
            if missing:
                details.append("missing=" + ",".join(missing))
            if stale:
                details.append("stale=" + ",".join(stale))
            if mismatched:
                details.append("mismatched=" + ",".join(mismatched))
            reason = ";".join(details) or "ack-incomplete"
        return ControlAckStatus(
            expected_generation=generation,
            expected_mode=mode,
            complete=complete,
            all_fresh=all_fresh,
            has_error=bool(errors),
            pending=bool(pending),
            acked_instances=tuple(acked),
            pending_instances=tuple(pending),
            missing_instances=tuple(missing),
            stale_instances=tuple(stale),
            mismatched_instances=tuple(mismatched),
            control_errors=errors,
            per_instance=per_instance,
            reason=reason,
        )

    def ack_status(
        self,
        expected_generation: int,
        expected_mode: Optional[str] = None,
        now: Optional[float] = None,
        *,
        mode: Optional[str] = None,
    ) -> ControlAckStatus:
        """Return whether every expected instance freshly applied a control."""
        if mode is not None:
            if expected_mode is not None and str(expected_mode) != str(mode):
                raise ValueError("expected_mode and mode disagree")
            expected_mode = mode
        return self._ack_status_from_views(
            self.read_all(now),
            expected_generation,
            expected_mode,
        )

    @staticmethod
    def _consensus(
        views: Mapping[str, EngineTelemetry],
        instance_ids: Tuple[str, ...],
        attribute: str,
        *,
        require_all: bool = True,
        ignore_none: bool = False,
    ):
        values = []
        for instance_id in instance_ids:
            view = views.get(instance_id)
            if view is None or view.stale:
                if require_all:
                    return None
                continue
            value = getattr(view, attribute)
            if value is None and ignore_none:
                continue
            values.append(value)
        if require_all and len(values) != len(instance_ids):
            return None
        if not values:
            return None
        first = values[0]
        return first if all(value == first for value in values) else None

    def aggregate(
        self,
        now: Optional[float] = None,
        expected_generation: Optional[int] = None,
        expected_mode: Optional[str] = None,
    ) -> dict:
        views = self.read_all(now)
        fresh = [view for view in views.values() if not view.stale]
        active = [view for view in fresh if view.n_scheduled_groups > 0]
        requested_generation = self._consensus(
            views, self.instance_ids, "requested_generation")
        applied_generation = self._consensus(
            views, self.instance_ids, "applied_generation")
        pending_generation = self._consensus(
            views,
            self.instance_ids,
            "pending_generation",
            require_all=False,
            ignore_none=True,
        )
        requested_mode = self._consensus(
            views, self.instance_ids, "requested_mode")
        applied_mode = self._consensus(
            views, self.instance_ids, "applied_mode")
        pending_mode = self._consensus(
            views,
            self.instance_ids,
            "pending_mode",
            require_all=False,
            ignore_none=True,
        )
        ack_generation = (
            requested_generation
            if expected_generation is None
            else int(expected_generation)
        )
        ack_mode = requested_mode if expected_mode is None else expected_mode
        ack = (
            None
            if ack_generation is None
            else self._ack_status_from_views(
                views, ack_generation, ack_mode)
        )
        chunked = {
            instance_id: view.chunked_prefill_enabled
            for instance_id, view in views.items()
        }
        result = {
            "available": len(fresh),
            "expected": len(self.instance_ids),
            "stale": len(views) - len(fresh),
            "missing": len(self.instance_ids) - len(views),
            "strict": bool(active) and not any(v.overlap for v in active),
            "omega": (
                sum(v.omega for v in active) / len(active) if active else 0.0),
            "phases": {
                instance_id: view.phase for instance_id, view in views.items()
            },
            "free_gpu_blocks": {
                instance_id: view.free_gpu_blocks
                for instance_id, view in views.items()
            },
            "requested_generations": {
                instance_id: view.requested_generation
                for instance_id, view in views.items()
            },
            "applied_generations": {
                instance_id: view.applied_generation
                for instance_id, view in views.items()
            },
            "pending_generations": {
                instance_id: view.pending_generation
                for instance_id, view in views.items()
            },
            "requested_modes": {
                instance_id: view.requested_mode
                for instance_id, view in views.items()
            },
            "applied_modes": {
                instance_id: view.applied_mode
                for instance_id, view in views.items()
            },
            "pending_modes": {
                instance_id: view.pending_mode
                for instance_id, view in views.items()
            },
            "control_errors": {
                instance_id: view.control_error
                for instance_id, view in views.items()
            },
            "chunked_prefill_enabled": chunked,
            "chunked_flags": dict(chunked),
            "requested_generation": requested_generation,
            "applied_generation": applied_generation,
            "pending_generation": pending_generation,
            "requested_mode": requested_mode,
            "applied_mode": applied_mode,
            "pending_mode": pending_mode,
            "ack_complete": bool(ack and ack.complete),
            "ack_generation": (
                ack.expected_generation if ack and ack.complete else None
            ),
            "ack_reason": "" if ack is None else ack.reason,
            "ack_status": None if ack is None else ack.to_dict(),
        }
        return result
