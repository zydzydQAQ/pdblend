# -*- coding: utf-8 -*-
"""Engine-step phase telemetry and strict temporal-PaDG purity gate."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class StepPhaseRecord:
    step: int
    t_s: float
    n_prefill_groups: int
    n_decode_groups: int
    n_scheduled_groups: int
    free_gpu_blocks: Optional[int] = None

    @property
    def overlap(self) -> bool:
        return self.n_prefill_groups > 0 and self.n_decode_groups > 0

    @property
    def phase(self) -> str:
        if self.overlap:
            return "overlap"
        if self.n_prefill_groups > 0:
            return "prefill"
        if self.n_decode_groups > 0:
            return "decode"
        return "idle"


def record_from_scheduler_output(step: int, output: Any,
                                 scheduler: Any = None,
                                 now: Optional[float] = None
                                 ) -> StepPhaseRecord:
    groups = list(getattr(output, "scheduled_seq_groups", ()) or ())
    n_prefill = getattr(output, "num_prefill_groups", None)
    if n_prefill is None:
        n_prefill = sum(
            1 for item in groups
            if bool(item.seq_group.is_prefill()))
    n_prefill = int(n_prefill)
    n_decode = max(len(groups) - n_prefill, 0)
    free_blocks = None
    manager = getattr(scheduler, "block_manager", None)
    getter = getattr(manager, "get_num_free_gpu_blocks", None)
    if callable(getter):
        try:
            free_blocks = int(getter())
        except (TypeError, ValueError):
            free_blocks = None
    return StepPhaseRecord(
        step=int(step), t_s=float(time.time() if now is None else now),
        n_prefill_groups=n_prefill, n_decode_groups=n_decode,
        n_scheduled_groups=len(groups), free_gpu_blocks=free_blocks)


def phase_purity_summary(records: List[StepPhaseRecord]) -> Dict[str, Any]:
    active = [r for r in records if r.n_scheduled_groups > 0]
    overlap = [r for r in active if r.overlap]
    return {
        "steps": len(records),
        "active_steps": len(active),
        "prefill_steps": sum(r.phase == "prefill" for r in active),
        "decode_steps": sum(r.phase == "decode" for r in active),
        "overlap_steps": len(overlap),
        "omega": (len(overlap) / len(active)) if active else 0.0,
        "strict": bool(active) and not overlap,
    }


class PhasePurityRecorder:
    """Wrap V0 Scheduler.schedule without changing its scheduling result."""

    def __init__(self):
        self.records: List[StepPhaseRecord] = []
        self._scheduler = None
        self._original: Optional[Callable] = None

    def attach(self, engine: Any, virtual_engine: int = 0) -> None:
        schedulers = getattr(engine, "scheduler", None)
        if not schedulers:
            raise RuntimeError("engine 未暴露 V0 scheduler；V1 不能用于 strict gate")
        scheduler = schedulers[int(virtual_engine)]
        original = scheduler.schedule

        def wrapped_schedule():
            result = original()
            output = result[1]
            self.records.append(record_from_scheduler_output(
                len(self.records), output, scheduler=scheduler))
            return result

        scheduler.schedule = wrapped_schedule
        self._scheduler = scheduler
        self._original = original

    def detach(self) -> None:
        if self._scheduler is not None and self._original is not None:
            self._scheduler.schedule = self._original
        self._scheduler = None
        self._original = None

    def summary(self) -> Dict[str, Any]:
        return phase_purity_summary(self.records)

    def write_json(self, path: str, **metadata: Any) -> None:
        payload = {
            "schema_version": 1,
            "summary": self.summary(),
            "metadata": metadata,
            "steps": [
                {**asdict(record), "phase": record.phase,
                 "overlap": record.overlap}
                for record in self.records
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
