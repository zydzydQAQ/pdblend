# -*- coding: utf-8 -*-
"""Strict PaDG engine-step purity accounting."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ecopadg.phase_purity import (
    PhasePurityRecorder,
    StepPhaseRecord,
    phase_purity_summary,
    record_from_scheduler_output,
)


def _group(prefill):
    return SimpleNamespace(
        seq_group=SimpleNamespace(is_prefill=lambda: bool(prefill)))


def test_record_counts_overlap_and_free_blocks():
    output = SimpleNamespace(
        scheduled_seq_groups=[_group(True), _group(False)],
        num_prefill_groups=1,
    )
    scheduler = SimpleNamespace(
        block_manager=SimpleNamespace(get_num_free_gpu_blocks=lambda: 42))
    row = record_from_scheduler_output(3, output, scheduler, now=1.5)
    assert row.phase == "overlap"
    assert row.overlap is True
    assert row.free_gpu_blocks == 42


def test_phase_purity_requires_active_xor_steps():
    pure = [
        StepPhaseRecord(0, 0.0, 1, 0, 1),
        StepPhaseRecord(1, 0.1, 0, 2, 2),
        StepPhaseRecord(2, 0.2, 0, 0, 0),
    ]
    summary = phase_purity_summary(pure)
    assert summary["strict"] is True
    assert summary["omega"] == 0.0
    mixed = pure + [StepPhaseRecord(3, 0.3, 1, 1, 2)]
    assert phase_purity_summary(mixed)["strict"] is False
    assert phase_purity_summary([])["strict"] is False


def test_recorder_wraps_and_restores_v0_scheduler():
    outputs = [
        SimpleNamespace(
            scheduled_seq_groups=[_group(True)], num_prefill_groups=1),
        SimpleNamespace(
            scheduled_seq_groups=[_group(False)], num_prefill_groups=0),
    ]

    class Scheduler:
        def __init__(self):
            self.i = 0
            self.block_manager = SimpleNamespace(
                get_num_free_gpu_blocks=lambda: 7)

        def schedule(self):
            out = outputs[self.i]
            self.i += 1
            return [], out, False

    scheduler = Scheduler()
    original = scheduler.schedule
    recorder = PhasePurityRecorder()
    recorder.attach(SimpleNamespace(scheduler=[scheduler]))
    scheduler.schedule()
    scheduler.schedule()
    assert recorder.summary()["strict"] is True
    recorder.detach()
    assert getattr(scheduler.schedule, "__func__", None) is getattr(
        original, "__func__", None)


def test_recorder_rejects_v1_engine():
    with pytest.raises(RuntimeError):
        PhasePurityRecorder().attach(SimpleNamespace())
