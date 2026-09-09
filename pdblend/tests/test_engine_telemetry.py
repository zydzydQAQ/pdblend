# -*- coding: utf-8 -*-
"""Patched vLLM atomic phase/KV telemetry reader."""
from __future__ import annotations

import json

from ecopadg.engine_telemetry import EngineTelemetry, FileTelemetryRegistry


def test_engine_telemetry_marks_stale_and_overlap():
    view = EngineTelemetry.from_dict({
        "instance_id": "mixed-0", "ts": 10.0, "phase": "overlap",
        "n_prefill_groups": 1, "n_decode_groups": 2,
        "n_scheduled_groups": 3, "overlap": True,
        "free_gpu_blocks": 99,
    }, now=13.0, stale_after_s=2.0)
    assert view.stale is True
    assert view.omega == 1.0
    assert view.free_gpu_blocks == 99


def test_registry_aggregates_phase_purity(tmp_path):
    for index, phase in enumerate(("prefill", "decode")):
        (tmp_path / ("mixed-%d.json" % index)).write_text(json.dumps({
            "instance_id": "mixed-%d" % index,
            "ts": 100.0,
            "phase": phase,
            "n_prefill_groups": int(phase == "prefill"),
            "n_decode_groups": int(phase == "decode"),
            "n_scheduled_groups": 1,
            "overlap": False,
            "free_gpu_blocks": 10 - index,
        }), encoding="utf-8")
    registry = FileTelemetryRegistry(
        str(tmp_path), ["mixed-0", "mixed-1"], stale_after_s=2.0)
    aggregate = registry.aggregate(now=101.0)
    assert aggregate["available"] == 2
    assert aggregate["strict"] is True
    assert aggregate["omega"] == 0.0
    assert aggregate["phases"]["mixed-0"] == "prefill"


def test_registry_missing_files_fail_closed(tmp_path):
    registry = FileTelemetryRegistry(str(tmp_path), ["mixed-0"])
    aggregate = registry.aggregate(now=1.0)
    assert aggregate["available"] == 0
    assert aggregate["strict"] is False


def test_registry_retains_stale_per_instance_view(tmp_path):
    (tmp_path / "mixed-0.json").write_text(json.dumps({
        "instance_id": "mixed-0",
        "ts": 10.0,
        "phase": "prefill",
        "n_prefill_groups": 1,
        "n_scheduled_groups": 1,
    }), encoding="utf-8")
    registry = FileTelemetryRegistry(
        str(tmp_path), ["mixed-0", "mixed-1"], stale_after_s=2.0)

    views = registry.read_all(now=13.0)
    assert views["mixed-0"].phase == "prefill"
    assert views["mixed-0"].stale is True
    assert "mixed-1" not in views
    aggregate = registry.aggregate(now=13.0)
    assert aggregate["available"] == 0
    assert aggregate["stale"] == 1
