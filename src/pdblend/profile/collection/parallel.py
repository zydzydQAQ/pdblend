"""Helpers for profiling independent engine instances concurrently.

The profiler owns engine lifecycle; this module only describes deterministic
sharding and evaluates the small interference comparison used to qualify a
parallel run.  Keeping these pure makes orchestration easy to test without a
GPU or vLLM installation.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


def partition_frequencies(freqs: Sequence[int], instance_count: int) -> tuple[tuple[int, ...], ...]:
    """Partition frequencies round-robin while preserving input order."""
    if instance_count < 1:
        raise ValueError("instance_count must be positive")
    shards = [[] for _ in range(instance_count)]
    for index, freq in enumerate(freqs):
        shards[index % instance_count].append(int(freq))
    return tuple(tuple(shard) for shard in shards)


def point_identity(*, freq_mhz: int, batch: int = 8, context_tokens: int = 1024) -> dict:
    return dict(freq_mhz=int(freq_mhz), batch=int(batch), context_tokens=int(context_tokens))


def relative_delta(parallel: float, isolated: float) -> float:
    if not all(isinstance(value, (int, float)) and math.isfinite(value)
               for value in (parallel, isolated)) or isolated <= 0:
        raise ValueError("interference values must be finite and isolated must be positive")
    return abs(float(parallel) - float(isolated)) / float(isolated)


def evaluate_interference(isolated: Mapping[str, float], parallel: Mapping[str, float],
                          *, limit: float = 0.05) -> dict:
    """Compare representative isolated and concurrent decode measurements."""
    timing = relative_delta(parallel["step_seconds"], isolated["step_seconds"])
    power = relative_delta(parallel["power_w"], isolated["power_w"])
    return dict(passed=timing <= limit and power <= limit,
                limit=limit, step_relative_error=timing,
                power_relative_error=power)


def qualified(summary: Mapping) -> bool:
    """Return true only for an explicitly completed, bounded comparison."""
    validation = summary.get("validation", {})
    comparisons = summary.get("comparisons", ())
    comparison_passed = bool(comparisons) and all(row.get("passed") is True for row in comparisons)
    bounded = validation.get("passed") is True or (
        comparison_passed and summary.get("overlapping_windows") is True)
    return bool(summary.get("complete") and summary.get("isolated") and
                summary.get("parallel") and bounded)


def common_window_overlap(instances: Sequence[Mapping], *, minimum_s: float = 2.) -> dict:
    """Check actual simultaneous sampling of every instance for every repeat."""
    windows = [item.get('repeats', []) for item in instances]
    if len(windows) < 2 or not windows[0] or any(len(rows) != len(windows[0]) for rows in windows):
        return dict(passed=False, overlap_seconds=[], reason='missing_or_mismatched_multi_instance_windows')
    overlaps = []
    try:
        for repeats in zip(*windows):
            starts, ends = [r['start_s'] for r in repeats], [r['end_s'] for r in repeats]
            if any(type(t) not in (int, float) or not math.isfinite(t) for t in starts+ends):
                raise ValueError('nonfinite measurement window')
            if any(b <= a for a, b in zip(starts, ends)):
                raise ValueError('nonpositive measurement window')
            overlaps.append(max(0., min(ends)-max(starts)))
    except (KeyError, TypeError, ValueError) as exc:
        return dict(passed=False, overlap_seconds=[], reason=str(exc))
    return dict(passed=all(seconds >= minimum_s for seconds in overlaps),
                overlap_seconds=overlaps, minimum_s=minimum_s)
