"""Attribute transition energy from the common, uninterrupted GPU sampler."""
from __future__ import annotations

import math


def interval_energy(samples, start, end, indices):
    """Interpolate boundary samples; never extrapolate beyond sampled time."""
    if not (math.isfinite(start) and math.isfinite(end)) or end < start:
        raise ValueError('invalid transition interval')
    if start == end:
        return 0.
    if len(samples) < 2 or start < samples[0][0] or end > samples[-1][0]:
        return None
    energy = 0.
    for (ta, wa), (tb, wb) in zip(samples, samples[1:]):
        if tb <= ta:
            raise ValueError('nonmonotonic power samples')
        lo, hi = max(start, ta), min(end, tb)
        if hi <= lo:
            continue
        pa, pb = sum(wa[i] for i in indices), sum(wb[i] for i in indices)
        if not all(math.isfinite(x) and x >= 0 for x in (pa, pb)):
            raise ValueError('invalid GPU power')
        left = pa + (pb-pa)*(lo-ta)/(tb-ta)
        right = pa + (pb-pa)*(hi-ta)/(tb-ta)
        energy += (left+right)*(hi-lo)/2.
    return energy


def measure_transitions(events, samples, gpu_ids, *, sampler_error=None, power_source=None):
    index = {gpu: i for i, gpu in enumerate(gpu_ids)}
    rows = []
    intervals = {gpu: [] for gpu in gpu_ids}
    for event in events:
        row = dict(event)
        if not set(row['gpus']) <= set(index):
            raise ValueError('transition references unmetered GPU')
        energy = (None if sampler_error else interval_energy(samples, row['started_s'], row['finished_s'],
                                                             [index[g] for g in row['gpus']]))
        row.update(energy_j=energy, energy_status='measured' if energy is not None else 'not_covered',
                   incremental_energy_j=None, counterfactual_measured=False)
        rows.append(row)
        for gpu in row['gpus']:
            intervals[gpu].append((row['started_s'], row['finished_s']))
    total, complete = 0., not bool(sampler_error)
    for gpu, spans in intervals.items():
        merged = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        for start, end in merged:
            value = interval_energy(samples, start, end, [index[gpu]]) if not sampler_error else None
            complete &= value is not None
            total += value or 0.
    return dict(schema=1, phases=rows, measured_union_energy_j=total if complete else None,
                energy_complete=complete, summed_parallel_phase_durations_are_walltime=False,
                power_source=power_source, sampler_error=sampler_error,
                incremental_energy_j=None, formal_eligible=False)
