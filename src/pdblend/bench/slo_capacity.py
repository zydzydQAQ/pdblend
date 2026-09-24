"""Read-only SLO capacity search over completed calibration/tuning trials.

This module neither runs experiments nor manufactures missing measurements.
Each rate needs independent repeated evidence. A capacity is an interval:
the greatest consistently passing rate and the smallest consistently failing
rate. Evaluation traces may only consume the frozen grid, never select it.
The caller must still apply campaign/profile qualification gates; this module
does not promote an observed SLO boundary into formal ranking eligibility.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Mapping, Sequence


def _positive(value, name):
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be finite and positive')


@dataclass(frozen=True)
class CapacityConfig:
    series_id: str
    slo_ttft_s: float
    slo_tpot_s: float
    required_repeats: int = 3
    min_requests_per_trial: int = 100
    relative_tolerance: float = .05
    initial_rate: float = 1.

    def __post_init__(self):
        if not isinstance(self.series_id, str) or not self.series_id:
            raise ValueError('an explicit model/system/dataset series_id is required')
        for name in ('slo_ttft_s', 'slo_tpot_s', 'initial_rate', 'relative_tolerance'):
            _positive(getattr(self, name), name)
        if self.relative_tolerance > .05:
            raise ValueError('capacity relative_tolerance cannot exceed 5%')
        if type(self.required_repeats) is not int or self.required_repeats < 3:
            raise ValueError('capacity needs at least three independent repeats')
        if type(self.min_requests_per_trial) is not int or self.min_requests_per_trial < 100:
            raise ValueError('P99 capacity trials need at least 100 offered requests')


@dataclass(frozen=True)
class CapacityTrial:
    series_id: str
    split: str
    rate_scale: float
    repeat_id: str
    evidence_id: str
    metrics: Mapping

    def __post_init__(self):
        _positive(self.rate_scale, 'rate_scale')
        for name in ('series_id', 'repeat_id', 'evidence_id'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(name + ' must identify real independent evidence')
        if self.split not in ('calibration', 'tuning'):
            raise ValueError('capacity selection only accepts calibration/tuning, never evaluation')
        if not isinstance(self.metrics, Mapping):
            raise ValueError('capacity metrics must be a measured mapping')


def trial_verdict(trial: CapacityTrial, config: CapacityConfig):
    """A known request failure fails SLO; absent timing evidence stays incomplete."""
    if trial.series_id != config.series_id:
        raise ValueError('capacity trials cannot mix model/system/dataset series')
    metrics = trial.metrics
    needed = ('offered_requests', 'successful_requests', 'joint_slo_requests',
              'unresolved_requests', 'ttft_samples', 'tpot_samples', 'measurement_usable',
              'slo_ttft_s', 'slo_tpot_s')
    missing = [name for name in needed if metrics.get(name) is None]
    if missing:
        return dict(verdict='incomplete', reasons=['missing:' + name for name in missing])
    for name in ('slo_ttft_s', 'slo_tpot_s'):
        _positive(metrics[name], name)
        if metrics[name] != getattr(config, name):
            raise ValueError('capacity evidence must use the same frozen SLO: ' + name)
    counts = {name: metrics[name] for name in needed
              if name not in ('measurement_usable', 'slo_ttft_s', 'slo_tpot_s')}
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError('request and timing sample counts must be nonnegative integers')
    if type(metrics['measurement_usable']) is not bool:
        raise ValueError('measurement_usable must be a boolean')
    offered, succeeded, good = (counts[k] for k in
                               ('offered_requests', 'successful_requests', 'joint_slo_requests'))
    if good > succeeded or succeeded + counts['unresolved_requests'] > offered:
        raise ValueError('capacity request counts are inconsistent')
    # The comparison reducer includes unresolved arrivals in failed_requests.
    if 'failed_requests' in metrics and (type(metrics['failed_requests']) is not int
            or metrics['failed_requests'] != offered - succeeded):
        raise ValueError('failed/successful requests must partition arrivals')
    if 'invalid_timing_requests' in metrics and (type(metrics['invalid_timing_requests']) is not int
            or not 0 <= metrics['invalid_timing_requests'] <= offered - succeeded):
        raise ValueError('invalid timing count cannot include successful requests')
    reasons = []
    if not metrics['measurement_usable']:
        reasons.append('measurement_unusable')
    if offered < config.min_requests_per_trial:
        reasons.append('insufficient_request_samples')
    if counts['unresolved_requests']:
        reasons.append('unresolved_requests')
    if counts['ttft_samples'] != succeeded or counts['tpot_samples'] != succeeded:
        reasons.append('incomplete_latency_samples')
    # All-failed trials have no successful latency distribution. Their 0%
    # success rate still provides an observed failing upper bound.
    if succeeded:
        for name in ('ttft_p99_s', 'tpot_p99_s'):
            value = metrics.get(name)
            if value is None:
                reasons.append('missing:' + name)
            elif type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(name + ' must be finite and nonnegative')
    if reasons:
        return dict(verdict='incomplete', reasons=reasons)
    failures = []
    if succeeded != offered or counts['unresolved_requests']:
        failures.append('success_rate_below_100_percent')
    if good / offered < .9:
        failures.append('joint_slo_below_90_percent')
    if succeeded and metrics['ttft_p99_s'] > config.slo_ttft_s:
        failures.append('ttft_p99_exceeds_slo')
    if succeeded and metrics['tpot_p99_s'] > config.slo_tpot_s:
        failures.append('tpot_p99_exceeds_slo')
    return dict(verdict='fail' if failures else 'pass', reasons=failures)


def capacity_state(trials: Sequence[CapacityTrial], config: CapacityConfig):
    """Return observed bounds plus the next repeat/rate, never an exact capacity.

    All repeats at a rate must agree. Mixed pass/fail results are unstable,
    rather than majority-voted into a claimed boundary. A failed rate below a
    passed one is nonmonotonic and must be investigated before more selection.
    """
    groups = defaultdict(list)
    evidence, repetitions, splits = set(), set(), set()
    for trial in trials:
        if not isinstance(trial, CapacityTrial):
            raise ValueError('capacity input must contain CapacityTrial records')
        if trial.evidence_id in evidence or (trial.rate_scale, trial.repeat_id) in repetitions:
            raise ValueError('duplicate capacity receipt or repeat cannot count as independent evidence')
        evidence.add(trial.evidence_id)
        repetitions.add((trial.rate_scale, trial.repeat_id))
        splits.add(trial.split)
        groups[trial.rate_scale].append(trial_verdict(trial, config))
    if len(splits) > 1:
        raise ValueError('freeze one calibration or tuning split for a capacity search')
    rates = []
    for rate, records in sorted(groups.items()):
        verdicts = [record['verdict'] for record in records]
        complete = [v for v in verdicts if v != 'incomplete']
        # Contradictory completed observations cannot be repaired by appending
        # repetitions. Surface them even when another attempt is incomplete or
        # the minimum repeat count has not yet been reached.
        if len(set(complete)) > 1:
            status = 'unstable'
        elif 'incomplete' in verdicts or len(complete) < config.required_repeats:
            status = 'incomplete'
        else:
            status = complete[0]
        rates.append(dict(rate_scale=rate, status=status, repeats=len(records),
            complete_repeats=len(complete), required_repeats=config.required_repeats, trials=records))
    passed = [row['rate_scale'] for row in rates if row['status'] == 'pass']
    failed = [row['rate_scale'] for row in rates if row['status'] == 'fail']
    lower, upper = max(passed, default=None), min(failed, default=None)
    incomplete = [row['rate_scale'] for row in rates if row['status'] == 'incomplete']
    unstable = [row['rate_scale'] for row in rates if row['status'] == 'unstable']
    nonmonotonic = lower is not None and upper is not None and lower >= upper
    width = (upper - lower) / lower if lower is not None and upper is not None and not nonmonotonic else None
    converged = (width is not None and (width <= config.relative_tolerance or
                 math.isclose(width, config.relative_tolerance, rel_tol=1e-12, abs_tol=0))
                 and not incomplete and not unstable and not nonmonotonic)
    next_rate = None
    if nonmonotonic:
        status, action = 'nonmonotonic', 'investigate'
    elif unstable:
        status, action = 'unstable_repeats', 'investigate'
    elif incomplete:
        status, action, next_rate = 'incomplete', 'repeat_rate', min(incomplete)
    elif converged:
        status, action = 'converged', 'freeze_evaluation_grid'
    elif lower is None and upper is None:
        status, action, next_rate = 'empty', 'measure_rate', config.initial_rate
    elif upper is None:
        status, action, next_rate = 'missing_upper_bound', 'measure_rate', lower * 2
    elif lower is None:
        status, action, next_rate = 'missing_lower_bound', 'measure_rate', upper / 2
    else:
        status, action, next_rate = 'bracketing', 'measure_rate', lower + (upper - lower) / 2
    if next_rate is not None and (not math.isfinite(next_rate) or next_rate <= 0
            or (status == 'bracketing' and not lower < next_rate < upper)):
        status, action, next_rate = 'numeric_search_limit', 'investigate', None
    return dict(schema='pdblend-slo-capacity-state/v1', series_id=config.series_id,
        selection_split=next(iter(splits), None), status=status, action=action,
        next_rate_scale=next_rate, passed_lower=lower, failed_upper=upper,
        relative_width=width, relative_tolerance=config.relative_tolerance,
        slo=dict(ttft_s=config.slo_ttft_s, tpot_s=config.slo_tpot_s,
                 success_rate=1., joint_slo_rate=.9),
        required_repeats=config.required_repeats, min_requests_per_trial=config.min_requests_per_trial,
        converged=converged, capacity_interval=(dict(passed_lower=lower, failed_upper=upper)
                                               if converged else None),
        rates=rates, evidence_ids=sorted(evidence))


def freeze_evaluation_grid(trials: Sequence[CapacityTrial], config: CapacityConfig,
                           *, multipliers=(.25, .5, .75, 1., 1.1)):
    """Select a future evaluation grid only from a complete tuning bracket.

    Multipliers apply to its passing lower bound; also retain the measured
    failing upper bound so the future evaluation includes both sides.
    """
    if not multipliers:
        raise ValueError('evaluation grid needs explicit positive multipliers')
    for value in multipliers:
        _positive(value, 'evaluation multiplier')
    if len(set(multipliers)) != len(multipliers):
        raise ValueError('evaluation grid multipliers must be unique')
    state = capacity_state(trials, config)
    if not state['converged']:
        raise ValueError('cannot freeze evaluation grid from ' + state['status'])
    rates = sorted({state['passed_lower'] * value for value in multipliers}
                   | {state['passed_lower'], state['failed_upper']})
    for rate in rates:
        _positive(rate, 'evaluation rate')
    return dict(schema='pdblend-frozen-capacity-evaluation-grid/v1', series_id=config.series_id,
        selection_split=state['selection_split'], target_split='evaluation',
        evaluation_used_for_selection=False, rate_scales=rates,
        capacity_interval=state['capacity_interval'], multipliers=list(multipliers),
        selection_evidence_ids=state['evidence_ids'], slo=dict(ttft_s=config.slo_ttft_s,
            tpot_s=config.slo_tpot_s, success_rate=1., joint_slo_rate=.9),
        required_repeats=config.required_repeats, min_requests_per_trial=config.min_requests_per_trial)


def main(argv=None):
    """Inspect a hash-bound capacity ledger; print a report without running jobs."""
    import argparse
    import json
    from .slo_capacity_receipts import read_capacity_ledger
    parser = argparse.ArgumentParser(description='Read-only SLO capacity evidence and next-rate decision')
    parser.add_argument('--ledger', required=True, help='pdblend-slo-capacity-ledger/v1 JSON manifest')
    args = parser.parse_args(argv)
    try:
        report = read_capacity_ledger(args.ledger)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
