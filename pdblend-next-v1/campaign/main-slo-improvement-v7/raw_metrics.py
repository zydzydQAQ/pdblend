"""Additional pure raw-metric checks; never edit a producer or an observation."""
import csv
import math
from pathlib import Path
import statistics


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def truth(value):
    return str(value).lower() in ('true', '1')


def match(actual, expected, field):
    # Frozen producers use NaN for the average of an empty latency series.
    # Retain that raw representation, but export an unknown value as JSON null.
    if expected is None:
        require(actual is None or type(actual) is float and math.isnan(actual),
                'undefined raw metric represented as a number: ' + field)
    else:
        require(number(actual) and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8),
                'recomputed raw metric differs: ' + field)


def audit_additional_metrics(summary, directory):
    """Run after the full workload, SLO and eight-GPU integration verifier."""
    with (Path(directory) / 'bench.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    require(rows, 'request denominator is empty')
    derived = {}
    for raw_key, summary_key in [('ttft_s', 'ttft_avg_s'), ('tpot_s', 'tpot_avg_s')]:
        values = [float(r[raw_key]) for r in rows if r.get(raw_key) not in (None, '')]
        require(all(math.isfinite(v) and v >= 0 for v in values), 'invalid raw latency: ' + raw_key)
        # Match the original metrics.summarize_bench denominator: all recorded
        # latencies, including partial requests if a latency was recorded.
        derived[summary_key] = statistics.fmean(values) if values else None
    energy, good = summary['energy_j'], summary['good_requests']
    require(number(energy) and energy >= 0 and type(good) is int and 0 <= good <= len(rows),
            'energy/good-request denominator invalid')
    derived['energy_per_good_request_j'] = energy / good if good else None
    derived['failed_requests'] = summary['n_expected'] - summary['completed_work_requests']
    derived['offered_requests'] = len(rows)
    derived['request_timeouts'] = sum(truth(r.get('request_timeout', False)) for r in rows)
    derived['admission_rejections'] = sum(truth(r.get('admission_rejection', False)) for r in rows)
    derived['input_tokens'] = sum(int(r['input_tokens']) for r in rows)
    for field, expected in derived.items():
        match(summary.get(field), expected, field)
    return dict(additional_raw_metrics_recomputed=True, normalized_metrics=derived,
                latency_denominator='all recorded per-request values; original producer convention',
                undefined_value_policy='raw NaN preserved; normalized JSON null, never zero')
