"""Independently enforce the preregistered B Dynamo arrival engineering gate."""
import csv
import math
from pathlib import Path

RULES_SHA = '25c58022b808d4e43dd7d1ef24a206b40bafe024f6074f9799b9f64f4bf7f135'


def recompute(rows, expected_count):
    if len(rows) != expected_count or len({x['idx'] for x in rows}) != expected_count:
        raise ValueError('arrival audit request count or identity differs')
    delays = []
    planned = []
    for row in rows:
        a, d = float(row['planned_arrival_s']), float(row['actual_dispatch_s'])
        if not (math.isfinite(a) and math.isfinite(d) and d >= a):
            raise ValueError('arrival audit has unknown or backwards dispatch time')
        delay = d - a
        if not math.isclose(delay, float(row['dispatch_delay_s']), rel_tol=0, abs_tol=1e-12):
            raise ValueError('recorded dispatch delay differs from actual timestamps')
        delays.append(delay)
        planned.append(a)
    delays.sort()
    where = (len(delays) - 1) * .99
    lower = int(where)
    p99 = delays[lower] + (delays[min(lower + 1, len(delays) - 1)] - delays[lower]) * (where - lower)
    if delays[-1] > 1.0 or p99 > .1:
        raise ValueError('preregistered arrival engineering limits exceeded')
    return dict(n_requests=len(rows), actual_dispatch_max_s=delays[-1],
                actual_dispatch_p99_s=p99, first_planned_s=min(planned))


def verify(p, cp, binding, directory, declaration_ref):
    reference = cp['execution_rules']
    p.need(reference['sha256'] == RULES_SHA, 'unrecognized arrival engineering rules')
    rules = p.checked(reference)
    p.need(binding['files'].get(reference['path']) == reference['sha256'],
           'arrival engineering rules not frozen in executed binding')
    p.need(rules['schema'] == 'B-cooperative-Dynamo8-arrival-engineering-v1'
           and rules['declaration'] == declaration_ref
           and rules['actual_dispatch_max_limit_s'] == 1.0
           and rules['actual_dispatch_p99_limit_s'] == .1
           and rules['percentile_method'] == 'linear at(n-1)*0.99',
           'arrival engineering policy or scientific declaration changed')
    for path, digest in rules['source_files'].items():
        p.need(p.sha(path) == digest, 'preregistered arrival rule implementation changed')
    p.need(p.checked(rules['cpu_validation'])['passed'] is True,
           'arrival rule CPU qualification is absent')
    reported = cp['arrival_fidelity_gate']
    p.need(reported['execution_rules'] == reference and reported['passed'] is True
           and reported['errors'] == [] and reported['max_limit_s'] == 1.0
           and reported['p99_limit_s'] == .1
           and reported['percentile_method'] == rules['percentile_method'],
           'checkpoint arrival qualification differs')
    raw = reported['raw_requests']
    expected_path = str(Path(directory) / 'bench.csv')
    p.need(raw['path'] == expected_path and p.sha(expected_path) == raw['sha256']
           and cp['artifacts'].get(expected_path) == raw['sha256'],
           'arrival engineering raw requests are not the executed checkpoint artifact')
    with open(expected_path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    actual = recompute(rows, cp['row']['n_requests'])
    p.need(rules['created_s'] < actual['first_planned_s'], 'arrival rules were declared after measurement began')
    for field in ('n_requests', 'actual_dispatch_max_s', 'actual_dispatch_p99_s'):
        p.need(math.isclose(reported[field], actual[field], rel_tol=0, abs_tol=1e-12),
               'checkpoint arrival value differs from raw timestamps: ' + field)
    return dict(passed=True, execution_rules=reference, raw_requests=raw,
                independent_raw_timestamp_recomputation=actual,
                arrival_gate_does_not_assert_complete_work_or_SLO=True)
