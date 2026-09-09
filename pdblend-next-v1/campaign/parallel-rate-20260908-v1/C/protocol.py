"""Immutable experiment contracts for the authorized SLO improvement campaign."""
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SNAPSHOT = REPO / 'campaign/five-system-results-v4/actual-snapshot-006'
DEADLINE = 1788872770.0400891
MODELS = ('7b', '14b', '32b')
DATASETS = ('alpaca', 'sharegpt', 'longbench')
BASELINES = ('mixed', 'distserve', 'dynamollm', 'ecoserve')
PINNED = {
    'points.csv': 'be823cc60792df7af04170e957019cfa7f15d39c78abddd21fd57815d0ecc716',
    'results.json': '1ddd027859c026ebbea6dce212fe8369098de1505f18d489af2a8f0be3e3febb',
    'manifest.json': 'abd30f4f9b758cd7ceb9eb056e038fa2244ab82581b0c02740f13ce1ea3954e8',
}
SCREEN = {
    '7b': {'alpaca': (9., 12.), 'sharegpt': (2., 3.), 'longbench': (1.5, 3.)},
    '14b': {'alpaca': (9., 12.), 'sharegpt': (1.5, 2.), 'longbench': (1., 1.25)},
    '32b': {'alpaca': (2.5, 4.), 'sharegpt': (.6, .8, 1., 2.), 'longbench': (.3, 1.)},
}
PAIR_FIELDS = ('model', 'dataset', 'rate_rps', 'seed', 'trace_sha256',
               'content_pairing_sha256', 'slo_ttft_s', 'slo_tpot_s',
               'n_expected', 'expected_generated_tokens')

def need(condition, why):
    if not condition:
        raise ValueError(why)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}

def checked(reference):
    need(sha(reference['path']) == reference['sha256'], 'changed reference: ' + reference['path'])
    return read(reference['path'])

def write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path if exclusive else path.with_name(path.name + '.tmp')
    with target.open('x' if exclusive else 'w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if not exclusive:
        target.replace(path)

def original_points():
    for name, digest in PINNED.items():
        need(sha(SNAPSHOT / name) == digest, 'original snapshot changed: ' + name)
    points = [p for p in read(SNAPSHOT / 'results.json')['points']
              if p['phase'] == 'main' and p['slo_scale'] == 1.]
    need(len(points) == len({p['cell_id'] for p in points}) == 450, 'original main grid incomplete')
    return points

def pair_identity(point):
    return tuple(point[key] for key in PAIR_FIELDS)

def verdict(candidate, baseline):
    need(pair_identity(candidate) == pair_identity(baseline), 'not an exact workload pair')
    energy, slo = candidate.get('energy_j'), candidate.get('slo_attainment')
    valid = (candidate.get('measurement_valid') is True and
             isinstance(energy, (int, float)) and math.isfinite(energy) and energy > 0 and
             isinstance(slo, (int, float)) and math.isfinite(slo) and 0 <= slo <= 1)
    baseline_energy = baseline['energy_j']
    target = min(.90, baseline['slo_attainment'])
    work = bool(valid and candidate.get('work_complete') is True and
                candidate['completed_work_requests'] == candidate['n_expected'] and
                candidate['generated_tokens'] == candidate['expected_generated_tokens'])
    # No statistical or numerical allowance is added to the user's criterion.
    energy_pass = bool(valid and energy <= baseline_energy)
    slo_pass = bool(valid and slo >= target)
    return dict(measurement_valid=bool(valid), work_pass=work,
                energy_pass=energy_pass, slo_pass=slo_pass,
                passed=work and energy_pass and slo_pass, slo_required=target,
                energy_change_pct=100 * (energy / baseline_energy - 1) if valid else None,
                slo_difference_pp=100 * (slo - baseline['slo_attainment']) if valid else None,
                baseline_system=baseline['system'], baseline_cell_id=baseline['cell_id'])
