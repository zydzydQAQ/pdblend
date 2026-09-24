import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pdblend.bench.comparison_campaign import binding
from pdblend.bench.cohort_dominance import compare_point
from pdblend.bench.measurement_compatibility import (
    BACKENDS, POWER, WHOLE_FILES, hydrate_receipt_evidence, load_compatibility,
    prepare_compatibility,
)
from pdblend.bench.resident_session import digest, write_new
from test_cohort_dominance import baseline_rows, point

ROOT = Path(__file__).resolve().parents[2]


def freeze(path, texts):
    files = {}
    for name, text in texts.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        files[name] = hashlib.sha256(text.encode()).hexdigest()
    write_new(path / 'manifest.json', dict(source_sha256=digest(files), files=files))
    return binding(path / 'manifest.json')


def old_text(text, cls, method):
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
    lines = text.splitlines(keepends=True)
    del lines[fn.lineno - 1:fn.end_lineno]
    text = ''.join(lines)
    if cls == 'PowerSampler':
        text = ''.join(line for line in text.splitlines(keepends=True) if not any(
            ('self.' + name + suffix) in line for name in ('frequency_readings', 'frequency_errors',
                'frequency_requested', '_frequency_lock') for suffix in (' =', ':')))
        text = text.replace("self.capture_frequency(reason='periodic')",
                            'self.frequency_samples.append((sample[0], [self.backend.current_freq(g) for g in self.gpus]))')
    return text


@pytest.fixture
def sources(tmp_path):
    texts = {name: (ROOT / 'src' / name).read_text() for name in (POWER, BACKENDS, *WHOLE_FILES)}
    before = dict(texts)
    before[POWER] = old_text(texts[POWER], 'PowerSampler', 'capture_frequency')
    before[BACKENDS] = old_text(texts[BACKENDS], 'PynvmlBackend', 'clock_diagnostics')
    return freeze(tmp_path / 'old', before), freeze(tmp_path / 'new', texts), texts


def test_explicit_review_preserves_distinct_sources_and_all_energy_core_hashes(sources, tmp_path):
    left, right, _ = sources
    review = prepare_compatibility(left, right)
    assert review['compatible'] and not review['source_identity_equal']
    assert review['changed_files'] == [BACKENDS, POWER]
    assert review['sources'][0]['energy_core'] == review['sources'][1]['energy_core']
    assert review['sources'][0]['measurement_source_sha256'] != review['sources'][1]['measurement_source_sha256']
    path = tmp_path / 'review.json'
    write_new(path, review)
    assert load_compatibility(path)['manifest_binding'] == binding(path)


@pytest.mark.parametrize('name,before,after', [
    (POWER, 'total += (prev_p + p) * 0.5 * dt', 'total += (prev_p + p) * 0.4 * dt'),
    (POWER, 'self._stop.wait(self.interval)', 'self._stop.wait(self.interval * 2)'),
    (BACKENDS, 'watts=mw/1000.', 'watts=mw/2000.'),
    (BACKENDS, "'temperature_c': lambda: float(self.temperature_c(gpu)),", "'temperature_c': lambda: 0.,"),
    (WHOLE_FILES[0], '', '# unrelated energy edit\n'),
])
def test_energy_cadence_core_whole_file_and_unreviewed_telemetry_changes_are_refused(sources, tmp_path, name, before, after):
    left, _, texts = sources
    changed = dict(texts)
    assert before in changed[name]
    changed[name] = after + changed[name] if not before else changed[name].replace(before, after, 1)
    right = freeze(tmp_path / 'changed', changed)
    with pytest.raises(ValueError, match='changed|unreviewed'):
        prepare_compatibility(left, right)


def test_claimed_compatibility_is_recomputed_not_trusted(sources, tmp_path):
    left, right, _ = sources
    review = prepare_compatibility(left, right)
    review['sources'][1]['measurement_source_sha256'] = 'forged'
    path = tmp_path / 'forged.json'
    write_new(path, review)
    with pytest.raises(ValueError, match='independently recomputed'):
        load_compatibility(path)
    Path(right['path']).parent.joinpath(POWER).write_text('changed source')
    with pytest.raises(ValueError, match='checksum'):
        prepare_compatibility(left, right)


def paired(sources, tmp_path):
    left, right, _ = sources
    path = tmp_path / 'compatible.json'
    write_new(path, prepare_compatibility(left, right))
    review = load_compatibility(path)
    candidate, baselines = point('pdblend', energy=80.), baseline_rows()
    for row, endpoint in [(candidate, review['sources'][1]), *[(r, review['sources'][0]) for r in baselines.values()]]:
        row['comparison_identity']['measurement_source_sha256'] = endpoint['measurement_source_sha256']
        row.update(evidence_source_manifest=endpoint['source_manifest'], raw_eight_gpu_meter_qualified=True,
                   measurement_evidence_binding={'receipt': 'hash-bound-test-evidence'})
    return candidate, baselines, (review,)


def test_cross_hash_requires_explicit_review_but_failed_baseline_clocks_remain_a_target(sources, tmp_path):
    candidate, baselines, reviews = paired(sources, tmp_path)
    baselines['ecoserve']['measurement_evidence_valid'] = False  # Frequency failed; meter passed.
    assert compare_point(candidate, baselines)['status'] == 'incomplete'
    result = compare_point(candidate, baselines, measurement_compatibility=reviews)
    assert result['observed_goal_met'] is True
    assert result['comparison_measurement_qualification'] == 'fail'
    assert result['baselines']['ecoserve']['measurement_compatibility'] == reviews[0]['manifest_binding']
    assert result['comparison_formal_eligible'] is False


@pytest.mark.parametrize('fault', ['raw_gate', 'binding', 'energy', 'source'])
def test_review_does_not_waive_raw_meter_complete_energy_or_exact_source_binding(sources, tmp_path, fault):
    candidate, baselines, reviews = paired(sources, tmp_path)
    target = baselines['ecoserve']
    if fault == 'raw_gate': target['raw_eight_gpu_meter_qualified'] = False
    if fault == 'binding': target.pop('measurement_evidence_binding')
    if fault == 'energy': target['tail_energy_kj'] = None
    if fault == 'source': target['evidence_source_manifest'] = {'path': 'other', 'sha256': 'other'}
    result = compare_point(candidate, baselines, measurement_compatibility=reviews)
    assert result['status'] == 'incomplete' and result['observed_goal_met'] is None


def test_candidate_frequency_failure_cannot_become_win_even_with_good_energy(sources, tmp_path):
    candidate, baselines, reviews = paired(sources, tmp_path)
    candidate['measurement_evidence_valid'] = False
    result = compare_point(candidate, baselines, measurement_compatibility=reviews)
    assert result['numerical_goal_met'] is True and result['observed_goal_met'] is False
    assert result['status'] == 'failed'


def test_receipt_hydration_requires_hash_bound_point_and_preserves_frequency_failure(sources, tmp_path):
    left, _, _ = sources
    review = prepare_compatibility(left, left)
    endpoint = review['sources'][0]
    row = point('pdblend', energy=80., revision=endpoint['source_sha256'])
    row['comparison_identity']['measurement_source_sha256'] = endpoint['measurement_source_sha256']
    point_path = tmp_path / 'window' / 'point.json'
    original = dict(name=row['point_id'], source_manifest=left,
                    engine_identity=dict(measurement_source_sha256=endpoint['measurement_source_sha256']))
    write_new(point_path, original)
    receipt_path = point_path.parent / 'receipt.json'
    write_new(receipt_path, dict(point_sha256=digest(original), artifacts={'point.json': binding(point_path)['sha256']},
        result=dict(measurement_evidence_valid=False, acceptance=dict(
            checked_gates=['metering.raw_eight_gpu_window'], missing_gates=['pdblend.physical_clocks'],
            gate_failures={'pdblend.physical_clocks': 'throttled'}))))
    row.update(receipt_path=str(receipt_path), receipt_sha256=binding(receipt_path)['sha256'])
    row['measurement_qualified'] = True  # A stale snapshot cannot override raw evidence.
    hydrated = hydrate_receipt_evidence(row)
    assert hydrated['raw_eight_gpu_meter_qualified'] is True
    assert hydrated['measurement_evidence_valid'] is False
    assert 'measurement_qualified' not in hydrated
    bad = deepcopy(row); bad['receipt_sha256'] = 'wrong'
    with pytest.raises(ValueError): hydrate_receipt_evidence(bad)
    bad = deepcopy(row); bad.pop('receipt_sha256')
    assert 'raw_eight_gpu_meter_qualified' not in hydrate_receipt_evidence(bad)
    bad.update(raw_eight_gpu_meter_qualified=True, measurement_evidence_binding={'fake': 'binding'})
    assert 'raw_eight_gpu_meter_qualified' not in hydrate_receipt_evidence(bad)
