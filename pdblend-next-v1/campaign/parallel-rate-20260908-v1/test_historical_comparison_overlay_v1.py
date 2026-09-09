"""Real historical selection and adversarial correction checks; no GPU actions."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest
import historical_comparison_overlay_v1 as correction

ROOT = Path(__file__).resolve().parent


@pytest.fixture(scope='module')
def actual():
    points = json.loads((ROOT/'reports/historical-scale-before-suffix-001/results.json').read_text())['points']
    audit = ROOT/'C/all-model-original-Dynamo90-arrival-audit-v2.json'
    sources = {}
    overlay = correction.verify(points,dict(path=str(audit),sha256=correction.sha(audit)),sources)
    return points,overlay,sources


def test_real_correction_preserves_master_and_all_raw_energy(actual):
    points,overlay,sources=actual
    before=copy.deepcopy(points)
    view=correction.comparison_view(points,overlay)
    assert points == before
    assert all(a.get('energy_j') == b.get('energy_j') for a,b in zip(points,view))
    assert len(sources) == 14
    assert sum(p['phase']=='main' and p['metrics_verified'] for p in points)==450
    assert sum(p['phase']=='main' and p['scientific_comparison_eligible'] for p in view)==448


def test_legacy_comparisons_exclude_exactly_the_two_diagnosed_executions(actual):
    points,overlay,_=actual
    spec=importlib.util.spec_from_file_location('legacy_historical_for_overlay',ROOT.parent/'five-system-results-v4/report.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    original=old.pairwise(points)
    current=old.pairwise(correction.comparison_view(points,overlay))
    changed=[b for a,b in zip(original,current) if a!=b]
    assert len(changed)==2
    assert {(p['model'],p['dataset'],p['rate_rps'],p['baseline']) for p in changed}=={
        ('32b','alpaca',4.,'dynamollm'),('32b','sharegpt',2.,'dynamollm')}
    assert all(not x['both_verified'] and x['primary_energy_ratio'] is None
               and x['j_per_good_ratio'] is None and x['slo_attainment_difference'] is None for x in changed)


def test_other_complete_low_slo_and_incomplete_observations_remain_visible(actual):
    points,overlay,_=actual
    view=correction.comparison_view(points,overlay)
    negatives=[(p,v) for p,v in zip(points,view) if p['metrics_verified']
               and p['cell_id'] not in correction.QUARANTINED and
               (not p['work_complete'] or p['slo_attainment']<.9)]
    assert negatives
    assert all(v['metrics_verified'] and v['scientific_comparison_eligible'] for p,v in negatives)


def test_raw_energy_cannot_be_silently_changed(actual):
    points,overlay,_=actual
    altered=copy.deepcopy(points)
    next(p for p in altered if p['cell_id'] in correction.QUARANTINED)['energy_j']+=1
    with pytest.raises(ValueError,match='energy or eligibility changed'):
        correction.comparison_view(altered,overlay)


def test_quarantine_cannot_be_converted_to_comparison_eligible(actual):
    points,overlay,_=actual
    altered=copy.deepcopy(overlay);altered['quarantined'][0]['scientific_comparison_eligible']=True
    with pytest.raises(ValueError,match='energy or eligibility changed'):
        correction.comparison_view(points,altered)


def test_unreviewed_additional_exclusion_is_rejected(actual):
    points,overlay,_=actual
    altered=copy.deepcopy(overlay)
    altered['quarantined'].append(dict(cell_id='unreviewed-low-slo-point'))
    with pytest.raises(ValueError,match='exact diagnosed'):
        correction.comparison_view(points,altered)


def test_erasing_a_quarantined_raw_record_is_rejected(actual):
    points,overlay,_=actual
    with pytest.raises(ValueError,match='missing quarantined'):
        correction.comparison_view([p for p in points if p['cell_id'] not in correction.QUARANTINED],overlay)


def test_changed_audit_cannot_authorize_exclusion(actual,tmp_path):
    points,_,_=actual
    changed=tmp_path/'audit.json';changed.write_text('{}')
    with pytest.raises(ValueError,match='unreviewed historical'):
        correction.verify(points,dict(path=str(changed),sha256=correction.sha(changed)),{})
