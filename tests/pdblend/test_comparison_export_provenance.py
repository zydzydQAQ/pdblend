import csv
from copy import deepcopy
import json

import pytest

from pdblend.bench import comparison_campaign as campaign
from pdblend.bench.comparison_export_provenance import energy_supplement_columns, load_baseline_inventory
from pdblend.bench.comparison_recorded import POLICY
from test_comparison_recorded import exported_fixture, write


def fixture_point():
    return dict(name='7b-distserve-alpaca-x1-seed701', system='distserve',
                model_id='Qwen2.5-7B-Instruct', dataset='alpaca', scale=1., seed=701,
                duration_s=150., slo=dict(ttft_s=1., tpot_s=.1),
                trace=dict(path='/frozen/original-trace.json', sha256='t'*64))


def test_real_gap_is_classified_without_changing_original_metadata_or_numeric_types():
    original = fixture_point()
    receipt = dict(path='/frozen/original-receipt.json', sha256='r'*64)
    current = dict(original, energy_supplement_of=receipt, duration_s=150)
    saved = deepcopy(current)
    result = energy_supplement_columns(current, {original['name']: dict(point=original, receipt=receipt)})
    assert result['is_energy_supplement']
    assert result['energy_supplement_classification'] == 'frozen_inventory_gap_supplement'
    assert result['energy_supplement_of'] == result['original_energy_supplement_of'] == receipt
    assert current == saved and type(current['duration_s']) is int


@pytest.mark.parametrize('change', [dict(name='7b-distserve-alpaca-x1.25-seed701'),
    dict(scale=1.25), dict(seed=702), dict(duration_s=151.), dict(dataset='sharegpt'),
    dict(trace=dict(path='/frozen/other-trace.json', sha256='x'*64)),
    dict(slo=dict(ttft_s=5., tpot_s=.1)), dict(scale=True),
    dict(energy_supplement_of=dict(path='/other/receipt.json', sha256='q'*64))])
def test_inherited_or_mismatched_gap_tag_is_retained_only_as_original_metadata(change):
    original = fixture_point()
    receipt = dict(path='/frozen/original-receipt.json', sha256='r'*64)
    current = dict(original, energy_supplement_of=receipt)
    current.update(change)
    result = energy_supplement_columns(current, {original['name']: dict(point=original, receipt=receipt)})
    assert not result['is_energy_supplement'] and result['energy_supplement_of'] == {}
    assert result['original_energy_supplement_of'] == current['energy_supplement_of']


def test_missing_inventory_never_promotes_an_unverified_tag():
    point = dict(fixture_point(), energy_supplement_of={'path':'unknown','sha256':'r'*64})
    result = energy_supplement_columns(point)
    assert not result['is_energy_supplement'] and result['energy_supplement_of'] == {}
    assert result['energy_supplement_classification'] == 'unverified_missing_frozen_inventory'


def test_export_uses_bound_gap_identity_and_preserves_historical_attempt(tmp_path):
    session, source, out = exported_fixture(tmp_path)
    spec = json.loads(source.read_text())
    original = spec['points'][0]
    window = session/'windows/mixed'
    receipt_path = window/'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['result']['metrics']['energy_service_j'] = None
    write(window/'result.json', receipt['result'])
    receipt['artifacts']['result.json'] = campaign.file_sha(window/'result.json')
    write(receipt_path, receipt)
    receipt_ref = dict(path=str(receipt_path), sha256=campaign.file_sha(receipt_path))
    inventory = dict(schema='pdblend-baseline-energy-gap-inventory/v1', frozen_baselines=[], gaps=[dict(
        point_id=original['name'], point_sha256=campaign.digest(original),
        point=dict(path=str(window/'point.json'), sha256=campaign.file_sha(window/'point.json')),
        receipt=receipt_ref)])
    inventory_path = tmp_path/'gap-inventory.json'; write(inventory_path, inventory)
    inventory_ref = dict(path=str(inventory_path), sha256=campaign.file_sha(inventory_path))
    gap = dict(original, revision='v2', energy_supplement_of=receipt_ref)
    boundary = dict(gap, name='mixed-boundary', scale=1.25,
                    trace=dict(path='/new/trace', sha256='e'*64), comparison_scope='boundary_full_group')
    spec['points'][0] = gap
    spec['points'].append(boundary)
    spec['baseline_comparison_policy'] = dict(frozen_baselines=inventory_ref)
    write(source, spec)
    original_bytes = receipt_path.read_bytes()
    loaded = load_baseline_inventory(inventory_ref, load_bound=campaign.load_bound)
    assert loaded['frozen_baselines'] == {} and list(loaded['gaps']) == ['mixed']
    campaign.export(source, out, session_roots=[session], analysis_policy=POLICY)
    rows = list(csv.DictReader(out.open()))
    historical = next(r for r in rows if r['point_id']=='mixed' and r['revision']=='v1')
    supplement = next(r for r in rows if r['point_id']=='mixed' and r['revision']=='v2')
    endpoint = next(r for r in rows if r['point_id']=='mixed-boundary')
    assert historical['is_energy_supplement'] == 'False'
    assert supplement['is_energy_supplement'] == 'True'
    assert json.loads(supplement['energy_supplement_of']) == receipt_ref
    assert endpoint['is_energy_supplement'] == 'False'
    assert json.loads(endpoint['energy_supplement_of']) == {}
    assert json.loads(endpoint['original_energy_supplement_of']) == receipt_ref
    assert receipt_path.read_bytes() == original_bytes
    assert historical['receipt_sha256'] == receipt_ref['sha256']
    assert len([r for r in rows if r['receipt_path'] == str(receipt_path)]) == 1
