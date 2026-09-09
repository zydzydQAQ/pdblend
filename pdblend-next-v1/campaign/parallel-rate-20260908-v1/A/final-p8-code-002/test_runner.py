"""CPU checks for request-failure boundaries and externally stored dynamic evidence."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location('final_p6_runner_under_test', Path(__file__).with_name('run.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize('change', [dict(work_complete=False), dict(failed_requests=1),
    dict(request_timeouts=1), dict(failed_requests=None), dict(request_timeouts=None)])
def test_any_request_failure_or_unknown_cannot_advance(change):
    summary = dict(work_complete=True, failed_requests=0, request_timeouts=0, slo_attainment=.01)
    summary.update(change)
    assert not runner.full_work(summary)


def test_full_work_low_slo_remains_observation():
    assert runner.full_work(dict(work_complete=True, failed_requests=0, request_timeouts=0, slo_attainment=.01))


def test_checkpoint_keeps_external_transition_and_inventory(tmp_path):
    output = tmp_path / 'results'
    operation = output / 'operations' / 'cell'
    operation.mkdir(parents=True)
    inventory = operation / 'inventory.final.json'
    inventory.write_text('{}')
    external = tmp_path / 'physical-runtime' / 'power.csv'
    external.parent.mkdir(); external.write_text('t_s,gpu0\n0,1\n')
    receipt = dict(measurement_valid=True, summary=dict(work_complete=True),
                   dynamic_artifacts={str(external): runner.sha(external)})
    (operation / 'receipt.json').write_text(json.dumps(receipt))
    binding = tmp_path / 'binding.json'; binding.write_text('{}')
    writes = {}
    common = SimpleNamespace(write=lambda p, v: writes.update({str(p): v}))
    cp = runner.checkpoint(common, binding, dict(cell_id='cell'), dict(cell_id='cell'), output, receipt)
    assert cp['artifacts'][str(external)] == runner.sha(external)
    assert cp['artifacts'][str(inventory)] == runner.sha(inventory)
    assert cp['receipt_sha256'] == runner.sha(operation / 'receipt.json')
    assert cp['measurement_valid'] is True
    external.write_text('changed')
    with pytest.raises(RuntimeError, match='transition artifact changed'):
        runner.checkpoint(common, binding, dict(cell_id='cell'), {}, output, receipt)


def test_failure_checkpoint_does_not_claim_valid_or_complete(tmp_path):
    operation = tmp_path / 'results' / 'operations' / 'failed'; operation.mkdir(parents=True)
    receipt = dict(measurement_valid=False, error='physical cleanup failed')
    (operation / 'receipt.json').write_text(json.dumps(receipt))
    binding = tmp_path / 'binding.json'; binding.write_text('{}')
    common = SimpleNamespace(write=lambda p, v: None)
    cp = runner.checkpoint(common, binding, dict(cell_id='failed'), {}, tmp_path / 'results', receipt)
    assert cp['measurement_valid'] is False and cp['work_complete'] is None


def test_original_workload_declaration_retains_all_rates_and_only_critical_repeats():
    declaration = runner.read(Path(__file__).with_name('work-declaration.json'))
    cells = declaration['cells']
    assert len(cells) == 36 and len({c['cell_id'] for c in cells}) == 36
    assert sum(c['repeat'] == 2 for c in cells) == 6
    assert len({c['original_cell_id'] for c in cells}) == 30
    for cell in cells:
        row = cell['source_row']
        assert cell['trace']['sha256'] == runner.sha(cell['trace']['path']) == row['trace_sha256']
        assert row['seed'] == 701 and row['arrival_window_s'] == 100
        assert row['slo_scale'] == 1 and row['system'] == 'pdblend'
        assert cell['arm'] == 'dynamic' and 'parallel-rate-p8' in cell['cell_id']
        assert set(cell['baseline_cell_ids']) == {'mixed', 'distserve', 'dynamollm', 'ecoserve'}


def test_actual_inventory_requires_complete_original_identity(tmp_path):
    a = Path(__file__).resolve().parent.parent
    spec = runner.read(a / 'load-p6-full-inputs-002/spec.json')
    base = runner.checked(spec['original_binding'])
    cap = runner.checked(spec['capacity_binding'])
    executor = runner.load(a.parent / 'hosts/14b-capacity-p6/capacity_executor.py', 'final_p6_inventory_cpu')
    ownership = runner.load(a / 'dynamic-execution-isolated-power-002/dynamic_ownership.py', 'final_p6_ownership_cpu')
    original = runner.checked(spec['config'])['instances']
    assert original == base['instances']
    valid = executor.Inventory(tmp_path / 'complete.json', original, cap['identity'])
    assert ownership.inventory(valid.path, base['instances'], identity=cap['identity'])['initial_ids'] == ['nextv3a6', 'nextv3a7']
    historical_minimal = runner.read(a / 'p4-minimal/fixed-release-001/configs/alpaca.json')['instances']
    missing = executor.Inventory(tmp_path / 'missing-container.json', historical_minimal, cap['identity'])
    with pytest.raises((RuntimeError, KeyError)):
        ownership.inventory(missing.path, base['instances'], identity=cap['identity'])
