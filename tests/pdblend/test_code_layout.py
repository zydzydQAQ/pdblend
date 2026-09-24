"""Compatibility contracts for the responsibility-based module layout."""
import importlib
import json
import pickle
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('old,new', [
    ('pdblend.profile.model', 'pdblend.profile.query.model'),
    ('pdblend.profile.power_table', 'pdblend.profile.query.power_table'),
    ('pdblend.profile.long_domain', 'pdblend.profile.query.long_domain'),
    ('pdblend.profile.profiler', 'pdblend.profile.collection.profiler'),
    ('pdblend.profile.calibration', 'pdblend.profile.calibration.core'),
    ('pdblend.profile.power_calibration', 'pdblend.profile.calibration.power_calibration'),
    ('pdblend.control.planner', 'pdblend.planner.pool'),
    ('pdblend.control.controller', 'pdblend.online.controller'),
    ('pdblend.control.policies.baselines', 'pdblend.legacy.baselines'),
    ('pdblend.proxy.router', 'pdblend.online.router'),
    ('pdblend.bench.tp_runtime', 'pdblend.online.tp_runtime'),
])
def test_old_module_is_canonical_object(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)


def test_old_path_monkeypatch_changes_function_globals(monkeypatch):
    old = importlib.import_module('pdblend.profile.calibration')
    new = importlib.import_module('pdblend.profile.calibration.core')
    replacement = object()
    monkeypatch.setattr(old, 'PerfModel', replacement)
    assert new.prepare_candidate.__globals__['PerfModel'] is replacement


def test_historical_pickle_globals_and_dataset_definitions():
    # Protocol-0 GLOBAL opcodes model the qualified names written before the move.
    from pdblend.profile.query.model import PerfModel, StaticState
    from pdblend.measure.datasets import Dataset
    assert pickle.loads(b'cpdblend.profile.model\nPerfModel\n.') is PerfModel
    assert pickle.loads(b'cpdblend.profile.model\nStaticState\n.') is StaticState
    assert pickle.loads(b'cpdblend.measure.datasets\nDataset\n.') is Dataset
    assert Dataset.__module__ == 'pdblend.measure.datasets'


def test_queries_import_without_sampling_or_fitting():
    script = '''
import pdblend.profile.query.model, pdblend.profile.query.versions
import pdblend.profile.query.long_domain
import sys
assert not any(n.startswith(('pdblend.profile.collection', 'pdblend.profile.calibration')) for n in sys.modules)
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
'''
    subprocess.run([sys.executable, '-B', '-c', script], check=True)


def test_new_source_inventory_binds_canonical_and_compatibility(tmp_path):
    from pdblend.source_inventory import implementation_hashes, verify_implementation
    package = tmp_path / 'pdblend'
    (package / 'profile/query').mkdir(parents=True)
    (package / 'profile/model.py').write_text('# compatibility\n')
    actual = package / 'profile/query/model.py'
    actual.write_text('# actual implementation\n')
    manifest = implementation_hashes(tmp_path)
    assert set(manifest) == {'pdblend/profile/model.py', 'pdblend/profile/query/model.py'}
    verify_implementation(manifest, root=tmp_path)
    actual.write_text('# changed implementation\n')
    with pytest.raises(ValueError, match='source implementation changed'):
        verify_implementation(manifest, root=tmp_path)


def test_old_calibration_module_cli():
    result = subprocess.run([sys.executable, '-B', '-m', 'pdblend.profile.calibration', '--help'],
                            check=True, capture_output=True, text=True)
    assert '--candidate-dir' in result.stdout


def test_calibration_package_child_import():
    import pdblend.profile.calibration.core as direct
    from pdblend.profile.calibration import core
    assert core is direct


def test_calibration_child_identity_when_collector_imports_first():
    script='''
from pdblend.profile.collection import short_domain_collect as collector
from pdblend.profile import short_domain as legacy
import sys
assert collector.sd is legacy
assert collector.sd.ShortClockMismatch is legacy.ShortClockMismatch
assert 'pdblend.profile.calibration.core.short_domain' not in sys.modules
'''
    subprocess.run([sys.executable,'-B','-c',script],check=True)
