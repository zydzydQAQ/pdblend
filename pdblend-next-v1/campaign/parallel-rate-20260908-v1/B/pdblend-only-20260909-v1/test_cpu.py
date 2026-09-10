"""CPU fixtures only: replay saved qualification, never call live restoration."""
import copy
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pdb_only_support as s


def load_local(name):
    spec = importlib.util.spec_from_file_location('B_pdb_only_test_' + name, s.HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CPU(unittest.TestCase):
    def test_retired_process_and_cleanup_fail_closed(self):
        with patch.object(s.p, 'active_owner', return_value=True):
            with self.assertRaisesRegex(AssertionError, 'still running'):
                s.retired_evidence(probe_processes=True)
        original = s.p.checked
        def broken(reference):
            result = original(reference)
            if isinstance(result, dict) and 'outer_cleanup_errors' in result:
                result = copy.deepcopy(result)
                result['clock_restore_complete'] = False
            return result
        with patch.object(s.p, 'checked', side_effect=broken):
            with self.assertRaises(AssertionError):
                s.retired_evidence()

    def test_only_pdb_loop_stops_at_each_cap(self):
        module = load_local('pipeline')
        obj = object.__new__(module.Pipeline)
        obj.state = dict(boundaries={})
        seen, selected = [], {}
        obj.save = lambda: None
        obj.guard = lambda: None
        obj.phase = lambda name: seen.append(('phase', name))
        obj.run = lambda name, argv: seen.append(('command', name))
        obj.measure = lambda ds, rate, system, tasks: seen.append(('measurement', ds, system))
        def select(ds):
            selected[ds] = selected.get(ds, 0) + 1
            return (dict(phase='pdblend', rate_rps=.5, next_tasks=[])
                if selected[ds] == 1 else dict(phase='baselines', cap_rate_rps=.5))
        obj.select = select
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, 'HERE', Path(tmp)), \
             patch.object(module, 'retired_evidence', return_value={}), patch.object(module, 'pdb_reuse', return_value=[]):
            obj.path = Path(tmp) / 'status.json'
            obj.execute()
        self.assertEqual(obj.state['phase'], 'all_B_PDB_groups_capped')
        self.assertFalse(obj.state['baseline_suite_complete'])
        self.assertEqual([x[1] for x in seen if x[0] == 'phase'], ['pdb-restore', 'pdb-spec', 'pdb-freeze'])
        self.assertEqual([x[2] for x in seen if x[0] == 'measurement'], ['pdblend'] * 3)
        with self.assertRaises(AssertionError):
            module.Pipeline.measure(obj, 'alpaca', .5, 'ecoserve', [])

    def test_saved_fixture_full_spec_freeze_release_validation(self):
        module = load_local('prepare_node')
        old_spec = s.p.read(s.OLD / 'qualification-spec-001.json')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copyfile(s.HERE / 'manifest.json', root / 'manifest.json')
            restore = root / 'pdb-restoration-001'
            s.p.save(restore / 'status.json', s.p.checked(old_spec['restoration']))
            s.p.save(restore / 'binding.json', s.p.checked(old_spec['binding']))
            s.p.save(restore / 'correctness/status.json', s.p.checked(old_spec['ordinary']))
            # Recorded qualification/raw clocks/requests are copied into a test-only
            # namespace. Only its spec reference changes to the fixture spec.
            shutil.copytree(s.OLD / 'qualification-001', root / 'pdb-qualification-001')
            with patch.object(module, 'HERE', root):
                spec_path = Path(module.run_phase('pdb-spec'))
                self.assertEqual(spec_path, root / 'pdb-qualification-spec.json')
                status_path = root / 'pdb-qualification-001/status.json'
                status = s.p.read(status_path)
                status['spec'] = s.p.ref(spec_path)
                s.p.save(status_path, status)
                entry = module.run_phase('pdb-freeze')
            prepare = s.p.load(s.RUNNER / 'prepare_release.py', 'B_pdb_only_fixture_release')
            release = prepare.prepare(declaration=s.source_contract()['declaration'],
                qualification=entry['qualification'], qualification_validator=entry['qualification_validator'],
                node='B', model='32b', dataset='alpaca', rate=.5, system='pdblend',
                out=root / 'test-release', predecessors=entry['predecessors'], stop_paths=[root / 'STOP'])
            runner = s.p.load(s.RUNNER / 'run_cells.py', 'B_pdb_only_fixture_loader')
            frozen, binding, _ = runner.load_release(release)
            self.assertEqual([r['repeat'] for r in frozen['rows']], [1, 2])
            self.assertEqual(binding['system'], 'pdblend')
            self.assertIn(str(root / 'manifest.json'), frozen['files'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
