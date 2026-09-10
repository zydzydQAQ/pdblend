"""CPU tests for the recovery barriers that protect immutable evidence."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import adapters
import cold_restore
from docker_equivalence import hostconfig_equivalence
import resume


class RecoveryBarriers(unittest.TestCase):
    def setUp(self):
        self.parent = adapters.p.read(adapters.control.ORIGINAL)
        self.old = adapters.p.read(self.parent['identity_file'])
        self.stopped = copy.deepcopy(self.old)
        for c in self.stopped:
            c['State'].update(Running=False, Pid=0, Paused=False, Restarting=False)

    def test_same_retained_stopped_containers_allowed(self):
        self.assertTrue(cold_restore.validate_inventory(self.parent, self.stopped, self.old))

    def test_running_target_rejected(self):
        self.stopped[0]['State']['Running'] = True
        with self.assertRaisesRegex(RuntimeError, 'unexpectedly already running'):
            cold_restore.validate_inventory(self.parent, self.stopped, self.old)

    def test_only_explicit_empty_dns_representations_equivalent(self):
        before = copy.deepcopy(self.stopped)
        for c in self.stopped:
            c['HostConfig']['Dns'] = []
        proof = cold_restore.validate_inventory(self.parent, self.stopped, self.old)
        self.assertEqual(len(proof), 4)
        self.assertTrue(all(row['differing_fields'] == ['Dns'] for row in proof))
        self.assertTrue(all(c['HostConfig']['Dns'] is None for c in before))
        self.assertTrue(all(c['HostConfig']['Dns'] is None for c in self.old))
        left, right = {'Dns': None, 'Memory': 5}, {'Dns': [], 'Memory': 5}
        self.assertTrue(hostconfig_equivalence(left, right)['equivalent'])
        self.assertTrue(hostconfig_equivalence(right, left)['equivalent'])
        self.assertIsNone(left['Dns'])

    def test_nonempty_missing_dns_and_other_hostconfig_changes_rejected(self):
        for changed in ({'Dns': ['8.8.8.8'], 'Memory': 5}, {'Memory': 5},
                        {'Dns': [], 'Memory': 6}, {'Dns': [], 'Memory': 5, 'NewKey': None}):
            with self.assertRaises(RuntimeError):
                hostconfig_equivalence({'Dns': None, 'Memory': 5}, changed)
        for change in ('dns', 'memory'):
            actual = copy.deepcopy(self.stopped)
            if change == 'dns':
                actual[0]['HostConfig']['Dns'] = ['8.8.8.8']
            else:
                actual[0]['HostConfig']['Memory'] += 1
            with self.assertRaises(RuntimeError):
                cold_restore.validate_inventory(self.parent, actual, self.old)

    def test_independent_eco_qualification_uses_same_scoped_rule(self):
        eco = adapters.g.load(adapters.HERE / 'qualify_eco_drained_baseline_v2.py', 'B32B_test_dns_qualifier')
        self.assertIs(eco.hostconfig_equivalence, hostconfig_equivalence)
        self.assertIs(cold_restore.hostconfig_equivalence, hostconfig_equivalence)

    def test_changed_image_or_mount_rejected(self):
        for key in ('image', 'mount'):
            inventory = copy.deepcopy(self.stopped)
            if key == 'image':
                inventory[0]['Image'] = 'sha256:changed'
            else:
                inventory[0]['Mounts'][0]['Source'] = '/wrong-model'
            with self.assertRaises(RuntimeError):
                cold_restore.validate_inventory(self.parent, inventory, self.old)

    def test_live_old_owner_rejected(self):
        read_bytes = Path.read_bytes
        with patch.object(Path, 'read_bytes', autospec=True, side_effect=lambda path:
                b'/sbin/init\0' if str(path) == '/proc/1/cmdline' else read_bytes(path)), \
                patch.object(adapters.control.r, 'alive', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'still alive'):
                cold_restore.host_processes()

    def test_sandbox_pid_namespace_rejected(self):
        with patch.object(Path, 'read_bytes', return_value=b'codex-linux-sandbox\0'):
            with self.assertRaisesRegex(RuntimeError, 'host PID namespace'):
                cold_restore.host_processes()

    def test_existing_attempt_or_checkpoint_never_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(resume, 'HERE', root), patch.object(resume, 'PIPELINE', root / 'pipeline-001'), \
                    patch.object(adapters.control, 'RESTORE', root / 'restore'), \
                    patch.object(adapters.control, 'QUAL', root / 'qual'):
                resume.fresh_outputs_required()
                perf = root / 'baseline-mixed-performance-001'
                perf.mkdir()
                (perf / 'failed-checkpoint.json').write_text('{}')
                with self.assertRaisesRegex(AssertionError, 'never automatically retry'):
                    resume.fresh_outputs_required()

    def test_output_adapters_do_not_mutate_historical_modules(self):
        self.assertEqual(adapters.historical.HERE, adapters.OLD)
        self.assertEqual(adapters.historical.RESTORE.parent, adapters.OLD)
        self.assertEqual(adapters.control.HERE, adapters.HERE)
        self.assertIs(adapters.control.terminal, adapters.historical.terminal)
        self.assertIs(adapters.verifier.b, adapters.control)

    def test_freeze_wrapper_and_loader_use_original_pdb_predecessor(self):
        """Exercise actual freeze + wrapper + loader; native gate is a test double.

        Genuine native qualification requires new GPU requests, so that one
        boundary is doubled here, then checked to reject the fixture unmocked.
        """
        prepare = adapters.g.load(adapters.HERE / 'prepare_baselines_v2.py', 'B32B_test_freeze')
        boundary, _ = adapters.historical.terminal()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qual = root / 'qualification'
            qual.mkdir()
            refs = {}
            for system, reference in adapters.control.OLD_BINDINGS.items():
                path = qual / (system + '.json')
                shutil.copyfile(reference, path)
                refs[system] = adapters.p.ref(path)
            (qual / 'bindings.json').write_text(json.dumps(refs))
            (root / 'baseline-boundary-001.json').write_text(json.dumps(boundary))
            for name in ('prepare_baselines_v2.py', 'baseline_control_v2.py', 'verify_baseline_v2.py', 'run_cells_v1.py'):
                shutil.copyfile(adapters.HERE / name, root / name)

            calls = []
            def qualified_fixture(reference):
                saved = adapters.p.checked(reference)
                self.assertEqual(saved['schema'], 'B32B-ascending-baseline-saved-qualification-v1')
                executed = adapters.p.checked(saved['executed_binding'])
                self.assertEqual(executed['system'], saved['system'])
                calls.append(saved['system'])
                return dict(passed=True, independently_recomputed=True, node='B', system=saved['system'],
                            binding=saved['executed_binding'], host_manifest=saved['host_manifest'])

            with patch.object(prepare, 'HERE', root), patch.object(adapters.control, 'QUAL', qual):
                with patch.object(adapters.verifier, 'verify', side_effect=qualified_fixture):
                    releases = prepare.prepare()
                    self.assertEqual(set(releases), set(resume.SYSTEMS))
                    for system, reference in releases.items():
                        release, binding, _ = adapters.cells.load_release(reference)
                        self.assertEqual(release['predecessors'], [adapters.p.ref(adapters.OLD / 'pdb-performance-001/status.json')])
                        self.assertEqual(len(release['rows']), 2)
                        self.assertEqual({row['system'] for row in release['rows']}, {system})
                        self.assertEqual(Path(binding['output']).parent.parent, root)
                    self.assertEqual(len(calls), 12)
                first = adapters.p.checked(releases['mixed'])
                with self.assertRaises(FileNotFoundError):
                    adapters.verifier.verify(first['qualification'])


if __name__ == '__main__':
    unittest.main()
