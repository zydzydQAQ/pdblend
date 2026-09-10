"""CPU-only release gates and byte-preserving assembly checks."""
import copy
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import assemble_results_v1 as a


def complete_gate_fixture():
    groups = [dict(model=m, dataset=d, node=n, complete=True, pdb_boundary_complete=True)
        for m, d, n in sorted(a.EXPECTED_GROUPS)]
    result = dict(groups=groups, complete=True, scope_complete=True, metric_audit_errors=[])
    checks = [dict(model=g['model'], dataset=g['dataset'], node=g['node'], pipeline_terminal=True)
        for g in groups]
    monitor = dict(scope_complete=True, finished_s=123.0, completion_scope='five_systems',
        groups_complete=9, metric_audit_errors=0, hydration_active=False,
        completion_checks=dict(raw_scope_complete=True, scope_complete=True,
            five_system_complete=True, supervisor_checks=checks))
    remaining = dict(total=dict(constant=0, variables={}), awaiting_evidence=[], raw_metric_audit_errors=0)
    ledger = dict(schema='uniform-setup-energy-ledger-v1', pending_or_invalid_evidence=[])
    return monitor, result, remaining, ledger


class Gates(unittest.TestCase):
    def test_only_exact_clean_nine_groups_pass(self):
        self.assertEqual(len(a.full_scope_gate(*complete_gate_fixture())), 9)

    def test_incomplete_audit_hydration_and_remaining_rejected(self):
        variants = [('monitor', 'scope_complete', False), ('monitor', 'finished_s', None),
            ('monitor', 'metric_audit_errors', 1), ('monitor', 'hydration_active', True),
            ('result', 'complete', False), ('result', 'metric_audit_errors', ['missing raw']),
            ('remaining', 'awaiting_evidence', ['unmirrored checkpoint']),
            ('remaining', 'raw_metric_audit_errors', 1),
            ('ledger', 'pending_or_invalid_evidence', ['active measurement'])]
        for category, key, value in variants:
            with self.subTest(category=category, key=key):
                values = complete_gate_fixture()
                dict(zip(('monitor', 'result', 'remaining', 'ledger'), values))[category][key] = value
                with self.assertRaises(ValueError): a.full_scope_gate(*values)
        for total in (dict(constant=1, variables={}), dict(constant=0, variables={'new_rates': 1})):
            values = complete_gate_fixture(); values[2]['total'] = total
            with self.assertRaises(ValueError): a.full_scope_gate(*values)

    def test_duplicated_group_and_wrong_host_rejected(self):
        values = complete_gate_fixture(); values[1]['groups'][1] = copy.deepcopy(values[1]['groups'][0])
        with self.assertRaises(ValueError): a.full_scope_gate(*values)
        values = complete_gate_fixture(); values[1]['groups'][0]['node'] = 'old-A'
        with self.assertRaises(ValueError): a.full_scope_gate(*values)

    def test_live_supervisor_or_partial_completion_gate_rejected(self):
        for key in ('raw_scope_complete', 'scope_complete', 'five_system_complete'):
            values = complete_gate_fixture(); values[0]['completion_checks'][key] = False
            with self.assertRaises(ValueError): a.full_scope_gate(*values)
        values = complete_gate_fixture()
        values[0]['completion_checks']['supervisor_checks'][0]['pipeline_terminal'] = False
        with self.assertRaises(ValueError): a.full_scope_gate(*values)

    def test_terminal_does_not_accept_live_or_error_states(self):
        state = dict(complete=True, finished_s=123, node_lease_held=False)
        self.assertTrue(a.clean_terminal(state))
        for key, value in [('complete', False), ('finished_s', None), ('node_lease_held', True), ('error', 'fault')]:
            self.assertFalse(a.clean_terminal(dict(state, **{key: value})))

    def test_all9_gate_fails_before_creating_any_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'current').mkdir()
            monitor, result, remaining, ledger = complete_gate_fixture(); monitor['scope_complete'] = False
            a.save(root / 'monitor.json', monitor)
            a.save(root / 'current/results.json', result)
            a.save(root / 'current/remaining-work.json', remaining)
            a.save(root / 'current/setup-energy-ledger.json', ledger)
            with self.assertRaises(ValueError):
                a.assemble(scope='all9', destination=root / 'destination', current=root / 'current',
                    monitor_path=root / 'monitor.json', sealed=root / 'no-input-needed')
            self.assertFalse((root / 'destination').exists())
            self.assertFalse((root / 'destination.zip').exists())
            self.assertEqual(list(root.glob('.destination-*')), [])


class Files(unittest.TestCase):
    def test_safe_archive_paths(self):
        for path in ('../outside', '/absolute', '.', 'safe/../../outside'):
            with self.assertRaises(ValueError): a.safe_member(path)
        self.assertEqual(a.safe_member('group/summary.csv'), Path('group/summary.csv'))

    def test_same_fd_read_rejects_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); target = root / 'state'; new = root / 'new'
            target.write_text('old'); new.write_text('new')
            original_stat = Path.stat
            def replacing_stat(path, *args, **kwargs):
                if path == target and new.exists(): os.replace(new, target)
                return original_stat(path, *args, **kwargs)
            with mock.patch.object(Path, 'stat', replacing_stat):
                with self.assertRaisesRegex(ValueError, 'changed during snapshot'): a.snapshot(target)

    def test_aggregate_signature_detects_value_and_duplicate_changes(self):
        row = dict(model='7b', dataset='alpaca', system='mixed', rate_rps='3', measurement_host='C')
        row.update({metric + suffix: '1' for metric in a.METRICS for suffix in ('_min', '_mean', '_max', '_n')})
        changed = dict(row, energy_j_mean='2')
        self.assertNotEqual(a.aggregate_signature([row]), a.aggregate_signature([changed]))
        with self.assertRaises(ValueError): a.aggregate_signature([row, row])


class ActualHostProbe(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.status = self.root / 'status.json'; self.lock = self.root / 'lease.lock'; self.lock.touch()
        self.state = dict(complete=True, finished_s=123, node_lease_held=False, pid=99999999, startticks='1')
        self.refresh()

    def refresh(self):
        a.save(self.status, self.state)
        self.request = dict(node='test', expected_hostname=socket.gethostname(),
            terminals=[a.snapshot(self.status)[1]], locks=[str(self.lock)])

    def probe(self):
        return subprocess.run([sys.executable, '-B', '-'], input=a.terminal_probe_script(self.request),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)

    def test_clean_actual_owner_and_lease_pass_without_mutating_lock(self):
        before = a.identity(self.lock.stat()); output = self.probe()
        self.assertEqual(output.returncode, 0, output.stderr)
        proof = json.loads(output.stdout)
        self.assertTrue(proof['passed']); self.assertTrue(proof['locks'][0]['probe_released'])
        self.assertEqual(before, a.identity(self.lock.stat()))

    def test_held_lease_rejected(self):
        with self.lock.open('r+') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try: self.assertNotEqual(self.probe().returncode, 0)
            finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def test_live_owner_changed_terminal_and_wrong_hostname_rejected(self):
        self.state['pid'] = os.getpid()
        self.state['startticks'] = Path('/proc/self/stat').read_text().rpartition(')')[2].split()[19]
        self.refresh(); self.assertNotEqual(self.probe().returncode, 0)
        self.state['pid'] = 99999999; self.refresh()
        self.status.write_text(self.status.read_text() + ' ')
        self.assertNotEqual(self.probe().returncode, 0)
        self.refresh(); self.request['expected_hostname'] = 'other-physical-node'
        self.assertNotEqual(self.probe().returncode, 0)


@unittest.skipUnless((a.SEALED / '7b-longbench/manifest.json').exists(), 'completed 7B fixture unavailable')
class RealSealedGroups(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.alpaca = a.SEALED / '7b-alpaca'
        cls.result = json.loads((cls.alpaca / 'results.json').read_text())
        cls.rows = list(csv.DictReader(io.StringIO((cls.alpaca / 'summary.csv').read_text())))
        cls.terminal = a.ROOT / 'C/uniform-rate-20260909-v2/pipeline-004/status.json'
        cls.proof = a.ROOT / 'C/uniform-rate-20260909-v2/clean-terminal-verification-001.json'

    def test_all_three_real_sealed_groups_and_zip_members(self):
        groups = [a.verify_group(a.SEALED / ('7b-' + dataset)) for dataset in a.DATASETS]
        self.assertEqual(sum(g['checks'][0]['coordinates'] for g in groups), 145)
        self.assertEqual(sum(g['checks'][0]['observed_normal_repeats'] for g in groups), 172)

    def test_missing_rate_or_metric_rejected(self):
        with self.assertRaises(ValueError): a.validate_science(self.result, self.rows[:-1])
        rows = copy.deepcopy(self.rows); rows[0]['generated_token_throughput_tps_mean'] = 'nan'
        with self.assertRaises(ValueError): a.validate_science(self.result, rows)

    def test_cross_host_trace_and_earlier_boundary_rejected(self):
        for key, value in [('measurement_host', 'B'), ('trace_sha256', '0' * 64)]:
            result = copy.deepcopy(self.result); result['observations'][0][key] = value
            with self.assertRaises(ValueError): a.validate_science(result, self.rows)
        result = copy.deepcopy(self.result)
        early = next(o for o in result['observations'] if o['system'] == 'pdblend' and o['rate_rps'] < 18)
        early['slo_attainment'] = .89
        with self.assertRaises(ValueError): a.validate_science(result, self.rows)

    def test_native_queue_rejection_cannot_be_relabelled_timeout(self):
        result = copy.deepcopy(self.result)
        observed = next(o for o in result['observations'] if o.get('independent_reclassification_only'))
        observed['request_timeouts'] = 170
        with self.assertRaises(ValueError): a.validate_science(result, self.rows)

    def test_changed_sealed_file_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / self.alpaca.name
            shutil.copytree(self.alpaca, target)
            (target / 'summary.csv').write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'file changed'): a.verify_group(target)

    def test_staged_copy_corruption_cannot_be_published(self):
        original_copytree = shutil.copytree
        def corrupt_copy(source, destination, *args, **kwargs):
            result = original_copytree(source, destination, *args, **kwargs)
            if (Path(destination) / 'manifest.json').exists():
                (Path(destination) / 'summary.csv').write_text('corrupted during copy\n')
            return result
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'combined'
            with mock.patch.object(shutil, 'copytree', corrupt_copy):
                with self.assertRaisesRegex(ValueError, 'copied sealed member differs'):
                    a.assemble(scope='7b', destination=destination, terminal_path=self.terminal)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix('.zip').exists())

    def test_merge_preserves_every_group_byte_and_relative_link(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'combined'
            output = a.assemble(scope='7b', destination=destination,
                terminal_path=self.terminal, terminal_proof_path=self.proof)
            self.assertEqual(output['scientific_coordinates'], 145)
            for dataset in a.DATASETS:
                name = '7b-' + dataset
                for source in (a.SEALED / name).rglob('*'):
                    if source.is_file():
                        self.assertEqual(source.read_bytes(), (destination / name / source.relative_to(a.SEALED / name)).read_bytes())
            for document in destination.rglob('*.md'):
                for link in re.findall(r'\]\(([^)]+)\)', document.read_text()):
                    if '://' not in link and not link.startswith(('/', '#')):
                        self.assertTrue((document.parent / link.split('#')[0]).exists(), str(document) + ': ' + link)
            manifest = json.loads((destination / 'manifest.json').read_text())
            with zipfile.ZipFile(output['archive']) as archive:
                for name, digest in manifest['files'].items():
                    self.assertEqual(a.hashlib.sha256(archive.read('combined/' + name)).hexdigest(), digest)
            with self.assertRaises(ValueError):
                a.assemble(scope='7b', destination=destination, terminal_path=self.terminal)


if __name__ == '__main__':
    unittest.main()
