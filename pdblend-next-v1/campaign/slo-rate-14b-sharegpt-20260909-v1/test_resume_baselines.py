"""CPU-only boundary tests; original.one is replaced before any execution test."""
import asyncio
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import resume_baselines as r
p = r.p


class ResumeBaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patch_here = patch.object(p, 'HERE', self.root)
        self.patch_here.start()
        self.node = self.root / 'A'
        self.node.mkdir()
        self.observations = []
        for rate, repeat, quality in (('.25', 1, 1.), ('.5', 1, .95), ('.75', 1, .92),
                                      ('1', 1, .84), ('1', 2, .85)):
            self.observations.append(self.observation(r.contract.make_row('A', rate, 'pdblend', repeat), quality))
        p.save(self.node / 'observations.json', self.observations)
        self.dead = dict(pid=999999999, startticks='0', finished_s=1., node_lease_held=False)
        self.last = dict(self.dead, complete=True, cleanup_complete=True)
        p.save(self.root / 'last.json', self.last)
        self.boundary = dict(node='A', observation_values=self.observations,
            decision=r.contract.evaluate_group('A', self.observations),
            last_cell_status=p.ref(self.root / 'last.json'))
        p.save(self.root / 'boundary.json', self.boundary)
        self.previous = dict(self.dead, node='A', phase='engineering_diagnosis',
            error="ValueError('baseline preparation failed')", last_cell_status=p.ref(self.root / 'last.json'),
            pdb_boundary=p.ref(self.root / 'boundary.json'))
        p.save(self.root / 'previous.json', self.previous)
        self.failed = dict(self.dead, node='A', phase='failed')
        p.save(self.root / 'failed.json', self.failed)
        self.repair = dict(node='A', implementation_unchanged=True, diagnosis='CPU interface fixture',
            failed_setup_status=p.ref(self.root / 'failed.json'))
        p.save(self.root / 'repair.json', self.repair)
        self.ready = {}
        for system in r.contract.SYSTEMS[1:]:
            handoff = {}
            for key in ('qualification', 'qualification_validator', 'binding', 'measurement_executor'):
                target = self.root / f'{system}-{key}.json'
                p.save(target, dict(fixture=system, kind=key))
                handoff[key] = p.ref(target)
            self.ready[system] = handoff
        p.save(self.root / 'ready.json', self.ready)
        self.args = SimpleNamespace(node='A', predecessor=self.root / 'previous.json',
            repair=self.root / 'repair.json', baseline_ready=self.root / 'ready.json', out=self.node / 'resume', boundary=None)
        self.args.out.mkdir()

    def tearDown(self):
        self.patch_here.stop()
        self.tmp.cleanup()

    def observation(self, row, quality=1.):
        raw = dict(row, measurement_valid=True, service_terminal_valid=True,
            strict_slo_recomputed=True, independently_recomputed=True,
            unknown_error_count=0, work_complete=True, slo_attainment=quality)
        target = self.root / ('audit-' + row['cell_id'] + '.json')
        p.save(target, raw)
        return dict(raw, audit_reference=p.ref(target), engineering_attempt=1)

    def update_previous(self, **changes):
        self.previous.update(changes)
        p.save(self.args.predecessor, self.previous)

    def update_failed(self, **changes):
        self.failed.update(changes)
        p.save(self.root / 'failed.json', self.failed)
        self.repair['failed_setup_status'] = p.ref(self.root / 'failed.json')
        p.save(self.args.repair, self.repair)

    def live(self):
        return dict(p.process_identity(os.getpid()), finished_s=1., node_lease_held=False)

    def test_valid_preflight_preserves_five_audits_and_boundary(self):
        paths = [self.node / 'observations.json', self.root / 'boundary.json'] + [
            Path(o['audit_reference']['path']) for o in self.observations]
        before = {path: path.read_bytes() for path in paths}
        _, observations, ready, boundary_ref = r.validate(self.args)
        self.assertEqual(len(observations), 5)
        self.assertEqual(set(ready), set(r.contract.SYSTEMS[1:]))
        self.assertEqual({path: path.read_bytes() for path in paths}, before)
        self.assertEqual(boundary_ref, p.ref(self.root / 'boundary.json'))

    def test_live_previous_last_and_failed_owners_are_rejected(self):
        with self.subTest(owner='previous'):
            self.update_previous(**self.live())
            with self.assertRaises(ValueError): r.validate(self.args)
            self.update_previous(**self.dead)
        with self.subTest(owner='last'):
            p.save(self.root / 'live-last.json', dict(self.last, **self.live()))
            self.update_previous(last_cell_status=p.ref(self.root / 'live-last.json'))
            with self.assertRaises(ValueError): r.validate(self.args)
            self.update_previous(last_cell_status=p.ref(self.root / 'last.json'))
        with self.subTest(owner='failed'):
            self.update_failed(**self.live())
            with self.assertRaises(ValueError): r.validate(self.args)

    def test_live_setup_descendant_is_rejected(self):
        self.update_failed(child=self.live())
        with self.assertRaises(ValueError): r.validate(self.args)

    def test_unsettled_previous_descendant_is_rejected(self):
        self.update_previous(unsettled_child=dict(self.live(), physical_lease_release_not_certified=True))
        with self.assertRaises(ValueError): r.validate(self.args)

    def test_last_measurement_must_match_confirmed_boundary(self):
        p.save(self.root / 'unrelated-last.json', dict(self.last, unrelated=True))
        self.update_previous(last_cell_status=p.ref(self.root / 'unrelated-last.json'))
        with self.assertRaises(ValueError): r.validate(self.args)

    def test_changed_original_audit_is_rejected(self):
        Path(self.observations[0]['audit_reference']['path']).write_text('{}')
        with self.assertRaises(ValueError): r.validate(self.args)

    def test_existing_baseline_performance_cannot_be_retried(self):
        row = r.contract.make_row('A', '.25', 'mixed', 1)
        p.save(self.node / 'observations.json', self.observations + [self.observation(row)])
        with self.assertRaises(ValueError): r.validate(self.args)

    def test_changed_handoff_is_rejected_before_performance(self):
        Path(self.ready['mixed']['qualification']['path']).write_text('{}')
        with self.assertRaises(ValueError): r.validate(self.args)

    def test_only_sixteen_unmeasured_baselines_are_submitted_once(self):
        _, observations, ready, _ = r.validate(self.args)
        prefix = copy.deepcopy(observations)
        calls = []
        async def fake_one(row, handoff, current, node_dir, out, state):
            calls.append(row)
            current.append(self.observation(row))
            return True
        state = dict(node='A', complete=False, attempts=[])
        with patch.object(r.original, 'one', fake_one):
            asyncio.run(r.execute(self.args, state, observations, ready))
        self.assertEqual(len(calls), 16)
        self.assertEqual(len({row['cell_id'] for row in calls}), 16)
        self.assertEqual({row['system'] for row in calls}, set(r.contract.SYSTEMS[1:]))
        for system in r.contract.SYSTEMS[1:]:
            self.assertEqual([row['rate_rps'] for row in calls if row['system'] == system], [.25, .5, .75, 1.])
        self.assertEqual(observations[:5], prefix)
        self.assertTrue(state['complete'])

    def test_performance_failure_is_not_retried(self):
        _, observations, ready, _ = r.validate(self.args)
        calls = []
        async def failed_one(*args):
            calls.append(args[0]['cell_id'])
            return False
        state = dict(node='A', complete=False, attempts=[])
        with patch.object(r.original, 'one', failed_one):
            asyncio.run(r.execute(self.args, state, observations, ready))
        self.assertEqual(len(calls), 1)
        self.assertFalse(state['complete'])

    def test_preventive_pause_requires_exact_predecessor_and_evidence(self):
        self.update_previous(phase='stopped_at_boundary', error=None)
        self.repair.update(preventive_setup_correction=True)
        p.save(self.args.repair, self.repair)
        self.boundary.update(paused_predecessor=p.ref(self.args.predecessor))
        self.args.boundary = self.root / 'paused-boundary.json'
        p.save(self.args.boundary, self.boundary)
        self.assertEqual(len(r.validate(self.args)[1]), 5)
        self.boundary['paused_predecessor']['sha256'] = '0' * 64
        p.save(self.args.boundary, self.boundary)
        with self.assertRaises(ValueError): r.validate(self.args)

    def test_handoff_drift_during_execution_stops_before_next_cell(self):
        _, observations, ready, _ = r.validate(self.args)
        calls = []
        async def fake_one(row, handoff, current, *args):
            calls.append(row)
            current.append(self.observation(row))
            Path(handoff['qualification']['path']).write_text('{}')
            return True
        with patch.object(r.original, 'one', fake_one), self.assertRaises(ValueError):
            asyncio.run(r.execute(self.args, dict(node='A', complete=False, attempts=[]), observations, ready))
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
