"""Real IPC/process transport with the exact original sampler and a CPU backend."""
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
import sampler_hooks as hooks

ROOT = Path(__file__).resolve().parent
ADAPTER = ROOT.parent / 'isolated-power-v2'

class ActualProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls.temp.name)
        spec = importlib.util.spec_from_file_location('fixture_original_sampler', ADAPTER / 'test_isolation.py')
        fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
        fake_sampler, _ = fixture.fake.__wrapped__(cls.tmp)
        cls.host_ref = fixture.m._CONTEXT[1]
        cls.adapter_ref = dict(path=str(ADAPTER / 'manifest.json'), sha256=hooks.sha(ADAPTER / 'manifest.json'))
        cls.raw = cls.tmp / 'owned-samplers'
        sys.path.insert(0, str(cls.tmp / 'host' / 'src'))
        cls.adapter = hooks.install(cls.raw, cls.host_ref, cls.adapter_ref)
        from ecopadg.measure.backends import PynvmlBackend
        cls.backend = staticmethod(lambda: PynvmlBackend(power_mode='instant'))

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_two_actual_invocations_same_context_no_mixed_ownership(self):
        seen = set()
        for _ in range(2):
            before = hooks.directories(self.raw)
            self.assertIs(hooks.install(self.raw, self.host_ref, self.adapter_ref), self.adapter)
            sampler = self.adapter.IsolatedPowerSampler(range(8), backend=self.backend(), sample_clocks=True)
            sampler.start(); sampler.wait_ready(); time.sleep(.04); sampler.stop()
            owned = hooks.directories(self.raw) - before
            self.assertEqual(len(owned), 1); self.assertFalse(owned & seen); seen.update(owned)
            files = hooks.completed_artifacts(owned, self.host_ref, self.adapter_ref)
            refs = hooks.sampler_references(owned)
            raw = hooks.checked(refs[0]['raw'])
            self.assertEqual(raw['samples'], sampler.samples)
            self.assertEqual(raw['metadata'], sampler.power_metadata)
            self.assertEqual(raw['frequency'], sampler.frequency_samples)
            self.assertIn(refs[0]['receipt']['path'], files)
            old = dict(files); sampler.stop()
            self.assertEqual(old, hooks.completed_artifacts(owned, self.host_ref, self.adapter_ref))

    def test_actual_prepared_primary_async_then_clean_stop(self):
        before = hooks.directories(self.raw)
        cell = SimpleNamespace(PowerSampler=self.adapter.IsolatedPowerSampler)
        async def original_cell(args):
            sampler = cell.PowerSampler(range(8), interval=.02, backend=self.backend())
            sampler.start()
            await asyncio.sleep(.1)
            self.assertGreaterEqual(len(sampler.samples), 2)
            await asyncio.to_thread(sampler.stop)
            return {'measurement_valid': True}
        cell.run_cell = original_cell
        result = asyncio.run(hooks.prepared_primary(None, cell, self.adapter))
        self.assertTrue(result['measurement_valid'])
        owned = hooks.directories(self.raw) - before
        self.assertEqual(len(owned), 1)
        self.assertTrue(hooks.completed_artifacts(owned, self.host_ref, self.adapter_ref))

    def test_wrong_host_manifest_source_rejected_before_observer(self):
        original = self.host_ref
        bad = dict(original, sha256='bad')
        before = hooks.directories(self.raw)
        with self.assertRaisesRegex(RuntimeError, 'reference changed'):
            hooks.install(self.raw, bad, self.adapter_ref)
        self.assertEqual(before, hooks.directories(self.raw))

if __name__ == '__main__': unittest.main()
