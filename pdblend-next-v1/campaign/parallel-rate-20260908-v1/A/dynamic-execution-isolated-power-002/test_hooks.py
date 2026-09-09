import asyncio
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace, ModuleType
import unittest
import sampler_hooks as hooks


class FakeSampler:
    instances = []
    readiness_failure = False
    def __init__(self, gpus, interval=.02, backend=None):
        self.power_source = dict(backend.power_source)
        self.error = None; self.samples = []; self.stopped = False; self.starts = 0
        self._process = None
        self._thread = SimpleNamespace(is_alive=lambda: not self.stopped)
        type(self).instances.append(self)
    def start(self):
        self.starts += 1; self._process = SimpleNamespace(poll=lambda: 0 if self.stopped else None)
    def wait_ready(self):
        time.sleep(.06)
        if type(self).readiness_failure:
            raise RuntimeError('readiness failure')
        self.samples = [[1, [1]*8], [2, [1]*8]]
    def stop(self):
        self.stopped = True


class PrimaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.old = {k: sys.modules.get(k) for k in ['ecopadg', 'ecopadg.measure', 'ecopadg.measure.backends']}
        for name in self.old:
            sys.modules[name] = ModuleType(name)
        self.backend = lambda **kw: SimpleNamespace(power_source=dict(mode='instant', field=186))
        sys.modules['ecopadg.measure.backends'].PynvmlBackend = self.backend
        FakeSampler.instances = []; FakeSampler.readiness_failure = False
        self.adapter = SimpleNamespace(IsolatedPowerSampler=FakeSampler)
        self.cell = SimpleNamespace(PowerSampler=FakeSampler)

    async def asyncTearDown(self):
        for key, value in self.old.items():
            if value is None: sys.modules.pop(key, None)
            else: sys.modules[key] = value

    async def test_slow_ready_nonblocking_and_only_one_worker(self):
        ticks = []
        async def tick():
            for _ in range(10): ticks.append(1); await asyncio.sleep(.01)
        async def run(args):
            self.assertGreaterEqual(len(ticks), 3)
            sampler = self.cell.PowerSampler(range(8), interval=.02, backend=self.backend())
            sampler.start(); self.assertEqual(sampler.starts, 1)
            sampler.stop(); return {'actual': True}
        self.cell.run_cell = run
        task = asyncio.create_task(tick())
        result = await hooks.prepared_primary(None, self.cell, self.adapter)
        await task
        self.assertEqual(result, {'actual': True})
        self.assertTrue(FakeSampler.instances[0].stopped)
        self.assertIs(self.cell.PowerSampler, FakeSampler)

    async def test_readiness_failure_before_controller_stops(self):
        FakeSampler.readiness_failure = True
        async def run(args): self.fail('controller called before ready')
        self.cell.run_cell = run
        with self.assertRaisesRegex(RuntimeError, 'readiness failure'):
            await hooks.prepared_primary(None, self.cell, self.adapter)
        self.assertTrue(FakeSampler.instances[0].stopped)

    async def test_controller_failure_before_consumption_stops(self):
        async def run(args): raise ValueError('controller failure')
        self.cell.run_cell = run
        with self.assertRaisesRegex(ValueError, 'controller failure'):
            await hooks.prepared_primary(None, self.cell, self.adapter)
        self.assertTrue(FakeSampler.instances[0].stopped)

    async def test_return_without_consumption_rejected(self):
        async def run(args): return {}
        self.cell.run_cell = run
        with self.assertRaisesRegex(RuntimeError, 'did not consume'):
            await hooks.prepared_primary(None, self.cell, self.adapter)
        self.assertTrue(FakeSampler.instances[0].stopped)

    async def test_wrong_scope_rejected_and_stopped(self):
        async def run(args): self.cell.PowerSampler([6, 7], backend=self.backend())
        self.cell.run_cell = run
        with self.assertRaisesRegex(RuntimeError, 'scope changed'):
            await hooks.prepared_primary(None, self.cell, self.adapter)
        self.assertTrue(FakeSampler.instances[0].stopped)

    async def test_double_primary_rejected(self):
        async def run(args):
            self.cell.PowerSampler(range(8), backend=self.backend())
            self.cell.PowerSampler(range(8), backend=self.backend())
        self.cell.run_cell = run
        with self.assertRaisesRegex(RuntimeError, 'once only'):
            await hooks.prepared_primary(None, self.cell, self.adapter)

    async def test_old_cached_factory_rejected(self):
        self.cell.PowerSampler = object
        with self.assertRaisesRegex(RuntimeError, 'cached the old'):
            await hooks.prepared_primary(None, self.cell, self.adapter)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.host = {'path': 'host', 'sha256': 'hostsha'}
        self.adapter = {'path': 'adapter', 'sha256': 'adaptersha'}
    def tearDown(self): self.tmp.cleanup()
    def add(self, name):
        directory = self.root / name; directory.mkdir()
        spec = dict(host_manifest=self.host, adapter_manifest=self.adapter,
                    gpus=list(range(8)), interval=.02, read_only=True)
        launch = dict(pid=123, read_only=True)
        receipt = dict(pid=123, complete=True, child_exited=True, reader_stopped=True,
                       returncode=0, error=None, raw_sample_count=2,
                       terminal=dict(pid=123, sampler_thread_stopped=True, error=None,
                                     emitted=2, counts=dict(samples=2, metadata=2, utilization=2, frequency=0)))
        (directory / 'final-raw.json').write_text(json.dumps(dict(
            power_source=dict(mode='instant'), sampling_error=None, samples=[[1, [1]*8], [2, [1]*8]],
            metadata=[{}, {}], utilization=[[1, [1]*8], [2, [1]*8]], frequency=[])))
        for file, value in [('spec', spec), ('launch', launch), ('receipt', receipt)]:
            (directory / (file+'.json')).write_text(json.dumps(value))
        return directory
    def test_two_cells_own_only_new_directories(self):
        one = self.add('sampler-one'); prior = hooks.directories(self.root)
        two = self.add('sampler-two')
        result = hooks.completed_artifacts(hooks.directories(self.root)-prior, self.host, self.adapter)
        self.assertEqual(set(result), {str(p) for p in two.glob('*')})
        self.assertFalse(any(str(one) in p for p in result))
    def test_failed_worker_rejected(self):
        directory = self.add('sampler-one'); path = directory/'receipt.json'
        result = json.loads(path.read_text()); result['complete'] = False; path.write_text(json.dumps(result))
        with self.assertRaisesRegex(RuntimeError, 'finish cleanly'):
            hooks.completed_artifacts({directory}, self.host, self.adapter)
    def test_missing_terminal_rejected(self):
        directory = self.add('sampler-one'); (directory/'receipt.json').unlink()
        with self.assertRaises(FileNotFoundError):
            hooks.completed_artifacts({directory}, self.host, self.adapter)
    def test_wrong_source_rejected(self):
        directory = self.add('sampler-one')
        with self.assertRaisesRegex(RuntimeError, 'scope changed'):
            hooks.completed_artifacts({directory}, dict(path='bad', sha256='bad'), self.adapter)
    def test_metadata_drop_rejected(self):
        directory = self.add('sampler-one'); path = directory/'receipt.json'
        result = json.loads(path.read_text()); result['terminal']['counts']['metadata'] = 1
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(RuntimeError, 'terminal is incomplete'):
            hooks.completed_artifacts({directory}, self.host, self.adapter)
    def test_bad_reference_hash_rejected(self):
        path = self.root/'reference.json'; path.write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'reference changed'):
            hooks.checked(dict(path=str(path), sha256='bad'))


if __name__ == '__main__': unittest.main()
