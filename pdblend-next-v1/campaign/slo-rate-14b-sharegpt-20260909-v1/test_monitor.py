import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

SPEC = importlib.util.spec_from_file_location('slo_monitor', Path(__file__).with_name('monitor.py'))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_root, self.old_workspace = m.HERE, m.WORKSPACE
        m.HERE = m.WORKSPACE = Path(self.temp.name)

    def tearDown(self):
        m.HERE, m.WORKSPACE = self.old_root, self.old_workspace
        self.temp.cleanup()

    def test_conflicting_immutable_file_is_preserved(self):
        path = m.HERE / 'file.json'
        path.write_text('original')
        mirror = m.Mirror(m.HERE / 'cache.json')
        with self.assertRaisesRegex(ValueError, 'conflict'):
            mirror.present({'path': str(path), 'sha256': 'a' * 64})
        self.assertEqual(path.read_text(), 'original')

    def test_exclusive_install_checks_hash(self):
        path, temporary = m.HERE / 'file.json', m.HERE / 'transfer.tmp'
        temporary.write_text('new')
        ref = {'path': str(path), 'sha256': hashlib.sha256(b'new').hexdigest()}
        mirror = m.Mirror(m.HERE / 'cache.json')
        mirror.install(ref, temporary)
        self.assertEqual(path.read_bytes(), b'new')
        self.assertEqual(mirror.downloaded, 1)

    def test_only_own_status_and_observations_are_mutable(self):
        raw = json.dumps({'schema': 'other-runtime-state'}).encode()
        record = dict(path=str(m.HERE / 'A/environment/status.json'),
            data=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaisesRegex(ValueError, 'invalid supervisor'):
            m.install_snapshot('A', record)
        self.assertFalse((m.HERE / 'A/environment/status.json').exists())

    def test_valid_observations_snapshot_mirrors_bytes(self):
        raw = b'[{"cell_id":"x"}]\n'
        path = m.HERE / 'A/observations.json'
        record = dict(path=str(path),data=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
        self.assertEqual(m.install_snapshot('A', record), [{'cell_id':'x'}])
        self.assertEqual(path.read_bytes(), raw)

    def test_reference_walk_does_not_follow_arbitrary_old_code_inventory(self):
        value = dict(files={'/root/workspace/old.py':'a'*64},
            artifacts={'/root/workspace/raw.csv':'b'*64},
            checkpoint={'path':'/root/workspace/cp.json','sha256':'c'*64})
        refs = list(m.references(value))
        self.assertEqual({x['path'] for x in refs}, {'/root/workspace/raw.csv','/root/workspace/cp.json'})

    def test_minimal_mirror_defers_qualification_and_large_auxiliary_raw(self):
        value = dict(qualification={'path':'/root/workspace/q.json','sha256':'a'*64},
            artifacts={'/root/workspace/power_metadata.jsonl':'b'*64,
                       '/root/workspace/bench.csv':'c'*64, '/root/workspace/power.csv':'d'*64},
            binding={'path':'/root/workspace/binding.json','sha256':'e'*64})
        self.assertEqual({x['path'] for x in m.minimal_references(value)},
            {'/root/workspace/bench.csv','/root/workspace/power.csv','/root/workspace/binding.json'})

    def test_snapshot_indexes_only_small_frozen_workload_metadata(self):
        directory = m.HERE / 'workloads/r0.25'
        directory.mkdir(parents=True)
        workload = directory / 'workload.json'
        workload.write_text('{"fixture":true}')
        manifest = directory / 'manifest.json'
        manifest.write_text('{}')
        (m.HERE / 'A').mkdir()
        (m.HERE / 'A/observations.json').write_text(json.dumps([dict(
            materialization_manifest={'path':str(manifest),'sha256':m.report.sha(manifest)},
            qualification={'path':str(m.HERE/'huge-qualification.json'),'sha256':'a'*64})]))
        before = {str(p):p.read_bytes() for p in m.HERE.rglob('*') if p.is_file()}
        result = subprocess.run([sys.executable, '-c', m.INDEX_SCRIPT, str(m.HERE), 'A'],
                                capture_output=True, check=True, text=True)
        indexed = json.loads(result.stdout)
        self.assertEqual(indexed['metadata_references'], [m.report.ref(workload)])
        self.assertEqual(before, {str(p):p.read_bytes() for p in m.HERE.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
