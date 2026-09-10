"""Delta status transfer tests use temporary files and never contact a host."""
import ast
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import monitor_delta_v1 as m
import monitor as original


class RemoteIndex(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'C'
        self.path = self.root / 'uniform-rate-20260909-v2/frequency/status.json'
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps(dict(state='old', padding='x' * 1024**2)))

    def index(self, known=None):
        result = subprocess.run([sys.executable, '-B', '-c', m.INDEX_SCRIPT, str(self.root)],
            input=json.dumps(known or {}).encode(), capture_output=True, check=True)
        return json.loads(result.stdout), len(result.stdout)

    def test_unchanged_megabyte_status_uses_small_marker(self):
        first, size = self.index()
        second, small = self.index({str(self.path): first[0]['remote_stat']})
        self.assertGreater(size, 1024**2)
        self.assertLess(small, 350)
        self.assertTrue(second[0]['unchanged'])
        self.assertNotIn('data', second[0])
        self.assertEqual(second[0]['remote_stat'], first[0]['remote_stat'])

    def test_changed_or_replaced_status_is_full_again(self):
        old = self.index()[0][0]
        self.path.write_text(json.dumps(dict(state='changed')))
        changed = self.index({str(self.path): old['remote_stat']})[0][0]
        self.assertEqual(json.loads(base64.b64decode(changed['data']))['state'], 'changed')
        replacement = self.path.with_suffix('.replacement')
        replacement.write_bytes(self.path.read_bytes())
        os.utime(replacement, ns=(changed['mtime_ns'], changed['mtime_ns']))
        os.replace(replacement, self.path)
        replaced = self.index({str(self.path): changed['remote_stat']})[0][0]
        self.assertIn('data', replaced)
        self.assertNotEqual(changed['remote_stat']['ino'], replaced['remote_stat']['ino'])

    def test_atomic_rename_during_read_never_pairs_old_bytes_with_new_stat(self):
        self.path.write_text('{"state":"old"}')
        real_open = Path.open
        replaced = []
        target = self.path

        class Reader:
            def __init__(self, stream): self.stream = stream
            def __enter__(self): return self
            def __exit__(self, *args): self.stream.close()
            def fileno(self): return self.stream.fileno()
            def seek(self, *args): return self.stream.seek(*args)
            def read(self, *args):
                raw = self.stream.read(*args)
                alternative = target.with_suffix('.next')
                with real_open(alternative, 'wb') as handle:
                    handle.write(b'{"state":"new"}')
                os.replace(alternative, target)
                replaced.append(True)
                return raw

        def intercepted(path, *args, **kwargs):
            stream = real_open(path, *args, **kwargs)
            return Reader(stream) if path == target and args == ('rb',) else stream

        output = io.StringIO()
        with mock.patch.object(sys, 'argv', ['index', str(self.root)]), mock.patch.object(sys, 'stdin', io.StringIO('{}')):
            with mock.patch.object(Path, 'open', intercepted), contextlib.redirect_stdout(output):
                exec(m.INDEX_SCRIPT, {})
        self.assertTrue(replaced)
        self.assertEqual(json.loads(output.getvalue()), [])
        next_snapshot = self.index()[0][0]
        self.assertEqual(json.loads(base64.b64decode(next_snapshot['data'])), {'state': 'new'})

    def test_failed_but_clean_energy_window_keeps_power_refs_only(self):
        status = dict(complete=False, passed=False, error='original export 404', finished_s=3,
            measurement_end_s=2, full_operation_energy_j=100, measurement_valid=True,
            node_lease_held=False, clock_restore_complete=True, cleanup_errors=[], sampling_error=None)
        power = self.path.parent / 'power'
        power.mkdir()
        for name in ('power.csv', 'power_source.json', 'power_metadata.jsonl'):
            (power / name).write_text('preserved raw bytes\n')
        self.path.write_text(json.dumps(status))
        snapshot = self.index()[0][0]
        self.assertEqual(len(snapshot['extra_refs']), 3)
        self.assertFalse(json.loads(base64.b64decode(snapshot['data']))['passed'])
        for name, value in [('node_lease_held', True), ('clock_restore_complete', False),
                            ('cleanup_errors', ['not restored']), ('sampling_error', 'missing'),
                            ('measurement_valid', False)]:
            with self.subTest(name=name):
                self.path.write_text(json.dumps(dict(status, **{name: value})))
                self.assertEqual(self.index()[0][0]['extra_refs'], [])
        # Preserve the original completed cold-restoration discovery path.
        self.path.write_text(json.dumps(dict(complete=True, finished_s=3, measurement_end_s=2,
            setup_and_correctness_energy_j=100)))
        self.assertEqual(len(self.index()[0][0]['extra_refs']), 3)

    def test_in_place_change_during_read_is_not_published(self):
        self.path.write_text('{"state":"old"}')
        real_open = Path.open
        target = self.path

        class Reader:
            def __init__(self, stream): self.stream = stream
            def __enter__(self): return self
            def __exit__(self, *args): self.stream.close()
            def fileno(self): return self.stream.fileno()
            def seek(self, *args): return self.stream.seek(*args)
            def read(self, *args):
                raw = self.stream.read(*args)
                with real_open(target, 'wb') as handle:
                    handle.write(b'{"state":"new"}')
                return raw

        def intercepted(path, *args, **kwargs):
            stream = real_open(path, *args, **kwargs)
            return Reader(stream) if path == target and args == ('rb',) else stream

        output = io.StringIO()
        with mock.patch.object(sys, 'argv', ['index', str(self.root)]), mock.patch.object(sys, 'stdin', io.StringIO('{}')):
            with mock.patch.object(Path, 'open', intercepted), contextlib.redirect_stdout(output):
                exec(m.INDEX_SCRIPT, {})
        self.assertEqual(json.loads(output.getvalue()), [])


class LocalMirror(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for obj, key, value in [(m, 'WORKSPACE', self.root), (m.c, 'ROOT', self.root),
                                (m, 'HERE', self.root / 'monitor')]:
            patch = mock.patch.object(obj, key, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.path = self.root / 'C/uniform-rate-20260909-v2/pipeline/status.json'
        self.ref = dict(path=str(self.root / 'C/measurements/audits/one.json'), sha256='a' * 64)
        self.raw = json.dumps(dict(observations=[self.ref], complete=False)).encode()
        self.remote = dict(size=len(self.raw), mtime_ns=1000000000, ctime_ns=2000000000, ino=30)
        self.snapshot = dict(path=str(self.path), remote_stat=self.remote, mtime_ns=self.remote['mtime_ns'],
            data=base64.b64encode(self.raw).decode(), extra_refs=[])
        self.cache = {}

    def unchanged(self):
        return dict(path=str(self.path), remote_stat=self.remote, unchanged=True)

    def test_unchanged_preserves_all_seeds_without_decoding(self):
        seeds = m.mirror_statuses('C', [self.snapshot], self.cache)
        self.assertEqual(seeds, [self.ref])
        with mock.patch.object(m.base64, 'b64decode', side_effect=AssertionError('must not decode unchanged raw')):
            self.assertEqual(m.mirror_statuses('C', [self.unchanged()], self.cache), seeds)

    def test_changed_local_file_forces_remote_full_request(self):
        m.mirror_statuses('C', [self.snapshot], self.cache)
        self.path.write_text('{"changed":"locally"}')
        with mock.patch.object(m.subprocess, 'run', return_value=mock.Mock(stdout=b'[]')) as remote:
            m.remote_statuses('C', self.cache)
            self.assertEqual(json.loads(remote.call_args.kwargs['input']), {})

    def test_same_bytes_replacement_forces_remote_full_request(self):
        m.mirror_statuses('C', [self.snapshot], self.cache)
        replacement = self.path.with_suffix('.next')
        replacement.write_bytes(self.raw)
        saved = self.cache[str(self.path)]['local_stat']
        os.utime(replacement, ns=(saved['mtime_ns'], saved['mtime_ns']))
        os.replace(replacement, self.path)
        with mock.patch.object(m.subprocess, 'run', return_value=mock.Mock(stdout=b'[]')) as remote:
            m.remote_statuses('C', self.cache)
            self.assertEqual(json.loads(remote.call_args.kwargs['input']), {})

    def test_local_change_after_remote_request_refetches_immediately(self):
        m.mirror_statuses('C', [self.snapshot], self.cache)
        self.path.write_text('{"changed":"between request and mirror"}')
        with mock.patch.object(m, 'remote_statuses', return_value=[self.snapshot]) as fetch:
            self.assertEqual(m.mirror_statuses('C', [self.unchanged()], self.cache), [self.ref])
            fetch.assert_called_once()
        self.assertEqual(self.path.read_bytes(), self.raw)

    def test_missing_cache_and_malformed_delta_fail_closed(self):
        with mock.patch.object(m, 'remote_statuses', return_value=[self.unchanged()]):
            with self.assertRaisesRegex(ValueError, 'lacks matching local'):
                m.mirror_statuses('C', [self.unchanged()], self.cache)
        with self.assertRaisesRegex(ValueError, 'bytes/stat differ'):
            m.mirror_statuses('C', [dict(self.snapshot, remote_stat=dict(self.remote, size=1))], self.cache)

    def test_local_rewrite_during_hash_is_not_cached(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(self.raw)
        with self.assertRaisesRegex(ValueError, 'changed during mirror'):
            m.mirrored_stat(self.path, b'not these bytes')

    def test_immutable_transfer_scientific_hydration_and_completion_unchanged(self):
        original_source = ast.parse(Path(original.__file__).read_text())
        new_source = ast.parse(Path(m.__file__).read_text())
        nodes = lambda tree: {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        old, new = nodes(original_source), nodes(new_source)
        for name in ('Mirror', 'references', 'scientific_references', 'signature', 'completion_state',
                     'safe_path', 'stat_key', 'remote_argv'):
            with self.subTest(name=name):
                self.assertEqual(ast.dump(old[name], include_attributes=False), ast.dump(new[name], include_attributes=False))
        original_mirror = old['mirror_statuses']
        original_mirror.name = 'mirror_full_statuses'
        self.assertEqual(ast.dump(original_mirror, include_attributes=False),
                         ast.dump(new['mirror_full_statuses'], include_attributes=False))
        self.assertEqual(m.TRANSFER_SCRIPT, original.TRANSFER_SCRIPT)


if __name__ == '__main__':
    unittest.main()
