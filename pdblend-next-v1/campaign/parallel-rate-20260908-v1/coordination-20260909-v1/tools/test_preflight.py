"""No SSH or GPU calls: validate default behavior and recovery-file boundaries."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

import preflight
import fetch_missing
import remote_readonly_probe as probe


class PreflightTests(unittest.TestCase):
    def test_default_never_connects(self):
        for node in ('A', 'C'):
            stream = io.StringIO()
            with patch.object(sys, 'argv', ['preflight.py', '--node', node]), \
                    patch.object(preflight.subprocess, 'run') as run, contextlib.redirect_stdout(stream):
                self.assertEqual(preflight.main(), 0)
            run.assert_not_called()
            result = json.loads(stream.getvalue())
            self.assertFalse(result['remote_connection_attempted'])
            self.assertFalse(result['gpu_work_started'])
            self.assertIsInstance(result['local_missing_retained_checkpoints'], int)

    def test_ssh_disables_password_and_rejects_argument_injection(self):
        for value in ['-oProxyCommand=bad', 'host;touch /tmp/unexpected', 'x$(id)', 'user@host']:
            with self.assertRaises(argparse.ArgumentTypeError):
                preflight.safe_host(value)
        args = argparse.Namespace(user='root', host='47.106.163.29', identity_file=None)
        source = "print('a'); print('$HOME `id`')"
        argv = preflight.ssh_argv(args, source)
        for option in ('-oBatchMode=yes', '-oStrictHostKeyChecking=yes',
                       '-oPasswordAuthentication=no', '-oKbdInteractiveAuthentication=no',
                       '-oUpdateHostKeys=no', '-oPermitLocalCommand=no'):
            self.assertIn(option, argv)
        self.assertEqual(shlex.split(argv[-1]), ['python3', '-B', '-c', source])

    def test_remote_hash_mismatch_missing_and_external_symlink(self):
        with tempfile.TemporaryDirectory(dir=preflight.HERE) as directory:
            p = Path(directory) / 'checkpoint.json'
            p.write_bytes(b'{"work_complete":true}\n')
            target = dict(path=str(p), expected_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), kinds=['checkpoint'])
            self.assertEqual(probe.file_check(target)['status'], 'match')
            self.assertEqual(probe.file_check(dict(target, expected_sha256='0' * 64))['status'], 'mismatch')
            self.assertEqual(probe.file_check(dict(target, path=str(p.with_name('absent.json'))))['status'], 'missing')
            link = p.with_name('outside.json')
            link.symlink_to('/etc/hostname')
            self.assertEqual(probe.file_check(dict(target, path=str(link)))['status'], 'outside_workspace')

    def test_snapshot_cannot_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory(dir=preflight.HERE) as directory:
            out = Path(directory) / 'snapshot.json'
            out.write_text('original\n')
            with patch.object(sys, 'argv', ['preflight.py', '--node', 'C', '--out', str(out)]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                preflight.main()
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(out.read_text(), 'original\n')

    def test_fetch_rejects_corrupted_bytes_before_creating_destination(self):
        with tempfile.TemporaryDirectory(dir=preflight.HERE) as directory:
            base = Path(directory)
            dest = base / 'missing.json'
            digest = hashlib.sha256(b'expected').hexdigest()
            snapshot = dict(ssh_host='47.106.163.29', ssh_user='root', remote=dict(
                identity=dict(hostname_matches=True, hostname='old-C', expected_hostname='old-C'),
                files=[dict(path=str(dest), local_status='missing', status='match', actual_sha256=digest, bytes=8)]))
            source = base / 'preflight.json'
            source.write_text(json.dumps(snapshot))
            returned = json.dumps(dict(path=str(dest), sha256=digest, bytes=3, data='YmFk'))
            fake = argparse.Namespace(returncode=0, stdout=returned, stderr='')
            with patch.object(sys, 'argv', ['fetch_missing.py', '--snapshot', str(source), '--out', str(base/'audit.json'), '--fetch']), \
                    patch.object(fetch_missing.subprocess, 'run', return_value=fake), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(fetch_missing.main(), 2)
            self.assertFalse(dest.exists())
            self.assertFalse(json.loads((base/'audit.json').read_text())['complete'])


if __name__ == '__main__':
    unittest.main()
