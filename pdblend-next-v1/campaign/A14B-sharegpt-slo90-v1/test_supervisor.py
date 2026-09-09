import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import supervisor as s
import runner
import trace_source


class Supervisor(unittest.TestCase):
    def test_missing_or_changed_validation_gate_cannot_launch(self):
        with tempfile.TemporaryDirectory() as d, patch.object(s, 'HERE', Path(d)):
            self.assertFalse(s.ready())
            package = Path(d) / 'package.json'
            runner.save(package, {'files': {}})
            runner.save(Path(d) / 'READY.json', dict(cpu_validation_passed=False,
                package_sha256=trace_source.file_sha(package)))
            self.assertFalse(s.ready())
            runner.save(Path(d) / 'READY.json', dict(cpu_validation_passed=True,
                package_sha256=trace_source.file_sha(package)))
            self.assertTrue(s.ready())
            package.write_text('{}\n')
            self.assertFalse(s.ready())

    def test_changed_source_cannot_be_staged(self):
        with tempfile.TemporaryDirectory() as d, patch.object(s, 'HERE', Path(d)):
            path = Path(d) / 'source.py'
            path.write_text('original')
            runner.save(Path(d) / 'package.json', {'files': {str(path): trace_source.file_sha(path)}})
            runner.save(Path(d) / 'READY.json', {})
            path.write_text('changed')
            with patch.object(s.subprocess, 'run') as process:
                with self.assertRaisesRegex(ValueError, 'changed'):
                    s.install('C')
                process.assert_not_called()

    def test_declined_claim_goes_back_to_waiting_without_second_host_in_parallel(self):
        with tempfile.TemporaryDirectory() as d, patch.object(s, 'OUT', Path(d)):
            runner.save(Path(d) / 'status.json', dict(host='A', claim_id='123'))
            with patch.object(s, 'status_of', return_value={'status': 'declined'}), \
                 patch.object(s, 'launch') as launch, patch.object(s, 'mirror') as mirror, \
                 patch.object(s.time, 'sleep', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    s.run()
                launch.assert_not_called()
                mirror.assert_not_called()
            state = runner.read(Path(d) / 'status.json')
            self.assertIsNone(state['claim_id'])
            self.assertIsNone(state['host'])
            self.assertEqual(state['declined_claims'], [{'host': 'A', 'claim_id': '123'}])

    def test_observed_failure_is_preserved_and_stops_dispatch(self):
        with tempfile.TemporaryDirectory() as d, patch.object(s, 'OUT', Path(d)):
            runner.save(Path(d) / 'status.json', dict(host='C', claim_id='123'))
            with patch.object(s, 'status_of', return_value={'status': 'failed', 'error': 'meter'}), \
                 patch.object(s, 'launch') as launch, patch.object(s, 'mirror'):
                self.assertEqual(s.run(), 2)
                launch.assert_not_called()
            self.assertEqual(runner.read(Path(d) / 'status.json')['execution']['error'], 'meter')


if __name__ == '__main__':
    unittest.main()
