import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import report_native_overlay_v1 as overlay


class OverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads((Path(__file__).parent / 'reports/current/results.json').read_text())
        cls.output = overlay.apply(cls.source)
        cls.index = next(overlay.DIRECTORY.glob('native-rejection-reconstruction-*/source-closure.json'))
        cls.definition = json.loads(cls.index.read_text())
        cls.checkpoint = cls.definition['checkpoint']

    def test_actual_report_all_numeric_values_and_pipeline_states_preserved(self):
        self.assertEqual(len(self.source['observations']), len(self.output['observations']))
        self.assertEqual(self.source['historical_observations'], self.output['historical_observations'])
        self.assertEqual(self.source['metric_audit_errors'], self.output['metric_audit_errors'])
        changed = []
        for original, updated in zip(self.source['observations'], self.output['observations']):
            if original['checkpoint'] == self.checkpoint:
                changed.append(updated)
                for key in overlay.SCIENTIFIC:
                    self.assertEqual(original[key], updated[key])
                self.assertEqual(updated['failure_class'], 'independently_diagnosed_capacity_rejection')
                self.assertEqual(updated['baseline_service_failure']['native_rejections'], 170)
                self.assertEqual(updated['baseline_service_failure']['actual_request_timeouts'], 0)
                self.assertTrue(updated['zero_output_diagnosis']['raw']['captured_s'] > 0)
            else:
                self.assertEqual(original, updated)
        self.assertEqual(len(changed), 1)
        for original, updated in zip(self.source['groups'], self.output['groups']):
            self.assertEqual(original['pipeline_state'], updated['pipeline_state'])
            self.assertEqual(original['declaration_contract'], updated['declaration_contract'])
            if (original['model'], original['dataset'], original['node']) == ('7b', 'alpaca', 'C'):
                self.assertTrue(updated['complete'])
                self.assertEqual(updated['decision']['phase'], 'complete')
                self.assertEqual(updated['report_contract'], overlay.CONTRACT)
            else:
                self.assertEqual(original, updated)

    def test_cache_hit_still_hashes_inputs_without_rerunning_full_audit(self):
        original_load = overlay.load
        def load(reference, name):
            self.assertNotEqual(reference, overlay.AUDITOR, 'cache hit reran full audit')
            return original_load(reference, name)
        with patch.object(overlay, 'load', side_effect=load), patch.object(overlay, 'file_hash', wraps=overlay.file_hash) as hashes:
            updated = overlay.apply(self.source)
            self.assertEqual(updated, self.output)
            self.assertGreater(hashes.call_count, 1400)

    def test_changed_report_checkpoint_sha_fails_closed(self):
        source = copy.deepcopy(self.source)
        target = next(r for r in source['observations'] if r['checkpoint'] == self.checkpoint)
        target['checkpoint']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'checkpoint SHA changed'):
            overlay.apply(source)

    def test_changed_report_scientific_value_and_host_fail_closed(self):
        for field, value in (('energy_j', -1), ('measurement_host', 'B')):
            source = copy.deepcopy(self.source)
            target = next(r for r in source['observations'] if r['checkpoint'] == self.checkpoint)
            target[field] = value
            before = copy.deepcopy(source)
            with self.subTest(field=field), self.assertRaises(ValueError):
                overlay.apply(source)
            self.assertEqual(source, before)

    def test_full_sha_rejects_changed_input_and_accepts_same_bytes_new_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input'; path.write_bytes(b'original')
            files = {str(path): hashlib.sha256(b'original').hexdigest()}
            first = overlay.fingerprint(files)
            replacement = Path(directory) / 'replace'; replacement.write_bytes(b'original'); replacement.replace(path)
            second = overlay.fingerprint(files)
            self.assertNotEqual(first, second)
            path.write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError, 'input changed'):
                overlay.fingerprint(files)

    def test_same_fd_snapshot_rejects_atomic_path_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.json'; path.write_text('{"old":true}')
            reference = overlay.ref(path); original_open = Path.open
            class ReplacingReader:
                def __init__(self, handle): self.handle = handle
                def __enter__(self): return self
                def __exit__(self, *args): self.handle.close()
                def fileno(self): return self.handle.fileno()
                def read(self, *args):
                    data = self.handle.read(*args)
                    replacement = Path(directory) / 'new'; replacement.write_text('{"new":true}'); replacement.replace(path)
                    return data
            def open_file(target, *args, **kwargs):
                handle = original_open(target, *args, **kwargs)
                return ReplacingReader(handle) if target == path and args == ('rb',) else handle
            with patch.object(Path, 'open', open_file), self.assertRaisesRegex(ValueError, 'snapshot read'):
                overlay.checked(reference)

    def test_seen_index_and_duplicate_target_fail_closed(self):
        key = str(self.index)
        previous = overlay._SEEN_INDICES[key]
        try:
            overlay._SEEN_INDICES[key] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'index changed'):
                overlay.apply(self.source)
        finally:
            overlay._SEEN_INDICES[key] = previous
        original_glob = Path.glob
        def glob(path, pattern):
            if path == overlay.DIRECTORY and pattern == 'native-rejection-reconstruction-*/source-closure.json':
                return iter((self.index, self.index))
            return original_glob(path, pattern)
        with patch.object(Path, 'glob', glob), self.assertRaisesRegex(ValueError, 'multiple derived'):
            overlay.apply(self.source)

    def test_removed_previously_seen_index_does_not_silently_restore_old_label(self):
        original = Path.is_file
        def is_file(path):
            return False if path == self.index else original(path)
        with patch.object(Path, 'is_file', is_file), self.assertRaisesRegex(ValueError, 'index disappeared'):
            overlay.apply(self.source)

    def test_no_target_preserves_every_other_group_and_observation(self):
        source = copy.deepcopy(self.source)
        source['observations'] = [r for r in source['observations'] if r['checkpoint'] != self.checkpoint]
        self.assertEqual(overlay.apply(source), source)


if __name__ == '__main__': unittest.main()
