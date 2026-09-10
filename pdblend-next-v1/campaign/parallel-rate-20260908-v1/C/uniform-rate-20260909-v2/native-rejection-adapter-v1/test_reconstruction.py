"""CPU regressions using immutable real native-refusal evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import native_queue as q
import reconstruct

HERE = Path(__file__).parent
SAVED = HERE.parent / 'native-rejection-reconstruction-001'


class ReconstructionTests(unittest.TestCase):
    def test_real_new_window_recomputes_same_classification_and_metrics(self):
        saved = q.p.read(SAVED / 'observation.json')
        result = reconstruct.audit(saved['checkpoint'])
        self.assertEqual(result['baseline_service_failure']['native_rejections'], 170)
        self.assertEqual(result['baseline_service_failure']['actual_request_timeouts'], 0)
        for key in ('energy_j', 'slo_attainment', 'ttft_avg_s', 'tpot_avg_s', 'gpu_util',
                    'request_throughput_rps', 'token_throughput_tps', 'actual_output_tokens'):
            self.assertEqual(result[key], saved[key])

    def test_changed_original_checkpoint_reference_rejected(self):
        saved = q.p.read(SAVED / 'observation.json')
        bad = dict(saved['checkpoint'], sha256='0' * 64)
        with self.assertRaises(ValueError): reconstruct.audit(bad)

    def test_future_release_full_cell_adapter_with_temporary_reference_fixture(self):
        # Exercise the future source-routing interface without changing or
        # publishing any real checkpoint. This temporary object is not a run.
        saved = q.p.read(SAVED / 'observation.json')
        cp = copy.deepcopy(q.p.checked(saved['checkpoint']))
        cp['release'] = q.p.ref(HERE / 'cpu-prepare-001/release.json')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cpu-only-release-reference-fixture.json'
            path.write_text(json.dumps(cp))
            adapter = q.p.load(HERE / 'cell_audit.py', 'test_future_cell_native_adapter')
            result = adapter.audit(q.p.ref(path))
            self.assertEqual(result['baseline_service_failure']['classification'],
                             'baseline_explicit_native_admission_queue_full')
            self.assertEqual(result['baseline_service_failure']['native_rejections'], 170)
            self.assertEqual(result['request_timeouts'], 0)


if __name__ == '__main__': unittest.main()
