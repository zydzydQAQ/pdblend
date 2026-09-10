"""Exercise the entire independent refusal audit with structured CP refs."""
import json
from pathlib import Path
import tempfile
import unittest
import metrics_native_refs_v3 as metrics


class NativeReferenceTest(unittest.TestCase):
    def test_real_pinned_refusal_evidence_accepts_only_exact_reference_conversion(self):
        source = metrics.ROOT / 'C/eco-drain37-v1/performance/results/checkpoints/eco-drain37-v1-7b-alpaca-r12-s701-w100-ecoserve-slo1-repeat1.json'
        cp = metrics.read(source)
        for key in ('binding', 'receipt'):
            cp[key] = dict(path=cp[key], sha256=cp[key + '_sha256'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference-format-fixture.json'
            path.write_text(json.dumps(cp))
            ids, proof = metrics.diagnosed_zero_output(metrics.ref(path))
            self.assertEqual(len(ids), 42)
            self.assertTrue(proof['no_accepted_request_stranded'])
            self.assertEqual(len(proof['actual_identity']), 8)
            self.assertEqual(proof['raw']['failed_requests'], 42)
            cp['receipt']['sha256'] = '0' * 64
            path.write_text(json.dumps(cp))
            with self.assertRaises(ValueError):
                metrics.diagnosed_zero_output(metrics.ref(path))


if __name__ == '__main__':
    unittest.main()
