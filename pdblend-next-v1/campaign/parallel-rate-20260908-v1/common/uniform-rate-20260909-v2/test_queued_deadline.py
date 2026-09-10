import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import metrics_queue_deadlines_v4 as metrics

HELPER=metrics.ROOT/'common/token-evidence-v2/audit_unforwarded_deadline_v1.py'
spec=importlib.util.spec_from_file_location('queued_zero_auditor',HELPER)
auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
CP=next((metrics.ROOT/'B/uniform-rate-20260909-v2/pipeline32-001').glob('0019*/measurement/results/checkpoints/*.json'))


class QueuedDeadlineTests(unittest.TestCase):
    def test_actual_deadline_has_exact_zero_and_keeps_slo(self):
        proof=auditor.audit(auditor.ref(CP))
        self.assertEqual([r['request_id'] for r in proof['zero_output_details']],['52'])
        self.assertEqual(proof['full_request_timing_count'],122)
        result=metrics.audit_checkpoint(CP)
        self.assertTrue(result['token_throughput_is_exact'])
        self.assertEqual(result['actual_output_tokens'],35126)
        self.assertEqual(result['request_timeouts'],16)
        self.assertEqual(result['completed_work_requests'],106)
        self.assertAlmostEqual(result['slo_attainment'],24/122)

    def test_forward_marker_rejects_zero_inference(self):
        original=Path.read_text
        def read(path,*args,**kwargs):
            text=original(path,*args,**kwargs)
            if path.name=='control.jsonl':
                rows=[json.loads(line) for line in text.splitlines()]
                for row in rows:
                    if row.get('kind')=='request_timing' and row.get('client_request_id')=='52':
                        row['forward_started_s']=row['queued_s']+.01
                return '\n'.join(map(json.dumps,rows))
            return text
        with patch.object(Path,'read_text',read):
            with self.assertRaisesRegex(ValueError,'native-forward boundary'):
                auditor.audit(auditor.ref(CP))

    def test_truncated_journal_does_not_prove_absence(self):
        original=Path.read_text
        def read(path,*args,**kwargs):
            text=original(path,*args,**kwargs)
            if path.name=='control.jsonl':
                return '\n'.join(line for line in text.splitlines() if '"client_request_id": "52"' not in line)
            return text
        with patch.object(Path,'read_text',read):
            with self.assertRaisesRegex(ValueError,'truncated or duplicated'):
                auditor.audit(auditor.ref(CP))

    def test_unrecognized_runtime_is_not_assumed_equivalent(self):
        with patch.object(auditor,'RUNTIME_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'forward-before-native source'):
                auditor.audit(auditor.ref(CP))


if __name__=='__main__':unittest.main()
