import csv
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
HELPER=ROOT/'common/token-evidence-v2/audit_controller_queue_429_v1.py'
spec=importlib.util.spec_from_file_location('controller429',HELPER)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
CP=next((ROOT/'C/uniform-rate-20260909-v2/pipeline-003/0023-dynamollm-alpaca-r18.0-normal/measurement/results/checkpoints').glob('*.json'))


class ControllerQueueTests(unittest.TestCase):
    def test_real_rejections_preserve_denominator_and_zero_timeouts(self):
        result=q.audit(q.ref(CP))
        self.assertEqual((result['n_expected'],result['controller_rejections'],result['controller_accepted_requests']),
                         (1806,345,1461))
        self.assertEqual(result['request_timeouts'],0)
        self.assertEqual(result['actual_output_tokens'],166630)
        self.assertTrue(result['token_throughput_is_exact'])
        self.assertTrue(all(r['actual_output_tokens']==0 for r in result['refusal_details']))

    def alter_request(self,edit):
        original=csv.DictReader
        def changed(stream,*args,**kwargs):
            rows=list(original(stream,*args,**kwargs))
            if rows and 'admission_rejection' in rows[0]:
                request=next(r for r in rows if r['admission_rejection']=='admission_queue_full')
                request.update(edit)
            return iter(rows)
        return patch.object(csv,'DictReader',changed)

    def test_foreign_429_body_is_not_accepted(self):
        with self.alter_request({'error':'RuntimeError: HTTP 429: {"error":{"code":"rate_limit"}}'}):
            with self.assertRaisesRegex(ValueError,'foreign or malformed'):q.audit(q.ref(CP))

    def test_timeout_cannot_be_relabelled_queue_rejection(self):
        with self.alter_request({'request_timeout':'True'}):
            with self.assertRaisesRegex(ValueError,'explicit controller rejection'):q.audit(q.ref(CP))

    def test_output_evidence_disallows_zero(self):
        with self.alter_request({'received_token_count':'1','received_token_ids':'[9]'}):
            with self.assertRaisesRegex(ValueError,'output evidence'):q.audit(q.ref(CP))

    def test_rejection_with_controller_timing_is_not_unforwarded(self):
        original=Path.read_text
        def changed(path,*args,**kwargs):
            text=original(path,*args,**kwargs)
            if path.name=='control.jsonl':
                rows=[json.loads(line) for line in text.splitlines()]
                next(r for r in rows if r.get('kind')=='request_timing')['client_request_id']='932'
                return '\n'.join(map(json.dumps,rows))
            return text
        with patch.object(Path,'read_text',changed):
            with self.assertRaisesRegex(ValueError,'journal is incomplete or a rejected request was forwarded'):q.audit(q.ref(CP))

    def test_unknown_source_is_not_assumed_equivalent(self):
        with patch.dict(q.SOURCES,{'src/ecopadg/serving/runtime.py':'0'*64}):
            with self.assertRaisesRegex(ValueError,'exact controller/collector source'):q.audit(q.ref(CP))

if __name__=='__main__':unittest.main()
