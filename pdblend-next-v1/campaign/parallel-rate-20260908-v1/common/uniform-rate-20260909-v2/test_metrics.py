import unittest
import hashlib
import json
from metrics import integrate, request_metrics, received_output_count


class MetricsTests(unittest.TestCase):
    def test_window_integration_excludes_outer_overlap(self):
        rows = [dict(t_s=t, power=100, util=50) for t in (0, 10, 20)]
        self.assertEqual(integrate(rows, 5, 15, ['power', 'util']), [1000, 500])
        with self.assertRaises(ValueError):
            integrate(rows, -1, 15, ['power'])

    def test_failed_requests_in_denominator_and_threshold_is_strict(self):
        trace = {'requests': [dict(idx=i, prompt_len=8, output_len=2) for i in range(3)]}
        config = dict(n_requests=3, slo_ttft_s=1., slo_tpot_s=.1)
        rows = [dict(idx=str(i), request_id=str(i), prompt_len='8', output_len='2', generated_tokens='2',
            token_count_source='server_usage', token_ids_verified='1', success='1', error='', request_timeout='0',
            input_tokens='8', ttft_s='.5', tpot_s='.05') for i in range(3)]
        rows[1]['ttft_s'] = '1'
        rows[2].update(success='0', error='timeout', generated_tokens='0', ttft_s='', tpot_s='', request_timeout='1')
        result = request_metrics(rows, trace, config)
        self.assertEqual(result['good_requests'], 1)
        self.assertEqual(result['slo_attainment'], 1 / 3)
        self.assertEqual(result['ttft_avg_s'], .75)
        self.assertFalse(result['work_complete'])
        self.assertEqual(result['request_timeouts'], 1)
        self.assertIsNone(result['actual_output_tokens'])

    def prefix(self):
        ids = [2, 2, 5]
        return dict(token_evidence_schema='2', received_token_count_exact='1', token_stream_opened='1',
            token_sequence_verified='1', token_stream_parse_error='0', received_token_ids=json.dumps(ids),
            received_token_events=json.dumps([dict(token_index=2, token_ids=[2, 2]), dict(token_index=3, token_ids=[5])]),
            received_token_count='3', output_len='8', output_token_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            token_count_source='missing', generated_tokens='0')

    def test_batched_prefix_audited_independently(self):
        row = self.prefix()
        self.assertEqual(received_output_count(row), 3)
        row['received_token_events'] = json.dumps([dict(token_index=3, token_ids=[2, 2, 4])])
        with self.assertRaises(ValueError):
            received_output_count(row)

    def test_duplicate_or_missing_index_fails_despite_exact_flag(self):
        row = self.prefix()
        events = json.loads(row['received_token_events'])
        events[1]['token_index'] = 2
        row['received_token_events'] = json.dumps(events)
        with self.assertRaises(ValueError):
            received_output_count(row)

    def test_partial_token_count_preserves_failed_slo_and_missing_tpot(self):
        row = dict(self.prefix(), idx='0', request_id='0', prompt_len='8', input_tokens='0',
            token_ids_verified='0', success='0', error='request_hard_timeout', request_timeout='1',
            n_text_chunks='2', ttft_s='.3', tpot_s='')
        trace = dict(requests=[dict(idx=0, prompt_len=8, output_len=8)])
        result = request_metrics([row], trace, dict(n_requests=1, slo_ttft_s=1., slo_tpot_s=.1))
        self.assertEqual(result['actual_output_tokens'], 3)
        self.assertEqual(result['generated_tokens'], 0)
        self.assertEqual(result['slo_attainment'], 0)
        self.assertFalse(result['work_complete'])
        self.assertIsNone(result['tpot_avg_s'])


if __name__ == '__main__':
    unittest.main()
