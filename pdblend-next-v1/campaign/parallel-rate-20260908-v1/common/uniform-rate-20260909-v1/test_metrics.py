import unittest
from metrics import integrate, request_metrics


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


if __name__ == '__main__':
    unittest.main()
