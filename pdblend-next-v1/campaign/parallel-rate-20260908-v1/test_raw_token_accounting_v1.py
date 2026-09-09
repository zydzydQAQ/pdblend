import copy
import unittest
from raw_metrics_v3 import token_accounting


class TokenAccountingTests(unittest.TestCase):
    def setUp(self):
        self.complete = dict(generated_tokens='512', token_count_source='server_usage',
            token_ids_verified='1', success='1', admission_rejection='', n_text_chunks='490',
            first_token_s='1', last_token_s='3')
        self.partial = dict(generated_tokens='0', token_count_source='missing',
            token_ids_verified='0', success='0', admission_rejection='', n_text_chunks='266',
            first_token_s='1', last_token_s='3')
        self.rejected = dict(generated_tokens='0', token_count_source='missing',
            token_ids_verified='0', success='0', admission_rejection='admission_queue_full',
            n_text_chunks='0', first_token_s='', last_token_s='', http_status='429',
            error='HTTP 429: {"error":{"type":"admission_rejection","code":"admission_queue_full"}}')

    def test_complete_exact(self):
        self.assertTrue(token_accounting([self.complete], 512)['generated_token_count_complete'])

    def test_chunks_are_not_tokens(self):
        value = token_accounting([self.complete, self.partial], 512)
        self.assertEqual(value['verified_generated_tokens'], 512)
        self.assertEqual(value['partial_output_chunks_without_verified_usage'], 266)
        self.assertTrue(value['recorded_generated_tokens_are_lower_bound'])
        self.assertFalse(value['generated_token_count_complete'])

    def test_rejected_before_service_exact_zero(self):
        self.assertTrue(token_accounting([self.rejected], 0)['generated_token_count_complete'])

    def test_rejection_after_output_unknown(self):
        row = dict(self.rejected, n_text_chunks='1', first_token_s='2')
        self.assertFalse(token_accounting([row], 0)['generated_token_count_complete'])

    def test_unknown_nonzero_not_claimed_lower_bound(self):
        value = token_accounting([dict(self.partial, generated_tokens='8')], 8)
        self.assertFalse(value['recorded_generated_tokens_are_lower_bound'])
        self.assertEqual(value['verified_generated_tokens'], 0)

    def test_no_output_timeout_still_unknown(self):
        row = dict(self.partial, n_text_chunks='0', first_token_s='', last_token_s='')
        value = token_accounting([row], 0)
        self.assertEqual(value['unverified_output_requests'], 1)
        self.assertEqual(value['unverified_partial_output_requests'], 0)

    def test_total_disagreement_rejected(self):
        with self.assertRaises(ValueError):
            token_accounting([self.complete], 0)

    def test_named_rejection_must_match_error(self):
        row = dict(self.rejected, admission_rejection='different_code')
        with self.assertRaises(ValueError):
            token_accounting([row], 0)


if __name__ == '__main__':
    unittest.main()
