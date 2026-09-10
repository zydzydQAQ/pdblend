import copy
import json
import unittest

from audit import classify_requests


class ServiceTerminalAuditTests(unittest.TestCase):
    def fixture(self):
        request = dict(request_id='0', prompt_len='8', output_len='2', input_tokens='8', generated_tokens='2',
            success='True', error='', token_count_source='server_usage', token_ids_verified='True',
            request_timeout='False', planned_arrival_s='100', actual_dispatch_s='100.001',
            request_deadline_s='220', http_status='200', admission_rejection='',
            first_token_s='101', last_token_s='102', n_text_chunks='2', token_stream_opened='True')
        timing = dict(kind='request_timing', client_request_id='0', request_id='native0',
            planned_arrival_s=100., actual_dispatch_s=100.001, hard_deadline_s=220.,
            forward_started_s=100.1, stream_end_s=102., cleanup_end_s=102.1, completed=True)
        ending = dict(kind='request_end', request_id='native0', at_s=102.05, completed=True)
        return [request], dict(requests=[dict(prompt_len=8, output_len=2)]), dict(n_requests=1, request_hard_timeout_s=120), [timing, ending]

    def timeout(self, forwarded=True):
        args = self.fixture(); request = args[0][0]
        request.update(success='False', error='request_hard_timeout', request_timeout='True',
            generated_tokens='1' if forwarded else '0', input_tokens='0', token_count_source='',
            token_ids_verified='False', n_text_chunks='1' if forwarded else '0',
            token_stream_opened='True' if forwarded else 'False',
            first_token_s='101' if forwarded else '', last_token_s='101' if forwarded else '')
        timing, ending = args[3]
        timing.update(completed=False, cleanup_end_s=220.1); timing.pop('stream_end_s')
        ending.update(completed=False, at_s=220.05)
        if not forwarded:
            timing.pop('forward_started_s')
        return args

    def test_complete_output_has_terminal_proof(self):
        result = classify_requests(*self.fixture())
        self.assertTrue(result['service_terminal_valid'])
        self.assertEqual(result['failed_request_ids'], [])

    def test_partial_stream_deadline_is_service_failure(self):
        result = classify_requests(*self.timeout())
        self.assertEqual(result['failed_request_ids'], ['0'])
        self.assertEqual(result['zero_output_ids'], [])

    def test_unforwarded_deadline_needs_positive_journal_proof(self):
        result = classify_requests(*self.timeout(False))
        self.assertEqual(result['zero_output_ids'], ['0'])
        args = self.timeout(False); args[3].clear()
        with self.assertRaisesRegex(ValueError, 'terminal evidence'):
            classify_requests(*args)

    def test_early_deadline_or_unknown_http_error_is_engineering(self):
        args = self.timeout(); args[3][0]['cleanup_end_s'] = 219
        args[3][1]['at_s'] = 218.9
        with self.assertRaisesRegex(ValueError, 'before declared deadline'):
            classify_requests(*args)
        args = self.timeout(); args[0][0].update(http_status='503', error='RuntimeError: HTTP 503: failure')
        with self.assertRaisesRegex(ValueError, 'isolated request deadline'):
            classify_requests(*args)

    def test_exact_queue_full_is_service_failure_without_native_forwarding(self):
        args = self.timeout(False); request = args[0][0]
        request.update(request_timeout='False', http_status='429', admission_rejection='admission_queue_full',
            error='RuntimeError: HTTP 429: ' + json.dumps(dict(error=dict(type='admission_rejection',
                code='admission_queue_full', message='admission queue full'))))
        events = args[3].copy(); args[3].clear()
        self.assertEqual(classify_requests(*args)['zero_output_ids'], ['0'])
        args[3].extend(events)
        with self.assertRaisesRegex(ValueError, 'was admitted'):
            classify_requests(*args)

    def test_unknown_rejection_and_missing_or_duplicate_requests_fail_closed(self):
        args = self.timeout(); args[0][0]['admission_rejection'] = 'unrecognized'
        with self.assertRaisesRegex(ValueError, 'unknown capacity rejection'):
            classify_requests(*args)
        args = self.fixture(); args[0].append(copy.deepcopy(args[0][0]))
        with self.assertRaisesRegex(ValueError, 'denominator differs'):
            classify_requests(*args)
        args = self.fixture(); args[3].pop()
        with self.assertRaisesRegex(ValueError, 'timing/end mismatch'):
            classify_requests(*args)


if __name__ == '__main__':
    unittest.main()
