import asyncio
import importlib.util
import json
from pathlib import Path
import unittest

import bench_vllm as bench


class Response:
    def __init__(self, events, status=200, error=None, body=''):
        self.events, self.status, self.error, self.body = events, status, error, body
        self.content = self.lines()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def text(self):
        return self.body

    async def lines(self):
        for event in self.events:
            data = event if isinstance(event, str) else json.dumps(event)
            yield ('data: ' + data + '\n').encode()
        if self.error:
            raise self.error


class Session:
    def __init__(self, response):
        self.response = response

    def post(self, *args, **kwargs):
        return self.response


def event(ids, index):
    return dict(choices=[dict(text='x')], token_ids=ids, token_index=index)


class CollectorTest(unittest.TestCase):
    def collect(self, response, length=3):
        result = asyncio.run(bench.send_request(Session(response), 'http://127.0.0.1:8080',
            'fixture', [1, 2], length, request_id='0', evaluation_protocol=bench.EVALUATION_V3))
        trace = dict(requests=[dict(prompt_len=2, output_len=length)])
        row = bench.bench_rows(trace, [result], 1, .1)[0]
        original = Path('/root/workspace/pdblend-next-v1/releases/five-system100-B32B-v1-runtime/benchmarks/scripts/bench_vllm.py')
        spec = importlib.util.spec_from_file_location('legacy_bench', original)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        old = legacy.bench_rows(trace, [result], 1, .1)[0]
        self.assertEqual(old, {key: row[key] for key in old})
        return row

    def test_timeout_retains_batched_prefix_without_changing_latency_or_success(self):
        row = self.collect(Response([event([7, 7, 9], 3)], error=asyncio.TimeoutError()))
        self.assertEqual(row['generated_tokens'], 0)
        self.assertEqual(row['received_token_count'], 3)
        self.assertEqual(row['received_token_count_exact'], 1)
        self.assertEqual(json.loads(row['received_token_ids']), [7, 7, 9])
        self.assertEqual(row['success'], 0)
        self.assertIsNone(row['tpot_s'])
        self.assertEqual(row['token_itl_exact'], 0)

    def test_cancel_retains_prefix(self):
        row = self.collect(Response([event([4], 1)], error=asyncio.CancelledError()))
        self.assertEqual((row['received_token_count'], row['received_token_count_exact']), (1, 1))
        self.assertTrue(row['request_timeout'])

    def test_success_usage_matches_sequence(self):
        row = self.collect(Response([event([4, 5], 2), event([6], 3),
            dict(usage=dict(completion_tokens=3, prompt_tokens=2)), '[DONE]']))
        self.assertEqual((row['success'], row['generated_tokens'], row['received_token_count_exact']), (1, 3, 1))

    def test_usage_mismatch_is_not_exact(self):
        row = self.collect(Response([event([4], 1),
            dict(usage=dict(completion_tokens=3, prompt_tokens=2)), '[DONE]']))
        self.assertEqual(row['success'], 0)
        self.assertEqual(row['received_token_count_exact'], 0)

    def test_duplicate_sequence_is_not_exact(self):
        row = self.collect(Response([event([4], 1), event([4], 1)]))
        self.assertEqual(row['received_token_count_exact'], 0)
        self.assertIn('out_of_order', row['error'])

    def test_text_without_token_ids_is_not_exact(self):
        row = self.collect(Response([dict(choices=[dict(text='abc')])], error=asyncio.TimeoutError()))
        self.assertEqual(row['received_token_count_exact'], 0)

    def test_missing_index_is_not_exact(self):
        row = self.collect(Response([dict(choices=[dict(text='abc')], token_ids=[4])], error=asyncio.TimeoutError()))
        self.assertEqual(row['received_token_count_exact'], 0)

    def test_503_is_not_automatically_proven_zero(self):
        row = self.collect(Response([], status=503, body='decode failed: bounded admission queue full'))
        self.assertEqual(row['received_token_count_exact'], 0)
        self.assertEqual(row['received_token_count'], 0)
        self.assertEqual(row['success'], 0)

    def test_malformed_json_invalidates_prefix_proof(self):
        row = self.collect(Response([event([4], 1), '{bad']))
        self.assertEqual(row['received_token_count_exact'], 0)


if __name__ == '__main__':
    unittest.main()
