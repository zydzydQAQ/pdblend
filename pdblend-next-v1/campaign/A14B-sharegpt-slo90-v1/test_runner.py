import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import protocol as p
import runner as r


class Queue(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def observation(self, system, scale, rate, attempt, good=89, offered=100, valid=True):
        directory = Path(self.temp.name) / str(attempt)
        directory.mkdir(exist_ok=True)
        receipt, summary = directory / 'receipt.json', directory / 'summary.json'
        receipt.write_text(json.dumps({'cpu_only': True, 'good': good}))
        summary.write_text(json.dumps({'cpu_only': True, 'offered': offered}))
        row = dict(cell_id='cpu-only-' + str(attempt), system=system, slo_scale=scale,
            rate_rps_decimal=rate, rate_rps=float(rate), executing_host='cpu-host', source_version='cpu-source',
            trace={'sha256': 'cpu-trace'}, status='running', attempt=attempt,
            receipt_path=str(receipt), summary_path=str(summary), point_result_path=str(directory / 'point-result.json'))
        verdict = dict(measurement_valid=valid, stop_eligible=valid, good_requests=good, offered_requests=offered,
            slo_attainment=good / offered, trace_sha256='cpu-trace',
            technical_error=None if valid else 'bad meter',
            **{key: row[key] for key in ('cell_id', 'system', 'slo_scale', 'rate_rps', 'executing_host', 'source_version')})
        meta = dict(schema='a14b-sharegpt-slo90-point-result-v1', protocol_id=p.PROTOCOL, complete=True,
            measurement_valid=valid, exit_status=0 if valid else 2, verdict=verdict,
            receipt_path=str(receipt), summary_path=str(summary),
            receipt_sha256=r.traces.file_sha(receipt), summary_sha256=r.traces.file_sha(summary),
            **{key: row[key] for key in ('cell_id', 'executing_host', 'source_version')})
        return row, meta

    def state(self):
        return dict(schema=1, protocol_id=p.PROTOCOL, scan=p.new_state(), records=[],
                    phase='pdblend', status='ready', next_attempt=1)

    def finish(self, state, good, offered=100, valid=True):
        key = r.next_point(state)
        self.assertIsNotNone(key)
        system, scale, rate = key
        row, meta = self.observation(system, scale, rate, len(state['records']) + 1, good, offered, valid)
        state['records'].append(row)
        state['status'] = 'running'
        return r.apply_result(state, len(state['records']) - 1, meta)

    def test_independent_endpoints_equality_and_all_baselines(self):
        s = self.state()
        s = self.finish(s, 91)
        self.assertEqual(r.next_point(s), ('pdblend', '0.5', '0.6'))
        s = self.finish(s, 90)
        self.assertEqual(r.next_point(s), ('pdblend', '0.5', '0.8'))
        s = self.finish(s, 89)
        self.assertEqual(r.next_point(s), ('pdblend', '2', '0.4'))
        s = self.finish(s, 89)
        self.assertEqual(s['scan']['scales']['0.5']['endpoint_rate'], '0.8')
        self.assertEqual(s['scan']['scales']['2']['endpoint_rate'], '0.4')
        seen = []
        while s['status'] != 'complete':
            seen.append(r.next_point(s))
            s = self.finish(s, 0)  # baseline failure never shortens its prefix
        expected = [(system, scale, rate) for system in p.BASELINES
                    for scale, rates in [('0.5', ['0.4', '0.6', '0.8']), ('2', ['0.4'])]
                    for rate in rates]
        self.assertEqual(seen, expected)
        self.assertIsNone(r.next_point(s))
        self.assertEqual(len(s['records']), 20)

    def test_invalid_below_target_does_not_establish_endpoint(self):
        s = self.finish(self.state(), 0, valid=False)
        self.assertEqual(s['status'], 'blocked')
        self.assertIsNone(s['scan']['scales']['0.5']['endpoint_rate'])
        self.assertIsNone(r.next_point(s))
        self.assertEqual(s['scan']['scales']['0.5']['next_index'], 0)

    def test_reconciliation_applies_sealed_point_once_without_dispatch(self):
        with tempfile.TemporaryDirectory() as d:
            state = self.state()
            meta = Path(d) / 'point-result.json'
            row, value = self.observation('pdblend', '.5', '.4', 1)
            row['point_result_path'] = str(meta)
            r.save(meta, value)
            state['records'].append(row)
            state['status'] = 'running'
            path = Path(d) / 'state.json'
            r.save(path, state)
            s = r.reconcile(path)
            self.assertEqual(r.next_point(s), ('pdblend', '2', '0.4'))
            self.assertEqual(r.reconcile(path), s)
            self.assertEqual(len(s['records']), 1)

    def test_crashed_unsealed_attempt_cannot_be_repeated(self):
        with tempfile.TemporaryDirectory() as d:
            state = self.state()
            state['records'].append(dict(status='running', point_result_path=str(Path(d) / 'missing')))
            path = Path(d) / 'state.json'
            r.save(path, state)
            self.assertEqual(r.reconcile(path)['status'], 'blocked')
            self.assertIsNone(r.next_point(r.read(path)))

    def test_valid_result_cannot_be_applied_twice(self):
        s = self.finish(self.state(), 100)
        with self.assertRaises(ValueError):
            r.apply_result(s, 0, {})

    def test_extension_rates_are_unbounded_declared_multiples(self):
        s = self.state()
        observed = []
        for _ in range(16):
            observed.append(r.next_point(s)[2])
            s = self.finish(s, 100)
        self.assertEqual(observed[-4:], ['9', '13.5', '20.25', '30.375'])

    def test_foreign_meta_cannot_advance_or_close_a_rate(self):
        for changed in ('cell_id', 'executing_host', 'source_version'):
            state = self.state()
            row, meta = self.observation('pdblend', '.5', '.4', 1)
            state['records'].append(row)
            state['status'] = 'running'
            meta[changed] = 'different'
            result = r.apply_result(state, 0, meta)
            self.assertEqual(result['status'], 'blocked')
            self.assertIsNone(result['scan']['scales']['0.5']['endpoint_rate'])

    def test_changed_receipt_and_nonzero_worker_exit_are_invalid(self):
        state = self.state()
        row, meta = self.observation('pdblend', '.5', '.4', 1)
        state['records'].append(row)
        state['status'] = 'running'
        Path(row['receipt_path']).write_text('changed')
        result = r.apply_result(state, 0, meta)
        self.assertEqual(result['status'], 'blocked')
        self.assertIn('digest differs', result['records'][0]['technical_error'])
        row, meta = self.observation('pdblend', '.5', '.4', 1)
        state['records'][0] = row
        meta['process_exit_status'] = 137
        self.assertEqual(r.apply_result(state, 0, meta)['status'], 'blocked')


if __name__ == '__main__':
    unittest.main()
