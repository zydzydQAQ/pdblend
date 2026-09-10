import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('slo_crosscheck', Path(__file__).with_name('crosscheck.py'))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class CrosscheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def ref(self, name):
        path = self.root / name
        return dict(path=str(path), sha256=c.sha(path))

    def csv(self, name, rows):
        with (self.root / name).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def fixture(self, offsets=(0., 50.), ttfts=(1., 2.5), config_scale=.5):
        requests = [dict(request_id=str(i), planned_arrival_s=100 + offsets[i], prompt_len=5,
            output_len=3, input_tokens=5, generated_tokens=3, success='1', error='',
            token_count_source='server_usage', token_ids_verified='1', request_timeout='False',
            ttft_s=ttfts[i], tpot_s=.05) for i in range(2)]
        power = [dict(t_s=t, **{f'gpu{i}_w': 10. for i in range(8)},
                      **{f'gpu{i}_util_pct': 25. for i in range(8)}) for t in (99., 150., 201.)]
        self.csv('bench.csv', requests)
        self.csv('power.csv', power)
        c.save(self.root / 'trace.json', dict(requests=[dict(arrival_s=t, prompt_len=5, output_len=3) for t in (0., 50.)]))
        c.save(self.root / 'runtime_config.json', dict(slo_ttft_s=5 * config_scale,
            slo_tpot_s=.15 * config_scale, slo_scale=config_scale, arrival_window_s=100.))
        c.save(self.root / 'summary.json', dict(measurement_start_s=100., measurement_end_s=200.,
            trace_sha256=self.ref('trace.json')['sha256'], comparison_system='pdblend',
            fixed_window=dict(effective_slo_s=dict(ttft=2.5, tpot=.075), arrival_window_s=100., slo_scale=.5)))
        for name in ('receipt.json', 'binding.json', 'audit.json'):
            c.save(self.root / name, {})
        c.save(self.root / 'checkpoint.json', dict(artifacts={str(self.root / 'runtime_config.json'):
            c.sha(self.root / 'runtime_config.json')}))
        row = dict(cell_id='fixture', measurement_host='A', system='pdblend', rate_rps=.25,
            measurement_valid=True, metric_source_integrity='verified', reference_only=False,
            raw_requests=self.ref('bench.csv'), raw_power=self.ref('power.csv'), summary=self.ref('summary.json'),
            checkpoint=self.ref('checkpoint.json'), receipt=self.ref('receipt.json'), binding=self.ref('binding.json'),
            audit_reference=self.ref('audit.json'), trace_reference=self.ref('trace.json'),
            trace_sha256=self.ref('trace.json')['sha256'], n_expected=2, completed_work_requests=2,
            good_requests=1, slo_attainment=.5, completion_fraction=1., measurement_duration_s=100.,
            energy_j=8000., gpu_util=.25, goodput_measurement_rps=.01, energy_per_good_request_j=8000.,
            failed_requests=0, request_timeouts=0, ttft_avg_s=1.75, tpot_avg_s=.05, work_complete=True,
            generated_tokens=6, generated_token_count_complete=True, energy_measured_gpu_count=8,
            energy_per_gpu_j=[1000.] * 8, gpu_util_per_gpu=[.25] * 8)
        return row

    def test_strict_slo_equality_loses_and_energy_clips_primary(self):
        result = c.check_row(self.fixture())
        self.assertEqual(result['metrics']['good_requests'], 1)
        self.assertEqual(result['metrics']['energy_j'], 8000.)

    def test_config_and_trace_changes_fail(self):
        for kwargs, message in ((dict(offsets=(0., 49.)), 'arrival offsets'),
                                (dict(config_scale=1.), 'actual runtime')):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                c.check_row(self.fixture(**kwargs))

    def test_altered_raw_hash_fails(self):
        row = self.fixture()
        (self.root / 'bench.csv').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'changed immutable'):
            c.check_row(row)

    def test_eight_gpu_series_required(self):
        power = [dict(t_s=t, **{f'gpu{i}_w': 10. for i in range(7)},
                      **{f'gpu{i}_util_pct': 25. for i in range(7)}) for t in (0., 1.)]
        with self.assertRaises(KeyError):
            c.integrate(power, 0., 1.)

    def test_pairing_error_is_independent_of_metric_pass(self):
        row = self.fixture()
        other = dict(row, cell_id='historical', reference_only=True, measurement_host='B', trace_sha256='0' * 64)
        ledger = c.build(self.root, [row, other])
        self.assertEqual(ledger['passed_count'], 1)
        self.assertFalse(ledger['trace_pairing'][0]['observed_trace_pairing_matches'])
        self.assertEqual(len(ledger['issues']), 1)

    def test_invalid_and_missing_rows_preserved(self):
        row = self.fixture()
        ledger = c.build(self.root, [dict(row, measurement_valid=False),
            dict(row, cell_id='pending', metric_source_integrity='missing')])
        self.assertEqual([item['status'] for item in ledger['entries']],
            ['engineering_invalid_preserved', 'awaiting_minimal_mirror'])
        self.assertEqual(ledger['failed_count'], 0)


if __name__ == '__main__':
    unittest.main()
