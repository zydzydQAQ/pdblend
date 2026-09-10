import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

import native_queue as q
import cell_audit

ROOT, HERE = q.ROOT, Path(__file__).parent


def fixture():
    common = dict(success='0', request_timeout='False', admission_rejection='',
        generated_tokens='0', n_text_chunks='0', first_token_s='', last_token_s='')
    good = dict(common, request_id='0', success='1', error='', http_status='200',
        token_count_source='server_usage', token_ids_verified='True', generated_tokens='64')
    refused = dict(common, request_id='1', http_status='503', error=q.NATIVE_ERROR)
    deadline = dict(common, request_id='2', http_status='200', request_timeout='True',
        error='request_hard_timeout', generated_tokens='9')
    events = []
    for key in ('0', '1', '2'):
        events.append(dict(kind='request_timing', client_request_id=key, request_id=key,
            forward_started_s=10 if key == '0' else 11, stream_end_s=12,
            cleanup_end_s=11.1, hard_deadline_s=130))
        if key != '2':
            events.append(dict(kind='admission', client_request_id=key,
                plan=dict(routes=[dict(decode_id='engine')])) )
    return [good, refused, deadline], events


class NativeCapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = q.p.load(q.ORIGINAL, 'test_original_native_queue')

    def test_mixed_refusal_deadline_partition(self):
        rows, events = fixture()
        details, deadlines = q.queue_details(rows, events, self.original)
        self.assertEqual([r['request_id'] for r in details], ['1'])
        self.assertEqual(deadlines, ['2'])
        self.assertEqual(rows[2]['generated_tokens'], '9')

    def test_unknown_503_not_generalized(self):
        rows, events = fixture()
        for key, value in (('error', 'RuntimeError: HTTP 503: scheduler stopped'),
                ('request_timeout', 'True'), ('n_text_chunks', '1'),
                ('first_token_s', '11.2'), ('generated_tokens', '1')):
            with self.subTest(key=key):
                bad = copy.deepcopy(rows); bad[1][key] = value
                with self.assertRaises((ValueError, AssertionError)):
                    q.queue_details(bad, events, self.original)

    def test_unknown_failure_and_duplicate_timing_rejected(self):
        rows, events = fixture()
        bad = copy.deepcopy(rows); bad[2]['http_status'] = '429'
        with self.assertRaises(ValueError): q.queue_details(bad, events, self.original)
        with self.assertRaises(ValueError): q.queue_details(rows, events + [events[0]], self.original)
        with self.assertRaises((ValueError, AssertionError)):
            q.queue_details(rows, [e for e in events if not (e['kind'] == 'admission' and e['client_request_id'] == '1')], self.original)

    def test_nonzero_native_output_and_no_active_work_rejected(self):
        rows, events = fixture()
        changed = copy.deepcopy(events); changed[0]['stream_end_s'] = 10.5
        with self.assertRaises(AssertionError): q.queue_details(rows, changed, self.original)

    def test_label_and_contract_keep_mixed_counts_separate(self):
        proof = dict(schema='C-Eco-native128-refusal-and-deadline-audit-v1',
            passed=True, independently_recomputed=True, checkpoint={'path': 'cp', 'sha256': 'same'},
            no_unknown_errors=True, cleanup_verified=True, native_rejection_request_ids=['1'],
            timeout_request_ids=['2'], native_rejections=1, request_timeouts=1)
        obs = dict(system='ecoserve', model='7b', measurement_host='C', work_complete=False,
            checkpoint=proof['checkpoint'], measurement_valid=True, zero_output_diagnosis=proof,
            n_expected=3, completed_work_requests=1, request_timeouts=1,
            baseline_service_failure=dict(classification='baseline_explicit_request_hard_timeout',
                independently_diagnosed=True, failed_request_ids=['1', '2']))
        result = cell_audit.classify(obs)
        classification = result['baseline_service_failure']
        self.assertEqual(classification['classification'], 'baseline_explicit_native_admission_queue_full')
        self.assertEqual((classification['native_rejections'], classification['actual_request_timeouts']), (1, 1))
        contract = q.p.load(HERE / 'contract.py', 'test_native_contract')
        old = q.p.load(HERE.parent / 'contract_capacity_v1.py', 'test_previous_contract')
        with tempfile.TemporaryDirectory() as tmp:
            diagnosis = Path(tmp) / 'diagnosis.json'
            q.p.save(diagnosis, dict(passed=True, independently_recomputed=True,
                checkpoint=obs['checkpoint'], classification=classification,
                no_unknown_errors=True, no_PDB_complete_boundary_claim=True))
            obs.update(failure_class='independently_diagnosed_capacity_rejection', diagnosis_reference=q.p.ref(diagnosis))
            self.assertTrue(contract.acceptable_baseline(obs))
            self.assertFalse(old.acceptable_baseline(obs))
            bad = copy.deepcopy(obs); bad['checkpoint']['sha256'] = 'different'
            self.assertFalse(contract.acceptable_baseline(bad))
            bad = copy.deepcopy(obs); bad['measurement_host'] = 'B'
            self.assertFalse(contract.acceptable_baseline(bad))

    def test_real_four_native_windows_all_metrics_equal(self):
        metrics = q.p.load(HERE / 'metrics.py', 'test_independent_native_metrics')
        paths = list((ROOT / 'common/uniform-rate-20260909-v2/diagnostics').glob('zero-output-*.json'))
        self.assertEqual(len(paths), 4)
        fields = ('n_expected', 'completed_work_requests', 'good_requests', 'generated_tokens',
            'expected_generated_tokens', 'slo_attainment', 'request_timeouts', 'ttft_avg_s',
            'tpot_avg_s', 'energy_j', 'gpu_util', 'request_throughput_rps', 'token_throughput_tps',
            'token_throughput_is_exact', 'actual_output_tokens', 'measurement_duration_s')
        for path in paths:
            old = q.p.read(path)
            actual = metrics.audit_checkpoint(old['checkpoint']['path'])
            for key in fields:
                self.assertEqual(actual[key], old[key], (path.name, key))
            proof = actual['zero_output_diagnosis']
            self.assertGreater(proof['native_rejections'], 0)
            self.assertEqual(proof['request_timeouts'], 0)
            self.assertEqual(proof['failed_requests'], old['failed_requests'])

    def test_source_equivalence_only_declared_cpu_branches(self):
        pairs = ((ROOT / 'common/uniform-rate-20260909-v2/metrics_native_refs_v3.py', HERE / 'metrics.py', {'diagnosed_zero_output'}),
            (HERE.parent / 'contract_capacity_v1.py', HERE / 'contract.py', {'acceptable_baseline'}),
            (HERE.parent / 'run_cells_v3.py', HERE / 'run_cells.py', {'execute'}))
        for before, after, permitted in pairs:
            def functions(path):
                return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name not in permitted}
            self.assertEqual(functions(before), functions(after))
        original = (HERE.parent / 'run_cells_v3.py').read_text()
        expected = original.replace("rejection = observation['baseline_service_failure'].get('classification') == 'baseline_explicit_controller_admission_queue_full'",
            "rejection = observation['baseline_service_failure'].get('classification') in ('baseline_explicit_controller_admission_queue_full', 'baseline_explicit_native_admission_queue_full')")
        self.assertEqual((HERE / 'run_cells.py').read_text(), expected)


if __name__ == '__main__': unittest.main()
