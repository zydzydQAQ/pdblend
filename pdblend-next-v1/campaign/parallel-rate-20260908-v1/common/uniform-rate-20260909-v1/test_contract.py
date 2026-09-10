"""CPU regression tests of scientific scheduling, pairing and strict endpoints."""
import copy
from decimal import Decimal
import tempfile
from pathlib import Path
import unittest
import contract as c


def workload(model='7b', dataset='sharegpt', rate='.25'):
    return dict(model=model, dataset=dataset, node=c.HOSTS[model], measurement_host=c.HOSTS[model],
        rate_rps=float(rate), rate_rps_decimal=c.number(rate), seed=701, arrival_seed=701,
        sampling_seed=20260907, arrival_window_s=100, trace_duration_s=100,
        trace_sha256='a'*64, content_pairing_sha256='b'*64,
        slo_ttft_s=c.SLOS[dataset][0], slo_tpot_s=c.SLOS[dataset][1],
        trace_reference=dict(path='/unopened/test-trace.json', sha256='a'*64))


def position(model='7b', dataset='sharegpt', rate='.25'):
    w = workload(model, dataset, rate)
    systems = {s: [c.new_task(c.row_for(w, c.HOSTS[model], s, r, 1)) for r in (1, 2)] for s in c.SYSTEMS}
    return dict(position_id=c.position_id(c.HOSTS[model], model, dataset, rate), node=c.HOSTS[model],
        model=model, dataset=dataset, rate_rps=float(rate), rate_rps_decimal=c.number(rate), workload=w,
        trace=w['trace_reference'], required_pdb_repeats=[1, 2], systems=systems, newly_added_coordinate=True)


def observation(p, repeat, slo=1., valid=True, complete=True):
    task = p['systems']['pdblend'][repeat-1]
    return dict(p['workload'], cell_id=task['cell_id'], system='pdblend', repeat=repeat,
                slo_attainment=slo, measurement_valid=valid, work_complete=complete)


class ContractTests(unittest.TestCase):
    def test_nine_user_grids(self):
        for m in c.MODELS:
            for d in c.DATASETS:
                rates = c.grid(m, d, c.INITIAL_LIMITS[m][d])
                ds = list(map(Decimal, rates))
                self.assertEqual(ds[0], c.step(m, d))
                self.assertTrue(all(b-a == c.step(m, d) for a, b in zip(ds, ds[1:])))
        self.assertEqual(c.grid('32b', 'longbench', '.5'), ['0.25', '0.5'])
        self.assertFalse(c.on_grid('32b', 'longbench', '.3'))
        self.assertFalse(c.on_grid('7b', 'alpaca', '18.75'))

    def test_decimal_validation(self):
        self.assertEqual(c.number('0.5000'), '0.5')
        for bad in (True, 0, -1, 'NaN', 'Infinity', 'nonsense'):
            with self.assertRaises(ValueError): c.number(bad)
        with self.assertRaises(ValueError): c.grid('7b', 'sharegpt', '.3')

    def test_joint_strict_latency_boundaries(self):
        for d, (ttft, tpot) in c.SLOS.items():
            self.assertTrue(c.strict_slo_pass(d, ttft*.9, tpot*.9, work_complete=True))
            self.assertFalse(c.strict_slo_pass(d, ttft, tpot*.9, work_complete=True))
            self.assertFalse(c.strict_slo_pass(d, ttft*.9, tpot, work_complete=True))
            self.assertFalse(c.strict_slo_pass(d, ttft*.9, tpot*.9, work_complete=False))
            self.assertFalse(c.strict_slo_pass(d, None, tpot*.9, work_complete=True))

    def test_any_loss_caps_before_other_repeat(self):
        p = position(); d = c.evaluate_rate(p, [observation(p, 1, .899)])
        self.assertTrue(d['cap_observed']); self.assertFalse(d['increase_rate_allowed'])
        self.assertEqual(d['status'], 'complete_current_rate_repeats')

    def test_straddling_repeats_never_averaged_into_pass(self):
        p = position(); d = c.evaluate_rate(p, [observation(p, 1, .878788), observation(p, 2, .939394)])
        self.assertEqual(d['status'], 'cap_complete_work_SLO_below_90')
        self.assertTrue(d['threshold_straddles']); self.assertFalse(d['increase_rate_allowed'])

    def test_exact_ninety_passes(self):
        p = position(); d = c.evaluate_rate(p, [observation(p, 1, .9), observation(p, 2, .9)])
        self.assertEqual(d['status'], 'advance'); self.assertFalse(d['cap_observed'])

    def test_invalid_or_incomplete_not_capacity_boundary(self):
        p = position()
        for kwargs in ({'valid': False}, {'complete': False}):
            d = c.evaluate_rate(p, [observation(p, 1, .1, **kwargs), observation(p, 2)])
            self.assertEqual(d['status'], 'stop_for_engineering_diagnosis')
            self.assertFalse(d['cap_observed']); self.assertFalse(d['increase_rate_allowed'])

    def test_cross_host_trace_and_repeat_rejected(self):
        p = position()
        for field, value in [('measurement_host', 'B'), ('trace_sha256', 'z'*64), ('seed', 1701), ('rate_rps', .5), ('repeat', 2)]:
            o = observation(p, 1); o[field] = value
            with self.assertRaises(ValueError): c.evaluate_rate(p, [o])

    def test_lower_gap_prevents_higher_dispatch_and_cap_prunes(self):
        p, q = position(rate='.25'), position(rate='.5')
        g = dict(model='7b', dataset='sharegpt', node='C', positions=[p, q], reused_observations=[], pairings=[])
        d = c.select_group(g, [observation(q, 1), observation(q, 2)])
        self.assertEqual(d['position_id'], p['position_id'])
        self.assertEqual(d['phase'], 'pdblend')
        d = c.select_group(g, [observation(p, 1, .89), observation(p, 2, .95)])
        self.assertEqual(d['phase'], 'baselines')
        self.assertEqual(len(d['baseline_tasks']), 8)
        self.assertTrue(all(t['row']['rate_rps'] == .25 for t in d['baseline_tasks']))

    def test_extension_uses_fixed_addition(self):
        p = position(rate='.25')
        g = dict(model='7b', dataset='sharegpt', node='C', positions=[p], reused_observations=[], pairings=[])
        d = c.select_group(g, [observation(p, 1), observation(p, 2)])
        self.assertEqual(d['next_rate_rps_decimal'], '0.5')

    def test_two_new_repeats_share_identical_trace(self):
        p = position()
        self.assertEqual(sum(len(ts) for ts in p['systems'].values()), 10)
        self.assertEqual({t['row']['trace_sha256'] for ts in p['systems'].values() for t in ts}, {'a'*64})

    def test_dynamic_audit_reuse_deduplicates_and_blocks_new_a(self):
        p = position(); g = dict(node='C', positions=[p], pairings=[], reused_observations=[])
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp)/'cp.json'; cp.write_bytes(c.encode(dict(row=p['systems']['pdblend'][0]['row'])))
            o = observation(p, 1); o['cell_id'] = 'original-actual-cell'; o['checkpoint'] = c.ref(cp)
            audit = Path(tmp)/'audit.json'
            audit.write_bytes(c.encode(dict(slo_threshold_comparison='strict_lt', observations=[o])))
            o['audit_reference'] = c.ref(audit)
            updated = c.apply_audited_reuse(g, [o])
            self.assertEqual(updated['positions'][0]['systems']['pdblend'][0]['action'], 'reuse')
            self.assertEqual(g['positions'][0]['systems']['pdblend'][0]['action'], 'execute')
            self.assertEqual(len(c.apply_audited_reuse(updated, [o])['reused_observations']), 1)
            with self.assertRaises(ValueError): c.apply_audited_reuse(dict(g, node='Anew20260909'), [o])
            audit.write_bytes(c.encode(dict(observations=[o])))
            o['audit_reference'] = c.ref(audit)
            with self.assertRaises(ValueError): c.apply_audited_reuse(g, [o])

    def test_partial_baseline_needs_independent_deadline_diagnosis(self):
        p = position(); g = dict(model='7b', dataset='sharegpt', node='C', positions=[p], pairings=[], reused_observations=[])
        base = dict(p['workload'], cell_id=p['systems']['distserve'][0]['cell_id'], system='distserve',
                    repeat=1, measurement_valid=True, work_complete=False, slo_attainment=.05)
        pdb = [observation(p, 1, .89), observation(p, 2, .95)]
        self.assertEqual(c.select_group(g, pdb + [base])['phase'], 'diagnosis')
        known = dict(base, failure_class='independently_diagnosed_capacity_deadline',
                     diagnosis_reference=dict(path='/already-audited/diagnosis.json', sha256='a'*64))
        result = c.select_group(g, pdb + [known])
        self.assertEqual(result['phase'], 'baselines')
        self.assertEqual(len(result['baseline_tasks']), 7)


if __name__ == '__main__':
    unittest.main()
