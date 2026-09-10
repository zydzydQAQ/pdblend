"""CPU acceptance tests for crossing, same-host reuse and metric completeness."""
import copy
import unittest
from decimal import Decimal
import contract as c


def position(rate='1.5', model='14b', dataset='alpaca', supplement=False):
    node = c.host(model, dataset)
    workload = dict(model=model, dataset=dataset, node=node, measurement_host=node,
        rate_rps=float(rate), rate_rps_decimal=c.number(rate), seed=701, trace_sha256='trace',
        content_pairing_sha256='content', slo_ttft_s=c.SLOS[dataset][0], slo_tpot_s=c.SLOS[dataset][1])
    tasks = {s: [c.new_task(c.row_for(workload, node, s, 1, 1))] for s in c.SYSTEMS}
    tasks['pdblend'].append(c.new_task(c.row_for(workload, node, 'pdblend', 2, 2)))
    return dict(position_id=c.position_id(node, model, dataset, rate), node=node, model=model, dataset=dataset,
        rate_rps=float(rate), rate_rps_decimal=c.number(rate), workload=workload, systems=tasks,
        metric_supplements=[c.new_task(c.row_for(workload, node, 'mixed', 1, 3, 'metric_supplement'))] if supplement else [])


def observation(task, attainment=1., complete=True, exact=True):
    result = copy.deepcopy(task['row'])
    result.update(measurement_valid=True, work_complete=complete, slo_attainment=attainment,
                  strict_slo_recomputed=True, token_throughput_is_exact=exact)
    return result


def group(*positions):
    return dict(node=positions[0]['node'], model=positions[0]['model'], dataset=positions[0]['dataset'],
                positions=list(positions), reused_observations=[], pairings=[])


class Decisions(unittest.TestCase):
    def test_exact_grids_and_host_migration(self):
        self.assertEqual(c.grid('32b', 'longbench', '.30'), ['0.05', '0.1', '0.15', '0.2', '0.25', '0.3'])
        self.assertFalse(c.on_grid('32b', 'longbench', '.30000000000000004'))
        self.assertEqual(c.host('14b', 'sharegpt'), 'B')
        self.assertEqual(c.host('14b', 'alpaca'), 'Anew20260909')
        with self.assertRaises(ValueError):
            c.number(True)

    def test_strict_slo_equality_is_miss(self):
        self.assertTrue(c.strict_slo_pass('alpaca', .999, .0999, work_complete=True))
        self.assertFalse(c.strict_slo_pass('alpaca', 1., .0999, work_complete=True))
        self.assertFalse(c.strict_slo_pass('alpaca', .999, .1, work_complete=True))

    def test_normal_one_repeat_advances(self):
        p = position()
        decision = c.select_group(group(p), [])
        self.assertEqual([t['repeat'] for t in decision['next_tasks']], [1])
        result = c.select_group(group(p), [observation(p['systems']['pdblend'][0], .90)])
        self.assertEqual(result['phase'], 'extension_declaration_required')
        self.assertEqual(result['next_rate_rps_decimal'], '3')

    def test_first_crossing_requires_second_and_straddle_still_caps(self):
        low, high = position(), position('3')
        first = observation(low['systems']['pdblend'][0], .899)
        result = c.select_group(group(low, high), [first])
        self.assertEqual(result['phase'], 'pdblend')
        self.assertEqual([t['repeat'] for t in result['next_tasks']], [2])
        second = observation(low['systems']['pdblend'][1], .95)
        result = c.select_group(group(low, high), [first, second])
        self.assertEqual(result['phase'], 'baselines')
        self.assertEqual(result['cap_rate_rps_decimal'], '1.5')
        self.assertEqual(result['decision']['cap_trigger_cell_id'], first['cell_id'])
        self.assertEqual(len(result['baseline_tasks']), 4)
        self.assertTrue(all(t['row']['rate_rps'] == 1.5 for t in result['baseline_tasks']))

    def test_preexisting_second_is_preserved_and_no_third(self):
        p = position()
        first = observation(p['systems']['pdblend'][0], .89)
        second = observation(p['systems']['pdblend'][1], .88)
        result = c.select_group(group(p), [first, second])
        self.assertEqual(result['phase'], 'baselines')
        self.assertEqual(result['decision']['cap_trigger_cell_id'], first['cell_id'])
        self.assertEqual(result['decision']['missing_cell_ids'], [])

    def test_incomplete_pdblend_is_diagnosis_not_capacity(self):
        p = position()
        result = c.select_group(group(p), [observation(p['systems']['pdblend'][0], .1, False)])
        self.assertEqual(result['phase'], 'diagnosis')
        self.assertFalse(result['decision']['cap_observed'])

    def test_cross_host_observation_rejected(self):
        p = position('.25', '14b', 'sharegpt')
        obs = observation(p['systems']['pdblend'][0])
        obs['measurement_host'] = 'Anew20260909'
        with self.assertRaisesRegex(ValueError, 'identity'):
            c.select_group(group(p), [obs])

    def test_supplement_cannot_hide_incomplete_work_or_change_repeats(self):
        p = position(supplement=True)
        obs = [observation(p['systems']['pdblend'][0], .89), observation(p['systems']['pdblend'][1], .95)]
        for system in c.SYSTEMS[1:]:
            row = observation(p['systems'][system][0], exact=system != 'mixed')
            if system == 'mixed':
                row.update(work_complete=False, failure_class='independently_diagnosed_capacity_deadline',
                           diagnosis_reference={'path': 'independent-proof', 'sha256': 'proof'})
            obs.append(row)
        result = c.select_group(group(p), obs)
        self.assertEqual(len(result['baseline_tasks']), 1)
        supplement = result['baseline_tasks'][0]
        self.assertEqual(supplement['row']['measurement_purpose'], 'metric_supplement')
        self.assertNotEqual(supplement['cell_id'], p['systems']['mixed'][0]['cell_id'])
        final = c.select_group(group(p), obs + [observation(supplement)])
        self.assertEqual(final['phase'], 'complete')
        self.assertFalse(obs[2]['work_complete'])

    def test_exact_audited_zero_can_cancel_reserved_supplement(self):
        p = position(supplement=True)
        obs = [observation(p['systems']['pdblend'][0], .89), observation(p['systems']['pdblend'][1], .95)]
        obs.extend(observation(p['systems'][s][0]) for s in c.SYSTEMS[1:])
        self.assertEqual(c.select_group(group(p), obs)['phase'], 'complete')

    def test_missing_exact_metric_cannot_mark_five_system_complete(self):
        p = position()
        obs = [observation(p['systems']['pdblend'][0], .89), observation(p['systems']['pdblend'][1], .95)]
        obs.extend(observation(p['systems'][s][0], exact=s != 'mixed') for s in c.SYSTEMS[1:])
        self.assertEqual(c.select_group(group(p), obs)['phase'], 'metric_supplement_declaration_required')


if __name__ == '__main__':
    unittest.main()
