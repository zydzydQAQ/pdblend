import copy
import tempfile
from pathlib import Path
import unittest

import cold_restore
import run_cells
import support as p


class MountEquivalence(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(Type='bind', Source='/workspace', Destination='/workspace', RW=True),
                     dict(Type='bind', Source='/models', Destination='/models', RW=False)]

    def test_exact_or_permuted_whole_dicts(self):
        self.assertFalse(cold_restore.mount_equivalence(self.rows, self.rows)['array_order_changed'])
        self.assertTrue(cold_restore.mount_equivalence(self.rows, self.rows[::-1])['array_order_changed'])

    def test_source_permission_destination_or_fields_cannot_change(self):
        for key, value in [('Source', '/other'), ('Destination', '/other'), ('RW', True), ('new', 1)]:
            other = copy.deepcopy(self.rows)
            other[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                cold_restore.mount_equivalence(self.rows, other)

    def test_duplicate_destination_rejected(self):
        with self.assertRaises(ValueError):
            cold_restore.mount_equivalence(self.rows + self.rows, self.rows + self.rows)


class DispatchOrder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / 'trace.json'
        path.write_text('{}\n')
        self.rows = [dict(cell_id='cell' + str(i), model='7b', dataset='alpaca', rate_rps=3.,
            system='pdblend', repeat=i, node='C', arrival_window_s=100., seed=701, slo_scale=1.,
            trace=str(path), trace_sha256=p.sha(path)) for i in (1, 2)]
        rows = self.rows
        class Contract:
            def lookup(self, declaration, model, dataset, rate, system, repeat):
                return rows[repeat - 1]
            def resolve_group(self, *args, **kwargs):
                return {}
            def select_group(self, group, observations):
                return dict(phase='pdblend', next_tasks=[dict(cell_id=r['cell_id']) for r in rows])
        self.contract = Contract()
        self.release = dict(rows=rows, declaration={}, node='C', scheduling_observations=[])
        self.binding = dict(model='7b', system='pdblend', configs={'alpaca': 'unchanged'})

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_or_both_repeats_allowed(self):
        run_cells.validate_rows(self.release, self.binding, self.contract)
        run_cells.validate_rows(dict(self.release, rows=self.rows[:1]), self.binding, self.contract)

    def test_second_repeat_cannot_skip_first(self):
        with self.assertRaisesRegex(ValueError, 'skips current'):
            run_cells.validate_rows(dict(self.release, rows=self.rows[1:]), self.binding, self.contract)

    def test_workload_or_binding_cannot_change(self):
        changed = copy.deepcopy(self.rows)
        changed[0]['rate_rps'] = 6.
        with self.assertRaises(ValueError):
            run_cells.validate_rows(dict(self.release, rows=changed), self.binding, self.contract)
        with self.assertRaises(ValueError):
            run_cells.validate_rows(self.release, dict(self.binding, system='mixed'), self.contract)


if __name__ == '__main__':
    unittest.main()
