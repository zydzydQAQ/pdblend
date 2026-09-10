"""Accounting boundary tests with small evidence only; no SSH or GPU work."""
import copy
import unittest
from unittest.mock import patch
import close_energy as c


class AccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = c.ROOT / 'A/observations.json'
        cls.row = c.report.normalize(c.report.read(path)[0], 'A', c.report.ref(path))
        cls.summary = c.checked(cls.row['summary'])
        cls.receipt = c.checked(cls.row['receipt'])

    def test_duplicate_cell_attempt_counted_once(self):
        rows, copies = c.deduplicate([self.row, copy.deepcopy(self.row)])
        self.assertEqual((len(rows), copies), (1, 1))

    def test_conflicting_duplicate_rejected(self):
        changed = dict(self.row, energy_j=self.row['energy_j'] + 1)
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            c.deduplicate([self.row, changed])

    def test_confirmation_repeat_is_distinct(self):
        repeated = dict(self.row, cell_id=self.row['cell_id'] + '-confirmation', repeat=2)
        self.assertEqual(len(c.deduplicate([self.row, repeated])[0]), 2)

    def test_same_host_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            c.nonoverlap([{'id': 'first', 'window': c.window(1, 5)}, {'id': 'second', 'window': c.window(4, 7)}])

    def test_touching_windows_are_disjoint(self):
        self.assertEqual(len(c.nonoverlap([{'id': 'first', 'window': c.window(1, 5)}, {'id': 'second', 'window': c.window(5, 7)}])), 2)

    def altered_primary(self, field, value):
        summary = copy.deepcopy(self.summary)
        summary[field] = value
        original = c.checked
        with patch.object(c, 'checked', side_effect=lambda ref: summary if ref == self.row['summary'] else original(ref)):
            return c.primary(self.row)

    def test_service_drain_outside_primary_rejected(self):
        with self.assertRaisesRegex(ValueError, 'tail outside'):
            self.altered_primary('drain_end_s', self.summary['measurement_end_s'] + 1)

    def test_incomplete_drain_rejected(self):
        with self.assertRaisesRegex(ValueError, 'drain incomplete'):
            self.altered_primary('drain_complete', False)

    def test_outer_cleanup_after_primary_is_accepted_without_addition(self):
        entry = c.primary(self.row)
        outer = entry['outer_operation_diagnostic_only']
        self.assertGreater(outer['receipt_finished_s'], entry['window']['end_s'])
        self.assertEqual(entry['energy_j'], self.row['energy_j'])
        self.assertFalse(outer['added_to_primary'])
        self.assertIsNone(outer['incremental_overhead_j'])

    def test_setup_failed_parent_is_not_added_again(self):
        setup = c.setup('A')
        self.assertEqual(len(setup['counted_windows']), 8)
        self.assertAlmostEqual(setup['measured_subtotal_j'], 1196698.9725937159)
        self.assertAlmostEqual(setup['direct_failed_operation_measured_subtotal_j'], 3.9666267113685603)
        self.assertTrue(setup['failed_setup_entries'])
        self.assertIsNone(setup['failed_parent_attempt_energy_j'])
        self.assertIsNone(setup['complete_setup_energy_j'])

    def test_large_or_missing_setup_receipt_uses_exact_small_projection(self):
        setup = c.setup('C')
        self.assertEqual(len(setup['counted_windows']), 7)
        self.assertAlmostEqual(setup['measured_subtotal_j'], 1176893.264467224)
        self.assertTrue(any('projection' in e for e in setup['receipt_checks']))


if __name__ == '__main__':
    unittest.main()
