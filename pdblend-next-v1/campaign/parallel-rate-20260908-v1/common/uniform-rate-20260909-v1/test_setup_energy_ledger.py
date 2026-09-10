import unittest
from setup_energy_ledger import subtract


class WindowAccounting(unittest.TestCase):
    def test_nested_transition_never_added_twice(self):
        self.assertEqual(subtract(12, 18, [(10, 20)]), [])

    def test_arrival_drain_and_cleanup_window_excluded_as_one(self):
        self.assertEqual(subtract(0, 150, [(1, 141)]), [(0, 1), (141, 150)])

    def test_overlapping_and_duplicate_exclusions(self):
        self.assertEqual(subtract(0, 20, [(3, 7), (5, 12), (3, 7), (16, 30)]),
                         [(0, 3), (12, 16)])


if __name__ == '__main__':
    unittest.main()
