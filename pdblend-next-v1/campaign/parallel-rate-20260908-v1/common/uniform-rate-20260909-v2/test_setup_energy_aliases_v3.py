import json
from pathlib import Path
import tempfile
import unittest

import setup_energy_ledger_v3 as ledger


class ReceiptAliases(unittest.TestCase):
    def fixture(self, root):
        original = root / 'A/uniform-example/physical/measurement.json'
        original.parent.mkdir(parents=True)
        receipt = dict(measurement_start_s=0., measurement_end_s=2., energy_j=16.,
            measurement_valid=True, gpu_indices=list(range(8)),
            power_evidence=dict(power_source_verified=True))
        original.write_text(json.dumps(receipt))
        (original.parent / 'power.csv').write_text(
            't_s,' + ','.join(f'gpu{i}_w' for i in range(8)) + '\n'
            + '\n'.join(str(t) + ',' + ','.join(['1'] * 8) for t in range(3)) + '\n')
        alias = root / 'A/uniform-example/recovery-inputs/measurement.json'
        alias.parent.mkdir(parents=True)
        alias.write_bytes(original.read_bytes())
        return original, alias

    def test_identical_recovery_receipt_is_one_physical_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, alias = self.fixture(root)
            value = ledger.collect(root)
            self.assertEqual(value['pending_or_invalid_evidence'], [])
            self.assertEqual(len(value['entries']), 1)
            entry = value['entries'][0]
            self.assertEqual(entry['receipt']['path'], str(original))
            self.assertEqual(entry['exclusive_energy_j'], 16.)
            self.assertEqual(entry['identical_receipt_aliases'], [ledger.m.ref(alias)])

    def test_changed_copy_is_an_unresolved_independent_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, alias = self.fixture(root)
            value = json.loads(alias.read_text())
            value['energy_j'] = 20.
            alias.write_text(json.dumps(value))
            result = ledger.collect(root)
            self.assertEqual(len(result['entries']), 1)
            self.assertEqual(result['entries'][0]['identical_receipt_aliases'], [])
            self.assertEqual(len(result['pending_or_invalid_evidence']), 1)

    def test_foreign_host_copy_cannot_supply_a_missing_meter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, alias = self.fixture(root)
            alias.unlink()
            foreign = root / 'B/uniform-example/physical/measurement.json'
            foreign.parent.mkdir(parents=True)
            foreign.write_bytes(original.read_bytes())
            result = ledger.collect(root)
            self.assertEqual(len(result['entries']), 1)
            self.assertEqual(result['entries'][0]['identical_receipt_aliases'], [])
            self.assertEqual(result['pending_or_invalid_evidence'][0]['node'], 'B')


if __name__ == '__main__':
    unittest.main()
