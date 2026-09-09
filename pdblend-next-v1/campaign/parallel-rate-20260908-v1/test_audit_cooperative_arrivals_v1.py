import copy
import csv
from pathlib import Path
import unittest

import audit_cooperative_arrivals_v1 as arrival
import final_selected_collect_v3 as collect


class ArrivalAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = collect.load_audit().p
        cls.cp = cls.p.read(next((collect.ROOT / 'B/cooperative-dynamo8-execution-001/results/checkpoints').glob('*.json')))
        cls.binding = cls.p.checked(cls.cp['binding'])
        cls.raw = cls.cp['arrival_fidelity_gate']['raw_requests']['path']
        with open(cls.raw, newline='') as stream:
            cls.rows = list(csv.DictReader(stream))

    def check(self, cp=None, binding=None):
        return arrival.verify(self.p, cp or self.cp, binding or self.binding,
                              Path(self.raw).parent, self.cp['declaration'])

    def test_actual_checkpoint(self):
        self.assertTrue(self.check()['passed'])

    def test_missing_binding_rule_pin(self):
        binding = copy.deepcopy(self.binding)
        del binding['files'][self.cp['execution_rules']['path']]
        with self.assertRaises(ValueError): self.check(binding=binding)

    def test_reported_max_cannot_hide_raw(self):
        cp = copy.deepcopy(self.cp)
        cp['arrival_fidelity_gate']['actual_dispatch_max_s'] = 0
        with self.assertRaises(ValueError): self.check(cp=cp)

    def test_reported_p99_cannot_hide_raw(self):
        cp = copy.deepcopy(self.cp)
        cp['arrival_fidelity_gate']['actual_dispatch_p99_s'] = 0
        with self.assertRaises(ValueError): self.check(cp=cp)

    def test_different_raw_file(self):
        cp = copy.deepcopy(self.cp)
        cp['arrival_fidelity_gate']['raw_requests']['path'] += '.unrelated'
        with self.assertRaises(ValueError): self.check(cp=cp)

    def test_relaxed_limits(self):
        cp = copy.deepcopy(self.cp)
        cp['arrival_fidelity_gate']['max_limit_s'] = 1000
        with self.assertRaises(ValueError): self.check(cp=cp)

    def test_max_limit_even_when_delay_field_matches(self):
        rows = copy.deepcopy(self.rows)
        row = rows[0]
        row['actual_dispatch_s'] = str(float(row['planned_arrival_s']) + 1.1)
        row['dispatch_delay_s'] = str(float(row['actual_dispatch_s']) - float(row['planned_arrival_s']))
        with self.assertRaises(ValueError): arrival.recompute(rows, len(rows))

    def test_p99_limit_below_max_limit(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row['actual_dispatch_s'] = str(float(row['planned_arrival_s']) + .2)
            row['dispatch_delay_s'] = str(float(row['actual_dispatch_s']) - float(row['planned_arrival_s']))
        with self.assertRaises(ValueError): arrival.recompute(rows, len(rows))

    def test_duplicate_request(self):
        rows = copy.deepcopy(self.rows)
        rows[1]['idx'] = rows[0]['idx']
        with self.assertRaises(ValueError): arrival.recompute(rows, len(rows))

    def test_unknown_time(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['actual_dispatch_s'] = 'nan'
        with self.assertRaises(ValueError): arrival.recompute(rows, len(rows))


if __name__ == '__main__': unittest.main()
