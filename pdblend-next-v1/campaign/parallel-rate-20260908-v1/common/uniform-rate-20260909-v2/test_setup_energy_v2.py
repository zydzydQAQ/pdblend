import csv
import json
from pathlib import Path
import tempfile
import unittest
import setup_energy_ledger_v2 as ledger


class FailedQualificationEnergyTests(unittest.TestCase):
    def fixture(self,root):
        root=Path(root);power=root/'power/power.csv';power.parent.mkdir()
        columns=['t_s']+[f'gpu{i}_w' for i in range(8)]
        with power.open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
            for t in (1,2,3):writer.writerow(dict(t_s=t,**{f'gpu{i}_w':1 for i in range(8)}))
        status=root/'status.json'
        value=dict(complete=False,passed=False,error='final export endpoint rejected',measurement_valid=True,
            measurement_start_s=1,measurement_end_s=3,finished_s=4,node_lease_held=False,
            clock_restore_complete=True,cleanup_errors=[],sampling_error=None,
            power_evidence=dict(power_source_verified=True),full_operation_energy_j=16)
        status.write_text(json.dumps(value));return status,value

    def test_valid_failed_qualification_energy_is_separate(self):
        with tempfile.TemporaryDirectory() as root:
            path,_=self.fixture(root);result,_=ledger.audit(path)
            self.assertEqual(result['full_window_energy_j'],16)
            self.assertEqual(result['category'],'failed_preparation_or_qualification')
            self.assertFalse(result['operation'])

    def test_unfinished_or_invalid_qualification_is_not_accepted(self):
        for update in ({'finished_s':None},{'node_lease_held':True},{'clock_restore_complete':False},
                       {'cleanup_errors':['native member still active']},{'measurement_valid':False}):
            with self.subTest(update=update),tempfile.TemporaryDirectory() as root:
                path,value=self.fixture(root);value.update(update);path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):ledger.audit(path)

    def test_failed_window_still_requires_exact_raw_integral(self):
        with tempfile.TemporaryDirectory() as root:
            path,value=self.fixture(root);value['full_operation_energy_j']=0;path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'raw eight-GPU integral'):ledger.audit(path)

if __name__=='__main__':unittest.main()
