import copy
import unittest
from final_completion_v1 import assess


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.points=[dict(model=m,dataset=d,cell_id=m+d,required_execution=True,
            measurement_valid=True,work_complete=True,version_id=m+d)
            for m in ('7b','14b','32b') for d in ('alpaca','sharegpt','longbench')]
        self.pairs=[dict(cell_id=p['cell_id'],baseline_system=s,passed=False)
            for p in self.points for s in ('mixed','distserve','dynamollm','ecoserve')]
        self.grid=[dict(original_cell_id=str(i),repeat_requirement_complete=True) for i in range(90)]
        self.baselines=[dict(cell_id='capacity-negative',measurement_valid=True,work_complete=False)]
        self.boundaries=[dict(model=p['model'],dataset=p['dataset'],requested_stop_condition_observed=True,
                              repeated_threshold_crossing_preserved=True) for p in self.points]

    def result(self):
        return assess(self.points,self.pairs,self.grid,self.baselines,self.boundaries)

    def test_negative_performance_does_not_block_completion(self):
        self.assertTrue(self.result()['complete'])

    def test_missing_second_repeat_blocks(self):
        self.points.append(dict(self.points[0],cell_id='missing-repeat2',measurement_valid=False,work_complete=None))
        self.assertFalse(self.result()['complete'])

    def test_failed_pdb_work_blocks(self):
        self.points[0]['work_complete']=False
        self.assertFalse(self.result()['complete'])

    def test_missing_baseline_blocks(self):
        self.pairs.pop()
        self.assertFalse(self.result()['complete'])

    def test_invalid_baseline_evidence_blocks(self):
        self.baselines[0]['measurement_valid']=False
        self.assertFalse(self.result()['complete'])

    def test_original_gap_cannot_be_hidden_by_new_rates(self):
        self.grid[0]['repeat_requirement_complete']=False
        self.assertFalse(self.result()['complete'])

    def test_unreached_stop_condition_requires_more_rate(self):
        self.boundaries[0]['requested_stop_condition_observed']=False
        self.assertFalse(self.result()['complete'])

    def test_second_source_version_cannot_be_spliced(self):
        self.points.append(dict(self.points[0],version_id='another-source'))
        self.assertFalse(self.result()['complete'])

    def test_explicitly_excluded_unmeasured_row_is_not_work_gap(self):
        self.points.append(dict(self.points[0],cell_id='above-firstloss',required_execution=False,
                                measurement_valid=False,work_complete=None))
        self.assertTrue(self.result()['complete'])


if __name__=='__main__':unittest.main()
