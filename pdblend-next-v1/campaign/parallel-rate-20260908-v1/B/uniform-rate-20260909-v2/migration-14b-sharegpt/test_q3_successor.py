import copy
import unittest
from unittest.mock import patch
import baseline_q3_successor as q

class HandoffTests(unittest.TestCase):
    def records(self):
        prior=dict(node='B',model='14b',datasets=['sharegpt'],finished_s=2,node_lease_held=False,
            error="ValueError('stop requested before next stage')",child=dict(exitcode=0),
            declaration={'path':'declaration','sha256':'a'},observations=[{'path':'observation','sha256':'b'}],
            last_cell_status={'path':'last','sha256':'c'})
        boundary=dict(prior,complete=True,pdb_boundary_complete=True,scope='pdblend')
        last=dict(complete=True,finished_s=1,failed=[],node_lease_held=False)
        return prior,boundary,last

    def check(self,prior,boundary,last,active=False):
        with patch.object(q.p,'active_owner',return_value=active),patch.object(q.p,'checked',return_value=last),patch.object(q.m,'decision',return_value=dict(pdb_boundary_complete=True,phase='baselines')):
            return q.checked_prior(prior,boundary)

    def test_only_planned_clean_boundary_accepted(self):
        self.assertTrue(self.check(*self.records())['pdb_boundary_complete'])
        for edit in [dict(error='unknown_failure'),dict(node='Anew20260909'),dict(model='32b'),dict(child=dict(exitcode=1)),dict(observations=[])]:
            with self.subTest(edit=edit):
                a,b,c=self.records();a.update(edit)
                with self.assertRaises(ValueError):self.check(a,b,c)

    def test_live_owner_and_invalid_last_measurement_rejected(self):
        with self.assertRaisesRegex(ValueError,'still owns'):self.check(*self.records(),active=True)
        a,b,c=self.records();c['node_lease_held']=True
        with self.assertRaisesRegex(ValueError,'last PDB'):self.check(a,b,c)

    def test_incomplete_boundary_never_advances(self):
        a,b,c=self.records();b['pdb_boundary_complete']=False
        with self.assertRaisesRegex(ValueError,'boundary is not complete'):self.check(a,b,c)

if __name__=='__main__':unittest.main()
