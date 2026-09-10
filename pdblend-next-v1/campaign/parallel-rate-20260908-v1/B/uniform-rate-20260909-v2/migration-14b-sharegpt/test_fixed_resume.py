import copy
import unittest
from unittest.mock import patch
import pipeline_v4 as m

class FixedResumeTests(unittest.TestCase):
    def run_case(self,change=None,active=False):
        plan=dict(resume_prior_pipeline='prior',reuse_complete_fixed_qualification=dict(path='qualification',sha256='q'))
        prior=dict(node='B',model='14b',datasets=['sharegpt'],finished_s=2,node_lease_held=False,observations=[],
            child=dict(exitcode=1,argv=[str(m.HERE/'stages.py'),'idle','--input','qualification']))
        qualified=dict(passed=True,finished_s=1,node_lease_held=False)
        if change:change(prior,qualified)
        def checked(reference):
            if reference=='prior':return prior
            if reference==plan['reuse_complete_fixed_qualification']:return dict(status='qualified-state')
            if reference=='qualified-state':return qualified
            raise AssertionError(reference)
        with patch.object(m.p,'checked',side_effect=checked),patch.object(m.p,'active_owner',return_value=active):
            return m.check_fixed_resume(plan)

    def test_completed_fixed_qualification_has_a_clean_resume(self):
        self.assertEqual(self.run_case()['path'],'qualification')

    def test_foreign_active_changed_or_partial_qualification_rejected(self):
        edits=[lambda p,q:p.update(model='32b'),lambda p,q:p.update(observations=['already-running-formal']),
               lambda p,q:p['child'].update(argv=['another.py','idle','qualification']),
               lambda p,q:p['child'].update(argv=[str(m.HERE/'stages.py'),'idle','another-qualification']),
               lambda p,q:q.update(passed=False),lambda p,q:q.update(node_lease_held=True)]
        for change in edits:
            with self.assertRaises(ValueError):self.run_case(change)
        with self.assertRaises(ValueError):self.run_case(active=True)

if __name__=='__main__':unittest.main()
