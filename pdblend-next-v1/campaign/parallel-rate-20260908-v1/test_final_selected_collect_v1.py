"""Bounded CPU checks for selection, continuation and actual process evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import final_selection_v1 as selection
import final_selected_collect_v1 as collector
import final_selected_baseline_v1 as baseline

ROOT=Path(__file__).resolve().parent


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.value=selection.read(ROOT/'final-selection-cpu-draft-v1.json')

    def validate(self,value,draft=True):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'selection.json';path.write_text(json.dumps(value))
            return selection.validate(path,selection.sha(path),draft=draft)

    def test_draft_verifies_all_reuse_and_actual_source_files(self):
        self.validate(self.value)

    def test_unpublished_A_cannot_generate_final_report(self):
        self.value['approved']=True
        with self.assertRaisesRegex(ValueError,'A actual final declaration'):
            self.validate(self.value,draft=False)

    def test_cross_model_series_never_best_point(self):
        self.assertTrue(selection.select_stage(self.value,'7b','p4',[]))
        self.assertFalse(selection.select_stage(self.value,'7b','p3',[]))
        self.assertFalse(selection.select_stage(self.value,'14b','p4',[]))
        self.assertTrue(selection.select_stage(self.value,'32b','p4',[]))

    def test_wrong_source_capacity_and_per_point_overrides_rejected(self):
        mutations=[lambda v:v['models']['7b'].update(actual_manifest=v['models']['14b']['actual_manifest']),
                   lambda v:v['models']['7b'].update(capacity_integration_v1=True),
                   lambda v:v['models']['32b'].update(include_cell_ids=['best']),
                   lambda v:v.update(no_per_point_version_selection=False)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                value=copy.deepcopy(self.value);mutate(value)
                with self.assertRaises(ValueError):self.validate(value)

    def test_missing_fullwork_cannot_establish_scope_loss(self):
        point=dict(cell_id='one',measurement_valid=False)
        origins={'one':[dict(status=dict(skipped_saturated=['one']))]}
        selection.annotate_scope([point],origins)
        self.assertFalse(point['required_execution'])
        observed=dict(cell_id='one',measurement_valid=True,work_complete=False)
        selection.annotate_scope([observed],origins)
        self.assertNotIn('required_execution',observed)


class IdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit=collector.load_audit();cls.p=cls.audit.p
        cls.cp=cls.p.read(next((ROOT/'C/boundary-baselines-p4v2/mixed/results/checkpoints').glob('*.json')))
        cls.binding=cls.p.read(cls.cp['binding'])
        cls.logical=cls.p.ref(ROOT/'C/boundary-p4v2/declaration.json')

    def test_actual_C_eight_process_chain(self):
        result=baseline.actual_identity(self.p,self.cp,self.binding,self.cp['receipt'])
        self.assertEqual(len(result),8)

    def test_hash_valid_but_changed_identity_content_rejected(self):
        after=Path(self.cp['receipt']).parent/'identity.after.json'
        mutations=[lambda x:x.pop(), lambda x:x.append(copy.deepcopy(x[0])),
                   lambda x:x[0]['container']['State'].update(Pid=999),
                   lambda x:x[0]['container']['State'].update(StartedAt='restarted'),
                   lambda x:x[0]['container'].update(Image='different'),
                   lambda x:x[0]['provenance'].update(model='different')]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                changed=self.p.read(after);mutate(changed)
                protocol=SimpleNamespace(need=self.p.need,sha=self.p.sha,
                    read=lambda path:changed if Path(path)==after else self.p.read(path))
                with self.assertRaises(ValueError):
                    baseline.actual_identity(protocol,self.cp,self.binding,self.cp['receipt'])

    def test_original_baseline_logical_reference_retained(self):
        row=self.cp['row'];logical=self.logical
        self.assertEqual(baseline.declaration_chain(self.p,self.cp,row,logical),logical)

    def test_unauthorized_new_declaration_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fake.json';path.write_text(json.dumps(dict(schema='unknown')))
            cp=copy.deepcopy(self.cp);cp['declaration']=dict(path=str(path),sha256=self.p.sha(path))
            with self.assertRaises(ValueError):
                baseline.declaration_chain(self.p,cp,cp['row'],self.logical)

    def test_actual_B_continuation_parent_row_and_observed_exclusion(self):
        reference=self.p.ref(ROOT/'B/baseline-reconciliation-001/declaration.json')
        instruction=self.p.checked(reference)
        row=instruction['remaining_cells'][0]
        cp=dict(row=row,declaration=reference)
        self.assertEqual(baseline.declaration_chain(self.p,cp,row,instruction['parent_declaration']),reference)
        changed=copy.deepcopy(row);changed['trace_sha256']='different'
        with self.assertRaises(ValueError):
            baseline.declaration_chain(self.p,dict(cp,row=changed),changed,instruction['parent_declaration'])
        observed=instruction['capacity_negatives'][0]['cell_id']
        parent=self.p.checked(instruction['parent_declaration'])
        old=next(r for r in parent['cells'] if r['cell_id']==observed)
        with self.assertRaises(ValueError):
            baseline.declaration_chain(self.p,dict(cp,row=old),old,instruction['parent_declaration'])

    def test_skip_status_without_verified_complete_loss_does_not_exclude(self):
        originals=self.p.original_points()
        grid=collector.logical_grid(originals,[],[])
        with self.assertRaisesRegex(ValueError,'independently verified complete loss'):
            selection.annotate_grid(grid,selection.read(ROOT/'final-selection-cpu-draft-v1.json'),
                                    originals,[],self.p)


if __name__=='__main__':
    unittest.main()
