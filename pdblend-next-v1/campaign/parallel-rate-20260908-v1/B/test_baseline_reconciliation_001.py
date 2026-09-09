import copy,json,sys,tempfile,unittest
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B));import baseline_reconciliation as q
D=B/'baseline-reconciliation-001/declaration.json';CID='parallel-rate-p4-completion-32b-sharegpt-r1.5-s701-w100-distserve-slo1-repeat1'
class Tests(unittest.TestCase):
 def test_actual_negative(self):
  d=q.audit(D);self.assertEqual(len(d['remaining_cells']),18);self.assertNotIn(CID,[r['cell_id'] for r in d['remaining_cells']]);self.assertIn(CID.replace('repeat1','repeat2'),[r['cell_id'] for r in d['remaining_cells']])
 def test_engineering_faults_rejected(self):
  cp=q.p.read(B/'baselines-completion-r1-001/results/checkpoints'/(CID+'.json'));r=q.p.checked(cp['receipt'])
  mutations=[lambda x:x.update(clock_restore_complete=False),lambda x:x.update(child_exitcode=1),lambda x:x.update(outer_cleanup_errors=['error']),lambda x:x['summary'].update(runtime_error='503'),lambda x:x['summary'].update(admission_rejections=1),lambda x:x['summary'].update(drain_complete=False),lambda x:x['summary'].update(energy_j=1),lambda x:x['summary'].update(request_timeouts=1)]
  for mutation in mutations:
   x=copy.deepcopy(r);mutation(x)
   with self.assertRaises(ValueError):q.capacity_negative(cp,x)
 def test_scope_mutations_rejected(self):
  d=q.p.read(D)
  for which in ['omit','duplicate','remove_known_negative','relax_deadline']:
   x=copy.deepcopy(d)
   if which=='omit':x['remaining_cells'].pop()
   if which=='duplicate':x['remaining_cells'].append(x['remaining_cells'][0])
   if which=='remove_known_negative':x['capacity_negatives']=[]
   if which=='relax_deadline':x['deadline_s']=1e20
   with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)/'declaration.json';p.write_text(json.dumps(x))
    with self.assertRaises(ValueError):q.audit(p)
if __name__=='__main__':unittest.main()
