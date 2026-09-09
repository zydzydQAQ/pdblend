"""CPU-only ledger counterexamples based on the first real completed cooperative CP."""
from pathlib import Path
import copy,json,sys
import pytest
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B));import cooperative_reconciliation_v1 as q
@pytest.fixture
def fixture(tmp_path,monkeypatch):
 original=B/'cooperative-dynamo8-execution-001';cp_path=next((original/'results/checkpoints').glob('*alpaca-r4*'));cp=q.p.read(cp_path);row=cp['row'];rows=q.p.read(q.original.DECL)['cells'];assert row==rows[0]
 stage=tmp_path/'fake-terminal-for-CPU';(stage/'results/checkpoints').mkdir(parents=True);q.p.write(stage/'results/checkpoints'/cp_path.name,cp)
 status=dict(pid=99999999,finished_s=1,node_lease_held=False,attempted=[row['cell_id']],completed=[row['cell_id']],failed=[]);q.p.write(stage/'status.json',status)
 d=dict(schema='B-cooperative-eight-capacity-reconciliation-v1',declaration=q.p.ref(q.original.DECL),execution_rules=q.p.ref(q.original.RULES),deadline_s=None,automatic_retries=False,source_files={str(B/'cooperative_reconciliation_v1.py'):q.p.sha(B/'cooperative_reconciliation_v1.py')},prior_stages=[dict(status=q.p.ref(stage/'status.json'))],capacity_negatives=[],remaining_cells=rows[1:],already_observed_count=1,original_group_count=8)
 actual_timing=q.original.timing_gate(q.p.checked(cp['receipt']),original/'results');assert actual_timing==cp['arrival_fidelity_gate'] and actual_timing['passed']
 monkeypatch.setattr(q.original,'timing_gate',lambda receipt,output:actual_timing)
 return tmp_path,stage,d,cp,status
@pytest.mark.parametrize('mutation',['none','omit_remaining','replay_observed','wrong_observed_count','changed_trace','active_owner','missing_CP','undeclared_failure','arrival_failure'])
def test_completed_prefix_must_not_omit_replay_or_waive_failure(fixture,monkeypatch,mutation):
 root,stage,d,cp,status=fixture
 if mutation=='omit_remaining':d['remaining_cells'].pop()
 if mutation=='replay_observed':d['remaining_cells'].insert(0,cp['row'])
 if mutation=='wrong_observed_count':d['already_observed_count']=2
 if mutation=='changed_trace':cp=copy.deepcopy(cp);cp['row']['trace_sha256']='0'*64;q.p.write(stage/'results/checkpoints'/(cp['row']['cell_id']+'.json'),cp)
 if mutation=='active_owner':status['node_lease_held']=True
 if mutation=='missing_CP':status['completed']=[]
 if mutation=='undeclared_failure':status['failed']=[dict(cell_id=cp['row']['cell_id'],error='unreviewed')]
 if mutation=='arrival_failure':monkeypatch.setattr(q.original,'timing_gate',lambda *args:dict(passed=False,errors=['late']))
 q.p.write(stage/'status.json',status);d['prior_stages'][0]['status']=q.p.ref(stage/'status.json');path=root/'declaration.json';q.p.write(path,d)
 if mutation=='none':assert len(q.audit(path)['remaining_cells'])==7
 else:
  with pytest.raises(ValueError):q.audit(path)
