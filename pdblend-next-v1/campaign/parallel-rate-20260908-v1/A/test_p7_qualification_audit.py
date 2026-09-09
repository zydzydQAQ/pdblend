import copy,json,sys
from pathlib import Path
import pytest
A=Path(__file__).resolve().parent;sys.path.insert(0,str(A));import p7_qualification_audit as m
@pytest.fixture
def work():
 out=A/'p7-autonomous-gate-001';s=m.fixed(m.ref(A/'p7-autonomous-gate-inputs-001/spec.json'));r=m.fixed(m.ref(out/'automatic_underload_gate/result.json'))
 return r,m.fixed(r['trace']),m.fixed(s['config']),m.prior.raw_measurement(r['raw_measurement']),json.loads((out/'automatic_underload_gate/requests.json').read_text())
def test_actual_fresh_gate_fullwork_low_slo_remains_valid(work):
 r,t,c,raw,rows=work;assert m.audit_rows(r,t,c,raw,rows,60)['n_good']==209
@pytest.mark.parametrize('field,value',[('success',0),('request_timeout',True),('token_ids_verified',0),('generated_tokens',0),('input_tokens',1),('idx',1),('request_id','1'),('slo_ok',2)])
def test_each_request_work_identity_failure_rejected(work,field,value):
 r,t,c,raw,rows=work;rows=copy.deepcopy(rows);rows[0][field]=value
 with pytest.raises(ValueError):m.audit_rows(r,t,c,raw,rows,60)
@pytest.mark.parametrize('field,delta',[('request_deadline_s',1),('planned_arrival_s',1)])
def test_original_arrival_and120_deadline_cannot_change(work,field,delta):
 r,t,c,raw,rows=work;rows=copy.deepcopy(rows);rows[0][field]+=delta
 with pytest.raises(ValueError):m.audit_rows(r,t,c,raw,rows,60)
@pytest.mark.parametrize('field,value',[('energy_j',1),('n_good',0),('failed_requests',1),('request_timeouts',1),('work_complete',False)])
def test_summary_or_failure_is_not_waived(work,field,value):
 r,t,c,raw,rows=work;r=dict(r);r[field]=value
 with pytest.raises(ValueError):m.audit_rows(r,t,c,raw,rows,60)
def test_gate_does_not_substitute900_duration(work):
 with pytest.raises(ValueError):m.audit_rows(*work,900)
