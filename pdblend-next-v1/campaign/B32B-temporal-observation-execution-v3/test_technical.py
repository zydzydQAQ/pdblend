import copy,json,sys,time
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import technical as t,common as c,run as r
from test_child import child,job

def test_actual_compact_bytes_engine_loader_and_original_child(tmp_path):
 _,j,s=job(tmp_path);p=Path(j['observation_spec']);old=p.read_bytes();assert len(old)>16384
 with pytest.raises(RuntimeError,match='loader rejected'):t.spec_preflight(p)
 t.compact_spec(p,s);proof=t.spec_preflight(p);assert proof['passed'] and proof['bytes']<16384 and not proof['writer_created'] and c.read(p)==s
 j['observation_spec_sha256']=c.sha(p);assert child.validate_job(j)==s


def evidence(tmp_path):
 a=c.read(t.AUTH);fields=['prior_claim','failed_status','failed_spec','failed_log','restoration_status','restored_binding','loader_review_manifest'];new=copy.deepcopy(a);new['files']={}
 for key in fields:
  p=tmp_path/key;p.write_bytes(Path(a[key]).read_bytes());new[key]=str(p);new['files'][str(p)]=c.sha(p)
 return new

def test_exact_retained_failure_and_successful_recovery_allows_only_second(tmp_path):
 a=evidence(tmp_path);assert t.verify_evidence(a,[a['prior_claim']],lambda _:False)==a

@pytest.mark.parametrize('mutation',['extra_claim','missing_claim','old_live','changed_bytes','recovery_invalid','prior_capture_complete','third_namespace'])
def test_other_or_unrestored_attempt_cannot_bypass_once(tmp_path,mutation):
 a=evidence(tmp_path);claims=[a['prior_claim']];live=lambda _:False
 if mutation=='extra_claim':claims.append('/another/execution-once.json')
 if mutation=='missing_claim':claims=[]
 if mutation=='old_live':live=lambda _:True
 if mutation=='changed_bytes':Path(a['failed_log']).write_text('changed')
 if mutation=='third_namespace':a['attempt_name']='B32B-temporal-observation-attempt-003'
 if mutation in ('recovery_invalid','prior_capture_complete'):
  key='restoration_status' if mutation=='recovery_invalid' else 'failed_status';p=Path(a[key]);d=c.read(p);d['measurement_valid' if mutation=='recovery_invalid' else 'capture_complete']=False if mutation=='recovery_invalid' else True;c.write(p,d);a['files'][str(p)]=c.sha(p)
 with pytest.raises(RuntimeError):t.verify_evidence(a,claims,live)


def test_runtime_full_mount_comparison_only_ignores_order():
 b={k:k for k in ('Id','Image','Config','HostConfig','Path','Args')};b.update(Mounts=[{'Destination':'/a','Source':'/x','RW':True},{'Destination':'/b','Source':'/y','RW':False}],State={'Running':True,'Pid':1,'StartedAt':'old'});a=copy.deepcopy(b);a['State'].update(Pid=2,StartedAt='new');a['Mounts'].reverse();r.preserve_container(b,a)
 a['Mounts'][0]['RW']=True
 with pytest.raises(RuntimeError,match='mount dictionaries'):r.preserve_container(b,a)
