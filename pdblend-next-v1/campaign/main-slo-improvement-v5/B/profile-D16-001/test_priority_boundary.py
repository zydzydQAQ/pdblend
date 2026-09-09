import ast,json,hashlib,time
from pathlib import Path
import pytest
SOURCE=Path('/root/workspace/pdblend-next-v1/campaign/main-slo-improvement-v5/B/profile-D16-001/run.py')
ns={'Path':Path,'json':json,'time':time,'hashlib':hashlib}
tree=ast.parse(SOURCE.read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('require','sha','predecessor')],type_ignores=[]),str(SOURCE),'exec'),ns)
@pytest.fixture
def fixture(tmp_path):
 root=tmp_path/'profile';root.mkdir();stage=tmp_path/'prior';(stage/'results/checkpoints').mkdir(parents=True)
 def write(p,x):p.write_text(json.dumps(x));return {'path':str(p),'sha256':ns['sha'](p)}
 release=write(tmp_path/'release.json',{});binding=write(tmp_path/'binding.json',{})
 rows=[dict(cell_id=f'cell{i}',model='32b',stage='screen_fixed2') for i in range(16)]
 write(stage/'declaration-order.json',rows)
 write(stage/'status.json',dict(complete=True,phase='complete',completed=[r['cell_id'] for r in rows],failed=[],pid=999999991))
 for row in rows:
  ref=write(stage/(row['cell_id']+'.json'),{'actual':True})
  write(stage/'results/checkpoints'/(row['cell_id']+'.json'),dict(row=row,measurement_valid=True,receipt=ref['path'],receipt_sha256=ref['sha256'],artifacts={ref['path']:ref['sha256']}))
 write(root/'priority-authorization.json',dict(authorized=True,automatic_retries=False,deadline_s=time.time()+10000,predecessor_release=release,predecessor_output=str(stage),predecessor_pid=999999991,original_binding=binding))
 ns['ROOT']=root
 return root,stage

def test_exact_terminal_prefix_accepted(fixture):
 result=ns['predecessor']();assert len(result['checkpoints'])==16 and result['old_baseline_or_ablation_called_complete'] is False
@pytest.mark.parametrize('mutation',['running','missing','tampered','deadline'])
def test_no_early_or_corrupt_release(fixture,mutation):
 root,stage=fixture;p=stage/'status.json';x=json.loads(p.read_text())
 if mutation=='running':x['phase']='running';x['complete']=False;p.write_text(json.dumps(x))
 if mutation=='missing':x['completed'].pop();p.write_text(json.dumps(x))
 if mutation=='tampered':(stage/'cell0.json').write_text('{}')
 if mutation=='deadline':p=root/'priority-authorization.json';x=json.loads(p.read_text());x['deadline_s']=time.time()+719;p.write_text(json.dumps(x))
 with pytest.raises(RuntimeError):ns['predecessor']()
