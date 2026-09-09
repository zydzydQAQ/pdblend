"""Adapt only an unfrozen workflow to the common no-task-deadline executor."""
import hashlib,json
from pathlib import Path
COMMON='campaign/parallel-rate-20260908-v1/common/execution-until-complete-v1'
DIGEST='77bcbbb68e20419e5bc469a838c71e1abfa789dc167d901501715fda0ff4a8a9'
def adapt(workflow):
 workflow=Path(workflow);name=workflow.name.removeprefix('workflow-')
 assert not (workflow.parent/(name+'-release')).exists(),'already frozen'
 dpath=workflow/'work-declaration.json';old=hashlib.sha256(dpath.read_bytes()).hexdigest();d=json.loads(dpath.read_text())
 d.update(deadline_s=None,campaign_lifecycle='until_declared_complete_v1')
 dpath.write_text(json.dumps(d,indent=2)+'\n');new=hashlib.sha256(dpath.read_bytes()).hexdigest()
 p=workflow/'protocol.py';s=p.read_text().replace('DEADLINE = 1788872770.0400891','DEADLINE = None');p.write_text(s)
 p=workflow/'runner.py';s=p.read_text().replace(old,new).replace('campaign/five-system-execution-v3/run.py',COMMON+'/run.py').replace('7c7dbe217243b42a8f93b57476ed457a6111e8f90c71ac4269130c9b46420f92',DIGEST)
 s=s.replace("deadline_s=p.DEADLINE, output=str(output)","deadline_s=None, campaign_lifecycle='until_declared_complete_v1', output=str(output)")
 s=s.replace("base['deadline_s'] = p.DEADLINE","base['deadline_s'] = None\n            base['campaign_lifecycle'] = 'until_declared_complete_v1'")
 s=s.replace("            p.need(time.time() + 900 < p.DEADLINE, 'insufficient ordinary gate and first-cell reserve')\n",'')
 s=s.replace(" or time.time() + 400 >= p.DEADLINE",'')
 p.write_text(s)
 p=workflow/'prepare_release.py';s=p.read_text().replace('campaign/five-system-execution-v3/run.py',COMMON+'/run.py').replace('campaign/five-system-execution-v3/child.py',COMMON+'/child.py')
 s=s.replace("created_s=time.time(), deadline_s=p.DEADLINE, implementation_id=host.name", "created_s=time.time(), deadline_s=None, campaign_lifecycle='until_declared_complete_v1', implementation_id=host.name")
 s=s.replace("p.REPO / 'campaign/pdblend-ablation-20260908-v1/execution.py')", "p.REPO / 'campaign/pdblend-ablation-20260908-v1/execution.py',\n                 p.REPO / '"+COMMON+"/manifest.json',\n                 p.REPO / '"+COMMON+"/cpu-validation.json')")
 p.write_text(s)
 for name in ['protocol.py','runner.py','prepare_release.py']:compile((workflow/name).read_text(),str(workflow/name),'exec')
 assert 'time.time() + 900 <' not in (workflow/'runner.py').read_text()
 assert 'time.time() + 400 >=' not in (workflow/'runner.py').read_text()
 return dict(workflow=str(workflow),declaration_sha256=new,campaign_lifecycle=d['campaign_lifecycle'],deadline_s=None)
if __name__=='__main__':
 import sys;print(json.dumps(adapt(Path(sys.argv[1]))))
