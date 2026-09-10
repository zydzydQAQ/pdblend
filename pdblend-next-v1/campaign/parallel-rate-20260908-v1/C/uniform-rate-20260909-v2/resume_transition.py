"""Wait for the current immutable pipeline to stop at its cell boundary, then resume exact observations."""
from pathlib import Path
import sys,time,subprocess,os
U=Path(__file__).resolve().parent
sys.path.insert(0,str(U.parents[1]/'common/uniform-rate-20260909-v2'))
import support as p
oldpath=U/'pipeline-001/status.json'
while p.active_owner(p.read(oldpath)):time.sleep(1)
old=p.read(oldpath)
p.need(old.get('error')=="ValueError('stop requested at cell boundary')",'unexpected prior termination')
p.need(old['finished_s'] and not old['node_lease_held'],'prior terminal missing')
last=p.checked(old['last_cell_status'])
p.need(last['complete'] and not last.get('error') and not last['failed'] and not last['node_lease_held'] and not p.active_owner(last),'last cell not cleanly complete')
plan=p.read(U/'plan-001.json')
plan.update(initial_observations=old['observations'],last_cell_status=old['last_cell_status'],previous_pipeline=p.ref(oldpath),stop_paths=[str(U/'STOP-002')])
p.need(not (U/'plan-002.json').exists(),'successor already exists')
p.save(U/'plan-002.json',plan)
argv=['python3','-B',str(U/'pipeline_v3.py'),'--plan',str(U/'plan-002.json'),'--out',str(U/'pipeline-002'),'--run']
with (U/'pipeline-002.log').open('xb') as f:
 child=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONPATH='/root/workspace/pdblend/.runtime-deps',PYTHONDONTWRITEBYTECODE='1'))
p.save(U/'pipeline-launch-002.json',dict(pid=child.pid,argv=argv,started_s=time.time(),preserved_observations=old['observations'],predecessor=p.ref(oldpath)))
