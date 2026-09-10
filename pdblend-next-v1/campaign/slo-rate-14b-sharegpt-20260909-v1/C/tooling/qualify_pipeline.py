"""Own only preparation/qualification, never serving rate measurements."""
from pathlib import Path
import json,subprocess,sys,time,os
import bootstrap as b
import power_selftest as p
from inputs import E,N
OUT=N/p.NODE
status=dict(schema='slo14-environment-pipeline-v1',node=p.NODE,pid=os.getpid(),started_s=time.time(),complete=False,steps=[])
def step(name,argv):
 status['phase']=name;status['steps'].append(dict(name=name,argv=argv,started_s=time.time()));b.save(OUT/'environment-status.json',status)
 with (OUT/(name+'.log')).open('x') as f:r=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
 status['steps'][-1].update(exitcode=r.returncode,finished_s=time.time());b.save(OUT/'environment-status.json',status)
 assert r.returncode==0,name+' failed'
try:
 step('retirement',[sys.executable,str(p.HERE/'retire.py')])
 step('bootstrap',[sys.executable,str(p.HERE/'cold_bootstrap.py'),'--spec',str(OUT/'preparation/bootstrap-spec.json'),'--out',str(OUT/'cold-bootstrap'),'--run'])
 from inputs import boot_record,fixed,idle
 boot=boot_record(OUT/'cold-bootstrap/status.json');fixedref=fixed(boot,OUT/'fixed-inputs')
 step('fixed',[sys.executable,str(p.HERE/'qualify_fixed.py'),'--spec',fixedref['path'],'--out',str(OUT/'fixed-qualification'),'--run'])
 idle_ref=idle(p.ref(OUT/'fixed-qualification/qualified.json'),OUT/'idle-inputs')
 step('idle',[sys.executable,str(p.HERE/'qualify_idle.py'),'--spec',idle_ref['path'],'--out',str(OUT/'pdb-qualification'),'--run'])
 import verify_idle
 verified=verify_idle.verify(p.ref(OUT/'pdb-qualification/qualified.json'))
 b.save(OUT/'pdb-ready.json',dict(node=p.NODE,qualification=p.ref(OUT/'pdb-qualification/qualified.json'),qualification_validator=p.ref(p.HERE/'verify_idle.py'),binding=verified['binding'],runtime_pythonpath=[str(p.HOST/'src'),str(p.HOST),str(p.METER),str(p.HERE),'/root/workspace/pdblend/.runtime-deps'],measurement_executor=p.ref(E/'run.py'),complete=True))
 status.update(complete=True,phase='pdb_ready')
except BaseException as e:status.update(error=repr(e),phase='failed');raise
finally:status['finished_s']=time.time();b.save(OUT/'environment-status.json',status)
