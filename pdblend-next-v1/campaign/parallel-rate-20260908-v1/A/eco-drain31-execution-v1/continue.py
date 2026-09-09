"""Execute an already restored/fresh-qualified declaration; never infer future gates."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from contract import *

def main(inputs_ref,out):
 inputs=checked(inputs_ref);need(not out.exists() and 'PDBLEND_NODE_LOCK_FD' not in os.environ,'new unleased orchestrator required');out.mkdir()
 state=dict(pid=os.getpid(),started_s=time.time(),phase='preflight',complete=False,node_lease_held=False,inputs=inputs_ref,steps=[])
 def save():
  state['updated_s']=time.time();temporary=out/'status.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(out/'status.json')
 def step(name,argv):
  state['phase']=name;save()
  with (out/(name+'.log')).open('x') as f:
   child=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT);state['child_pid']=child.pid;save();code=child.wait()
  state.pop('child_pid',None);state['steps'].append(dict(name=name,exit_code=code));save();need(code==0,'stage failed; no automatic retry: '+name)
 try:
  scope=inputs['execution_scope'];binding=inputs['qualified_binding'];checked(scope);checked(binding)
  release_dir=Path(inputs['release_directory']);performance=Path(inputs['performance_directory']);need(not release_dir.exists() and not performance.exists(),'immutable new output directories required')
  step('freeze',[sys.executable,str(HERE/'prepare.py'),'--scope',scope['path'],'--scope-sha256',scope['sha256'],'--binding',binding['path'],'--binding-sha256',binding['sha256'],'--out',str(release_dir)])
  release=release_dir/'release.json';args=[sys.executable,str(HERE/'run.py'),'--release',str(release),'--out',str(performance)]
  step('validate',args);step('measure',args+['--run']);terminal=read(performance/'status.json')
  need(terminal['complete'] and not terminal['failed'] and terminal['node_lease_held'] is False and not alive(terminal['pid']),'complete/clean/exited Eco queue required')
  state.update(phase='complete',complete=True,performance_terminal=ref(performance/'status.json'))
 except BaseException as exc:state.update(phase='needs_attention',error=repr(exc));raise
 finally:state['finished_s']=time.time();save()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--inputs-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(dict(path=str(a.inputs.resolve()),sha256=a.inputs_sha256),a.out.resolve())
