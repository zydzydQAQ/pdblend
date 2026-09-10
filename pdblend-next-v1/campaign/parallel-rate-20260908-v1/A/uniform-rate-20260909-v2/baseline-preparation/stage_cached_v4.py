"""Sequential fresh-node baseline creation, independent gates, then immutable handoffs."""
import argparse,fcntl,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent/'qualification-v2';U=HERE.parents[1];ROOT=U.parents[1]
sys.path.insert(0,str(ROOT/'common/uniform-rate-20260909-v2'))
import support as p
sys.path.insert(0,str(HERE))
from qualify import child

def terminal(pipeline,out):
 state=p.read(pipeline/'status.json')
 p.need('8tp1' in state.get('completed_baseline_stages',[]) and not state['node_lease_held'] and not state.get('error'),'previous resident stage incomplete')
 last=p.checked(state['last_cell_status']);p.need(last['complete'] and last['finished_s'] and not p.active_owner(last) and not last['node_lease_held'] and not last['failed'],'previous resident measurement active/failed')
 release=p.checked(last['release'])
 value=dict(schema='uniform-v2-new-A-baseline-layout-terminal',complete=True,node_lease_held=False,finished_s=time.time(),node='Anew20260909',model='14b',last_cell_status=state['last_cell_status'],observations=state['observations'],declaration=state['declaration'],binding=release['binding'])
 p.save(out/'predecessor-terminal.json',value);return out/'predecessor-terminal.json'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--definition',type=Path,required=True);a=ap.parse_args();definition=p.checked(p.ref(a.definition))
 for reference in definition['sources']:p.need(p.sha(reference['path'])==reference['sha256'],'baseline stage source changed')
 out=Path(definition['out']);p.need(not out.exists(),'fresh baseline stage output required')
 lock=(U/'baseline-stage.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);out.mkdir(parents=True)
 state=dict(schema='uniform-v2-baseline-stage-status',node='Anew20260909',model='14b',stage=definition['stage'],pid=os.getpid(),startticks=p.process_identity(os.getpid())['startticks'],started_s=time.time(),complete=False,node_lease_held=False,phase='create',definition=p.ref(a.definition));p.save(out/'status.json',state)
 try:
  predecessor=(Path(definition['pipeline'])/'pdblend-terminal.json' if definition['stage']=='8tp1' else terminal(Path(definition['pipeline']),out))
  producer=HERE.parent/'producer_v2.py';bootstrap=out/'bootstrap-stage';qualification=out/'qualification'
  child([sys.executable,'-B',str(producer),'--spec',definition['template'],'--predecessor-terminal',str(predecessor),'--out',str(bootstrap),'--run'],out,state)
  state['phase']='fresh_native_qualification';p.save(out/'status.json',state)
  child([sys.executable,'-B',str(HERE/'qualify.py'),'--bootstrap',str(bootstrap/'bootstrap/binding.json'),'--out',str(qualification),'--node','Anew20260909','--hostname','iZwz9274emxme9019d2sjgZ','--profile',definition['profile'],'--run'],out,state)
  bindings=p.read(qualification/'bindings.json')
  for system,reference in bindings.items():
   state['phase']='qualification-cache-'+system;p.save(out/'status.json',state)
   cache_out=out/'qualification-caches'/system
   child([sys.executable,'-B',str(HERE.parent/'optional_cache_v4.py'),'--helper',definition['cache_helper']['path'],'--qualification',reference['path'],'--validator',str(HERE/'verify.py'),'--out',str(cache_out)],out,state)
   selection=p.read(cache_out/'selection.json')
   p.need(selection['qualification']==reference and selection['original_validator']==p.ref(HERE/'verify.py') and not selection['interrupted'],'cache selection input changed/stopped')
   cached_validator=selection['qualification_validator']
   state.setdefault('qualification_caches',{})[system]=p.ref(cache_out/'selection.json')
   state['phase']='collector-'+system;p.save(out/'status.json',state)
   destination=out/'formal'/system
   child([sys.executable,'-B',definition['meter_binding']['path'],'--binding',reference['path'],'--out',str(destination),'--native-validator',cached_validator['path']],out,state)
   qref=p.ref(destination/'qualified.json')
   # CLI above independently replays native/raw evidence before publishing qref.
   handoff=dict(node='Anew20260909',model='14b',system=system,qualification=qref,qualification_validator=definition['meter_binding'],
    predecessors=[p.ref(qualification/'status.json')],extra_files=[p.ref(a.definition)])
   for dataset in p.checked(reference)['configs']:
    p.save(out/'handoffs'/(dataset+'-'+system+'.json'),handoff)
  state.update(complete=True,phase='qualified',bindings=p.ref(qualification/'bindings.json'))
 except BaseException as exc:state.update(error=repr(exc),phase='stopped_failure');raise
 finally:state.update(finished_s=time.time(),node_lease_held=False);p.save(out/'status.json',state);lock.close()
if __name__=='__main__':main()
