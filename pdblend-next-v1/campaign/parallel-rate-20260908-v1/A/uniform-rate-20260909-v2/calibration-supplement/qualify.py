"""Audit terminal supplement, build same-identity certificate, run original gates."""
import argparse,json,os,signal,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;U=H.parents[1]/'uniform-rate-20260909-v1'
sys.path.insert(0,str(U/'dynamic-producer'))
import fresh_support as f
import producer as p
import audit as original_audit
import combine

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--supplement',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args()
 f.need(a.run,'explicit original gate execution required');out=a.out.resolve();f.need(not out.exists(),'fresh qualification root required');out.mkdir(parents=True)
 state=dict(schema='new-A-mixed-capacity-producer-status-v2',started_s=time.time(),pid=os.getpid(),startticks=p.startticks(os.getpid()),complete=False,node_lease_held=False,stages=[])
 def update(**kw):
  state.update(kw,updated_s=time.time());tmp=out/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out/'status.json')
 p.CHILD_UPDATE=update
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,p.request_stop)
 update(phase='independent_mixed_calibration_audit')
 try:
  parent=f.read(a.supplement/'status.json')
  f.need(parent['complete'] and parent['phase']=='measured_pending_independent_audit' and f.no_live_pid(parent['pid'],parent['startticks']),'supplement owner must terminate before proof')
  composition=combine.compose(U/'dynamic-qualification-001/layout_calibration',a.supplement,out/'certificate')
  proof=combine.verify(composition);certificate=proof['certificate'];update(composition=composition,certificate=certificate)
  root=f.read(a.supplement/'prepared-root.json')
  for path in sorted(H.glob('*.py')):f.add(root['files'],f.ref(path))
  f.add(root['files'],f.ref(H/'driver-source-audit.json'))
  paired=f.ref(U/'dynamic-qualification-001/layout_calibration/cycle-1-under_load-layout2to3/result.json')
  for mode in ('automatic_underload_gate','qualification900'):
   f.need(not p.stop_requested(),'stop before original autonomous stage')
   spec,run=p.prepare_stage(root,out,mode,certificate=certificate,paired=paired)
   update(phase=mode,current_spec=spec,current_output=str(run))
   p.command([sys.executable,'-B',str(f.DRIVER/'capacity_load_calibrate.py'),'--spec',spec['path'],'--spec-sha256',spec['sha256'],'--out',str(run),'--run'],out/(mode+'.log'))
   audit=original_audit.audit_stage(run,spec,mode);aref=f.save(out/(mode+'-audit.json'),audit)
   state['stages'].append(dict(mode=mode,spec=spec,output=str(run),audit=aref));update()
  base=f.checked(f.checked(state['stages'][-1]['spec'])['original_binding'])
  base.update(independent_capacity_qualification_granted=True,isolated_power_adapter=f.ref(U/'isolated-power/manifest.json'))
  bref=f.save(out/'qualified-base-binding.json',base)
  update(complete=True,phase='qualified',binding=bref,finished_s=time.time())
  q=dict(schema='new-A-mixed-fresh-dynamic-capacity-qualification-v2',node='Anew20260909',model='14b',binding=bref,status=f.ref(out/'status.json'),stages=state['stages'],composition=composition,certificate=certificate,fixed_qualification=root['fixed_qualification'],fixed_validator=root['fixed_validator'],source_identity=root['compatibility'],files=f.tree_files(out),source_files={str(path):f.sha(path) for path in H.glob('*.py')})
  qref=f.save(out/'qualified.json',q)
  f.save(out/'handoff.json',dict(node='Anew20260909',model='14b',system='pdblend',qualification=qref,qualification_validator=f.ref(H/'verify.py'),formal_execution_started=False))
 except BaseException as exc:
  update(complete=False,phase='stopped_failure',error=repr(exc),finished_s=time.time());raise
if __name__=='__main__':main()
