"""Declare and run only the approved eight supplemental development windows."""
import argparse, copy, json, os, signal, sys, time
from pathlib import Path
H=Path(__file__).resolve().parent
U=H.parents[1]/'uniform-rate-20260909-v1'
sys.path.insert(0,str(U/'dynamic-producer'))
import fresh_support as f
import producer as p
from capacity_load_calibrate import generate_trace

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--run',action='store_true');args=ap.parse_args()
 out=args.out.resolve();f.need(not out.exists(),'fresh supplement root required');out.mkdir(parents=True)
 status=dict(schema='new-A-matched-low-idle-supplement-owner-v1',pid=os.getpid(),startticks=p.startticks(os.getpid()),started_s=time.time(),complete=False,node_lease_held=False)
 def update(**fields):
  status.update(fields,updated_s=time.time());tmp=out/'status.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(out/'status.json')
 p.CHILD_UPDATE=update
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,p.request_stop)
 update(phase='prepare')
 try:
  options=argparse.Namespace(fixed_qualification=U/'idle-qualification-002/qualified.json',fixed_validator=U/'verify_idle_budget2.py')
  root=p.prepare_root(options,out)
  oldcap=f.read(U/'dynamic-qualification-001/layout_calibration-inputs/capacity-binding.json')
  f.need(root['capacity']['identity']==oldcap['identity'],'supplement changes real model/node/control/profile identity')
  template,run=p.prepare_stage(root,out,'layout_calibration')
  spec=copy.deepcopy(f.checked(template))
  templates=f.read(f.A/'load-alpaca-inputs-001/templates.json')
  if isinstance(templates,dict):templates=templates['templates']
  seeds=(1701,1702,1703)
  for cycle,seed in zip(spec['cycles'],seeds):
   trace=generate_trace(templates,[dict(name='low',duration_s=60,rate_rps=1.5)],seed,spec['demand_domain_sha256'])
   cycle['low']=f.save(out/'supplement-inputs'/f'low-seed-{seed}.json',trace);f.add(spec['files'],cycle['low'])
  for path in sorted(H.glob('*.py')):f.add(spec['files'],f.ref(path))
  f.add(spec['files'],f.ref(H/'driver-source-audit.json'))
  spec.update(supplement_kind='matched-low-1.5-three-pairs-plus-one-idle-pair-v1',native_idle_poll_interval_s=.1,
      selected_phase_count=8,selected_phase_names=['cycle-1-idle-layout2','cycle-1-low-layout2','cycle-1-low-layout3','cycle-1-idle-layout3','cycle-2-low-layout2','cycle-2-low-layout3','cycle-3-low-layout2','cycle-3-low-layout3'],
      original_failed_layout=f.ref(U/'dynamic-qualification-001/layout_calibration/status.json'),original_producer_failure=f.ref(U/'dynamic-qualification-001/status.json'),
      reason='New-node low .3 evidence failed SLO; one idle window exceeded native sampling gap. Unchanged SLO/control identity; independent new1.5 endpoint; original negatives retained.',control_algorithm_changed=False)
  sref=f.save(out/'supplement-spec.json',spec)
  f.save(out/'prepared-root.json',root)
  update(spec=sref,output=str(run),phase='cpu_validation')
  argv=[sys.executable,'-B',str(H/'supplement_driver.py'),'--spec',sref['path'],'--spec-sha256',sref['sha256'],'--out',str(run)]
  p.command(argv,out/'cpu-validation.log')
  if args.run:
   update(phase='layout_calibration')
   p.command(argv+['--run'],out/'layout_calibration.log')
   update(phase='measured_pending_independent_audit',complete=True,finished_s=time.time())
  else:update(phase='cpu_prepared_only',complete=True,hardware_actions=False,finished_s=time.time())
 except BaseException as exc:
  update(phase='stopped_failure',complete=False,error=repr(exc),finished_s=time.time());raise
if __name__=='__main__':main()
