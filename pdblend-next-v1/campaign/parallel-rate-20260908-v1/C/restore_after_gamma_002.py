"""Restore the same stopped C7B residents after the external task cleanly released.

No synthetic previous serving identity: the actual predecessor is an empty node.
Original restart/native/power/cleanup primitives are reused byte-for-byte.
"""
import argparse,asyncio,importlib.util,json,os,signal,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'baseline-after-gamma-restore-002'
PREVIOUS=OUT/'empty-node-binding.json'
PAUSE=HERE/'gamma-priority-pause-002/receipt.json'
OLD=HERE/'restore_boundary_baselines_p4v2.py'
def load():
 s=importlib.util.spec_from_file_location('c_retained_after_gamma_original_adapter',OLD);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
m=load();m.OUT=OUT;m.PREVIOUS=PREVIOUS

def pause_check():
 receipt=m.read(PAUSE)
 m.require(receipt['user_authorized'] and receipt['external_owner_exited'] and receipt['external_child_exited'] and receipt['all_containers_stopped'] and receipt['owned_cleanup_complete'] and receipt['clock_restore_complete'],'external task not at clean terminal boundary')
 for p,h in receipt['files'].items():m.require(m.sha(p)==h,'external terminal source/status changed')
 status=m.read(next(v['snapshot']['path'] for v in receipt['original_observations'] if v['original_path'].endswith('/owner-status.json')))
 flag=Path(receipt['stop_owner_path']);m.require(flag.is_file() and ('stop' in flag.read_text().lower() or 'priority' in flag.read_text().lower()),'external pause flag no longer requests stop')
 m.require(status['phase']=='failed' and status['node_lease_held'] is False and not Path('/proc',str(status['pid'])).exists(),'external owner resumed')
 m.require(m.read(PREVIOUS)['instances']==[] and m.read(PREVIOUS)['configs']=={} and m.read(PREVIOUS)['actual_predecessor']=='empty-node-after-external-cleanup','previous state must remain explicitly empty')
 return receipt
original_verify=m.verify_release
def verify_release(path,digest):pause_check();return original_verify(path,digest)
m.verify_release=verify_release

def prepare():
 m.require(not OUT.exists(),'fresh restore attempt required');OUT.mkdir()
 previous=dict(schema='empty-node-restore-predecessor-v1',model='7b',system='pdblend',hostname=m.HOST,protocol_id='per-dataset-slo-five-system-fixed-window-v1',deadline_s=None,campaign_lifecycle='until_declared_complete_v1',instances=[],configs={},files={str(PAUSE):m.sha(PAUSE)},large_inputs={},actual_predecessor='empty-node-after-external-cleanup',serving_binding=False,restore_only=True)
 previous['files'].update(m.read(PAUSE)['files']);m.write(PREVIOUS,previous)
 deps=dict(m.read(HERE/'baseline-boundary-restore-p4v2/adapter-manifest.json')['files']);deps[str(Path(__file__).resolve())]=m.sha(__file__);deps[str(PAUSE)]=m.sha(PAUSE);deps.update(previous['files']);deps[str(HERE/'gamma-priority-pause-001/containers.before.json')]=m.sha(HERE/'gamma-priority-pause-001/containers.before.json')
 m.write(OUT/'adapter-manifest.json',dict(files=deps,unchanged_hardware_primitives=str(m.PARENT),actual_empty_predecessor=True))
 m.write(OUT/'deployment.json',m.expected());print(json.dumps(dict(prepared=True,spec=str(OUT/'deployment.json'),sha256=m.sha(OUT/'deployment.json'))))

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
 if a.prepare:prepare();return
 m.package_check();spec=m.read(OUT/'deployment.json');m.validate_spec(spec);pause_check()
 if not a.run:print(json.dumps(dict(passed=True,cpu_only=True,actual_empty_predecessor=True)));return
 m.require('PDBLEND_NODE_LOCK_FD' not in os.environ,'fresh unique lease required');host=Path(spec['host_release']);sys.path[:0]=[str(host/'src'),str(host),'/root/workspace/pdblend/.runtime-deps']
 from ecopadg.serving.campaign import node_lease
 async def run():
  task=asyncio.current_task();loop=asyncio.get_running_loop();stop=False
  def cancel():
   nonlocal stop
   if not stop:stop=True;task.cancel()
  for sig in (signal.SIGTERM,signal.SIGINT):loop.add_signal_handler(sig,cancel)
  return await m.bind_parent(spec).launch(OUT/'deployment.json')
 with node_lease():r=asyncio.run(run())
 print(json.dumps(dict(restored=r['complete'],measurement_valid=r['measurement_valid'],fresh_27_required=True)))
if __name__=='__main__':main()
