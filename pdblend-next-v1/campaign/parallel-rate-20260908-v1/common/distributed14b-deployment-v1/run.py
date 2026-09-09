"""One explicit deployment operation under an exclusive original node lease."""
import argparse,asyncio,json,os,signal
from pathlib import Path
import deploy
def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
 spec=deploy.read(a.spec);deploy.validate_spec(spec)
 if not a.run:print(json.dumps(dict(cpu_only=True,hardware_actions=False,spec_valid=True)));return
 deploy.require('PDBLEND_NODE_LOCK_FD' not in os.environ,'fresh unleased deployment owner required')
 deploy.adapter.load_runtime(spec['host_release'],spec['common_dir'])
 from ecopadg.serving.campaign import node_lease
 async def execute(lease):
  task=asyncio.current_task();loop=asyncio.get_running_loop();cancelled=False
  def stop():
   nonlocal cancelled
   if not cancelled:cancelled=True;task.cancel()
  for sig in (signal.SIGTERM,signal.SIGINT):loop.add_signal_handler(sig,stop)
  return await deploy.execute(a.spec,a.out,run=True,lease=lease)
 with node_lease() as lease:result=asyncio.run(execute(lease))
 print(json.dumps(dict(complete=result['complete'],measurement_valid=result['measurement_valid'],binding=result['binding_base'])))
if __name__=='__main__':main()
