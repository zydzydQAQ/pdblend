"""Retire the recorded finished predecessor under the shared node lease, with all-eight energy."""
import asyncio,json,os,socket,sys,time
from pathlib import Path
import power_selftest as p
import bootstrap as b
from inputs import E,N
sys.path[:0]=[str(p.METER),str(p.HOST/'src'),str(p.HOST),'/root/workspace/pdblend/.runtime-deps']
async def run():
 from ecopadg.serving.campaign import node_lease
 from capacity_backend import TransitionMeter
 from meter_evidence import install
 import aiohttp
 common=b.load(E/'run.py','slo14_retire_common')
 out=N/p.NODE/'retirement';assert not out.exists()
 predecessor=p.ROOT/({'A':'A/uniform-rate-20260909-v2/pipeline-007/status.json','C':'C/uniform-rate-20260909-v2/pipeline-004/status.json'}[p.NODE])
 old=p.read(predecessor);assert old['complete'] and old['finished_s'] and not old['node_lease_held']
 allowed={'A':'/A/uniform-rate-20260909-v2/baseline-lb-distserve-002/','C':'/AC-baseline-deployment-prepared-v1/C-resident/'}[p.NODE]
 with node_lease():
  p.actual_identity();out.mkdir(parents=True);state=dict(node=p.NODE,hostname=socket.gethostname(),started_s=time.time(),predecessor=p.ref(predecessor),complete=False,node_lease_held=True,owned_stops=[],errors=[]);b.save(out/'status.json',state)
  ids=(await common.command('docker','ps','-q')).split();inspect=json.loads(await common.command('docker','inspect',*ids)) if ids else []
  for item in inspect:
   serialized=json.dumps(item)
   assert allowed in serialized,'unrelated live Docker container: '+item['Name']
  b.save(out/'containers.before.json',inspect)
  # No active predecessor experiment/controller process may survive.
  for proc in Path('/proc').glob('[0-9]*'):
   try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
   except (OSError,UnicodeError):continue
   if str(p.ROOT) in cmd and any(x in cmd for x in ['/pipeline.py','/pipeline_v','/run_cells.py','/qualify_fixed.py','/qualify_idle.py']):
    raise RuntimeError('live predecessor scheduler/qualification: '+cmd[:400])
  finish=install(out/'isolated-samplers',p.ref(p.HOST/'manifest.json'),p.ref(p.ADAPTER),p.ref(p.HOOKS));meter=await TransitionMeter(out/'power').start()
  try:
   for item in inspect:
    result=await common.command('docker','stop','--time','30',item['Id']);state['owned_stops'].append(dict(id=item['Id'],result=result));b.save(out/'status.json',state)
   assert not (await common.command('docker','ps','-q')).strip()
   assert not (await common.command('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits')).strip()
   state['complete']=True
  finally:
   state['measurement']=await meter.finish();finish();state['complete']=state['complete'] and state['measurement']['measurement_valid'];state.update(finished_s=time.time(),node_lease_held=False);b.save(out/'status.json',state)
  assert state['complete']
asyncio.run(run())
