import asyncio,importlib.util,json,sys,os
from pathlib import Path
A=Path(__file__).resolve().parent;H=A.parent/'hosts/14b-capacity-p6';sys.path[:0]=[str(H/'src'),str(H),'/root/workspace/pdblend/.runtime-deps']
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
from ecopadg.serving.campaign import node_lease
async def run():
 import aiohttp
 from ecopadg.measure.backends import PynvmlBackend
 c=load(A.parent/'common/execution-until-complete-v1/run.py','fresh_oracle_common')
 out=A/'p6-original-oracle-001';c.require(not out.exists(),'fresh oracle output');out.mkdir()
 spec=c.read(A/'load-p6-gate-inputs-002/spec.json');binding=c.read(spec['original_binding']['path']);cap=c.read(spec['capacity_binding']['path'])
 profile=spec['profiles'];c.require(c.sha(profile['path'])==profile['sha256'],'actual profile changed');binding['files'][profile['path']]=profile['sha256'];c.validate_binding(binding);c.write(out/'binding.json',binding)
 sys.path.insert(0,str(A/'p4-minimal'));helper=load(A/'p4-minimal/runner.py','fresh_oracle_helper')
 hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
 async with aiohttp.ClientSession(trust_env=False) as session:
  await c.identity(session,binding);await helper.measured_ordinary(c,session,binding,hardware,out/'setup');await c.identity(session,binding)
 old=c.read(cap['correctness_oracles']['path']);measured=c.read(out/'setup/ordinary/status.json');cases=[]
 for case in old['cases']:
  rows=[r for r in measured['replies'] if r['prompt_length']==case['prompt_length']]
  c.require(len(rows)==2 and {r['instance_id'] for r in rows}=={i['id'] for i in binding['instances']},'both actual originals required')
  tokenids=[r['response']['token_ids'] for r in rows];c.require(tokenids[0]==tokenids[1] and len(tokenids[0])==64,'full64 numerical agreement required')
  cases.append(dict(prompt_length=case['prompt_length'],prompt=case['prompt'],token_ids=tokenids[0],max_tokens=64))
 rp=out/'setup/ordinary/status.json';c.write(out/'oracles.json',dict(schema='capacity-correctness-oracles-v1',measured=True,model_source_identity=cap['identity'],source=dict(path=str(rp),sha256=c.sha(rp)),cases=cases))
 print(json.dumps(dict(passed=True,oracle=str(out/'oracles.json'),sha256=c.sha(out/'oracles.json'))))
if __name__=='__main__':
 with node_lease():asyncio.run(run())
