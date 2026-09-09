"""Unchanged natural profiler in an isolated process, with an identity-gated HTTP observer."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit
import aiohttp
from capacity import validate_runtime_capacity, validate_batch16_observation
import late_context

ROOT=Path(__file__).resolve().parent
HELPER=ROOT.parent/'budget-profiling-v2-candidate'

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
 sys.modules[name]=module;spec.loader.exec_module(module);return module

class DispatchGuard:
 def __init__(self,journal):self.journal=journal;self.identity_verified=False
 def record(self,method,url,kwargs):
  parsed=urlsplit(str(url));route=parsed.path
  if method.upper()=='GET':return
  if not self.identity_verified:raise RuntimeError('complete byte identity gate has not passed; no write HTTP allowed')
  if parsed.hostname!='127.0.0.1' or parsed.port!=33500 or route not in ('/v1/completions','/control','/drain','/cancel'):
   raise RuntimeError('write outside the authorized observation instance/routes')
  rid=(kwargs.get('headers') or {}).get('X-Request-Id')
  if route=='/v1/completions' and (not isinstance(rid,str) or not rid.startswith('pdb-profile-')):
   raise RuntimeError('request ownership missing')
  self.journal.write(json.dumps(dict(port=parsed.port,method=method,route=route,request_id=rid,
   body=kwargs.get('json'),started_s=time.time()))+'\n')
  self.journal.flush() # All bytes are sent to the OS before the first network await.

async def main():
 sys.path.insert(0,str(ROOT));outer=load('batch_outer_check',ROOT/'run.py');outer.check()
 observation=load('original_batch_observation',ROOT/'observation.frozen.py')
 sys.path.insert(0,str(HELPER));module=load('unaltered_natural_profiler',HELPER/'run.py')
 import evidence
 outer.require(Path(evidence.__file__).resolve()==HELPER/'evidence.py','wrong frozen profiler evidence')
 expected=json.loads((ROOT/'expected-identity.json').read_text())
 journal=(ROOT/'dispatch.jsonl').open('x',buffering=1);guard=DispatchGuard(journal)
 original_request=aiohttp.ClientSession._request
 async def observed_request(session,method,url,**kwargs):
  guard.record(method,url,kwargs)
  return await original_request(session,method,url,**kwargs)
 class VerifiedProfiler(module.Profiler):
  async def identity(self):
   actual=await super().identity()
   auxiliary=json.loads((ROOT/'auxiliary-vllm-sources.json').read_text())
   script='import hashlib,json,pathlib,sys;print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in json.loads(sys.argv[1])}))'
   observed=json.loads(await module.command('docker','exec',self.args.container,'python3','-c',script,json.dumps(sorted(auxiliary))))
   outer.require(observed==auxiliary,'original transfer source hash changed')
   actual['live_vllm_and_serving_source_sha256'].update(observed)
   number=getattr(self,'identity_number',0)+1;self.identity_number=number
   directory=ROOT/'child-identities';directory.mkdir(exist_ok=True)
   with (directory/f'{number:04d}.json').open('x') as handle:
    json.dump(actual,handle,indent=2,allow_nan=False);handle.write('\n')
   observation.validate_identity(actual,expected)
   provenance=await self.http('/provenance')
   receipt=late_context.source_record(actual,provenance,expected,observed_s=time.time())
   source_directory=ROOT/'source-order';source_directory.mkdir(exist_ok=True)
   late_context.write_new(source_directory/f'{number:04d}.json',receipt)
   guard.identity_verified=True
   return actual
  async def idle(self):
   state=await super().idle()
   if getattr(self,'active_point',None) is not None:
    proof=validate_runtime_capacity(state,require_accepting=False)
    self.capacity_observations.append(proof)
   return state
  async def measure(self,point,path):
   self.active_point=point;self.capacity_observations=[]
   try:raw=await super().measure(point,path)
   finally:self.active_point=None
   raw['actual_capacity_gate_observations']=self.capacity_observations
   if not raw.get('error'):
    try:
     events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines() if line]
     raw['batch_coverage_observation']=observation.validate_tp2_observation(raw,events)
     raw['short512_batch16_evidence']=validate_batch16_observation(raw,events)
    except Exception as exc:raw['error']=repr(exc)
    module.write(path/'raw.json',raw)
   return raw
 module.Profiler=VerifiedProfiler;aiohttp.ClientSession._request=observed_request
 try:
  good=await module.run(observation.arguments())
  report=late_context.derive_all(ROOT,original_evidence=evidence,tp2_validator=observation.validate_tp2_observation)
  return good and report['complete']
 finally:aiohttp.ClientSession._request=original_request;journal.close()
def verify_owned_parent(parent_pid):
 if type(parent_pid) is not int or parent_pid<=1 or os.getppid()!=parent_pid:
  raise RuntimeError('child must be owned by the explicit outer runner')
 status=json.loads((ROOT/'status.json').read_text())
 if status.get('verified_for_controls') is not True or status.get('phase')!='natural_batch_observations':
  raise RuntimeError('outer identity/capacity gate has not released this child')

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--parent-pid',type=int,required=True);args=parser.parse_args()
 verify_owned_parent(args.parent_pid)
 raise SystemExit(0 if asyncio.run(main()) else 1)
