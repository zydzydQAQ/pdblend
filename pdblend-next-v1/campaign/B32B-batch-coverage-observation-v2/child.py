"""Unchanged natural profiler in an isolated process, with an identity-gated HTTP observer."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit
import aiohttp

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
   actual=await super().identity();observation.validate_identity(actual,expected)
   guard.identity_verified=True
   return actual
  async def measure(self,point,path):
   raw=await super().measure(point,path)
   if not raw.get('error'):
    try:
     events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines() if line]
     raw['batch_coverage_observation']=observation.validate_tp2_observation(raw,events)
    except Exception as exc:raw['error']=repr(exc)
    module.write(path/'raw.json',raw)
   return raw
 module.Profiler=VerifiedProfiler;aiohttp.ClientSession._request=observed_request
 try:return await module.run(observation.arguments())
 finally:aiohttp.ClientSession._request=original_request;journal.close()
if __name__=='__main__':raise SystemExit(0 if asyncio.run(main()) else 1)
