#!/usr/bin/env python3
"""GPU worker for a preregistered independent decode validation partition."""
import argparse,asyncio,json,time
from pathlib import Path
from pdblend.profile.profiler import Profiler
from pdblend.profile.merge import sha256
from pdblend.engine.launcher import Fleet
from pdblend.engine.client import EngineClient
from pdblend.bench.gates import random_prompt

p=argparse.ArgumentParser();p.add_argument('plan',type=Path);p.add_argument('out',type=Path);p.add_argument('--port',type=int,required=True);a=p.parse_args()
plan=json.loads(a.plan.read_text());candidate=Path(plan['candidate'])
if sha256(candidate)!=plan['candidate_sha256']:raise ValueError('candidate changed after preregistration')
class ValidationProfiler(Profiler):
 async def _wait_progress(self,live,tasks,targets,timeout_s=600):
  return await super()._wait_progress(live,tasks,targets,timeout_s)
prof=ValidationProfiler('Qwen2.5-7B-Instruct',[0],freqs=(900,),mixed_freqs=(),out_dir=a.out,base_port=a.port)
prof.raw['validation']=dict(plan=plan,started_s=time.time(),purpose='independent_holdout',generation_budget=512)
try:
 with Fleet(prof.specs,a.out/'logs') as fleet:
  fleet.start_all();inst=fleet[prof.specs[0].instance_id]
  prof.raw['kv_capacity_tokens']=prof._kv_capacity(inst);prof._lock(900,[0])
  async def run():
   async with EngineClient(inst.spec.instance_id,inst.spec.base_url) as client:
    await client.complete(random_prompt(256,1),8,'warmup')
    for j,pt in enumerate(plan['points']):
     b,c=pt['batch'],pt['context_tokens']
     if b*(c+512)>.9*prof.raw['kv_capacity_tokens']:
      prof.raw.setdefault('skipped',[]).append(dict(**pt,reason='full_generation_KV_capacity'));prof._checkpoint();continue
     row=await prof._decode_batch(client,[0],b,c,64,f'v-{j}-{c}-{b}')
     row.update(freq_mhz=900,validation_label=pt.get('label','grid'))
     prof.raw['decode'].append(row);prof._checkpoint()
     print(json.dumps(dict(batch=b,context=c,step=row['step_seconds'])),flush=True)
  asyncio.run(run())
 prof.raw['validation'].update(finished_s=time.time(),complete=True)
finally:
 prof.meter.reset_all();prof._checkpoint()
