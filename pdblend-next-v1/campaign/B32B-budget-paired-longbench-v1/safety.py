"""Frozen-derived HTTP identity and native TP2 cleanup helpers; no GPU workload entry point."""

import asyncio

import csv

import hashlib

import json

import os

from pathlib import Path

import signal

import time

import aiohttp

from ecopadg.serving.campaign import node_lease

from ecopadg.serving.backend import ClockOwner

from ecopadg.serving.measurement import power_evidence, save_raw

from ecopadg.measure.backends import PynvmlBackend

from ecopadg.measure.power import PowerSampler, trapezoid_energy

from ecopadg.metrics import clip_power_window

ROOT = Path(__file__).resolve().parent

OLD = ROOT.parent/'B32B-engine-v3-candidate-v2'

RELEASE = ROOT.parents[1]/'releases/io-v3-runtime'

PORTS = (33500, 33501)

IDS = ('nextv3b0', 'nextv3b1')

NAMES = tuple('pdb-v2-'+x for x in IDS)

IMAGE = 'sha256:fd4ba34686c028ec6ba0ae17220f833b24c2f45f29535f735066f5a7a27004c2'

BUDGET = dict(schema_version=1, max_num_batched_tokens=8192, max_num_seqs=32)

RESIDUALS = ('active','running','waiting','kv_allocations','transfer_allocations',
    'transfer_buffered_tensors','transfer_inflight_receives','transfer_inflight_sends')

def require(ok, message):
    if not ok: raise RuntimeError(message)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temporary.replace(path)

def check_ack(r, *, startup=False):
    require(not r.get('error') and not r.get('runtime_error') and r.get('transport_healthy') is True, 'unhealthy owner/transport')
    require(r.get('generation') == r.get('acknowledged_generation') and r.get('scheduler_budget_pending') is None, 'pending or false ACK')
    owners = [x.get('controls',{}).get('runtime') for x in r.get('scheduler_io',[])]
    require(owners and all(x and x.get('generation') == r['generation'] and not x.get('error') for x in owners), 'cache/owner generation differs')
    require(0 <= time.time()-r.get('timestamp',0) <= 1, 'stale owner snapshot')
    if startup:
        require(r.get('scheduler_budget_effective') == dict(max_num_batched_tokens=8192,max_num_seqs=32), 'startup budget not restored')

def is_idle(r): return all(k in r for k in RESIDUALS) and not any(r[k] for k in RESIDUALS)

def check_drain(before, proof):
    require(proof.get('drained') is True and proof.get('accepting') is False
        and proof.get('generation') == before['generation']+1
        and proof.get('drain_proof_type') == 'synchronous_put_owner_barrier', 'invalid native owner drain')
    ranks = proof.get('transfers',[])
    require(proof.get('send_counters_verified') is True and len(ranks)==2, 'missing TP2 send proof')
    require(all(r.get('listener_alive') is True and r.get('send_counters_observed') is True
        and r.get('send_healthy') is True and r.get('send_started')==r.get('send_completed') and r.get('send_failed')==0
        and not any(r.get(k) for k in ('buffered_tensors','inflight_receives','inflight_sends','buffered_gpu_bytes','allocations'))
        for r in ranks), 'rank residual/unhealthy transport')

class Safety:
    def remaining(self, limit):
            if self.deadline is None: return limit
            remaining = self.deadline-time.monotonic()
            if remaining <= 0: raise TimeoutError('outer cleanup deadline reached')
            return min(limit, remaining)

    async def command(self,*args,timeout=12):
            p = await asyncio.create_subprocess_exec(*args,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
            try: out,_ = await asyncio.wait_for(p.communicate(),self.remaining(timeout))
            except BaseException:
                if p.returncode is None: p.kill(); await p.wait()
                raise
            require(p.returncode==0,out.decode(errors='replace')[-1000:]); return out.decode()

    async def http(self,port,route,body=None,*,label='',timeout=10):
            row = dict(port=port,route=route,request=body,label=label,started_s=time.time())
            try:
                async with self.session.request('GET' if body is None else 'POST',f'http://127.0.0.1:{port}{route}',
                    json=body,timeout=aiohttp.ClientTimeout(total=self.remaining(timeout))) as response:
                    text=await response.text()
                    try: value=json.loads(text)
                    except ValueError: value=text
                    row.update(status=response.status,body=value)
                    require(response.status==200,route+': '+str(value)[:500]); return value
            except BaseException as exc: row['error']=repr(exc); raise
            finally: row['finished_s']=time.time(); self.log.write(json.dumps(row,allow_nan=False)+'\n')

    async def runtime(self,port,label='runtime'): return await self.http(port,'/runtime',label=label,timeout=5)

    async def settled_idle(self,port,timeout=20):
            until=time.monotonic()+self.remaining(timeout)
            while True:
                r=await self.runtime(port,'wait-settled-idle')
                require(not r.get('error') and not r.get('runtime_error') and r.get('transport_healthy') is True,'unhealthy owner')
                if is_idle(r) and r.get('scheduler_budget_pending') is None and r.get('generation')==r.get('acknowledged_generation'):
                    check_ack(r); return r
                require(time.monotonic()<until,'request/budget did not settle'); await asyncio.sleep(.02)

    async def identity(self,label):
            manifest=json.loads((ROOT/'manifest.json').read_text())
            for path,digest in manifest['frozen_inputs'].items(): require(sha(path)==digest,'frozen input changed: '+path)
            for path,digest in manifest['files'].items(): require(sha(ROOT/path)==digest,'wrapper/frozen validator changed: '+path)
            release=json.loads((RELEASE/'manifest.json').read_text())
            for path,digest in release['files'].items(): require(sha(RELEASE/path)==digest,'release changed: '+path)
            objects=json.loads(await self.command('docker','inspect',*NAMES)); rows=[]
            for i,obj in enumerate(objects):
                require(obj['Image']==IMAGE and obj['State']['Running'],'image/container identity changed')
                env={x.partition('=')[0]:x.partition('=')[2] for x in obj['Config']['Env'] if x.startswith('NCCL_')}
                require(env==json.loads((OLD/'transport-environment.json').read_text())['nccl_environment'],'NCCL environment changed')
                prov=await self.http(PORTS[i],'/provenance',label=label)
                expected={str(RELEASE/p):v for p,v in release['files'].items() if p.startswith('src/ecopadg/serving/') and p.endswith('.py')}
                require(prov.get('source_files_at_import')==expected,'source import identity differs')
                require(prov.get('instance_id')==IDS[i] and prov.get('tp')==2 and prov.get('model')=='/models/Qwen2.5-32B-Instruct'
                    and prov.get('cuda_visible_devices')==f'{2*i},{2*i+1}','wrong model/TP/GPU')
                patches=json.loads((OLD/'candidate-manifest.json').read_text())['image_patch_files']
                code='import hashlib,json,pathlib;print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in '+repr(list(patches))+'}))'
                observed=json.loads(await self.command('docker','exec',NAMES[i],'python3','-c',code))
                require(observed==patches,'image patch bytes changed')
                r=await self.runtime(PORTS[i],label); check_ack(r,startup=True)
                require(is_idle(r) and r.get('accepting') is True and r['role']=='mixed' and r['mode']=='continuous','not idle continuous mixed')
                rows.append(dict(container=obj,provenance=prov,runtime=r,nccl_environment=env,image_patch_sha256=observed))
            write(ROOT/('identity.'+label+'.json'),rows); return rows

    async def resume(self,port):
            current=await self.runtime(port,'finally-resume-generation'); target=current['generation']+1
            await self.http(port,'/control',dict(generation=target,role='mixed',mode='continuous',admit_prefill=True,
                admit_decode=True,scheduler_budget=BUDGET),label='finally-startup-budget-resume',timeout=20)
            r=await self.settled_idle(port,timeout=12); check_ack(r,startup=True)
            require(r['generation']==target and r.get('accepting') is True and r['role']=='mixed' and r['mode']=='continuous'
                and r['admit_prefill'] and r['admit_decode'],'restore ACK/admission differs'); return r

    async def restore_one(self,port):
            row=self.state.setdefault('cleanup',{}).setdefault(str(port),{})
            error=None
            try:
                # The proof phase has its own bound, reserving time for unconditional admission restoration.
                async def proof():
                    before=await self.settled_idle(port,timeout=18)
                    barrier=await self.http(port,'/drain',dict(expected_generation=before['generation']),label='outer-native-drain',timeout=12)
                    row['drain']=barrier; check_drain(before,barrier)
                await asyncio.wait_for(proof(),self.remaining(30))
            except BaseException as exc: error=exc; row['proof_error']=repr(exc)
            finally:
                try: row['restored']=await self.resume(port)
                except BaseException as exc: row['resume_error']=repr(exc); raise
            if error is not None: raise error
