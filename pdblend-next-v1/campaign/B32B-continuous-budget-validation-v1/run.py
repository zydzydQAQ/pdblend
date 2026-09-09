"""One independent continuous budget gate; temporal correctness remains failed."""
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


class Gate:
    def __init__(self):
        require(not (ROOT/'status.json').exists() and not (ROOT/'outer-http.jsonl').exists(), 'existing evidence retained')
        self.log = (ROOT/'outer-http.jsonl').open('x', buffering=1)
        self.child = None; self.child_log = None; self.clocks = None
        self.verified = False; self.deadline = None; self.event_offsets = {}
        self.state = dict(complete=False, passed=False, phase='preflight', started_s=time.time(),
            scope='independent continuous scheduler budget correctness only',
            total_runtime_gate='failed', temporal_correctness='unfixed; original strict failure retained',
            work_timeout_s=390, cleanup_timeout_s=90, original_validator_unchanged=True,
            baseline_execution=False, performance_matrix=False)
        self.save()

    def save(self): write(ROOT/'status.json',self.state)

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

    async def work(self,hardware):
        self.state['identity_before']=await self.identity('before')
        self.verified=True; self.state['verified_for_controls']=True
        for name in IDS:
            path=OLD/'runtime'/(name+'.control.events.jsonl'); require(path.is_file(),'owner timeline missing')
            self.event_offsets[name]=path.stat().st_size
        self.clocks=await asyncio.to_thread(ClockOwner,hardware,(0,1,2,3))
        await self.clocks.set((0,1,2,3),2520,verify_rise=False)
        self.state['clock_policy']=dict(gpus=[0,1,2,3],mhz=2520,verify_rise=False)
        self.child_log=(ROOT/'child.log').open('xb')
        self.state.update(phase='unchanged_continuous_validator',validator_start_s=time.time()); self.save()
        self.child=await asyncio.create_subprocess_exec('python3','-u',str(ROOT/'child.py'),cwd=str(ROOT),
            stdin=asyncio.subprocess.DEVNULL,stdout=self.child_log,stderr=asyncio.subprocess.STDOUT,start_new_session=True)
        self.state['child_pid']=self.child.pid; self.save()
        code=await self.child.wait(); self.state.update(child_exitcode=code,validator_end_s=time.time())
        result=json.loads((ROOT/'validation/result.json').read_text()); self.state['validator_passed']=result.get('passed') is True
        cross=[]
        for tokens in (8192,1024,2048):
            key='budget_'+str(tokens)
            a=result['instances'].get('33500',{}).get('checks',{}).get(key,{}).get('outputs')
            b=result['instances'].get('33501',{}).get('checks',{}).get(key,{}).get('outputs')
            cross.append(dict(budget=tokens,prompt_lengths=[128,7168],output_tokens_each=64,
                available=a is not None and b is not None,exact_equal=a is not None and a==b))
        self.state['cross_replica_outputs']=cross; self.save()
        require(code==0 and result.get('passed') is True,'unchanged continuous validator failed')
        require(all(x['available'] and x['exact_equal'] for x in cross),'cross-replica output token mismatch')
        self.state['checks_passed']=True

    async def stop_child(self):
        if self.child is None or self.child.returncode is not None: return
        self.state['child_terminated_for_cleanup']=True
        self.child.terminate()
        try: await asyncio.wait_for(self.child.wait(),self.remaining(3))
        except asyncio.TimeoutError:
            self.child.kill(); await asyncio.wait_for(self.child.wait(),self.remaining(2))
        self.state['child_final_exitcode']=self.child.returncode

    def owned_ids(self):
        path=ROOT/'dispatch.jsonl'; result=set()
        if not path.exists(): return result
        content=path.read_bytes()
        lines=content.splitlines(keepends=True)
        for line in lines:
            if not line.endswith(b'\n'):
                # A killed writer may leave the last dispatch line incomplete. The HTTP await
                # follows that write, so no request can have been sent from an incomplete line.
                self.state['dispatch_incomplete_tail']=line.decode(errors='replace')
                continue
            row=json.loads(line)
            if row['route']!='/v1/completions': continue
            port,rid=row['port'],row.get('request_id')
            require(port in PORTS and isinstance(rid,str) and rid.startswith('budget-'),'unexpected dispatch ownership')
            result.add((port,rid))
        return result

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

    async def cleanup(self):
        started=time.monotonic(); self.deadline=started+80
        self.state['cleanup_start_s']=time.time()
        try:
            await self.stop_child()
            if self.verified:
                try: ids=self.owned_ids()
                except BaseException as exc:
                    ids=set(); self.state['dispatch_parse_error']=repr(exc)
                self.state['owned_request_ids']=sorted(ids)
                async def cancel(port,rid):
                    try: await self.http(port,'/cancel',dict(request_id=rid),label='outer-owned-cancel',timeout=8)
                    except BaseException as exc: self.state.setdefault('cancel_errors',[]).append(dict(port=port,id=rid,error=repr(exc)))
                await asyncio.gather(*(cancel(p,r) for p,r in ids))
                outcomes=await asyncio.gather(*(self.restore_one(p) for p in PORTS),return_exceptions=True)
                errors=[repr(x) for x in outcomes if isinstance(x,BaseException)]
                if errors: self.state['drain_errors']=errors
                self.state['cleanup_complete']=not errors and not self.state.get('dispatch_parse_error')
                if not errors:
                    self.state['identity_after']=await asyncio.wait_for(self.identity('after'),self.remaining(12))
                    for a,b in zip(self.state['identity_before'],self.state['identity_after']):
                        require(a['container']['Id']==b['container']['Id'] and a['container']['State']['StartedAt']==b['container']['State']['StartedAt']
                            and a['provenance']==b['provenance'],'resident engine changed')
        except BaseException as exc: self.state.update(cleanup_complete=False,cleanup_error=repr(exc))
        finally:
            self.deadline=started+90 # Reserve the final ten seconds for releasing every owned clock.
            try:
                if self.clocks is not None: await asyncio.wait_for(self.clocks.close(),self.remaining(10))
            except BaseException as exc: self.state.update(cleanup_complete=False,clock_cleanup_error=repr(exc))
            self.state['cleanup_elapsed_s']=time.monotonic()-started
            self.state['cleanup_end_s']=time.time()
            self.deadline=None

    def capture_events(self):
        for name,offset in self.event_offsets.items():
            try:
                with (OLD/'runtime'/(name+'.control.events.jsonl')).open('rb') as f:
                    f.seek(offset); raw=f.read()
                (ROOT/(name+'.events.jsonl')).write_bytes(raw)
            except BaseException as exc:
                self.state.setdefault('event_capture_errors',[]).append(dict(instance_id=name,error=repr(exc)))

    async def run(self):
        hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
        sampler=PowerSampler(range(8),interval=.02,backend=hardware,sample_clocks=True); sampler.start()
        start=None; failure=None
        async with aiohttp.ClientSession(trust_env=False) as self.session:
            try:
                until=time.monotonic()+5
                while True:
                    rows=list(sampler.samples); meta=list(sampler.power_metadata[:len(rows)])
                    require(not sampler.error and time.monotonic()<until,'eight-GPU instant sampling failed')
                    if len(rows)>=2 and power_evidence(rows,sampler.power_source,meta)['power_source_verified']: break
                    await asyncio.sleep(.02)
                start=self.state['measurement_start_s']=time.time()
                await asyncio.wait_for(self.work(hardware),390)
            except BaseException as exc: failure=exc; self.state['error']=repr(exc)
            finally:
                await self.cleanup()
                end=self.state['measurement_end_s']=time.time()
                await asyncio.sleep(.15)
                self.capture_events() # Secondary event-file I/O must never skip full failure power evidence.
                await asyncio.to_thread(sampler.stop)
                dest=ROOT/'power'; dest.mkdir()
                save_raw(dest,[],sampler.samples,sampler.utilization_samples,power_source=sampler.power_source,power_metadata=sampler.power_metadata)
                with (dest/'clocks.csv').open('w',newline='') as f:
                    w=csv.writer(f); w.writerow(['t_s']+[f'gpu{i}_sm_mhz' for i in range(8)])
                    w.writerows([t]+list(v) for t,v in sampler.frequency_samples)
                evidence=power_evidence(sampler.samples,sampler.power_source,sampler.power_metadata)
                self.state.update(power_evidence=evidence,sampling_error=sampler.error)
                try: self.state['total_node_energy_j']=trapezoid_energy(clip_power_window(sampler.samples,start,end,pad_s=0)) if start else None
                except BaseException as exc: self.state['integration_error']=repr(exc)
                self.state['measurement_valid']=bool(start and evidence['power_source_verified'] and not sampler.error
                    and self.state.get('cleanup_complete') and not self.state.get('integration_error'))
                self.state['evidence_complete']=not self.state.get('event_capture_errors')
                self.state.update(complete=True,passed=bool(self.state.get('checks_passed') and self.state['measurement_valid']
                    and self.state['evidence_complete']),finished_s=time.time())
                self.save(); self.log.close()
                if self.child_log: self.child_log.close()
        if isinstance(failure,(asyncio.CancelledError,KeyboardInterrupt,SystemExit)): raise failure


async def main():
    task=asyncio.current_task(); interrupted=False
    def cancel():
        nonlocal interrupted
        if not interrupted: interrupted=True; task.cancel()
    for sig in (signal.SIGINT,signal.SIGTERM): asyncio.get_running_loop().add_signal_handler(sig,cancel)
    await Gate().run()


if __name__=='__main__':
    with node_lease(): asyncio.run(main())
