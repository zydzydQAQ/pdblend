"""Bounded isolated-child B batch observations; no serving profile change."""
import argparse
import asyncio
import importlib.util
import sys
import csv
import hashlib
import json
import math
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


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def check():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for path,digest in manifest['files'].items():require(sha(ROOT/path)==digest,'observation source changed: '+path)
    for path,digest in manifest['frozen_references'].items():require(sha(path)==digest,'frozen reference changed: '+path)
    return manifest


def verify_frozen_bytes(b):
    # The prepare receipt already validated every trace contract. Recheck every
    # frozen byte, including those traces, without repeating large JSON decoding
    # while the timing-sensitive instant NVML sampler shares this interpreter.
    freeze=b.read(b.ROOT/'freeze.json')
    require(freeze['package_manifest_sha256']==b.sha(b.ROOT/'package-manifest.json'),'frozen package manifest differs')
    for relative,digest in b.read(b.ROOT/'package-manifest.json')['files'].items():
        require(b.sha(b.ROOT/relative)==digest,'frozen package bytes changed: '+relative)
    for path,digest in freeze['dependencies']['files'].items():
        require(b.sha(path)==digest,'frozen dependency bytes changed: '+path)
    return freeze


def predecessor():
    base=ROOT.parent/'B32B-deadline-24h-v1'
    checkpoints=sorted((base/'checkpoints/probe').glob('*.json'))
    require(len(checkpoints)==2,'both original fixed300 seeds must finish before long coverage observation')
    for path in checkpoints:
        row=json.loads(path.read_text())
        receipt=json.loads((base/'receipts'/(row['cell_id']+'.json')).read_text())
        require(row.get('measurement_valid') is True and receipt.get('screen_valid') is True
            and receipt.get('outer_cleanup',{}).get('complete') is True,'fixed300 predecessor native or measurement invalid')
    return dict(checkpoints={str(path):sha(path) for path in checkpoints},scope='Both fixed300 probes terminal; node lease and actual idle gate checked before any mutation')


def check_long_capacity(evidence):
    runtime=evidence['instances'][0]['runtime']
    require(type(runtime.get('free_kv_tokens')) is int and runtime['free_kv_tokens']>=8*(4096+256),
        'real free KV cannot hold all8 full long request allocations')


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temporary.replace(path)


def check_ack(r, *, startup=False):
    require(not r.get('error') and not r.get('runtime_error') and r.get('transport_healthy') is True, 'unhealthy owner/transport')
    require(r.get('generation') == r.get('acknowledged_generation') and r.get('scheduler_budget_pending') is None, 'pending or false ACK')
    owners = [x.get('controls',{}).get('runtime') for x in r.get('scheduler_io',[])]
    require(len(owners)==1 and all(x and x.get('generation') == r['generation'] and not x.get('error') for x in owners), 'cache/owner generation differs')
    require(r.get('acknowledged_generations')==[r['generation']] and r.get('observed_control_generation')==r['generation'], 'owner generation observation differs')
    require(0 <= time.time()-r.get('timestamp',0) <= 1, 'stale owner snapshot')
    if startup:
        require(r.get('scheduler_budget_effective') == dict(max_num_batched_tokens=8192,max_num_seqs=32), 'startup budget not restored')


def is_idle(r): return all(k in r for k in RESIDUALS) and not any(r[k] for k in RESIDUALS)


def check_drain(before, proof):
    require(proof.get('drained') is True and proof.get('accepting') is False
        and proof.get('generation') == before['generation']+1
        and proof.get('drain_proof_type') == 'synchronous_put_owner_barrier', 'invalid native owner drain')
    observed=proof.get('transfer_observed_s')
    require(type(observed) in (int,float) and math.isfinite(observed) and 0<=time.time()-observed<=1,
        'native transfer proof timestamp missing or stale')
    ranks=proof.get('transfers')
    require(proof.get('send_counters_verified') is True and isinstance(ranks,list) and len(ranks)==2,
        'missing TP2 send proof')
    zero_fields=('buffered_tensors','inflight_receives','inflight_sends','buffered_gpu_bytes')
    counters=('send_started','send_completed','send_failed')
    for rank in ranks:
        require(isinstance(rank,dict) and all(k in rank for k in zero_fields+counters+('allocations',)),
            'required native TP rank observation absent')
        require(rank.get('listener_alive') is True and rank.get('send_counters_observed') is True
            and rank.get('send_healthy') is True,'rank listener or send accounting unhealthy/unobserved')
        require(all(type(rank[k]) is int and rank[k]>=0 for k in counters)
            and rank['send_started']==rank['send_completed'] and rank['send_failed']==0,
            'rank send counters absent/invalid/unsettled')
        require(all(type(rank[k]) is int and rank[k]==0 for k in zero_fields)
            and isinstance(rank['allocations'],dict) and not rank['allocations'],'rank residue or missing zero observation')


class Gate:
    def __init__(self):
        require(not (ROOT/'status.json').exists() and not (ROOT/'outer-http.jsonl').exists(), 'existing evidence retained')
        self.log = (ROOT/'outer-http.jsonl').open('x', buffering=1)
        self.child = None; self.child_log = None; self.clocks = None
        self.verified = False; self.deadline = None; self.event_offsets = {}; self.hardware=None; self.b=None
        self.state = dict(complete=False, passed=False, phase='preflight', started_s=time.time(),
            scope='6 natural batch8 TP2 long4096 observations; no profile publication or KV certification',
            total_runtime_gate='failed', temporal_correctness='unfixed; original strict failure retained',
            work_timeout_s=600, cleanup_timeout_s=90, original_profiler_unchanged=True,
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
        check()
        if self.b is None:
            base=ROOT.parent/'B32B-load-matrix-user-slo-v1';sys.path.insert(0,str(base))
            self.b=load('frozen_b_batch_identity',base/'run.py')
        if label=='before':
            self.state['frozen_identity_gate']=await asyncio.to_thread(verify_frozen_bytes,self.b)
        evidence=await self.b.live(self.session,expected_inventory=json.loads((self.b.ROOT/'freeze.json').read_text())['inventory'])
        write(ROOT/('identity.'+label+'.json'),evidence)
        return evidence

    async def work(self,hardware):
        self.hardware=hardware
        self.state['identity_before']=await self.identity('before')
        check_long_capacity(self.state['identity_before'])
        self.verified=True;self.state['verified_for_controls']=True;self.save()
        for name in IDS:
            path=OLD/'runtime'/(name+'.control.events.jsonl');require(path.is_file(),'owner timeline missing')
            self.event_offsets[name]=path.stat().st_size
        self.child_log=(ROOT/'child.log').open('xb')
        self.state.update(phase='natural_batch_observations',observation_start_s=time.time());self.save()
        self.child=await asyncio.create_subprocess_exec('python3','-u',str(ROOT/'child.py'),cwd=str(ROOT),
            stdin=asyncio.subprocess.DEVNULL,stdout=self.child_log,stderr=asyncio.subprocess.STDOUT,start_new_session=True)
        self.state['child_pid']=self.child.pid;self.save()
        code=await self.child.wait();self.state.update(child_exitcode=code,observation_end_s=time.time())
        report=json.loads((ROOT/'results/campaign.json').read_text());self.state['child_campaign']=report
        require(code==0 and report.get('complete') is True and len(report.get('points',[]))==6,'natural batch observations failed or incomplete')
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
        path=ROOT/'dispatch.jsonl';result=set()
        if not path.exists():return result
        for line in path.read_bytes().splitlines(keepends=True):
            if not line.endswith(b'\n'):
                self.state['dispatch_incomplete_tail']=line.decode(errors='replace');continue
            row=json.loads(line)
            if row['route']!='/v1/completions':continue
            port,rid=row['port'],row.get('request_id')
            require(port==33500 and isinstance(rid,str) and rid.startswith('pdb-profile-'),'unexpected dispatch ownership')
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
        started=time.monotonic();self.deadline=started+80
        self.state['cleanup_start_s']=time.time();child_stopped=False
        try:
            await self.stop_child()
            require(self.child is None or self.child.returncode is not None,'owned child still alive; fail-stop before native/clock actions')
            child_stopped=True;self.state['child_exit_confirmed']=True
            if self.verified:
                try:ids=self.owned_ids()
                except BaseException as exc:ids=set();self.state['dispatch_parse_error']=repr(exc)
                self.state['owned_request_ids']=sorted(ids)
                async def cancel(port,rid):
                    try:await self.http(port,'/cancel',dict(request_id=rid),label='outer-owned-cancel',timeout=8)
                    except BaseException as exc:self.state.setdefault('cancel_errors',[]).append(dict(port=port,id=rid,error=repr(exc)))
                await asyncio.gather(*(cancel(p,r) for p,r in ids))
                outcomes=await asyncio.gather(*(self.restore_one(p) for p in PORTS),return_exceptions=True)
                errors=[repr(x) for x in outcomes if isinstance(x,BaseException)]
                if errors:self.state['drain_errors']=errors
                self.state['cleanup_complete']=not errors and not self.state.get('dispatch_parse_error')
                if not errors:self.state['identity_after']=await asyncio.wait_for(self.identity('after'),self.remaining(12))
            else:self.state['cleanup_complete']=True;self.state['identity_failure_no_control']=True
        except BaseException as exc:self.state.update(cleanup_complete=False,cleanup_error=repr(exc))
        finally:
            self.deadline=started+90
            try:
                if self.verified and child_stopped:
                    # The child owned these locks; only acquire after its exit is proven.
                    self.clocks=ClockOwner(self.hardware,(0,1,2,3))
                    await asyncio.wait_for(self.clocks.close(),self.remaining(10))
                    self.state['clock_release_complete']=True
                elif self.verified:self.state['clock_release_skipped_child_may_be_alive']=True
            except BaseException as exc:self.state.update(cleanup_complete=False,clock_cleanup_error=repr(exc))
            self.state['cleanup_elapsed_s']=time.monotonic()-started;self.state['cleanup_end_s']=time.time();self.deadline=None

    def capture_events(self):
        for name,offset in self.event_offsets.items():
            try:
                with (OLD/'runtime'/(name+'.control.events.jsonl')).open('rb') as f:
                    f.seek(offset); raw=f.read()
                (ROOT/(name+'.events.jsonl')).write_bytes(raw)
                events=[json.loads(line) for line in raw.splitlines() if line]
                require(all(e.get('mode')=='continuous' and e.get('role')=='mixed' and 0<=e.get('tokens',-1)<=8192 for e in events),'temporal/overbudget owner execution observed')
            except BaseException as exc:
                self.state.setdefault('event_capture_errors',[]).append(dict(instance_id=name,error=repr(exc)))

    async def monitored_work(self,hardware,sampler):
        task=asyncio.create_task(self.work(hardware))
        try:
            while True:
                require(not sampler.error,'outer instant sampler failed: '+str(sampler.error))
                done,_=await asyncio.wait((task,),timeout=.02)
                if done:
                    await task
                    require(not sampler.error,'outer instant sampler failed: '+str(sampler.error))
                    return
        finally:
            if not task.done():task.cancel()
            await asyncio.gather(task,return_exceptions=True)

    async def run(self):
        try:
            hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
            sampler=PowerSampler(range(8),interval=.02,backend=hardware,sample_clocks=True);sampler.start()
        except BaseException as exc:
            self.state.update(complete=True,passed=False,measurement_valid=False,error=repr(exc),phase='hardware_preflight_failed')
            self.save();self.log.close();raise
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
                await asyncio.wait_for(self.monitored_work(hardware,sampler),600)
            except BaseException as exc: failure=exc; self.state['error']=repr(exc)
            finally:
                self.state['phase']='cleanup';self.save()
                await self.cleanup()
                self.state['phase']='save_evidence';self.save()
                end=self.state['measurement_end_s']=time.time()
                until=time.monotonic()+3
                while not sampler.samples or sampler.samples[-1][0]<end:
                    if sampler.error or time.monotonic()>=until:
                        self.state['sampling_tail_error']='outer power tail not bracketed';break
                    await asyncio.sleep(.01)
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
                self.state['observed_raw_energy_j']=trapezoid_energy(sampler.samples)
                self.state['measurement_valid']=bool(start and self.verified and not self.state.get('sampling_tail_error') and evidence['power_source_verified'] and not sampler.error
                    and self.state.get('cleanup_complete') and not self.state.get('integration_error'))
                self.state['evidence_complete']=not self.state.get('event_capture_errors')
                for path,digest in json.loads((ROOT/'protected-baselines.json').read_text())['files'].items():
                    try:require(sha(path)==digest,'protected baseline changed: '+path)
                    except BaseException as exc:self.state.setdefault('preservation_errors',[]).append(repr(exc))
                self.state['baseline_preservation_verified']=not self.state.get('preservation_errors')
                self.state['evidence_complete'] &= self.state['baseline_preservation_verified']
                self.state.update(phase='finished',complete=True,passed=bool(self.state.get('checks_passed') and self.state['measurement_valid']
                    and self.state['evidence_complete']),finished_s=time.time())
                self.save(); self.log.close()
                if self.child_log: self.child_log.close()
        if isinstance(failure,(asyncio.CancelledError,KeyboardInterrupt,SystemExit)): raise failure


async def main():
    check();prior=predecessor()
    task=asyncio.current_task();interrupted=False;gate=None
    def cancel():
        nonlocal interrupted
        if not interrupted:
            interrupted=True
            if gate is not None and gate.state.get('phase') in ('cleanup','save_evidence'):
                gate.state['stop_requested_during_cleanup']=True;gate.save()
            else:task.cancel()
    for sig in (signal.SIGINT,signal.SIGTERM): asyncio.get_running_loop().add_signal_handler(sig,cancel)
    gate=Gate();gate.state['predecessor']=prior;gate.save();await gate.run();require(gate.state.get('passed') is True,'batch observation failed; original evidence retained')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    check()
    if args.check:print(json.dumps(dict(package_valid=True,planned_points=6,gpu_executed=False)))
    else:
        with node_lease():asyncio.run(main())
