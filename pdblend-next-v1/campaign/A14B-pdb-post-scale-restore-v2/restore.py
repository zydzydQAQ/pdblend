"""A-only retained PDB restart after actual model main150 and scale90.

Default check/prepare are CPU-only. No serving requests or scale runner exist here.
"""
from __future__ import annotations
import argparse
import asyncio
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time

ROOT=Path(__file__).resolve().parent
CAMPAIGN=ROOT.parent
DEPLOY=CAMPAIGN/'AC-baseline-deployment-v2/deploy.py'
COMMON=CAMPAIGN/'five-system-execution-v3/run.py'
BARRIER=CAMPAIGN/'A14B-resident-restore-v3/release_gate.py'
DEADLINE=1788872770.0400891
HOST='iZwz92bdfqihqp38tekqjyZ'
PROTOCOL='per-dataset-slo-five-system-fixed-window-v1'
import contracts

def require(ok,why):
    if not ok: raise RuntimeError(why)
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def load(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def verify_files(files):
    for p,h in files.items():require(sha(p)==h,'frozen file changed: '+str(p))
def package_check():
    m=read(ROOT/'manifest.json');verify_files({str(ROOT/p):h for p,h in m['files'].items()});verify_files(m['dependencies'])
def valid_sha(s):return isinstance(s,str) and len(s)==64 and all(x in '0123456789abcdef' for x in s)
def normalized_mounts(rows):return sorted(json.dumps(r,sort_keys=True) for r in rows)

def static_container(actual,original):
    for k in ('Id','Image','Name','Path','Args','Config','HostConfig'):
        require(actual.get(k)==original.get(k),'retained container identity/settings changed: '+k)
    require(normalized_mounts(actual['Mounts'])==normalized_mounts(original['Mounts']),'retained mounts changed')
    return True

def validate_previous(previous):
    require(previous==contracts.read(contracts.PREVIOUS),'exact current resident baseline binding required')
    require([(i['tp'],i['gpus']) for i in previous['instances']]==[(1,[j]) for j in range(8)],'eight current resident TP1s required')
    require(all(i['native_kind']=='legacy_sync_put' for i in previous['instances']),'actual legacy baseline native scope')
    return True

def plan(out):
    return contracts.build(out)

def prepare(a):
    package_check();require(not a.out.exists(),'new restore directory required')
    spec=plan(a.out)
    spec['files'].update(read(ROOT/'manifest.json')['dependencies'])
    spec['files'].update({str(ROOT/p):h for p,h in read(ROOT/'manifest.json')['files'].items()})
    spec['files'][str(ROOT/'manifest.json')]=sha(ROOT/'manifest.json')
    verify_files(spec['files']);write(a.out/'deployment.json',spec)
    return dict(spec=str(a.out/'deployment.json'),sha256=sha(a.out/'deployment.json'),hardware_actions=False,
        runnable_only_after_actual_main150_scale90_and_fresh_lease=True,correctness_verified=False)

def validate_spec(spec):
    expected=plan(spec['out'])
    for k,v in expected.items():
        if k!='files':require(spec.get(k)==v,'restore declaration changed: '+k)
    require(all(spec['files'].get(p)==h for p,h in expected['files'].items()),'original source freeze missing')
    for name,h in read(ROOT/'manifest.json')['files'].items():
        require(spec['files'].get(str(ROOT/name))==h,'restore implementation not frozen')
    return True

class Limit:
    def __init__(self,end,guard=None):
        require(type(end) in (int,float) and math.isfinite(end),'finite absolute deadline required')
        self.end=float(end);self.mono=time.monotonic()+max(0.,self.end-time.time());self.guard=guard
    def left(self):
        if self.guard is not None:self.guard()
        seconds=min(self.end-time.time(),self.mono-time.monotonic())
        if seconds<=0:raise asyncio.TimeoutError('restore deadline exhausted; no new operation')
        return seconds
    async def sleep(self,seconds):await asyncio.sleep(min(seconds,self.left()));self.left()

async def command(argv,limit,records,cap=40):
    limit.left();r=dict(argv=argv,started_s=time.time());records.append(r);process=None
    try:
        process=await asyncio.create_subprocess_exec(*argv,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
        r['pid']=process.pid
        raw,_=await asyncio.wait_for(process.communicate(),min(cap,limit.left()))
        r.update(returncode=process.returncode,output=raw.decode(errors='replace'))
        require(process.returncode==0,'command failed: '+r['output'][-1000:]);limit.left();return r['output']
    except BaseException as e:r['error']=repr(e);raise
    finally:
        if process is not None and process.returncode is None:process.kill();await process.wait()
        r['finished_s']=time.time()

async def http(session,i,route,body,limit,records):
    import aiohttp
    limit.left();r=dict(id=i['id'],url=i['url'],route=route,request=body,started_s=time.time());records.append(r)
    try:
        async with session.request('GET' if body is None else 'POST',i['url']+route,json=body,
                timeout=aiohttp.ClientTimeout(total=min(10,limit.left()))) as response:
            text=await response.text();r['status']=response.status
            try:r['body']=json.loads(text)
            except ValueError:r['body']=text
            require(response.status==200,'native HTTP failed: '+text[:500]);limit.left();return r['body']
    except BaseException as e:r['error']=repr(e);raise
    finally:r['finished_s']=time.time()

async def idle(session,i,limit,records,common):
    while True:
        raw=await http(session,i,'/runtime',None,limit,records)
        try:common.idle(raw,i)
        except RuntimeError:await limit.sleep(.05)
        else:
            require(raw['generation']==raw.get('acknowledged_generation'),'real owner ACK missing')
            return raw

async def native(session,i,limit,records,common,result):
    """Bounded original native drain/resume; real v3 counters and budget ACK remain required."""
    result.update(complete=False,errors=[])
    try:
        require(i['native_kind'] in ('legacy_sync_put','v3') and i['tp'] in (1,2),'original observed native scope')
        before=await idle(session,i,limit,records,common);result['before']=before
        proof=await http(session,i,'/drain',dict(expected_generation=before['generation']),limit,records)
        result['proof']=proof;common.barrier(before,proof,i)
        paused=await idle(session,i,limit,records,common)
        require(paused['generation']==proof['generation'] and paused.get('accepting') is False,'drain generation changed')
        payload=dict(generation=paused['generation']+1,role='mixed',mode='continuous',admit_prefill=True,admit_decode=True)
        if i['native_kind']=='v3':payload['scheduler_budget']=dict(schema_version=1,max_num_batched_tokens=8192,max_num_seqs=32)
        result['resumed']=dict(before=paused,control=payload)
        reply=await http(session,i,'/control',payload,limit,records)
        after=await idle(session,i,limit,records,common);result['resumed'].update(reply=reply,after=after)
        require(reply.get('generation')==payload['generation']==after['generation'] and after.get('accepting') is True
            and all(after.get(k)==v for k,v in payload.items() if k!='scheduler_budget'),'actual resume not applied')
        if 'scheduler_budget' in payload:
            require(all(after.get('scheduler_budget_effective',{}).get(k)==v for k,v in payload['scheduler_budget'].items() if k!='schema_version'),'resting budget ACK differs')
        result['complete']=True
    except asyncio.CancelledError:result['errors'].append('CancelledError');raise
    except Exception as e:result['errors'].append(repr(e));raise
    finally:result['finished_s']=time.time()

def verify_stopped(spec,inventory):
    by={c['Name'].lstrip('/'):c for c in inventory}
    expected_running={i['container']['name'] for i in read(spec['previous_binding'])['instances']}
    require({n for n,c in by.items() if c['State']['Running']}==expected_running,'unexpected running container')
    for n,original in spec['expected_containers'].items():
        c=by[n];static_container(c,original)
        require(c['State']['Running'] is False and c['State']['Pid']==0 and not c['State'].get('Paused')
            and not c['State'].get('Restarting') and not c['State'].get('Dead'),'resident is not stopped')
    return {n:copy.deepcopy(by[n]) for n in spec['expected_containers']}

def verify_restarted(spec,before,after):
    require(set(after)==set(spec['expected_containers']),'restarted resident set differs')
    for n,old in before.items():
        c=after[n];static_container(c,spec['expected_containers'][n])
        s=c['State'];require(s['Running'] is True and type(s['Pid']) is int and s['Pid']>0
            and s['StartedAt']>old['State']['StartedAt'] and s['StartedAt']!=spec['expected_containers'][n]['State']['StartedAt']
            and not s.get('Restarting') and not s.get('Dead') and not s.get('Paused'),'fresh restarted process not proven')
    require(len({c['State']['Pid'] for c in after.values()})==2,'duplicate resident host PID')

def archive_runtime(spec,out):
    result={}
    for i in spec['instances']:
        runtime=Path(read(i['config'])['runtime_dir'])
        for suffix in ('.control.json','.control.events.jsonl'):
            source=runtime/(i['id']+suffix);raw=source.read_bytes()
            require(suffix!='.control.events.jsonl' or not raw or raw.endswith(b'\n'),'old owner event prefix truncated')
            dest=Path(out)/'old-runtime'/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:f.write(raw)
            require(sha(source)==sha(dest),'stopped runtime changed while archiving')
            result[str(source)]=dict(archive=str(dest),sha256=sha(dest),bytes=len(raw),
                mutation='engine_initialize_replaces_control' if suffix=='.control.json' else 'append_only_owner_events')
    write(Path(out)/'runtime-archive.json',result);return result

def verify_prefixes(records):
    for p,r in records.items():
        require(sha(r['archive'])==r['sha256'],'runtime archive changed')
        if r['mutation']=='append_only_owner_events':
            with Path(p).open('rb') as f:prefix=f.read(r['bytes'])
            require(len(prefix)==r['bytes'] and hashlib.sha256(prefix).hexdigest()==r['sha256'],'old owner prefix altered')

async def ready(session,i,expected,limit,records,common):
    while True:
        try:
            p=await http(session,i,'/provenance',None,limit,records)
            require(all(p.get(k)==v for k,v in expected.items() if k!='pid') and type(p.get('pid')) is int
                and p['pid']>0,'fresh source/model provenance differs')
            r=await http(session,i,'/runtime',None,limit,records)
            if r.get('generation')==0 and r.get('acknowledged_generation')==-1:
                # This is an unacknowledged initial state, never an invented ACK.
                require(r.get('id')==i['id'] and r.get('accepting') is True and r.get('transport_healthy') is True
                    and not r.get('error') and not r.get('runtime_error')
                    and all(k in r and not r[k] for k in ('active','running','waiting','kv_allocations',
                        'transfer_allocations','transfer_buffered_tensors','transfer_inflight_receives')),
                    'initial owner not idle/accepting')
                await http(session,i,'/control',dict(generation=1,role='mixed',mode='continuous',admit_prefill=True,admit_decode=True),limit,records)
            r=await idle(session,i,limit,records,common)
            require(r.get('accepting') is True,'fresh owner not accepting');return p,r
        except (OSError,ConnectionError):await limit.sleep(.2)

def energy_evidence(sampler,start,end,power_evidence):
    require(start<end and len(sampler.samples)>=2,'nonempty measured operation needed')
    rows=sampler.samples;require(rows[0][0]<=start and rows[-1][0]>=end,'all8 power does not bracket full operation')
    require(all(len(v)==8 and all(math.isfinite(x) and x>=0 for x in v) for _,v in rows),'all8 power gap/nonfinite')
    require(all(b[0]>a[0] for a,b in zip(rows,rows[1:])),'power time not increasing')
    clocks=sampler.frequency_samples
    require(clocks and clocks[0][0]<=start and clocks[-1][0]>=end
        and all(len(v)==8 and all(math.isfinite(x) and x>0 for x in v) for _,v in clocks),'actual all8 clocks incomplete')
    evidence=power_evidence(rows,sampler.power_source,sampler.power_metadata)
    require(not sampler.error and evidence['power_source_verified'],'instant power source missing/error')
    common=load(CAMPAIGN/'AC-baseline-binding-v2/gate_evidence.py','resident_restore_energy')
    return dict(energy_j=common.integrate(rows,start,end),power_evidence=evidence,
        max_power_gap_s=max(b[0]-a[0] for a,b in zip(rows,rows[1:])),clock_rows=len(clocks),gpu_count=8)

async def launch(spec_path):
    """Explicit future deployment; caller main holds the real fresh node lease."""
    import aiohttp
    from ecopadg.measure.backends import PynvmlBackend
    from ecopadg.measure.power import PowerSampler
    from ecopadg.serving.measurement import save_raw,power_evidence
    package_check();spec=read(spec_path);validate_spec(spec);verify_files(spec['files'])
    require(spec['hostname']==socket.gethostname()==HOST and spec['operation']=='restart-retained-pdb-after-model-scale','wrong restore host/operation')
    common=load(COMMON,'resident_restore_common');deploy=load(DEPLOY,'resident_restore_deploy');barrier=load(BARRIER,'resident_restore_release')
    release=barrier.verify_release(spec['model_main_release'],spec['model_main_release_sha256'])
    require(barrier.process_scan()['no_live_serving_child'],'serving child alive before recovery')
    previous=read(spec['previous_binding']);validate_previous(previous);common.validate_binding(previous)
    common.validate_binding(read(contracts.PDB))  # Original target weight/large-input stat and source guards.
    proof=contracts.completed_scale()
    require(time.time()+840<DEADLINE,'original deadline cannot accommodate recovery and cleanup')
    out=Path(spec['out']);require(not (out/'deployment-receipt.json').exists(),'restore attempt already exists')
    result=dict(started_s=time.time(),complete=False,measurement_valid=False,created=[],restarted=[],start_intents=[],stopped=[],
        commands=[],http_records=[],errors=[],model_main_release=release,predecessor_proof=[proof],
        new_containers_created=False,created_field_semantics='legacy binder identity alias for restarted existing IDs',
        output_correctness_verified=False,scale_executed=False,clock_controls_issued=False)
    pre=Limit(min(time.time()+120,DEADLINE-840));sampler=None;failure=None
    async with aiohttp.ClientSession(trust_env=False) as session:
        # No GPU mutation, source/control-file writes, or restart before these checks.
        try:
            names=(await command(['docker','ps','-aq'],pre,result['commands'])).split()
            inventory=json.loads(await command(['docker','inspect',*names],pre,result['commands']))
            stopped=verify_stopped(spec,inventory);write(out/'containers.before.json',inventory)
            for i in previous['instances']:
                c=next(c for c in inventory if c['Id']==i['container']['id'])
                require(c['State']['StartedAt']==i['container']['StartedAt'] and c['State']['Running'],'previous live identity changed')
                p=await http(session,i,'/provenance',None,pre,result['http_records'])
                require(all(p.get(k)==v for k,v in i['provenance'].items()),'previous provenance changed')
                await idle(session,i,pre,result['http_records'],common)
            require(barrier.process_scan()['no_live_serving_child'],'serving child appeared during preflight')
        except BaseException as e:
            result.update(error=repr(e),finished_s=time.time(),hardware_actions=False)
            write(out/'deployment-receipt.json',result);raise
        hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
        sampler=PowerSampler(range(8),interval=.02,backend=hardware,sample_clocks=True);sampler.start()
        try:
            await deploy.await_power_ready(sampler,power_evidence)
            require(time.time()+840<DEADLINE,'preflight exhausted recovery reserve')
            result['operation_start_s']=time.time()
            limit=Limit(min(time.time()+720,DEADLINE-120),lambda:require(not sampler.error,'active power sampler failed'))
            result['runtime_archive']=archive_runtime(spec,out)
            result['previous_native_restore']={i['id']:{} for i in previous['instances']}
            for i in previous['instances']:
                await native(session,i,limit,result['http_records'],common,result['previous_native_restore'][i['id']])
            for i in previous['instances']:
                await command(['docker','stop','--time','30',i['container']['id']],limit,result['commands'])
                result['stopped'].append(i['container']['name'])
            while True:
                free=await asyncio.to_thread(deploy.free_gpu_snapshot,hardware);result['gpus_after_stop']=free
                if all(x['used_bytes']==0 and not x['compute_pids'] and not x['graphics_pids'] for x in free['gpus']):break
                await limit.sleep(.2)
            for i in spec['instances']:
                name=i['container_name'];cid=stopped[name]['Id']
                intent=dict(name=name,container_id=cid,issued_s=time.time());result['start_intents'].append(intent)
                write(out/'start-intents'/(i['id']+'.json'),intent)
                await command(['docker','start',cid],limit,result['commands'],cap=30)
                result['restarted'].append(dict(id=i['id'],name=name,container_id=cid))
            started=json.loads(await command(['docker','inspect',*spec['expected_containers']],limit,result['commands']))
            after={c['Name'].lstrip('/'):c for c in started};verify_restarted(spec,stopped,after)
            result['new_provenance']={};result['startup']={};result['new_native_restore']={}
            for i in spec['instances']:
                p,r=await ready(session,i,spec['expected_provenance'][i['id']],limit,result['http_records'],common)
                result['new_provenance'][i['id']]=p;result['startup'][i['id']]=r
                rr=result['new_native_restore'].setdefault(i['id'],{})
                await native(session,i,limit,result['http_records'],common,rr)
            verify_prefixes(result['runtime_archive']);verify_files(spec['files'])
            final=json.loads(await command(['docker','inspect',*spec['expected_containers']],limit,result['commands']))
            final_by={c['Name'].lstrip('/'):c for c in final};verify_restarted(spec,stopped,final_by)
            require(all(final_by[n]['State']['StartedAt']==after[n]['State']['StartedAt'] and final_by[n]['State']['Pid']==after[n]['State']['Pid'] for n in after),'resident changed during recovery')
            require(set((await command(['docker','ps','--format','{{.Names}}'],limit,result['commands'])).split())==set(after),'unexpected running engine after recovery')
            write(out/'containers.after.json',final);result['created']=copy.deepcopy(result['restarted']);result['complete']=True
        except BaseException as e:failure=e;result['errors'].append(repr(e))
        finally:
            if not result['complete']:
                cleanup=Limit(min(time.time()+120,DEADLINE));result['cleanup_deadline_s']=cleanup.end
                async def stop_owned(x):
                    try:await command(['docker','stop','--time','10',x['container_id']],cleanup,result['commands'],cap=20)
                    except asyncio.CancelledError:raise
                    except Exception as e:result['errors'].append('stop retained target: '+repr(e))
                await asyncio.gather(*(stop_owned(x) for x in result['start_intents']))
            result['operation_end_s']=time.time()
            try:await asyncio.sleep(.12);await asyncio.to_thread(sampler.stop)
            except BaseException as e:result['errors'].append('sampler stop: '+repr(e))
            power=out/'deployment-power';power.mkdir(exist_ok=True)
            try:
                save_raw(power,[],sampler.samples,sampler.utilization_samples,power_source=sampler.power_source,power_metadata=sampler.power_metadata)
                with (power/'clocks.csv').open('x') as f:
                    w=csv.writer(f);w.writerow(['t_s']+[f'gpu{j}_sm_mhz' for j in range(8)]);w.writerows((t,*v) for t,v in sampler.frequency_samples)
            except BaseException as e:result['errors'].append('raw save: '+repr(e))
            try:
                measured=energy_evidence(sampler,result['operation_start_s'],result['operation_end_s'],power_evidence)
                result.update(all8_operation_energy_j=measured['energy_j'],power_evidence=measured['power_evidence'],clock_evidence=measured)
                result['measurement_valid']=result['complete'] and not result['errors']
            except Exception as e:result['errors'].append('measurement: '+repr(e));result['all8_operation_energy_j']=None
            result.update(finished_s=time.time(),sampling_error=sampler.error,
                scope='measured retained PDB restart only; fresh continuous output correctness required; PD remains disabled')
            result['artifacts']={str(p):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='deployment-receipt.json'}
            write(out/'deployment-receipt.json',result)
        if failure is not None:raise failure
        require(result['complete'] and result['measurement_valid'],'recovery failed; no correctness or serving successor')
        return result

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('check');q.add_argument('--spec',type=Path)
    q=sub.add_parser('prepare')
    q.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('restore');q.add_argument('--spec',type=Path,required=True);q.add_argument('--run',action='store_true')
    a=p.parse_args();package_check()
    if a.command=='prepare':result=prepare(a)
    elif a.command=='check' or not a.run:
        if a.spec:validate_spec(read(a.spec));verify_files(read(a.spec)['files'])
        result=dict(cpu_check=True,hardware_actions=False,scale_executed=False,actual_recovery_verified=False)
    else:
        require('PDBLEND_NODE_LOCK_FD' not in os.environ,'fresh exclusive lease required')
        spec=read(a.spec);host=Path(spec['host_release']);sys.path[:0]=[str(host/'src'),str(host),'/root/workspace/pdblend/.runtime-deps']
        from ecopadg.serving.campaign import node_lease
        async def supervised():
            task=asyncio.current_task();loop=asyncio.get_running_loop();sent=False
            def cancel():
                nonlocal sent
                if not sent:sent=True;task.cancel()
            for sig in (signal.SIGINT,signal.SIGTERM):loop.add_signal_handler(sig,cancel)
            return await launch(a.spec)
        with node_lease():result=asyncio.run(supervised())
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
