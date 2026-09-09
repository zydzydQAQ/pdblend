"""HTTP-only four-request diagnostic child. Parent owns identity/lease/GPU/power.

Exact mismatches are observations, not exceptions that discard the control pair.
"""
from __future__ import annotations
import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import time

ROOT=Path(__file__).resolve().parent
CHECKS=ROOT.parent/'B32B-legacy-baseline-correctness-v1/checks.py'
CHECKS_SHA='e0965f922ae42245b275d9c17342c31689290c61bcac5dfaf33a2a352fd29931'
SPEC=ROOT/'request-spec.json'
SPEC_SHA='9b3b218388534462fc1de0be9526c5ed6155bc7971d4fd837fd40d2857d78370'
LABELS=('golden-first','golden-second','temporal-first','temporal-second')

def require(ok,why):
    if not ok:raise RuntimeError(why)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def load_checks():
    require(sha(CHECKS)==CHECKS_SHA,'frozen legacy request/control/native helper changed')
    s=importlib.util.spec_from_file_location('temporal_original_legacy_checks',CHECKS);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def validate_job(job):
    require(sha(SPEC)==SPEC_SHA,'original four-request declaration changed')
    require(Path(job['binding_path']).is_absolute() and sha(job['binding_path'])==job['binding_sha256']
        and read(job['binding_path'])==job['binding'],'job binding does not equal frozen actual file')
    binding=job['binding'];ii=binding['instances']
    require(binding['model']=='32b' and len(ii)==1 and ii[0]['tp']==2 and ii[0]['native_kind']=='legacy_sync_put',
        'single actual B diagnostic TP2 required')
    require(not ii[0].get('scheduler_cache_observed') and 'restore_budget_tokens' not in ii[0],
        'legacy diagnostic cannot use a v3 scheduler namespace')
    require(Path(job['observation_spec']).is_absolute() and sha(job['observation_spec'])==job['observation_spec_sha256'],
        'observation spec SHA differs')
    spec=read(job['observation_spec']);original=read(SPEC);expected=copy.deepcopy(original)
    expected['output_dir']=spec['output_dir'];require(spec==expected,'only output_dir may differ from frozen four-request spec')
    capture=Path(spec['output_dir']);require(capture.is_absolute() and ROOT.parent in capture.parents and '..' not in capture.parts,
        'independent absolute campaign capture path required')
    require([x['label'] for x in spec['requests']]==list(LABELS),'four request order differs')
    require(len({x['request_uuid'] for x in spec['requests']})==4,'four disjoint UUIDs required')
    require(spec['request_ids']==[spec['requests'][1]['request_uuid'],spec['requests'][3]['request_uuid']],
        'only golden-second and temporal-second UUIDs may be observed')
    for k in ('work_deadline_s','cleanup_deadline_s'):
        require(type(job[k]) in (int,float) and math.isfinite(job[k]),'finite actual epoch deadline required')
    require(time.time()<job['work_deadline_s']<job['cleanup_deadline_s']<=spec['global_deadline_s']
        and job['work_deadline_s']-time.time()<=390.1,'original global/work deadline exceeded')
    require(0<job['cleanup_deadline_s']-job['work_deadline_s']<=90,'original cleanup cap exceeded')
    require(Path(job['output_dir']).is_absolute(),'absolute child output required')
    return spec

class PhaseDeadline:
    def __init__(self,end):self.end=end;self.mono=time.monotonic()+max(0.,end-time.time());self.revoked=False
    def remaining(self):
        require(not self.revoked,'phase HTTP permission revoked after cancellation')
        remaining=min(self.end-time.time(),self.mono-time.monotonic())
        if remaining<=0:raise asyncio.TimeoutError('absolute diagnostic phase deadline exhausted')
        return remaining

def deadline_checks(base):
    class Checks(base.Checks):
        """Reuse frozen HTTP/control/native semantics; cap every real send centrally."""
        def set_phase(self,end):self.limit=PhaseDeadline(end)
        async def http(self,i,path,payload=None,rid=None,timeout=45):
            phase=self.limit
            remaining=phase.remaining() # No coroutine/request construction after expiry.
            try:
                return await asyncio.wait_for(super().http(i,path,payload,rid,timeout=min(timeout,remaining)),remaining)
            except asyncio.CancelledError:
                # Cleanup can replace self.limit before cancelling unfinished work.
                # Revoking that work must not revoke the newly owned cleanup phase.
                phase.revoked=True;raise
        def own(self,i,rid):
            super().own(i,rid);self.ownlog.flush();os.fsync(self.ownlog.fileno())
    return Checks

class Observation:
    def __init__(self,job,spec,checker,base,out,status):
        self.job=job;self.spec=spec;self.c=checker;self.base=base;self.out=out;self.status=status
        self.i=job['binding']['instances'][0];self.outputs={};self.status['completed_request_labels']=[]
    def save(self):
        self.status['completed_requests']=len(self.outputs)
        write(self.out/'status.json',self.status)
        write(self.out/'full-outputs.json',dict(token_ids_by_request_uuid=self.outputs,
            completed_request_labels=self.status['completed_request_labels'],all_four_full64=len(self.outputs)==4,
            observation_spec_sha256=self.job['observation_spec_sha256'],binding_sha256=self.job['binding_sha256'],
            scope='independently generated native-default full64; original temporal gate unchanged'))
    async def run(self):
        await self.c.idle(self.i)
        for request in self.spec['requests']:self.c.own(self.i,request['request_uuid'])
        response=await self.c.http(self.i,'/native-reference',dict(requests=self.spec['requests'],
            work_deadline_s=self.job['work_deadline_s']),timeout=self.c.limit.remaining())
        require(response.get('complete') is True,'native driver incomplete')
        values=response['token_ids_by_request_uuid'];ids=[r['request_uuid'] for r in self.spec['requests']]
        require(set(values)==set(ids) and all(len(values[r])==64 and all(type(x) is int and x>=0 for x in values[r]) for r in ids),'four complete actual64 outputs')
        self.outputs=values;self.status['completed_request_labels']=list(LABELS)
        differences=[self.base.difference(values[ids[j]],values[ids[j+2]]) for j in range(2)]
        self.status.update(complete=True,exact_passed=not any(differences),first_differences=differences,
            original_temporal_gate_changed=False,reference_is_independent_numerical_implementation=False)
        self.save()

async def execute(job_path,*,session_factory=None,checker_class=None):
    """Parent invokes one subprocess. No parent measurement or capture proof is asserted."""
    job=read(job_path);out=Path(job['output_dir']);require(not out.exists(),'new child directory required');out.mkdir(parents=True)
    status=dict(schema=1,pid=os.getpid(),started_s=time.time(),complete=False,completed_requests=0,exact_passed=False,cleanup_complete=False,
        finished_s=None,measurement_valid=False,phase='validating',errors=[],capture_complete_not_checked=True,
        job_path=str(Path(job_path).resolve()),job_sha256=sha(job_path),checks_source_sha256=CHECKS_SHA)
    write(out/'status.json',status);checker=None;observation=None;failure=None;signal_cancelled=False
    task=asyncio.current_task();loop=asyncio.get_running_loop()
    def cancel():
        nonlocal signal_cancelled
        if not signal_cancelled:signal_cancelled=True;task.cancel()
    for sig in (signal.SIGTERM,signal.SIGINT):loop.add_signal_handler(sig,cancel)
    try:
        spec=validate_job(job);base=load_checks()
        if session_factory is None:
            import aiohttp
            session_factory=lambda:aiohttp.ClientSession(trust_env=False)
        async with session_factory() as session:
            cls=checker_class or deadline_checks(base);checker=cls(session,job['binding'],out/'checks')
            checker.set_phase(job['work_deadline_s']);observation=Observation(job,spec,checker,base,out,status);observation.save()
            try:
                status['phase']='four-native-reference-requests';observation.save()
                await asyncio.wait_for(observation.run(),checker.limit.remaining())
            except BaseException as e:failure=e;status['errors'].append(repr(e));status['cancelled']=isinstance(e,asyncio.CancelledError)
            finally:
                status['phase']='cleanup';status['cleanup_started_s']=time.time()
                end=min(job['cleanup_deadline_s'],time.time()+90);status['actual_cleanup_deadline_s']=end
                checker.set_phase(end) # Explicit owned cleanup phase; no work retry.
                try:
                    result=await asyncio.wait_for(checker.cleanup(),checker.limit.remaining())
                    status['cleanup_complete']=bool(result and checker.state.get('cleanup',{}).get('complete')
                        and not checker.limit.revoked and time.time()<=end)
                except BaseException as e:
                    status['errors'].append('cleanup: '+repr(e));status['cleanup_complete']=False
                status['cleanup_finished_s']=time.time()
                if time.time()>end:status['cleanup_complete']=False
                checker.state['complete']=status['complete'];checker.state['passed']=bool(status['exact_passed'] and status['cleanup_complete'])
                checker.save();checker.close()
                status['finished_s']=time.time();status['phase']='terminal'
                observation.save()
    except BaseException as e:
        failure=e;status['errors'].append('setup/terminal: '+repr(e));status['finished_s']=time.time();status['phase']='terminal'
        if checker is not None:
            try:checker.close()
            except Exception:pass
        write(out/'status.json',status)
        if not (out/'full-outputs.json').exists():write(out/'full-outputs.json',{'token_ids_by_request_uuid':{},'all_four_full64':False})
    finally:
        for sig in (signal.SIGTERM,signal.SIGINT):loop.remove_signal_handler(sig)
    status['child_exit_ok']=bool(status['complete'] and status['cleanup_complete'] and not status['errors'])
    write(out/'status.json',status)
    return status

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True);a=p.parse_args()
    status=asyncio.run(execute(a.job));print(json.dumps(status,indent=2,allow_nan=False))
    raise SystemExit(0 if status['child_exit_ok'] else 1)

if __name__=='__main__':main()
