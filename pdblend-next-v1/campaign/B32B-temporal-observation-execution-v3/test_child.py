"""Bounded CPU tests run the frozen HTTP/control/native helper against a fake transport.

No networking or engine/model imports. These are control-flow proofs, not GPU outputs.
"""
import asyncio,copy,importlib.util,json,time
from pathlib import Path
import pytest

P=Path(__file__).with_name('child.py')
s=importlib.util.spec_from_file_location('temporal_execution_child_tested',P);child=importlib.util.module_from_spec(s);s.loader.exec_module(child)

def job(tmp_path,seconds=5):
    spec=child.read(child.SPEC);spec['output_dir']=str(child.ROOT.parent/'future-cpu-capture-not-created/capture')
    sp=tmp_path/'spec.json';child.write(sp,spec)
    binding=dict(model='32b',instances=[dict(id='diag-only',url='http://unused.invalid:1',tp=2,native_kind='legacy_sync_put')])
    bp=tmp_path/'binding.json';child.write(bp,binding)
    d=dict(binding=binding,binding_path=str(bp),binding_sha256=child.sha(bp),observation_spec=str(sp),observation_spec_sha256=child.sha(sp),
        output_dir=str(tmp_path/'child'),work_deadline_s=time.time()+seconds,cleanup_deadline_s=time.time()+seconds+2)
    jp=tmp_path/'job.json';child.write(jp,d);return jp,d,spec

class Response:
    status=200
    def __init__(self,session,path,payload,rid):self.session=session;self.path=path;self.payload=payload;self.rid=rid
    async def __aenter__(self):self.result=await self.session.handle(self.path,self.payload,self.rid);return self
    async def __aexit__(self,*args):pass
    async def text(self):return json.dumps(self.result)

class Session:
    def __init__(self,spec,*,difference=False,bad_hold=False,bad_ack=False,slow_drain=0):
        self.spec=spec;self.requests={x['request_uuid']:x for x in spec['requests']};self.sent=[];self.active={};self.pending={}
        self.generation=1;self.role='mixed';self.mode='continuous';self.prefill=True;self.decode=True;self.accepting=True
        self.difference=difference;self.bad_hold=bad_hold;self.bad_ack=bad_ack;self.slow_drain=slow_drain;self.closed=False
    async def __aenter__(self):return self
    async def __aexit__(self,*args):self.closed=True
    def request(self,method,url,json=None,headers=None,timeout=None):
        path='/'+url.split('/',3)[3];rid=(headers or {}).get('X-Request-Id');self.sent.append(dict(path=path,body=copy.deepcopy(json),rid=rid,at_s=time.time()))
        return Response(self,path,json,rid)
    def runtime(self):
        waiting=len(self.pending)
        if self.bad_hold and not self.prefill:waiting=0
        return dict(id='diag-only',generation=self.generation,acknowledged_generation=self.generation-1 if self.bad_ack and self.generation>1 else self.generation,
            timestamp=time.time(),transfer_observed_s=time.time(),transport_healthy=True,runtime_error=None,error=None,
            role=self.role,mode=self.mode,admit_prefill=self.prefill,admit_decode=self.decode,accepting=self.accepting,
            active=len(self.active)+len(self.pending),running=len(self.active),waiting=waiting,kv_allocations={rid:len(req['body']['prompt']) for rid,req in self.active.items()},
            transfer_allocations={},transfer_buffered_tensors=0,transfer_inflight_receives=0)
    @staticmethod
    def ranks():return [dict(buffered_tensors=0,inflight_receives=0,buffered_gpu_bytes=0,allocations={},listener_alive=True) for _ in range(2)]
    async def handle(self,path,payload,rid):
        await asyncio.sleep(0)
        if path=='/runtime':return self.runtime()
        if path=='/control':
            assert payload['generation']==self.generation+1 and 'scheduler_budget' not in payload
            self.generation=payload['generation'];self.role=payload['role'];self.mode=payload['mode'];self.prefill=payload['admit_prefill'];self.decode=payload['admit_decode'];self.accepting=True
            return copy.deepcopy(payload)
        if path=='/cancel':
            self.active.pop(payload['request_id'],None);self.pending.pop(payload['request_id'],None);return dict(transfers=self.ranks())
        if path=='/drain':
            await asyncio.sleep(self.slow_drain);assert not self.active and not self.pending
            self.generation+=1;self.accepting=False;self.prefill=False
            return dict(drained=True,accepting=False,generation=self.generation,drain_proof_type='synchronous_put_owner_barrier',transfers=self.ranks())
        assert path=='/v1/completions'
        req=self.requests[rid];assert payload==req['body']
        self.pending[rid]=req
        try:
            while not self.prefill:await asyncio.sleep(.001)
            self.pending.pop(rid,None);self.active[rid]=req
            await asyncio.sleep(.21 if req['label'].endswith('-first') and not req['label'].startswith('golden') else .004)
            values=list(range(64))
            if self.difference and req['label']=='temporal-second':values[31]=999
            return dict(token_ids=values,usage=dict(prompt_tokens=req['prompt_length'],completion_tokens=64))
        finally:self.active.pop(rid,None);self.pending.pop(rid,None)

def execute(jp,session):return asyncio.run(child.execute(jp,session_factory=lambda:session))

def test_all_six_prescribed_requests_and_real_helper_native_cleanup(tmp_path):
    jp,j,s=job(tmp_path);session=Session(s);r=execute(jp,session)
    assert r['complete'] and r['completed_requests']==6 and r['exact_passed'] and r['cleanup_complete'] and r['child_exit_ok'] and session.closed
    rows=[x for x in session.sent if x['path']=='/v1/completions']
    assert [x['rid'] for x in rows]==[x['request_uuid'] for x in s['requests']]
    assert [x['body'] for x in rows]==[x['body'] for x in s['requests']]
    outputs=child.read(Path(j['output_dir'])/'full-outputs.json');assert outputs['all_six_full64'] and len(outputs['token_ids_by_request_uuid'])==6
    own=[json.loads(x) for x in (Path(j['output_dir'])/'checks/owned.jsonl').read_text().splitlines()]
    assert [x['request_id'] for x in own]==[x['request_uuid'] for x in s['requests']]
    proof=child.read(Path(j['output_dir'])/'checks/checks.json')['cleanup']['instances']['diag-only']['proof'];assert len(proof['transfers'])==2
    assert r['phases']['temporal']['held']['waiting']==1 and r['phases']['continuous']['first_still_pending_at_second_dispatch']
    assert session.mode=='continuous' and session.prefill and session.decode and session.accepting and not session.active

def test_temporal_exact_failure_retained_then_continuous_full_control(tmp_path):
    jp,j,s=job(tmp_path);r=execute(jp,Session(s,difference=True))
    assert r['complete'] and r['completed_requests']==6 and not r['exact_passed'] and r['cleanup_complete'] and r['child_exit_ok']
    assert r['phases']['temporal']['first_differences'][1]==dict(position_one_based=32,reference=31,observed=999)
    assert r['phases']['continuous']['exact_passed']
    assert child.read(Path(j['output_dir'])/'full-outputs.json')['token_ids_by_request_uuid'][s['requests'][3]['request_uuid']][31]==999

@pytest.mark.parametrize('kwargs',[dict(bad_hold=True),dict(bad_ack=True)])
def test_control_or_hold_failure_stops_dependent_phase_and_preserves_partial_raw(tmp_path,kwargs):
    jp,j,s=job(tmp_path);session=Session(s,**kwargs);r=execute(jp,session)
    assert not r['complete'] and not r['child_exit_ok'] and r['errors'] and r['finished_s'] is not None
    assert not any(x['rid']==s['requests'][4]['request_uuid'] for x in session.sent)
    assert (Path(j['output_dir'])/'checks/http.jsonl').read_text()
    if kwargs.get('bad_hold'):assert r['cleanup_complete'] and r['completed_requests']==2

def test_work_timeout_cancels_owned_tasks_without_revoking_new_cleanup(tmp_path):
    jp,j,s=job(tmp_path,seconds=.06);session=Session(s);r=execute(jp,session)
    assert not r['complete'] and r['completed_requests']==2 and r['cleanup_complete'] and not r['child_exit_ok']
    assert any(x['path']=='/cancel' for x in session.sent) and not session.active and not session.pending
    assert all(x['rid'] in {z['request_uuid'] for z in s['requests']} for x in session.sent if x['rid'])

def test_external_cancel_during_work_still_performs_owned_cleanup(tmp_path):
    async def run():
        jp,j,s=job(tmp_path);session=Session(s);task=asyncio.create_task(child.execute(jp,session_factory=lambda:session));await asyncio.sleep(.065);task.cancel();r=await task
        assert r['cancelled'] and r['cleanup_complete'] and not r['complete'] and r['finished_s'] is not None
    asyncio.run(run())

def test_cancelled_work_http_revokes_captured_phase_only_and_retains_http_record(tmp_path):
    async def run():
        _,j,s=job(tmp_path);base=child.load_checks();session=Session(s);c=child.deadline_checks(base)(session,j['binding'],tmp_path/'direct-checks')
        c.set_phase(time.time()+2);old=c.limit;req=s['requests'][2]
        t=asyncio.create_task(c.http(c.instances[0],'/v1/completions',req['body'],req['request_uuid']));await asyncio.sleep(.01)
        c.set_phase(time.time()+2);new=c.limit;t.cancel()
        with pytest.raises(asyncio.CancelledError):await t
        assert old.revoked and not new.revoked
        await c.runtime(c.instances[0]);c.close()
        assert 'CancelledError' in (tmp_path/'direct-checks/http.jsonl').read_text()
    asyncio.run(run())

def test_cleanup_deadline_blocks_frozen_finally_resume_after_timeout(tmp_path):
    async def run():
        _,j,s=job(tmp_path);base=child.load_checks();session=Session(s,slow_drain=.3);c=child.deadline_checks(base)(session,j['binding'],tmp_path/'deadline-checks')
        deadline=time.time()+.025;c.set_phase(deadline);start=time.monotonic()
        try:result=await asyncio.wait_for(c.cleanup(),c.limit.remaining())
        except asyncio.TimeoutError:result=False
        assert not result and time.monotonic()-start<.1
        assert not [x for x in session.sent if x['at_s']>deadline]
        assert [x['path'] for x in session.sent][-1]=='/drain'
        c.close();assert 'CancelledError' in (tmp_path/'deadline-checks/http.jsonl').read_text()
    asyncio.run(run())

@pytest.mark.parametrize('field',['binding','spec','deadline'])
def test_bad_job_never_constructs_http_session_and_keeps_terminal_status(tmp_path,field):
    jp,j,s=job(tmp_path)
    if field=='binding':j['binding']['instances'][0]['tp']=1
    if field=='spec':v=child.read(j['observation_spec']);v['requests'][0]['body']['max_tokens']=1;child.write(j['observation_spec'],v);j['observation_spec_sha256']=child.sha(j['observation_spec'])
    if field=='deadline':j['work_deadline_s']=time.time()-1
    child.write(jp,j)
    def forbidden():raise AssertionError('session must not be constructed')
    r=asyncio.run(child.execute(jp,session_factory=forbidden));assert not r['complete'] and not r['cleanup_complete'] and r['completed_requests']==0 and r['phase']=='terminal' and r['errors']

def test_output_directory_cannot_overwrite_existing_attempt(tmp_path):
    jp,j,s=job(tmp_path);Path(j['output_dir']).mkdir();(Path(j['output_dir'])/'keep').write_text('retained')
    with pytest.raises(RuntimeError,match='new child'):execute(jp,Session(s))
    assert (Path(j['output_dir'])/'keep').read_text()=='retained'
