"""One isolated B observation. Default checks CPU only; --run is explicit."""
import argparse,asyncio,copy,csv,json,os,shutil,signal,socket,sys,time
from pathlib import Path
import common as c
import archive
import technical


def deadline_limits(now):
    c.require(c.finite(now) and now+900<c.DEADLINE,'insufficient original absolute deadline')
    return dict(started_s=now,startup_end_s=now+150,work_end_s=now+540,
        cleanup_end_s=now+630,restore_end_s=now+870,end_s=now+900)

def assert_free(snapshot):
    rows=snapshot.get('gpus',[])
    c.require([r.get('gpu') for r in rows]==list(range(8)),'all eight physical GPU observations required')
    c.require(all(r.get('memory_api')=='nvmlDeviceGetMemoryInfo_v2' and type(r.get('used_bytes')) is int
        and r['used_bytes']==0 and r.get('compute_pids')==[] and r.get('graphics_pids')==[] for r in rows),
        'physical GPU memory/processes have not exited')

def preserve_container(before,after):
    for key in ('Id','Image','Config','HostConfig','Path','Args'):
        c.require(before.get(key)==after.get(key),'preserved container configuration differs: '+key)
    def mounts(value):
        rows=value['Mounts'];c.require(len({x['Destination'] for x in rows})==len(rows),'duplicate mount destination')
        return sorted(json.dumps(x,sort_keys=True) for x in rows)
    c.require(mounts(before)==mounts(after),'preserved full mount dictionaries differ')
    c.require(after['State']['Running'] is True and after['State']['StartedAt']!=before['State']['StartedAt']
        and after['State']['Pid']!=before['State']['Pid'],'restart must expose new real process provenance')



def sample_window(samples,clocks,start,end):
    c.require(c.finite(start) and c.finite(end) and start<end,'finite positive operation window required')
    for name,rows in (('power',samples),('clock',clocks)):
        c.require(len(rows)>=2 and rows[0][0]<=start and rows[-1][0]>=end,'real '+name+' prefix/tail missing')
        previous=None
        for t,values in rows:
            c.require(c.finite(t) and (previous is None or t>previous) and len(values)==8
                and all(c.finite(v) and v>=0 for v in values),'invalid all8 '+name+' vector/time')
            previous=t

def launch_identity(inspection,instance):
    c.require(inspection['State']['Running'] is True and inspection['Image']==instance['image'],'actual image/process differs')
    c.require(inspection['Config']['Cmd']==['python3',instance['engine_entry'],'--config',instance['config']],'actual engine startup command differs')
    env=dict(x.split('=',1) for x in inspection['Config']['Env'])
    expected=dict(x.split('=',1) for x in instance['environment'])
    c.require(all(env.get(k)==v for k,v in expected.items()),'actual diagnostic environment differs')
    actual={(m['Type'],m['Source'],m['Destination'],m['RW']) for m in inspection['Mounts']}
    want={(m['Type'],m['Source'],m['Destination'],m['RW']) for m in instance['mounts']}
    c.require(actual==want and inspection['HostConfig']['NetworkMode']=='host' and inspection['HostConfig']['IpcMode']=='host','actual engine mount/network differs')


def owner_evidence(spec,out):
    obs=c.read(spec['observation_spec']);cfg=c.read(spec['diagnostic_instance']['config'])
    source=Path(cfg['runtime_dir'])/(cfg['id']+'.control.events.jsonl')
    data=source.read_bytes();c.require(len(data)<=4*1024*1024 and data.endswith(b'\n'),'missing/big/incomplete owner events')
    dest=Path(out)/'diagnostic-owner.events.jsonl';dest.write_bytes(data)
    events=[json.loads(line) for line in data.splitlines()];counts={}
    for request in obs['requests']:
        selected=[e for e in events if request['request_uuid'] in e.get('request_ids',[]) and e.get('tokens',0)>0]
        c.require(len(selected)==64,'actual owner must schedule exactly64 steps for each short full64 request')
        expected='temporal' if request['label'].startswith('temporal-') else 'continuous'
        c.require(all(e.get('mode')==expected and e.get('role')=='mixed' for e in selected),'actual owner mode/role differs')
        if expected=='temporal':c.require(not any(e.get('prefill') and e.get('decode') for e in selected),'temporal real owner overlapped prefill/decode')
        counts[request['request_uuid']]=dict(label=request['label'],steps=len(selected),mode=expected)
    return dict(complete=True,file=str(dest),sha256=c.sha(dest),requests=counts)

class Operation:
    def __init__(self,spec):
        self.spec=spec;self.out=Path(spec['results']);self.stop=False;self.deadline=None;self.restoring=False
        self.anchor_wall=time.time();self.anchor_mono=time.monotonic()
        self.state=dict(schema=1,complete=False,hardware_executed=False,performance_evidence=False,
            original_temporal_failure_preserved=True,automatic_retries=False,phase='identity',
            old_performance_binding_reusable=False,fresh_correctness_and_binding_required=True,errors=[],commands=[])
        self.deploy=c.module('deployment');self.engine=c.module('executor');self.Checks=c.module('checks')
        self.engine.command=self.command # Same identity logic, bounded kill/reap subprocess implementation.
        self.child=None;self.child_log=None;self.diag_intent=False;self.old_stop_intent=False
        self.identity_verified=False;self.controls_owned=False;self.diag_binding=None;self.clock=None;self.sampler=None;self.hardware=None
        self.old_identity=None;self.original=None;self.terminal={};self.limits=None
    def save(self):c.write(self.out/'status.json',self.state)
    def phase(self,name,when=None):self.state['phase']=name;self.state.setdefault('stages',[]).append(dict(phase=name,started_s=time.time() if when is None else when));self.save()
    def remaining(self,cap=30):
        c.require(self.deadline is not None and self.before_deadline(),'operation stage deadline exceeded')
        if not self.restoring:c.require(not self.stop,'diagnostic STOP requested')
        return max(.001,min(cap,self.deadline-time.time(),self.anchor_mono+self.deadline-self.anchor_wall-time.monotonic()))
    def before_deadline(self):
        return time.time()<self.deadline and time.monotonic()<self.anchor_mono+self.deadline-self.anchor_wall
    async def command(self,*argv,timeout=30):
        row=dict(argv=list(argv),started_s=time.time());self.state['commands'].append(row);self.save()
        try:
            result=await self.deploy.command(*argv,timeout=self.remaining(timeout));row['stdout']=result;return result
        except BaseException as exc:row['error']=repr(exc);raise
        finally:row['finished_s']=time.time();self.save()
    async def inspected(self,names):return json.loads(await self.command('docker','inspect',*names,timeout=12))
    async def sources(self,name,expected):
        # Stdlib reads only: no import vllm/torch and no CUDA context.
        script='import hashlib,json,sys; print(json.dumps({p:hashlib.sha256(open(p,"rb").read()).hexdigest() for p in json.loads(sys.argv[1])}))'
        actual=json.loads(await self.command('docker','exec',name,'python3','-c',script,json.dumps(list(expected)),timeout=12))
        c.require(actual==expected,'actual installed source changed: '+name);return actual
    async def gpu_free(self,label,cap=20):
        until=min(self.deadline,time.time()+cap);mono_until=time.monotonic()+cap;latest=None
        while time.time()<until and time.monotonic()<mono_until and self.before_deadline():
            latest=await asyncio.to_thread(self.deploy.free_gpu_snapshot,self.hardware)
            try:assert_free(latest);c.write(self.out/(label+'.json'),latest);return latest
            except RuntimeError:await asyncio.sleep(.1)
        c.write(self.out/(label+'.json'),latest);raise RuntimeError('GPU processes/memory not released: '+label)
    async def native(self,session,binding,label,owned=()):
        owner=self
        class BoundedChecks(self.Checks.Checks):
            async def http(self,i,path,payload=None,rid=None,timeout=45):
                return await super().http(i,path,payload,rid,timeout=min(timeout,owner.remaining(timeout)))
        check=BoundedChecks(session,binding,self.out/label)
        try:
            for iid,rid in owned:check.owned.add((iid,rid))
            complete=await check.cleanup();c.require(complete,'native proof/resume failed: '+label)
            return check.state['cleanup']
        finally:check.close()
    async def source_identity(self,session,i,inspection,diagnostic=False):
        actual=await self.engine.http(session,i,'/provenance',timeout=self.remaining(3))
        cfg=c.read(i.get('config',i.get('engine_config')))
        expected=dict(instance_id=i['id'],tp=2,model=c.MODEL,dtype='bfloat16',max_model_len=8192,
            cuda_visible_devices=','.join(map(str,i['gpus'])),source_files_at_import=
            {str(p):c.sha(p) for p in Path(self.spec['diagnostic_instance']['engine_entry']).parent.glob('*.py')})
        c.require(all(actual.get(k)==v for k,v in expected.items()),'new process actual engine provenance differs')
        c.require(all(cfg.get(k)==v for k,v in dict(tp=2,max_num_batched_tokens=8192,max_num_seqs=32,max_model_len=8192,model=c.MODEL).items()),'actual startup work limit differs')
        sources=await self.sources(inspection['Name'].lstrip('/'),self.spec['installed_diagnostic_sources' if diagnostic else 'installed_parent_sources'])
        return dict(container=inspection,provenance=actual,installed_sources=sources)
    async def ready(self,session,i,diagnostic=False):
        until=self.deadline;last=None;verified=None
        while time.time()<until and self.before_deadline():
            self.remaining()
            try:
                raw=await self.engine.http(session,i,'/runtime',timeout=self.remaining(2))
                c.require(raw.get('id')==i['id'] and not raw.get('error') and not raw.get('runtime_error'),'unhealthy restored/new engine')
                if verified is None:
                    inspection=(await self.inspected([i['container_name']]))[0]
                    launch_identity(inspection,i)
                    verified=await self.source_identity(session,i,inspection,diagnostic)
                if raw.get('generation')==0 and raw.get('acknowledged_generation')==-1:
                    c.require(self.Checks.is_idle(raw) and raw.get('transport_healthy') is True,'startup not genuinely idle')
                    await self.engine.http(session,i,'/control',dict(generation=1,role='mixed',mode='continuous',admit_prefill=True,admit_decode=True),timeout=self.remaining(8))
                    continue
                self.Checks.check_ack(raw,i);c.require(self.Checks.is_idle(raw) and raw.get('accepting') is True,'new owner not idle/accepting')
                verified['runtime']=raw;return verified
            except Exception as exc:last=repr(exc);await asyncio.sleep(.15)
        raise TimeoutError('ready failed: '+str(last))
    async def create_diagnostic(self,session):
        self.phase('build_diagnostic_image')
        i=copy.deepcopy(self.spec['diagnostic_instance']);context=Path(self.spec['image_context'])
        # The exact local parent tag avoids a registry lookup for a bare sha FROM.
        await self.command('docker','tag',c.IMAGE,self.spec['parent_tag'],timeout=8)
        parent=json.loads(await self.command('docker','image','inspect',self.spec['parent_tag'],timeout=8))[0]
        c.require(parent['Id']==c.IMAGE,'local build parent tag differs')
        await self.command('docker','build','--network','none','--pull=false','--iidfile',str(self.out/'image.iid'),str(context),timeout=60)
        i['image']=(self.out/'image.iid').read_text().strip();c.require(i['image'].startswith('sha256:') and i['image']!=c.IMAGE,'diagnostic derived image missing')
        self.state['diagnostic_image']=i['image'];self.diag_intent=True;self.state['creation_intent']=i;self.save()
        self.phase('start_diagnostic_container');await self.command(*self.deploy.docker_start_arguments(i),timeout=25)
        identity=await self.ready(session,i,True);self.state['diagnostic_identity']=identity;c.write(self.out/'diagnostic-identity.before.json',identity)
        ins=identity['container'];bound=dict(id=i['id'],tp=2,gpus=i['gpus'],role='mixed',url=i['url'],port=i['port'],kv_port=i['kv_port'],
            native_kind='legacy_sync_put',scheduler_cache_observed=False,engine_config=i['config'],
            container=dict(name=i['container_name'],id=ins['Id'],image=ins['Image'],StartedAt=ins['State']['StartedAt']),provenance=identity['provenance'])
        self.diag_binding=dict(schema=1,model='32b',hostname=c.NODE,deadline_s=c.DEADLINE,instances=[bound],configs={},
            host_release=str(c.HOST),output=str(self.out/'child'),protocol_id=c.PROTOCOL,system='diagnostic',performance_evidence=False)
        c.write(self.out/'diagnostic-binding.json',self.diag_binding)
        c.write(self.out/'diagnostic-full-identity.json',await self.engine.identity(session,self.diag_binding))
        live=await asyncio.to_thread(self.deploy.free_gpu_snapshot,self.hardware)
        for gpu in live['gpus']:
            if gpu['gpu'] in i['gpus']:c.require(gpu['used_bytes']>0 and gpu['compute_pids'] and not gpu['graphics_pids'],'diagnostic TP rank not physically resident')
            else:c.require(gpu['used_bytes']==0 and not gpu['compute_pids'] and not gpu['graphics_pids'],'foreign GPU process appeared during isolated diagnostic')
        c.write(self.out/'diagnostic-gpu-processes.json',live)
        (self.out/'diagnostic-docker-top.txt').write_text(await self.command('docker','top',i['container_name'],'-eo','pid,ppid,args',timeout=8))
        return i
    async def stop_child(self):
        if self.child is None:self.terminal['http_child_exited']=True;return
        if self.child.returncode is None:
            self.child.send_signal(signal.SIGINT)
            try:await asyncio.wait_for(self.child.wait(),self.remaining(75))
            except asyncio.TimeoutError:
                self.child.kill();await asyncio.wait_for(self.child.wait(),self.remaining(5))
        c.require(self.child.returncode is not None,'owned child exit is unconfirmed')
        self.terminal['http_child_exited']=True;self.state['child_exitcode']=self.child.returncode
    async def run_child(self):
        self.phase('six_http_requests')
        issued_s=time.time()
        child_work_end=min(self.limits['work_end_s'],issued_s+390)
        child_cleanup_end=min(self.limits['cleanup_end_s'],child_work_end+90)
        c.require(issued_s<child_work_end<child_cleanup_end,'no bounded child work/cleanup interval remains')
        self.deadline=child_work_end
        self.state['child_limits']=dict(issued_s=issued_s,work_end_s=child_work_end,cleanup_end_s=child_cleanup_end)
        job=dict(binding=self.diag_binding,binding_path=str(self.out/'diagnostic-binding.json'),binding_sha256=c.sha(self.out/'diagnostic-binding.json'),
            observation_spec=self.spec['observation_spec'],observation_spec_sha256=self.spec['observation_spec_sha256'],
            output_dir=str(self.out/'child'),work_deadline_s=child_work_end,cleanup_deadline_s=child_cleanup_end)
        c.write(self.out/'job.json',job);self.child_log=(self.out/'child.log').open('xb')
        env=dict(os.environ);env.pop('PDBLEND_NODE_LOCK_FD',None);env['PYTHONPATH']=str(c.HOST/'src')+':'+str(c.HOST)+':/root/workspace/pdblend/.runtime-deps'
        self.child=await asyncio.create_subprocess_exec(sys.executable,str(c.ROOT/'child.py'),'--job',str(self.out/'job.json'),
            stdout=self.child_log,stderr=asyncio.subprocess.STDOUT,env=env,close_fds=True)
        self.state['child_pid']=self.child.pid;self.state['child_argv']=[sys.executable,str(c.ROOT/'child.py'),'--job',str(self.out/'job.json')];self.save()
        while self.child.returncode is None:
            if self.stop or (c.ROOT/'STOP').exists() or (self.out/'STOP').exists() or self.sampler.error or not self.before_deadline():
                self.state['work_interrupt_reason']='STOP/deadline/sensor';self.deadline=child_cleanup_end;self.restoring=True
                await self.stop_child();break
            await asyncio.sleep(.1)
        self.terminal['http_child_exited']=self.child.returncode is not None;self.state['child_exitcode']=self.child.returncode
        child=c.read(self.out/'child/status.json');self.state['child_status']=child
        c.require(child.get('complete') is True and child.get('cleanup_complete') is True,'six-request child/cleanup not complete')
        return child
    async def stop_diagnostic(self):
        self.phase('stop_diagnostic_before_original_restore')
        if self.diag_intent:
            name=self.spec['diagnostic_instance']['container_name']
            # A daemon may have created the unique name even when docker run timed out.
            listing=(await self.command('docker','ps','-a','--format','{{.Names}}',timeout=8)).split()
            if name in listing:
                before=(await self.inspected([name]))[0];c.write(self.out/'diagnostic-before-stop.json',before)
                await self.command('docker','stop','--time','10',name,timeout=20)
                after=(await self.inspected([name]))[0]
                c.require(after['State']['Running'] is False and after['State']['Pid']==0,'diagnostic container did not stop')
                c.write(self.out/'diagnostic-after-stop.json',after)
                # Keep the stopped diagnostic container; collect all logs even on failure.
                try:(self.out/'diagnostic-docker.log').write_text(await self.command('docker','logs',name,timeout=8))
                except BaseException as exc:self.state['errors'].append('docker log '+repr(exc))
        self.terminal['diagnostic_container_stopped']=True
        await self.gpu_free('all8-free-before-original-restart',20)
        self.terminal['diagnostic_gpu_workers_gone']=True
    async def restore_original(self,session):
        c.require(self.terminal.get('http_child_exited') is True and self.terminal.get('diagnostic_gpu_workers_gone') is True,
            'never restart original models while an owned HTTP/GPU worker remains')
        self.phase('restore_all_four_original_containers');self.deadline=self.limits['restore_end_s']
        names=[i['container']['name'] for i in self.original['instances']]
        # Every old endpoint restarts; no stale live NCCL peer or old StartedAt binding survives.
        outcomes=await asyncio.gather(*(self.command('docker','start',name,timeout=25) for name in names),return_exceptions=True)
        c.require(not any(isinstance(x,BaseException) for x in outcomes),'one or more original container starts failed')
        deployment=c.read(c.DEPLOYMENT);items=copy.deepcopy(deployment['instances'])
        identities=await asyncio.gather(*(self.ready(session,i,False) for i in items),return_exceptions=True)
        c.write(self.out/'restored-ready.json',[dict(error=repr(x)) if isinstance(x,BaseException) else x for x in identities])
        c.require(not any(isinstance(x,BaseException) for x in identities),'one or more original owners not ready')
        fresh=copy.deepcopy(self.original);fresh.update(configs={},output=str(self.out/'future-restored-correctness'),
            output_correctness_verified=False,formal_eligible=False,correctness_gate_required_before_performance=True)
        for i,before,actual in zip(fresh['instances'],self.old_identity,identities):
            preserve_container(before['container'],actual['container'])
            c.require(before['provenance']==actual['provenance'],'original imported source/model/namespace changed')
            i['role']='mixed';i['container']['StartedAt']=actual['container']['State']['StartedAt'];i['provenance']=actual['provenance']
        fresh.pop('mechanism_proof',None);fresh.pop('correctness_evidence',None)
        fresh['restoration_evidence']=str(self.out/'status.json');fresh['old_correctness_is_historical_only']=True
        self.state['restored_native']=await self.native(session,fresh,'restored-native')
        c.write(self.out/'restored-bootstrap.binding.json',fresh)
        c.write(self.out/'restored-identity.after.json',await self.engine.identity(session,fresh))
        c.write(self.out/'restored-gpu-processes.json',await asyncio.to_thread(self.deploy.free_gpu_snapshot,self.hardware))
        self.state['restored_binding']=str(self.out/'restored-bootstrap.binding.json');self.state['restored_binding_sha256']=c.sha(self.out/'restored-bootstrap.binding.json')
        self.state['runtime_prefix_after_restore']=archive.verify_prefix_after_restore(fresh,self.state['runtime_stopped'],self.state['restored_native'],self.deadline)
        c.write(self.out/'runtime-prefix-after-restore.json',self.state['runtime_prefix_after_restore'])
        self.state['all_original_restored']=True
    async def run(self):
        import aiohttp
        from ecopadg.measure.backends import PynvmlBackend
        from ecopadg.measure.power import PowerSampler,trapezoid_energy
        from ecopadg.serving.measurement import save_raw,power_evidence
        from ecopadg.metrics import clip_power_window
        from ecopadg.serving.backend import ClockOwner
        self.out.mkdir();self.state['pid']=os.getpid();self.state['started_s']=time.time();self.save()
        self.deadline=min(c.DEADLINE-900,time.time()+120) # Read-only identity budget; no action until complete.
        started=None;ended=None;child_status=None
        loop=asyncio.get_running_loop()
        for sig in (signal.SIGINT,signal.SIGTERM):loop.add_signal_handler(sig,lambda:setattr(self,'stop',True))
        async with aiohttp.ClientSession(trust_env=False) as session:
            try:
                c.main_gate(self.spec['main_proof']);self.original=c.read(self.spec['previous_binding']);c.binding_scope(self.original)
                c.require(c.sha(self.spec['previous_binding'])==self.spec['previous_binding_sha256'],'prior binding changed')
                self.engine.validate_binding(self.original)
                self.old_identity=await self.engine.identity(session,self.original)
                for i in self.original['instances']:await self.sources(i['container']['name'],self.spec['installed_parent_sources'])
                c.write(self.out/'original-identity.before.json',self.old_identity)
                self.identity_verified=True
                self.state['runtime_before_control']=archive.capture(self.original,self.out/'runtime-before-control',self.deadline)
                self.save()
                c.require(not self.stop,'STOP before hardware ownership')
                self.hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
                self.sampler=PowerSampler(range(8),interval=.02,backend=self.hardware,sample_clocks=True);self.sampler.start()
                await self.deploy.await_power_ready(self.sampler,power_evidence)
                started=time.time();self.controls_owned=True;self.limits=deadline_limits(started);self.deadline=self.limits['startup_end_s'];self.state['limits']=self.limits
                self.phase('clock_setup',started);self.clock=ClockOwner(self.hardware,tuple(range(8)));await asyncio.wait_for(self.clock.set(range(8),2520,verify_rise=False),self.remaining(15))
                self.state['hardware_executed']=True
                self.phase('native_all_original_before_stop');await self.native(session,self.original,'original-before-stop-native')
                names=[i['container']['name'] for i in self.original['instances']]
                self.phase('stop_all_original');self.old_stop_intent=True;self.state['original_stop_intent']=names;self.save()
                results=await asyncio.gather(*(self.command('docker','stop','--time','10',name,timeout=20) for name in names),return_exceptions=True)
                c.require(not any(isinstance(x,BaseException) for x in results),'original stop failed')
                stopped=await self.inspected(names);c.require(all(r['State']['Running'] is False and r['State']['Pid']==0 for r in stopped),'original containers still running')
                c.write(self.out/'original-stopped.json',stopped);await self.gpu_free('all8-free-before-diagnostic')
                # Refuse name/ports before declaring any new creation intent.
                existing=(await self.command('docker','ps','-a','--format','{{.Names}}')).split()
                c.require(self.spec['diagnostic_instance']['container_name'] not in existing,'diagnostic name already exists: no retry')
                sockets=[]
                try:
                    for port in [34501,*range(34732,34764)]:
                        s=socket.socket();s.bind(('127.0.0.1',port));sockets.append(s)
                finally:
                    for s in sockets:s.close()
                await self.create_diagnostic(session)
                self.deadline=self.limits['work_end_s'];child_status=await self.run_child()
            except BaseException as exc:self.state['errors'].append('primary '+repr(exc))
            finally:
                self.restoring=True
                if self.limits is None:self.deadline=min(time.time()+90,c.DEADLINE)
                else:self.deadline=self.limits['cleanup_end_s']
                try:await self.stop_child()
                except BaseException as exc:self.state['errors'].append('child terminal '+repr(exc))
                if self.terminal.get('http_child_exited') is True and self.diag_binding is not None:
                    try:
                        owned=[];log=self.out/'child/checks/owned.jsonl'
                        if log.exists():
                            allowed={r['request_uuid'] for r in c.read(self.spec['observation_spec'])['requests']}
                            for line in log.read_text().splitlines():
                                row=json.loads(line);c.require(row['request_id'] in allowed and row['instance_id']==self.diag_binding['instances'][0]['id'],'foreign owned request log');owned.append((row['instance_id'],row['request_id']))
                        self.phase('parent_diagnostic_native_cleanup');self.state['diagnostic_native']=await self.native(session,self.diag_binding,'parent-diagnostic-native',owned)
                    except BaseException as exc:self.state['errors'].append('diagnostic native '+repr(exc))
                if self.old_stop_intent and self.terminal.get('http_child_exited') is True:
                    # Failure during a partial old stop must finish stopping all original peers too.
                    self.deadline=self.limits['restore_end_s']
                    try:
                        names=[i['container']['name'] for i in self.original['instances']]
                        results=await asyncio.gather(*(self.command('docker','stop','--time','10',name,timeout=20) for name in names),return_exceptions=True)
                        c.require(not any(isinstance(x,BaseException) for x in results),'cannot establish all-old-stopped restoration boundary')
                        await self.stop_diagnostic()
                        stopped=await self.inspected(names)
                        self.phase('archive_all_original_runtime_before_restart')
                        self.state['runtime_stopped']=archive.capture(self.original,self.out/'runtime-stopped-before-restart',self.deadline,
                            stopped=stopped,initial=self.state['runtime_before_control'])
                        self.save();await self.restore_original(session)
                    except BaseException as exc:self.state['errors'].append('original restoration '+repr(exc))
                elif self.identity_verified and self.controls_owned and not self.old_stop_intent:
                    # No old process restart happened: retain its original binding but verify admission.
                    try:self.state['unchanged_original_native']=await self.native(session,self.original,'unchanged-original-native')
                    except BaseException as exc:self.state['errors'].append('unchanged original recovery '+repr(exc))
                if self.clock is not None:
                    try:
                        self.phase('release_all8_clocks');self.deadline=self.limits['end_s'];await asyncio.wait_for(self.clock.close(),self.remaining(15));self.state['clock_restore_complete']=True
                    except BaseException as exc:self.state['errors'].append('clock release '+repr(exc));self.state['clock_restore_complete']=False
                else:self.state['clock_restore_complete']=True
                ended=time.time();self.state.update(operation_start_s=started,operation_end_s=ended,process_terminal=self.terminal)
                if self.child_log:self.child_log.close()
                if self.sampler is not None:
                    try:await asyncio.sleep(.15);await asyncio.to_thread(self.sampler.stop)
                    except BaseException as exc:self.state['errors'].append('sampler stop '+repr(exc))
                    try:
                        power=self.out/'power';power.mkdir();save_raw(power,[],self.sampler.samples,self.sampler.utilization_samples,
                            power_source=self.sampler.power_source,power_metadata=self.sampler.power_metadata)
                        with (power/'clocks.csv').open('x',newline='') as f:
                            w=csv.writer(f);w.writerow(['t_s']+[f'gpu{i}_sm_mhz' for i in range(8)]);w.writerows([t,*v] for t,v in self.sampler.frequency_samples)
                        self.state['power_evidence']=power_evidence(self.sampler.samples,self.sampler.power_source,self.sampler.power_metadata)
                        self.state['sampling_error']=self.sampler.error
                        sample_window(self.sampler.samples,self.sampler.frequency_samples,started,ended)
                        self.state['full_operation_energy_j']=trapezoid_energy(clip_power_window(self.sampler.samples,started,ended,pad_s=0))
                        for k,row in enumerate(self.state.get('stages',[])):
                            end=self.state['stages'][k+1]['started_s'] if k+1<len(self.state['stages']) else ended
                            row.update(ended_s=end,energy_j=trapezoid_energy(clip_power_window(self.sampler.samples,row['started_s'],end,pad_s=0)))
                    except BaseException as exc:self.state['errors'].append('power evidence '+repr(exc))
                # Secondary capture/source I/O cannot prevent raw failure-energy preservation.
                try:
                    c.verify_files(self.spec['files']);self.engine.validate_binding(self.original) if not self.old_stop_intent and self.original else None
                except BaseException as exc:self.state['errors'].append('frozen inputs after '+repr(exc))
                try:
                    if child_status is None and (self.out/'child/status.json').is_file():child_status=c.read(self.out/'child/status.json')
                    if self.terminal.get('diagnostic_gpu_workers_gone'):self.state['owner_evidence']=owner_evidence(self.spec,self.out)
                except BaseException as exc:self.state['owner_evidence_error']=repr(exc)
                try:
                    if child_status:
                        self.state['capture']=c.frozen_capture(self.spec['observation_spec'],c.read(self.spec['observation_spec'])['output_dir'],
                            self.out/'capture-frozen',self.out/'child/full-outputs.json',child_status,self.terminal)
                except BaseException as exc:self.state['capture_error']=repr(exc)
                self.state['measurement_valid']=bool(started and self.state.get('all_original_restored') and self.state.get('clock_restore_complete')
                    and not self.state.get('sampling_error') and self.state.get('power_evidence',{}).get('power_source_verified')
                    and self.state.get('full_operation_energy_j') is not None and not self.state['errors'])
                self.state.update(complete=True,phase='terminal',finished_s=time.time(),child_status=child_status,
                    capture_complete=self.state.get('capture',{}).get('capture_complete') is True,
                    exact_passed=child_status.get('exact_passed') if child_status else None,
                    observation_completed=bool(self.state['measurement_valid'] and self.state.get('owner_evidence',{}).get('complete') and self.state.get('capture',{}).get('capture_complete')))
                self.save()
        return self.state

def no_prior_claims():
    technical.verify()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path);p.add_argument('--spec-sha256');p.add_argument('--run',action='store_true');a=p.parse_args()
    c.package_check()
    if not a.run:print(json.dumps(dict(cpu_only=True,hardware_actions=False,one_attempt=True)));return
    c.require(a.spec is not None and a.spec_sha256 is not None and c.sha(a.spec)==a.spec_sha256,'explicit pinned prepared spec SHA required');spec=c.read(a.spec);c.verify_files(spec['files'])
    c.require(Path(spec['results']).parent==a.spec.resolve().parent and Path(spec['results']).name=='results' and a.spec.resolve().parent.parent==c.ROOT.parent
        and a.spec.resolve().parent.name=='B32B-temporal-observation-attempt-003' and spec['attempt_claim']==str(c.ROOT/'execution-once.json'),'isolated attempt namespace differs')
    c.require(spec['hostname']==socket.gethostname()==c.NODE and spec['deadline_s']==c.DEADLINE,'wrong node/deadline')
    technical.spec_preflight(spec['observation_spec'])
    c.require(spec['previous_binding']==c.read(technical.AUTH)['restored_binding'] and spec['previous_binding_sha256']==c.read(technical.AUTH)['files'][spec['previous_binding']],'authorized fresh binding differs')
    c.require(spec['candidate_manifest_sha256']==c.CANDIDATE_SHA and spec['parent_image']==c.IMAGE,'wrong diagnostic candidate')
    c.require(not os.environ.get('PDBLEND_NODE_LOCK_FD'),'diagnostic must acquire a fresh node lease')
    no_prior_claims()
    c.require(not (c.ROOT/'STOP').exists() and not Path(spec['results']).exists(),'STOP/existing attempt; no automatic retry')
    sys.path[:0]=[str(c.HOST/'src'),str(c.HOST),'/root/workspace/pdblend/.runtime-deps']
    from ecopadg.serving.campaign import node_lease
    with node_lease():
        no_prior_claims()
        c.main_gate(spec['main_proof']);deadline_limits(time.time())
        # Atomic package-global one-shot claim survives failed/partial executions.
        with Path(spec['attempt_claim']).open('x') as f:json.dump(dict(spec=str(a.spec.resolve()),sha256=c.sha(a.spec),pid=os.getpid(),claimed_s=time.time()),f)
        result=asyncio.run(Operation(spec).run())
    c.require(result['observation_completed'],'diagnostic incomplete; retain evidence, do not retry automatically')
if __name__=='__main__':main()
