"""No docker, HTTP, NVML or GPU: counterexamples to real execution boundaries."""
import asyncio,copy,importlib.util,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import common as c
import run as r


def test_default_modules_no_hardware_import():
    assert not any(x in sys.modules for x in ('torch','pynvml','vllm'))

def test_same_frozen_checks_module_bytes():
    checks=c.module('checks')
    assert c.sha(checks.__file__)==c.MODULES['checks'][1]
    with pytest.raises(RuntimeError,match='native TP ranks'):checks.check_ranks([],2)
    with pytest.raises(RuntimeError,match='missing observation'):checks.check_ranks([{},{}],2)

@pytest.mark.parametrize('value',[float('nan'),c.DEADLINE-899,c.DEADLINE])
def test_no_new_deadline_reset(value):
    with pytest.raises(RuntimeError):r.deadline_limits(value)

def test_budget_absolute_and_original_deadline():
    limits=r.deadline_limits(c.DEADLINE-1000)
    assert limits['end_s']==c.DEADLINE-100
    assert limits['startup_end_s']<limits['work_end_s']<limits['cleanup_end_s']<limits['restore_end_s']<limits['end_s']

def free():return dict(gpus=[dict(gpu=i,memory_api='nvmlDeviceGetMemoryInfo_v2',used_bytes=0,compute_pids=[],graphics_pids=[]) for i in range(8)])

@pytest.mark.parametrize('change',[lambda x:x['gpus'].pop(),lambda x:x['gpus'][2].update(used_bytes=1),lambda x:x['gpus'][3].update(compute_pids=[44]),lambda x:x['gpus'][5].update(graphics_pids=[45]),lambda x:x['gpus'][0].pop('used_bytes')])
def test_native_container_exit_cannot_replace_physical_free(change):
    evidence=free();change(evidence)
    with pytest.raises(RuntimeError):r.assert_free(evidence)

def test_free_all_eight():r.assert_free(free())

@pytest.mark.parametrize('missing',['http_child_exited','diagnostic_container_stopped','diagnostic_gpu_workers_gone'])
def test_writer_complete_does_not_replace_terminal(missing):
    terminal=dict(http_child_exited=True,diagnostic_container_stopped=True,diagnostic_gpu_workers_gone=True);terminal[missing]=False
    with pytest.raises(RuntimeError):c.capture_gate(dict(complete=True,cleanup_complete=True,completed_requests=6),terminal)

def test_exact_difference_is_retained_not_capture_filter():
    c.capture_gate(dict(complete=True,cleanup_complete=True,completed_requests=6,exact_passed=False),
        dict(http_child_exited=True,diagnostic_container_stopped=True,diagnostic_gpu_workers_gone=True))

def test_incomplete_native_rejects_capture():
    with pytest.raises(RuntimeError):c.capture_gate(dict(complete=True,cleanup_complete=False,completed_requests=6),
        dict(http_child_exited=True,diagnostic_container_stopped=True,diagnostic_gpu_workers_gone=True))

@pytest.mark.parametrize('change',[lambda x:x['State'].update(StartedAt='old'),lambda x:x['State'].update(Pid=11),lambda x:x.update(Image='other'),lambda x:x.update(Config={'Env':['changed']})])
def test_restoration_does_not_reuse_started_at_or_changed_source(change):
    before=dict(Id='a',Image='i',Config={},HostConfig={},Mounts=[],Path='python',Args=['entry'],State=dict(StartedAt='old',Pid=11,Running=True))
    after=copy.deepcopy(before);after['State'].update(StartedAt='new',Pid=12);change(after)
    with pytest.raises(RuntimeError):r.preserve_container(before,after)

def test_restore_container_exact_except_process_identity():
    before=dict(Id='a',Image='i',Config={},HostConfig={},Mounts=[],Path='python',Args=['entry'],State=dict(StartedAt='old',Pid=11,Running=True))
    after=copy.deepcopy(before);after['State'].update(StartedAt='new',Pid=12);r.preserve_container(before,after)

@pytest.mark.parametrize('change',[lambda x:x.update(complete=False),lambda x:x['steps'][0].update(complete=False),lambda x:x['steps'][0].update(exitcode=1),lambda x:x.update(phase='dynamollm-resident-main')])
def test_real_main_terminal_required(change):
    state=dict(pid=1,complete=True,phase='main_incomplete_correctness',finished_s=10,steps=[dict(pid=2,complete=True,exitcode=0)])
    change(state)
    with pytest.raises(RuntimeError):c.process_gate(state,live=lambda _:False)

def test_live_process_blocks_even_if_terminal_file():
    state=dict(pid=1,complete=True,phase='main_incomplete_correctness',finished_s=10,steps=[dict(pid=2,complete=True,exitcode=0)])
    with pytest.raises(RuntimeError):c.process_gate(state,live=lambda x:x==2)


def fixture_main(tmp_path,monkeypatch):
    sequence=tmp_path/'sequence';sequence.mkdir();monkeypatch.setattr(c,'SEQUENCE',sequence)
    source=tmp_path/'workloads.json';cells=[];groups={};proof_files={}
    for system in ('mixed','dynamollm','distserve'):
        out=tmp_path/system;out.mkdir();(out/'checkpoints').mkdir();(out/'invocations').mkdir()
        records=[]
        for n in range(30):
            row=dict(cell_id=system+str(n),system=system,phase='main',dataset=('alpaca','sharegpt','longbench')[n%3]);cells.append(row)
            receipt=out/(row['cell_id']+'.json');c.write(receipt,dict(measurement_valid=True,child_stopped=True,clock_restore_complete=True,
                finished_s=20,child_pid=100+n,restoration={'a':dict(complete=True)},summary=dict(slo_attainment=.2)))
            cp=out/'checkpoints'/(row['cell_id']+'.json');cpvalue=dict(row=row,measurement_valid=True,work_complete=False,receipt=str(receipt),
                receipt_sha256=c.sha(receipt),artifacts={str(receipt):c.sha(receipt)});c.write(cp,cpvalue)
            records.append(dict(cell_id=row['cell_id'],checkpoint=str(cp),checkpoint_sha256=c.sha(cp),receipt_sha256=c.sha(receipt)))
        c.write(out/'invocations/a.json',dict(phase='main',system=system,complete=True,finished_s=21,pid=99,error=None))
        groups[system]=dict(complete=True,completed=30,binding=str(out/'binding.json'),records=records)
    c.write(source,dict(model='32b',protocol_id=c.PROTOCOL,cells=cells))
    for system,g in groups.items():
        out=tmp_path/system;c.write(out/'binding.json',dict(model='32b',hostname=c.NODE,system=system,deadline_s=c.DEADLINE,
            protocol_id=c.PROTOCOL,files={str(source):c.sha(source)},output=str(out)))
    proof=sequence/'main-proof.json';c.write(proof,dict(model='32b',hostname=c.NODE,protocol_id=c.PROTOCOL,deadline_s=c.DEADLINE,
        source_manifest=str(source),source_sha256=c.sha(source),baseline_systems=groups))
    c.write(sequence/'status.json',dict(complete=True,phase='main_incomplete_correctness',finished_s=22,pid=98,
        steps=[dict(complete=True,exitcode=0,pid=99)],main_proof=str(proof),main_proof_sha256=c.sha(proof)))
    return proof

def test_three_main_raw_gate_keeps_valid_incomplete_and_qfail(tmp_path,monkeypatch):
    proof=fixture_main(tmp_path,monkeypatch);v=c.main_gate(proof,live=lambda _:False)
    assert set(v['groups'])=={'mixed','dynamollm','distserve'} and v['ecoserve_gate_passed'] is False and v['global_scale_released'] is False

@pytest.mark.parametrize('mutation',['raw','missing30','native','childlive','foreignbinding'])
def test_three_main_gate_rejects_invalid_or_live_evidence(tmp_path,monkeypatch,mutation):
    proof=fixture_main(tmp_path,monkeypatch);live=lambda _:False
    if mutation=='raw':(tmp_path/'mixed/mixed0.json').write_text('{}')
    if mutation=='missing30':(tmp_path/'distserve/checkpoints/distserve29.json').unlink()
    if mutation=='native':
        rc=tmp_path/'mixed/mixed0.json';v=c.read(rc);v['restoration']['a']['complete']=False;c.write(rc,v)
        # Even if an adversary recomputes the CP hash, real native gate must reject.
        cp=tmp_path/'mixed/checkpoints/mixed0.json';d=c.read(cp);d['receipt_sha256']=c.sha(rc);d['artifacts'][str(rc)]=c.sha(rc);c.write(cp,d)
        p=c.read(proof);p['baseline_systems']['mixed']['records'][0].update(checkpoint_sha256=c.sha(cp),receipt_sha256=c.sha(rc));c.write(proof,p)
        sp=c.SEQUENCE/'status.json';s=c.read(sp);s['main_proof_sha256']=c.sha(proof);c.write(sp,s)
    if mutation=='childlive':live=lambda pid:pid==100
    if mutation=='foreignbinding':
        p=tmp_path/'mixed/binding.json';v=c.read(p);v['model']='7b';c.write(p,v)
    with pytest.raises((RuntimeError,FileNotFoundError)):c.main_gate(proof,live=live)

class FakeChild:
    def __init__(self):self.returncode=None;self.signals=[]
    def send_signal(self,s):self.signals.append(s)
    def kill(self):self.signals.append('kill');self.returncode=-9
    async def wait(self):self.returncode=0;return 0

def bare_operation(tmp_path):
    op=object.__new__(r.Operation);op.out=tmp_path;op.state={'errors':[]};op.deadline=time.time()+20;op.restoring=True;op.stop=False;op.terminal={}
    op.anchor_wall=time.time();op.anchor_mono=time.monotonic()
    op.spec={'diagnostic_instance':{'container_name':'own-diag'}};op.child=None;op.diag_intent=True;op.save=lambda:None;op.phase=lambda x:None
    return op

def test_owned_child_must_really_exit_before_restore(tmp_path):
    async def case():
        op=bare_operation(tmp_path);op.child=FakeChild();await op.stop_child()
        assert op.terminal['http_child_exited'] and op.child.signals and op.child.returncode==0
        op.terminal['http_child_exited']=False
        with pytest.raises(RuntimeError,match='never restart'):await op.restore_original(None)
    asyncio.run(case())

def test_diagnostic_stop_failure_does_not_assert_workers_free(tmp_path):
    async def case():
        op=bare_operation(tmp_path);calls=[]
        async def cmd(*args,**kw):
            calls.append(args)
            if args[1]=='ps':return 'own-diag\n'
            raise RuntimeError('stop failed')
        async def inspect(names):return [{'State':{'Running':True},'Id':'own'}]
        async def free(*args):calls.append('free')
        op.command=cmd;op.inspected=inspect;op.gpu_free=free
        with pytest.raises(RuntimeError):await op.stop_diagnostic()
        assert not op.terminal.get('diagnostic_gpu_workers_gone') and 'free' not in calls
    asyncio.run(case())

def test_diagnostic_creation_timeout_still_stops_owned_name(tmp_path):
    async def case():
        op=bare_operation(tmp_path);calls=[];inspections=0
        async def cmd(*args,**kw):
            calls.append(args)
            return 'own-diag\n' if args[1]=='ps' else ''
        async def inspect(names):
            nonlocal inspections
            inspections+=1;return [dict(Id='own',State=dict(Running=inspections==1,Pid=123 if inspections==1 else 0))]
        async def free(*args):calls.append('free')
        op.command=cmd;op.inspected=inspect;op.gpu_free=free
        await op.stop_diagnostic()
        assert any(x[:2]==('docker','stop') for x in calls if isinstance(x,tuple))
        assert op.terminal['diagnostic_gpu_workers_gone'] and calls[-1]=='free'
    asyncio.run(case())


def test_wall_regression_cannot_extend_stage(tmp_path,monkeypatch):
    op=bare_operation(tmp_path);op.deadline=op.anchor_wall+10
    monkeypatch.setattr(r.time,'time',lambda:op.anchor_wall-500)
    monkeypatch.setattr(r.time,'monotonic',lambda:op.anchor_mono+11)
    with pytest.raises(RuntimeError,match='deadline'):op.remaining()


def launch_fixture():
    i=dict(image='frozen',engine_entry='/entry',config='/config',environment=['CUDA_VISIBLE_DEVICES=2,3','NCCL_P2P_DISABLE=1'],mounts=[dict(Type='bind',Source='/x',Destination='/x',RW=True)])
    actual=dict(Image='frozen',State={'Running':True},Config={'Cmd':['python3','/entry','--config','/config'],'Env':['CUDA_VISIBLE_DEVICES=2,3','NCCL_P2P_DISABLE=1']},
        Mounts=copy.deepcopy(i['mounts']),HostConfig=dict(NetworkMode='host',IpcMode='host'))
    return i,actual

@pytest.mark.parametrize('bad',['env','cmd','mount','image'])
def test_actual_launch_identity_precedes_initial_control(bad):
    i,a=launch_fixture()
    if bad=='env':a['Config']['Env'][0]='CUDA_VISIBLE_DEVICES=0,1'
    if bad=='cmd':a['Config']['Cmd'][-1]='/foreign-config'
    if bad=='mount':a['Mounts'][0]['RW']=False
    if bad=='image':a['Image']='foreign'
    with pytest.raises(RuntimeError):r.launch_identity(a,i)


def test_owner_events_actual_modes_and_64_steps(tmp_path):
    obs=dict(requests=[dict(label='temporal-second',request_uuid='rid')]);obs_path=tmp_path/'spec.json';c.write(obs_path,obs)
    cfg=tmp_path/'engine.json';c.write(cfg,dict(id='test',runtime_dir=str(tmp_path)))
    spec=dict(observation_spec=str(obs_path),diagnostic_instance={'config':str(cfg)})
    event=tmp_path/'test.control.events.jsonl'
    events=[dict(request_ids=['rid'],tokens=1,role='mixed',mode='temporal',prefill=0,decode=1) for _ in range(64)]
    event.write_text(''.join(json.dumps(e)+'\n' for e in events));assert r.owner_evidence(spec,tmp_path)['complete']
    events[31]['prefill']=1;event.write_text(''.join(json.dumps(e)+'\n' for e in events))
    with pytest.raises(RuntimeError,match='overlapped'):r.owner_evidence(spec,tmp_path)


@pytest.mark.parametrize('bad',['nan','seven','tail','regress','clock'])
def test_complete_all8_sample_brackets(bad):
    power=[(0,[50.]*8),(1,[60.]*8),(2,[55.]*8)];clocks=[(0,[2520]*8),(2,[1500]*8)]
    if bad=='nan':power[1][1][0]=float('nan')
    if bad=='seven':power[1][1].pop()
    if bad=='tail':power.pop()
    if bad=='regress':power[1]=(-1,power[1][1])
    if bad=='clock':clocks[1][1].pop()
    with pytest.raises(RuntimeError):r.sample_window(power,clocks,.1,1.9)


@pytest.mark.parametrize('failure',['identity','power_ready'])
def test_precontrol_failure_never_sends_cleanup_and_keeps_power(tmp_path,monkeypatch,failure):
    import types
    events=[]
    def mod(name,**fields):
        m=types.ModuleType(name);m.__dict__.update(fields);monkeypatch.setitem(sys.modules,name,m)
    class Sampler:
        def __init__(self,*a,**k):self.samples=[(0,[1]*8),(1,[1]*8)];self.frequency_samples=[(0,[1]*8),(1,[1]*8)];self.utilization_samples=[];self.power_source={};self.power_metadata=[];self.error=None
        def start(self):events.append('sampler_start')
        def stop(self):events.append('sampler_stop')
    def backend(**kw):events.append('backend_read');return object()
    def clock(*a,**kw):events.append('CLOCK_CONTROL');raise AssertionError('must not acquire clocks')
    def raw(*a,**kw):events.append('power_saved')
    for name in ('ecopadg','ecopadg.measure','ecopadg.serving'):mod(name)
    mod('ecopadg.measure.backends',PynvmlBackend=backend)
    mod('ecopadg.measure.power',PowerSampler=Sampler,trapezoid_energy=lambda x:0)
    mod('ecopadg.serving.measurement',save_raw=raw,power_evidence=lambda *a:{'power_source_verified':False})
    mod('ecopadg.metrics',clip_power_window=lambda *a,**kw:[])
    mod('ecopadg.serving.backend',ClockOwner=clock)
    import aiohttp
    class Session:
        async def __aenter__(self):return self
        async def __aexit__(self,*a):pass
    monkeypatch.setattr(aiohttp,'ClientSession',lambda **kw:Session())
    monkeypatch.setattr(c,'main_gate',lambda p:{})
    monkeypatch.setattr(c,'binding_scope',lambda b:None)
    monkeypatch.setattr(c,'verify_files',lambda f:None)
    monkeypatch.setattr(r.archive,'capture',lambda *a,**kw:dict(complete=True))
    prev=tmp_path/'binding.json';c.write(prev,{'instances':[]})
    spec=dict(results=str(tmp_path/'results'),previous_binding=str(prev),previous_binding_sha256=c.sha(prev),
        main_proof='unused',installed_parent_sources={},files={})
    op=r.Operation(spec)
    async def identity(*a):
        if failure=='identity':raise RuntimeError('identity mismatch')
        return []
    async def readiness(*a):raise RuntimeError('not two complete frames')
    async def native(*a,**k):events.append('NATIVE_CONTROL');raise AssertionError('must not control')
    op.engine.identity=identity;op.engine.validate_binding=lambda b:None;op.deploy.await_power_ready=readiness;op.native=native
    status=asyncio.run(op.run())
    assert status['complete'] and status['measurement_valid'] is False
    assert 'CLOCK_CONTROL' not in events and 'NATIVE_CONTROL' not in events
    if failure=='power_ready':assert 'sampler_stop' in events and 'power_saved' in events
    else:assert 'backend_read' not in events
