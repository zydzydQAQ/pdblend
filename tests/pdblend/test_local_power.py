import asyncio
import copy
import json
import time
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdblend.profile import local_power as lp, power_calibration as pc
from pdblend.profile.identity import sha256_value
from test_power_calibration import power_model, setup, Sampler


def inventory(now=100, peers=None):
    uuids=['g'+str(i) for i in range(8)]
    return dict(allocated_gpu_uuids=uuids[:4],lease_manifest_sha256='manifest',lease_id='lease',
        last_updated_s=now,inventory=[dict(uuid=u,pids=[123] if i<4 else []) for i,u in enumerate(uuids)],
        peer_jobs=peers or [],peer_snapshots=[])


@pytest.mark.parametrize('fault',['peer','outside_pid','stale','uuid','manifest','history','missing_gpu'])
def test_isolated_inventory_rejects_unqualified_environment(fault):
    value=inventory()
    if fault=='peer':value['peer_jobs']=[{'job_id':'other'}]
    if fault=='outside_pid':value['inventory'][-1]['pids']=[456]
    if fault=='stale':value['last_updated_s']=1
    if fault=='uuid':value['allocated_gpu_uuids'][0]='x'
    if fault=='manifest':value['lease_manifest_sha256']='x'
    if fault=='history':value['peer_snapshots']=[dict(at_s=99,peers=['other'])]
    if fault=='missing_gpu':value['inventory'].pop()
    with pytest.raises(ValueError):lp.isolated_inventory(value,uuids=['g0','g1','g2','g3'],started_s=90,now_s=100,manifest_sha256='manifest')


def test_isolation_accepts_only_own_group_without_claiming_parallel():
    lp.isolated_inventory(inventory(),uuids=['g0','g1','g2','g3'],started_s=90,now_s=100,manifest_sha256='manifest')


@pytest.mark.asyncio
async def test_peer_activity_mid_window_interrupts_measurement(tmp_path):
    profiler=SimpleNamespace(out_dir=tmp_path)
    gate=lp.IsolatedGroup(profiler); calls=[]; ready=asyncio.Event()
    def check():
        calls.append(1)
        if len(calls)>1:ready.set();raise ValueError('peer appeared')
    gate.check=check
    with pytest.raises(RuntimeError,match='changed inventory'):
        async with gate.measurement():await ready.wait();await asyncio.sleep(30)
    assert len(calls)==2


def package_fixture(root):
    own=root/'training';own.mkdir();prior=root/'prior';prior.mkdir();proposal=root/'proposal';proposal.mkdir()
    env=dict(image_digest='image',vllm='vllm',torch='torch',cuda='cuda',hardware_id='8xL20-lease',source_hash='s'*64,gpu_uuids=['g0','g1','g2','g3'])
    base=power_model();base.model='/models/Qwen2.5-14B-Instruct';base.decode_power_overrides={}
    base.save(proposal/'base.json');candidate=copy.deepcopy(base)
    raw=dict(system='pdblend',model_id='Qwen2.5-14B-Instruct',tp=4,pp=1,model_hash='model',tokenizer_hash='tok',
        environment=env,holdout_independent=False,prefill=[],decode=[],mixed=[])
    for f in pc.FREQUENCIES:
        nodes=[]
        for b in (1,128):
            for c in (256,1024,4096):
                reps=[]
                for i in range(3):
                    file=own/f'{f}-{b}-{c}-{i}.json'
                    file.write_text(json.dumps(dict(power=[[1,[150]*4],[6,[150]*4]],frequency=[[2,[f]*4]],
                        start_token_counts=[5+i]*b,end_token_counts=[15+i]*b)))
                    reps.append(dict(effective_context_tokens=c+10+i,power_w=600.,step_seconds=.1,steady_window_s=5,
                        min_steps=10,power_samples=2,frequency_samples=1,samples_file=file.name,samples_sha256=pc.digest(file)))
                raw['decode'].append(dict(freq_mhz=f,batch=b,context_tokens=c,effective_context_tokens=c+11,step_seconds=.1,power_w=600.,repeats=reps))
                nodes.append(dict(batch=b,nominal_context_tokens=c,context_min=c+10,context_max=c+12,power_w=600.))
        candidate.decode_power_overrides[f]=dict(kind=pc.KIND,batch_interpolation='linear',nodes=nodes)
    raw['identity_sha256']=sha256_value(raw);(own/'raw.json').write_text(json.dumps(raw));candidate.save(proposal/'candidate.json')
    old=copy.deepcopy(raw);old['holdout_candidate_sha256']=pc.digest(proposal/'base.json')
    old['decode']=[];old['identity_sha256']=sha256_value({k:v for k,v in old.items() if k!='identity_sha256'})
    (prior/'raw.json').write_text(json.dumps(old))
    (prior/'frozen-fit.json').write_text(json.dumps(dict(candidate_sha256=pc.digest(proposal/'base.json'))))
    plan=dict(candidate_sha256=pc.digest(proposal/'candidate.json'),training_raw_sha256=pc.digest(own/'raw.json'),
        points=[dict(freq_mhz=f,batch=b,context_tokens=1024,max_tokens=500,repeats=3,settle_s=2,measure_s=5,
             purpose='independent_power_holdout') for f in pc.FREQUENCIES for b in (1,128)])
    (proposal/'plan.json').write_text(json.dumps(plan))
    args=dict(candidate=proposal/'candidate.json',plan=proposal/'plan.json',base_candidate=proposal/'base.json',
        training_raw=own/'raw.json',prior_holdout=prior,out=root/'package')
    lp.prepare(**args)
    return args,env


def test_package_binds_own_training_preserves_timing_and_rejects_tampering(tmp_path):
    args,_=package_fixture(tmp_path);package=args['out'];manifest,plan,model=lp.load_package(package)
    assert manifest['training_windows_checked']==108 and len(plan['points'])==12
    assert json.loads((package/'retained-timing-audit.json').read_text())['passed'] is None
    assert not manifest['formal_eligible'] and not manifest['full_profile_qualified']
    before=args['candidate'].read_bytes();(package/'candidate.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'):lp.load_package(package)
    assert args['candidate'].read_bytes()==before


def test_candidate_and_plan_cannot_expand_domain_or_borrow_model(tmp_path):
    args,_=package_fixture(tmp_path);m,plan,model=lp.load_package(args['out'])
    bad=copy.deepcopy(plan);bad['points'][0]['context_tokens']=8000
    with pytest.raises(ValueError,match='memory|support'):lp.validate_plan(bad,model,m['candidate_sha256'])
    train=json.loads(args['training_raw'].read_text());train['model_id']='Qwen2.5-7B-Instruct'
    with pytest.raises(ValueError,match='14B training'):lp.verify_training(train,args['training_raw'].parent,model,pc.PerfModel.load(args['base_candidate']))


def test_twelve_point_runner_loads_once_and_keeps_qualification_separate(tmp_path,monkeypatch):
    from pdblend.profile import profiler as pm
    from pdblend.engine import launcher,client as cm
    args,env=package_fixture(tmp_path);out=tmp_path/'measurement';events=[]
    clock,_,launches,background=setup(tmp_path);original={p:p.read_bytes() for p in (tmp_path/'prior').iterdir()}
    class Profiler:
        def __init__(self,model,gpus,**kw):
            self.out_dir=kw['out_dir'];self.out_dir.mkdir();self.tp=4
            self.raw=dict(config={},decode=[],prefill=[],mixed=[],environment=env)
            self.model_spec=SimpleNamespace(model_hash='model',tokenizer_hash='tok')
            self.specs=[SimpleNamespace(instance_id='p',base_url='http://unused',gpus=gpus)]
            self.meter=SimpleNamespace(sampler=lambda g:Sampler(clock),reset_all=lambda:events.append('reset'))
        def _checkpoint(self):(self.out_dir/'raw.json').write_text(json.dumps(self.raw))
        def _kv_capacity(self,inst):return 2441936
        def _lock(self,f,g):clock.freq=f
    class Fleet:
        def __init__(self,specs,logs):self.specs=specs
        def __enter__(self):return self
        def __exit__(self,*a):events.append('stop')
        def start_all(self):events.append('load')
        def __getitem__(self,key):return SimpleNamespace(spec=self.specs[0])
    class Client:
        def __init__(self,*a):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*a):pass
    class Gate:
        def __init__(self,p):self.p=p
        async def qualify_external(self,*a):events.append('qualification')
        @asynccontextmanager
        async def measurement(self):
            yield
            evidence=dict(complete=True,passed=True,measurement_complete=True,parallel=False)
            path=out/'isolation.json';path.write_text(json.dumps(evidence))
            self.p.raw['local_power_isolation']=dict(samples_file=path.name,samples_sha256=pc.digest(path))
        def write(self,*a):events.append('error')
    actual=pc.collect_power_point
    async def collect(*a,**kw):return await actual(*a,**kw,_clock=clock,_sleep=clock.sleep,_background_factory=background)
    monkeypatch.setattr(pm,'Profiler',Profiler);monkeypatch.setattr(pm,'_load_flock',nullcontext)
    monkeypatch.setattr(launcher,'Fleet',Fleet);monkeypatch.setattr(cm,'EngineClient',Client)
    monkeypatch.setattr(lp,'IsolatedGroup',Gate);monkeypatch.setattr(pc,'collect_power_point',collect);monkeypatch.setattr(pc.asyncio,'sleep',clock.sleep)
    result=pc.run(package=args['out'],model_path='/models/Qwen2.5-14B-Instruct',gpus=[0,1,2,3],base_port=15000,
        out=out,panel_adapter=lp.Adapter('isolated-host'))
    assert result['complete'] and result['power_passed'],result
    assert result['reused_timing_passed'] is None and not result['calibration_components_passed']
    assert result['concurrency_qualified'] and not result['full_profile_qualified'] and not result['formal_eligible']
    assert result['measured_decode_points']==12 and len(launches)==12
    assert events==['load','qualification','stop','reset']
    assert {p:p.read_bytes() for p in (tmp_path/'prior').iterdir()}==original
    audit=json.loads((out/'power-only-audit.json').read_text())
    assert all(v['windows']==v['expected_windows']==6 for v in audit['by_frequency'].values())


def test_completed_combined_timing_binds_original_sources_without_relabel(tmp_path):
    import shutil
    args,_=package_fixture(tmp_path);prior=args['prior_holdout'];original=tmp_path/'completed';original.mkdir()
    raw=json.loads(args['training_raw'].read_text());raw['holdout_candidate_sha256']=pc.digest(args['base_candidate'])
    for row in raw['decode']:
        for rep in row['repeats']:shutil.copy2(args['training_raw'].parent/rep['samples_file'],original/rep['samples_file'])
    (original/'mixed.json').write_text('{}')
    raw['mixed']=[dict(valid=True,base_step_s=.1,alone_prefill_s=.1,probe_ttft_s=.2,
        samples_file='mixed.json',samples_sha256=pc.digest(original/'mixed.json')) for _ in range(12)]
    inherited=copy.deepcopy(raw);inherited['decode']=inherited['decode'][:1];inherited['mixed']=[]
    for rep in inherited['decode'][0]['repeats']:shutil.copy2(original/rep['samples_file'],prior/rep['samples_file'])
    (prior/'raw.json').write_text(json.dumps(inherited))
    plan=dict(prefill=[],decode=[{k:r[k] for k in ('freq_mhz','batch','context_tokens')} for r in raw['decode']])
    manifest=dict(candidate_sha256=pc.digest(args['base_candidate']),plan=plan)
    (original/'frozen-fit.json').write_text(json.dumps(manifest))
    combined=copy.deepcopy(raw);combined['decode'][0]['evidence_source']='original'
    combined['evidence_sources']=dict(original=dict(raw_sha256=pc.digest(prior/'raw.json')))
    raw['decode']=raw['decode'][1:]
    (original/'raw.json').write_text(json.dumps(raw));(original/'combined-holdout.json').write_text(json.dumps(combined))
    completion=dict(complete=True,independent_holdout=True,candidate_sha256=pc.digest(args['base_candidate']),
        raw_sha256=pc.digest(original/'raw.json'),combined_holdout_sha256=pc.digest(original/'combined-holdout.json'),calibration_passed=False)
    (original/'completion.json').write_text(json.dumps(completion))
    before={p:p.read_bytes() for p in original.iterdir()}
    audit,files=lp.bind_original_timing(original,prior,pc.PerfModel.load(args['base_candidate']),pc.digest(args['base_candidate']))
    assert audit['passed'] and original/'combined-holdout.json' in files
    assert {p:p.read_bytes() for p in original.iterdir()}==before
    assert json.loads((original/'completion.json').read_text())['calibration_passed'] is False
    (prior/'raw.json').write_text('{}')
    with pytest.raises(ValueError,match='inherited source'):lp.bind_original_timing(original,prior,pc.PerfModel.load(args['base_candidate']),pc.digest(args['base_candidate']))
