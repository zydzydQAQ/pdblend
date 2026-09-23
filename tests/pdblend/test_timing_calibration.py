import copy
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdblend.profile import timing_calibration as tc
from pdblend.profile.model import PerfModel, StaticState


def model():
    fs=tc.FREQUENCIES
    base=PerfModel(freqs=fs,prefill_time={f:(.01,0.,0.) for f in fs},
        prefill_power={f:(200.,0.) for f in fs},decode_time={f:(.1,0.,0.,0.) for f in fs},
        decode_power={f:(600.,0.) for f in fs},static={f'active_idle@{f}':StaticState(100.) for f in fs},
        model='/models/Qwen2.5-32B-Instruct',tp=4,pp=1,system='pdblend',kv_capacity_tokens=400000,
        bounded_coverage=dict(prefill_tokens=[128,8192]))
    base.decode_overrides={f:dict(kind='split_b1_relative',coefficients=[.1,0,0,0,.1,0],
        domain=dict(batch=[1,256],context=[256,4096],max_batch_context=360000)) for f in fs}
    candidate=dict(kind=tc.KIND,system='pdblend',model_id='Qwen2.5-32B-Instruct',tp=4,pp=1,
        batch_knots=[16,32,64],context_degree=0,formal_eligible=False,
        residual_seconds={str(f):.004 for f in fs},domains={str(f):copy.deepcopy(base.decode_overrides[f]['domain']) for f in fs})
    return tc.TimingOverlay(base,candidate)


class Clock:
    def __init__(self):self.now=1000.;self.live=[];self.freq=1500;self.step=.1
    def __call__(self):return self.now
    async def sleep(self,seconds):
        self.now+=seconds
        for r in self.live:
            while r.token_times_s[-1]+self.step<=self.now+1e-8:
                r.token_times_s.append(r.token_times_s[-1]+self.step)


class Sampler:
    error=None
    def __init__(self,clock):self.clock=clock
    def start(self):self.start_s=self.clock()
    def stop(self):
        self.samples=[(self.start_s+.5,[150.]*4),(self.clock(),[150.]*4)]
        self.frequency_samples=[(self.start_s+1,[self.clock.freq]*4)]


def setup(tmp_path):
    clock=Clock();launches=[]
    profiler=SimpleNamespace(out_dir=tmp_path,raw=dict(kv_capacity_tokens=400000),
        meter=SimpleNamespace(sampler=lambda g:Sampler(clock)))
    @asynccontextmanager
    async def background(profiler,client,point,tag):
        launches.append(tag)
        clock.live=[SimpleNamespace(token_times_s=[clock.now-1.6+i*.1 for i in range(17)]) for _ in range(point['batch'])]
        yield clock.live,[]
    return clock,profiler,launches,background


def point(batch=32):
    return dict(freq_mhz=1500,batch=batch,context_tokens=1024,max_tokens=500,repeats=3,
        settle_s=2.,measure_s=5.,purpose=tc.PURPOSE)


def test_overlay_exactly_preserves_original_domain_and_outer_predictions():
    m=model()
    for f in tc.FREQUENCIES:
        for b in (1,2,8,12,16,64,128,256):
            for c in (256,1024,4096):
                assert m.decode_supported(b,c,f)==m.base.decode_supported(b,c,f)
                if m.decode_supported(b,c,f):assert m.step_seconds(b,c,f)==m.base.step_seconds(b,c,f)
    assert m.step_seconds(24,1024,1500)==pytest.approx(.102)
    assert m.step_seconds(32,1024,1500)==pytest.approx(.104)
    assert m.step_seconds(48,1024,1500)==pytest.approx(.102)
    for args in ((32,5000,1500),(32,1024,1600)):
        with pytest.raises(ValueError,match='coverage'):m.step_seconds(*args)
    bad=copy.deepcopy(m.candidate);bad['domains']['900']['context'][1]+=1
    with pytest.raises(ValueError,match='coverage'):tc.TimingOverlay(m.base,bad)


def test_constant_residual_uses_training_windows_and_requires_context_shapes():
    base=model().base.decode_overrides[1500]
    rows=[dict(batch=32,context_tokens=c,repeats=[dict(effective_context_tokens=c+10+i,step_seconds=.104) for i in range(3)]) for c in (256,1024,4096)]
    assert tc.fit_residual(rows,base)==pytest.approx([.004])
    with pytest.raises(ValueError,match='two training context'):tc.fit_residual(rows[:1],base)


@pytest.mark.asyncio
async def test_timing_windows_one_prefill_preserve_actual_context_and_resume(tmp_path):
    clock,profiler,launches,background=setup(tmp_path);p=point();binding=dict(candidate_sha256='c',plan_sha256='p')
    checkpoints=[]
    row=await tc.collect_point(profiler,None,[0,1,2,3],p,model=model(),binding=binding,
        on_window=lambda x:checkpoints.append(x),_clock=clock,_sleep=clock.sleep,_background_factory=background)
    assert len(launches)==1 and len(checkpoints)==3 and row['prefill_runs']==1
    contexts=[r['effective_context_tokens'] for r in row['repeats']]
    assert contexts[0]>1024 and contexts==sorted(contexts) and len(set(contexts))==3
    raw=dict(decode=[row]);audit=tc.audit_fresh(raw,tmp_path,[p],model(),binding)
    assert audit['passed'] and len(audit['points'])==3
    resumed=await tc.collect_point(profiler,None,[0,1,2,3],p,model=model(),binding=binding,
        previous=row['repeats'],_clock=clock,_sleep=clock.sleep,_background_factory=background)
    assert resumed==row and len(launches)==1
    assert tc.resume_windows(raw,tmp_path,[p],binding)=={tc.point_key(p)}
    evidence=json.loads((tmp_path/row['repeats'][0]['samples_file']).read_text())
    assert evidence['purpose']==tc.PURPOSE and evidence['binding']==binding
    corrupt=copy.deepcopy(raw);corrupt['decode'][0]['repeats'][1]=copy.deepcopy(row['repeats'][0])
    with pytest.raises(ValueError,match='duplicate'):tc.resume_windows(corrupt,tmp_path,[p],binding)
    (tmp_path/row['repeats'][0]['samples_file']).write_text('{}')
    with pytest.raises(ValueError,match='checksum'):tc.resume_windows(raw,tmp_path,[p],binding)


@pytest.mark.asyncio
async def test_timing_failure_is_preserved_without_retries_and_frequency_checked(tmp_path):
    clock,profiler,launches,background=setup(tmp_path);p=point();binding={};clock.step=.13
    row=await tc.collect_point(profiler,None,[0,1,2,3],p,model=model(),binding=binding,
        _clock=clock,_sleep=clock.sleep,_background_factory=background)
    audit=tc.audit_fresh(dict(decode=[row]),tmp_path,[p],model(),binding)
    assert not audit['passed'] and len(launches)==1
    assert len([e for e in audit['failures'] if e['metric']=='independent_timing_window'])==3
    assert all(r['samples_sha256'] for r in row['repeats'])
    # A matching raw observation but wrong clock is still a failed gate.
    clock.step=.1;clock.freq=1200
    row=await tc.collect_point(profiler,None,[0,1,2,3],p,model=model(),binding=binding,
        _clock=clock,_sleep=clock.sleep,_background_factory=background)
    audit=tc.audit_fresh(dict(decode=[row]),tmp_path,[p],model(),binding)
    assert len([e for e in audit['failures'] if e['metric']=='timing_frequency_identity'])==3


@pytest.mark.asyncio
async def test_partial_timing_resume_keeps_old_raw_and_declares_new_prefill(tmp_path):
    clock,profiler,launches,background=setup(tmp_path);p=point();saved=[]
    def interrupt(repeats):
        saved[:]=repeats;raise RuntimeError('interrupt')
    with pytest.raises(RuntimeError,match='interrupt'):
        await tc.collect_point(profiler,None,[0,1,2,3],p,model=model(),binding={},on_window=interrupt,
            _clock=clock,_sleep=clock.sleep,_background_factory=background)
    original=(tmp_path/saved[0]['samples_file']).read_bytes()
    row=await tc.collect_point(profiler,None,[0,1,2,3],p,model=model(),binding={},previous=saved,
        _clock=clock,_sleep=clock.sleep,_background_factory=background)
    assert row['prefill_runs']==2 and len(row['repeats'])==3 and len(launches)==2
    assert (tmp_path/saved[0]['samples_file']).read_bytes()==original
    tc.resume_windows(dict(decode=[row]),tmp_path,[p],{})


@pytest.mark.asyncio
async def test_early_identity_failure_writes_failed_receipt(tmp_path,monkeypatch):
    def reject(package):raise ValueError('wrong model hash')
    monkeypatch.setattr(tc,'load_package',reject)
    with pytest.raises(ValueError,match='model hash'):
        await tc.collect_existing(profiler=None,client=None,gpus=[0,1,2,3],package=tmp_path/'package',out=tmp_path/'out')
    receipt=json.loads((tmp_path/'out/completion.json').read_text())
    assert not receipt['complete'] and not receipt['timing_passed'] and not receipt['formal_eligible']
    assert not receipt['energy_comparable']


def resident(tmp_path,clock):
    parent=tmp_path/'power';parent.mkdir(exist_ok=True)
    evidence=parent/'samples/external-interference.json'
    tc.atomic_json(evidence,dict(complete=True,passed=True,cohort_id='power-4-4'))
    environment=dict(image_digest='image',source_hash='source',gpu_uuids=['uuid0','uuid1','uuid2','uuid3'],
        torch='torch',cuda='cuda',vllm='vllm',hardware_id='8xL20-lease')
    p=SimpleNamespace(out_dir=parent,meter=SimpleNamespace(sampler=lambda g:Sampler(clock)),
        model_spec=SimpleNamespace(model_hash='model',tokenizer_hash='tokenizer'),
        raw=dict(system='pdblend',model_id='Qwen2.5-32B-Instruct',tp=4,pp=1,environment=environment,
            kv_capacity_tokens=400000,decode=[dict(power_only=True)],
            external_interference=dict(complete=True,passed=True,cohort_id='power-4-4',measured_mode='parallel',
                samples_file='samples/external-interference.json',samples_sha256=tc.digest(evidence)),
            concurrency_environment=dict(lease_id='lease',allocated_gpu_uuids=environment['gpu_uuids'])),
        _lock=lambda f,g:setattr(clock,'freq',f))
    return p


def test_resident_binding_uses_uuid_source_and_exact_external_receipt(tmp_path):
    p=resident(tmp_path,Clock());one=tc.resident_binding(p)
    p.raw['environment']['gpu_uuids'][0]='different-physical-gpu'
    assert tc.resident_binding(p)!=one
    (p.out_dir/'samples/external-interference.json').write_text('{}')
    with pytest.raises(ValueError,match='qualification'):tc.resident_binding(p)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure',[False,True])
async def test_resident_runner_separate_receipts_and_parent_unchanged(tmp_path,monkeypatch,failure):
    clock,_,launches,background=setup(tmp_path);clock.step=.13 if failure else .1
    p=resident(tmp_path,clock);old=copy.deepcopy(p.raw);package=tmp_path/'package';package.mkdir()
    tc.atomic_json(package/'manifest.json',{})
    manifest=dict(model_hash='model',tokenizer_hash='tokenizer',candidate_sha256='c',plan_sha256='p',
        inputs={},training_environment=copy.deepcopy(p.raw['environment']))
    plan=dict(points=[point()]);m=model()
    monkeypatch.setattr(tc,'load_package',lambda path:(manifest,plan,m))
    monkeypatch.setattr(tc,'reuse_original',lambda manifest,model:dict(passed=True))
    monkeypatch.setattr(tc.asyncio,'sleep',clock.sleep)
    original=tc.collect_point
    async def collect(*args,**kwargs):
        return await original(*args,**kwargs,_clock=clock,_sleep=clock.sleep,_background_factory=background)
    monkeypatch.setattr(tc,'collect_point',collect)
    result=await tc.collect_existing(profiler=p,client=None,gpus=[0,1,2,3],package=package,out=tmp_path/'timing')
    assert result['complete'] and result['timing_passed'] is (not failure)
    assert result['receipt_sha256']==tc.digest(tmp_path/'timing/completion.json')
    assert p.raw==old and len(launches)==1
    completion=json.loads((tmp_path/'timing/completion.json').read_text())
    assert completion['timing_passed'] is (not failure) and not completion['formal_eligible'] and not completion['energy_comparable']
    assert 'receipt_sha256' not in completion
    raw=json.loads((tmp_path/'timing/raw.json').read_text())
    assert raw['external_interference']==old['external_interference']
    assert raw['parent_power_artifact_root']==str(p.out_dir.resolve())
    # Same container ordinals on a different lease cannot inherit old windows.
    p.raw['environment']['gpu_uuids'][0]='replacement-uuid'
    with pytest.raises(ValueError,match='checkpoint'):
        await tc.collect_existing(profiler=p,client=None,gpus=[0,1,2,3],package=package,out=tmp_path/'timing')
    assert len(launches)==1
