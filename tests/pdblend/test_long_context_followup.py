import asyncio
import copy
import json
from contextlib import asynccontextmanager,nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdblend.profile import long_context_followup as lf,long_context_collect as lc,profiler as pm
from pdblend.profile.wave import atomic_json
from pdblend.profile.identity import sha256_value


class Clock:
    def __init__(self):self.now=1000.;self.live=[];self.freq=900;self.step=.025
    def __call__(self):return self.now
    async def sleep(self,seconds):
        self.now+=seconds
        for r in self.live:
            while r.token_times_s[-1]+self.step<=self.now+1e-8:r.token_times_s.append(r.token_times_s[-1]+self.step)
    @asynccontextmanager
    async def background(self,profiler,client,point,tag):
        self.live=[SimpleNamespace(token_times_s=[self.now-.4+i*.025 for i in range(17)],error=None) for _ in range(point['batch'])]
        yield self.live,[]


class Sampler:
    error=None
    def __init__(self,clock,tp):self.clock,self.tp=clock,tp
    def start(self):self.begin=self.clock()
    def stop(self):
        watts=getattr(self.clock,'watts',100)
        self.samples=[(self.begin+.5,[watts]*self.tp),(self.clock(),[watts]*self.tp)]
        self.frequency_samples=[(self.begin+1,[self.clock.freq]*self.tp)]


async def fixture(root,model='Qwen2.5-32B-Instruct'):
    tp,batches=lf.MODELS[model];prior=root/'prior';prior.mkdir();clock=Clock()
    p=pm.Profiler.__new__(pm.Profiler);p.out_dir=prior;p.tp=tp;p.parallel_layout={}
    p.raw=dict(schema=2,system='pdblend',model_id=model,model='/models/'+model,model_hash='m',tokenizer_hash='t',tp=tp,pp=1,
        profile_key={},parallel_layout={},role='mixed',freqs=list(lf.FREQUENCIES),config={},
        environment=dict(image_digest='image',source_hash='source',torch='torch',vllm='vllm',cuda='cuda',hardware_id='8xL20-lease',
            gpu_uuids=['g'+str(i) for i in range(tp)]),kv_capacity_tokens=200000,prefill=[],decode=[],mixed=[],transfer=[],static={})
    p.meter=SimpleNamespace(sampler=lambda g:Sampler(clock,tp))
    points=[dict(freq_mhz=f,batch=b,context_tokens=c,max_tokens=1024,repeats=3,settle_s=2,measure_s=5,purpose='training_extension')
        for f in lf.FREQUENCIES for b in batches for c in (5120,7168)]
    plan=dict(system='pdblend',model_id=model,tp=tp,pp=1,training=points,fit_existing_holdout=False)
    atomic_json(prior/'training-plan.json',plan);p.raw['training_plan_sha256']=lc.digest(prior/'training-plan.json')
    for point in points:
        clock.freq=point['freq_mhz']
        p.raw['decode'].append(await lc.collect_training_point(p,None,list(range(tp)),point,
            _clock=clock,_sleep=clock.sleep,_background_factory=clock.background))
    p._checkpoint();atomic_json(prior/'completion.json',dict(complete=True,raw_sha256=lc.digest(prior/'raw.json')))
    package=root/'package';lf.prepare(prior=prior,out=package)
    return package,p


@pytest.mark.asyncio
async def test_prepare_only_own_complete_training_and_early_end_budget(tmp_path):
    package,p=await fixture(tmp_path)
    m,plan,prior=lf.load_package(package)
    assert len(plan['training'])==12 and all(x['max_tokens']==512 and x['training_derived_min_output_tokens']<512 for x in plan['training'])
    assert m['expected_holdout_points']==24 and m['exact_batches']==[1,4]
    assert not m['fit_existing_holdout'] and not m['formal_eligible']
    # Pure CPU preflight refuses a measured fast configuration needing >512 outputs.
    rows=copy.deepcopy(prior['decode'])
    for row in rows:
        for rep in row['repeats']:rep['step_seconds']=.005
    assert lf.output_budget(rows)>512
    (p.out_dir/'raw.json').write_text('{}')
    with pytest.raises(ValueError,match='input changed'):lf.load_package(package)


@pytest.mark.asyncio
async def test_exact_batch_candidate_never_extrapolates_or_fits_holdout(tmp_path):
    _,p=await fixture(tmp_path);endpoint=copy.deepcopy(p.raw)
    endpoint['decode']=[]
    for row in p.raw['decode']:
        if row['context_tokens']!=7168:continue
        r=copy.deepcopy(row);r['context_tokens']=7680
        for rep in r['repeats']:rep['effective_context_tokens']+=512
        endpoint['decode'].append(r)
    candidate=lf.fit_candidate(p.raw,endpoint)
    for b in (2,3,5):
        with pytest.raises(ValueError,match='batch'):lf.predict(candidate,'step_seconds',900,b,6500)
    with pytest.raises(ValueError,match='domain'):lf.predict(candidate,'step_seconds',900,1,8000)
    assert lf.predict(candidate,'step_seconds',900,1,6500)==pytest.approx(.025,rel=.002)
    endpoint['decode'][0]['independent_holdout']=True
    with pytest.raises(ValueError,match='holdout'):lf.fit_candidate(p.raw,endpoint)


def test_14b_training_only_binds_own_speed_without_reusing_prefill_or_holdout(tmp_path):
    raw=dict(schema=2,system='pdblend',model_id='Qwen2.5-14B-Instruct',tp=1,pp=1,model_hash='m',tokenizer_hash='t',
        kv_capacity_tokens=47232,prefill=[dict(freq_mhz=900,input_tokens=128,legacy_unbound=True)],decode=[])
    for f in lf.FREQUENCIES:
        for b in (1,4):
            reps=[]
            for i in range(3):
                p=tmp_path/f'{f}-{b}-{i}.json';p.write_text('{}')
                reps.append(dict(step_seconds=.03,steady_window_s=5,min_steps=100,power_samples=2,frequency_samples=1,
                    samples_file=p.name,samples_sha256=lc.digest(p)))
            raw['decode'].append(dict(freq_mhz=f,batch=b,context_tokens=4096,repeats=reps))
    raw['identity_sha256']=sha256_value(raw);source=tmp_path/'raw.json';atomic_json(source,raw)
    provenance=tmp_path/'original-training.json';atomic_json(provenance,dict(training_raw=str(source),training_raw_sha256=lc.digest(source)))
    out=tmp_path/'training-only';manifest=lf.prepare_training_only(training_manifest=provenance,out=out)
    plan=json.loads((out/'training-plan.json').read_text())
    assert len(plan['training'])==36 and plan['holdout']==[] and not manifest['fit_performed']
    assert all(p['max_tokens']==min(1024,8192-p['context_tokens']) for p in plan['training'])
    assert all(p['batch']*(p['context_tokens']+p['max_tokens'])<=.9*47232 for p in plan['training'])
    raw['holdout_candidate_sha256']='a holdout is not training'
    raw['identity_sha256']=sha256_value({k:v for k,v in raw.items() if k!='identity_sha256'});atomic_json(source,raw)
    atomic_json(provenance,dict(training_raw=str(source),training_raw_sha256=lc.digest(source)))
    with pytest.raises(ValueError,match='original 14B'):
        lf.prepare_training_only(training_manifest=provenance,out=tmp_path/'invalid')


@pytest.mark.parametrize('fault',[None,'early_end','power_error'])
def test_full_staged_runner_loads_once_freezes_before_holdout_and_never_mutates_prior(tmp_path,monkeypatch,fault):
    from pdblend.profile import wave
    from pdblend.engine import launcher,client as cm
    package,prior_profiler=asyncio.run(fixture(tmp_path));m,plan,source=lf.load_package(package)
    prior_bytes={p:p.read_bytes() for p in prior_profiler.out_dir.rglob('*.json')}
    out=tmp_path/'measurement';events=[];clock=Clock();actual_collect=lc.collect_bounded_decode_point
    class Profiler:
        _checkpoint=pm.Profiler._checkpoint
        _snapshot_concurrency_environment=pm.Profiler._snapshot_concurrency_environment
        resume=pm.Profiler.resume
        def _decode_grid(self):return []
        def __init__(self,model,gpus,**kw):
            self.out_dir=kw['out_dir'];self.out_dir.mkdir(exist_ok=True);self.tp=2;self.parallel_layout={}
            self.model=model;self.pp=1;self.system='pdblend';self.role='mixed';self.freqs=lf.FREQUENCIES;self.mixed_freqs=(1500,2100,2520)
            self.raw=dict(copy.deepcopy(source),prefill=[],decode=[],mixed=[],transfer=[],static={},config={})
            self.model_spec=SimpleNamespace(model_id=source['model_id'],model_hash='m',tokenizer_hash='t')
            self.specs=[SimpleNamespace(instance_id='p',base_url='http://unused',gpus=gpus)]
            self.meter=SimpleNamespace(sampler=lambda g:Sampler(clock,2),reset_all=lambda:events.append('reset'))
        def _kv_capacity(self,instance):return 200000
        def _lock(self,f,g):clock.freq=f
    class Fleet:
        def __init__(self,specs,*a):self.specs=specs
        def __enter__(self):return self
        def __exit__(self,*a):events.append('stop')
        def start_all(self):events.append('load')
        def __getitem__(self,key):return SimpleNamespace(spec=self.specs[0])
    @asynccontextmanager
    async def client(*a):events.append('client_enter');yield object();events.append('client_exit')
    class Wave:
        async def qualify_external(self,p,f):
            events.append('qualification');path=out/'qualification.json';atomic_json(path,dict(complete=True,passed=True,cross_job=True,seq=events.count('qualification')))
            p.raw['external_interference']=dict(complete=True,passed=True,cross_job=True,samples_file=path.name,samples_sha256=lc.digest(path))
        @asynccontextmanager
        async def measurement(self):events.append('measurement_enter');yield;events.append('measurement_exit')
        def write(self,*args):events.append(('error',args))
    async def collect(profiler,client,gpus,point,**kw):
        if point['purpose']=='independent_holdout_repair':
            assert (out/'long-candidate/candidate.json').is_file() and (out/'long-candidate/manifest.json').is_file()
            assert len(json.loads((out/'raw.json').read_text())['decode'])==12
            events.append('holdout')
            if fault=='early_end':raise lc._EarlyEnd('real stream exhausted bounded max_tokens')
            if fault=='power_error':clock.watts=130
        else:events.append('training')
        return await actual_collect(profiler,client,gpus,point,**kw,_clock=clock,_sleep=clock.sleep,_background_factory=clock.background)
    monkeypatch.setattr(pm,'Profiler',Profiler);monkeypatch.setattr(pm,'_load_flock',nullcontext)
    monkeypatch.setattr(launcher,'Fleet',Fleet);monkeypatch.setattr(cm,'EngineClient',client)
    monkeypatch.setattr(wave.ProfileWave,'from_environment',lambda:Wave())
    monkeypatch.setattr(lc,'collect_bounded_decode_point',collect);monkeypatch.setattr(lc.asyncio,'sleep',clock.sleep)
    result=lc.run(plan_path=package/'endpoint-training-plan.json',training_raw_path=Path(m['inputs']['raw.json']['path']),
        model_path='/models/'+m['model_id'],gpus=[0,1],base_port=10000,out=out,resident_followup=lf.ResidentFollowup(package))
    if fault=='early_end':
        assert not result['complete'] and result['status']=='inconclusive' and not result['extended_calibration_passed'],result
        assert result['endpoint_training_complete'] and result['resident_followup']['measured_points']==0
    else:
        assert result['complete'] and result['status']=='passed' and result['extended_calibration_passed']==(fault is None),result
    assert result['resident_followup']['expected_points']==24 and not result['formal_eligible']
    assert result['fit_performed'] and result['candidate_fit_performed'] and not result['training_collector_fit_performed']
    assert result['holdout_points_consumed_by_training']==0 and not result['candidate_fit_uses_holdout']
    assert result['fresh_holdout_points']==(0 if fault=='early_end' else 24)
    assert ('extended_candidate_fit' not in result['missing_gates'])
    assert ('fresh_independent_holdout' in result['missing_gates'])==(fault=='early_end')
    assert ('extended_candidate_calibration' in result['missing_gates'])==(fault=='power_error')
    assert events.count('load')==events.count('stop')==1 and events.count('training')==12 and events.count('holdout')==24
    assert events.index('client_enter')<events.index('training')<events.index('holdout')<events.index('client_exit')<events.index('measurement_exit')<events.index('stop')
    assert {p:p.read_bytes() for p in prior_profiler.out_dir.rglob('*.json')}==prior_bytes
    new=json.loads((out/'raw.json').read_text());fresh=json.loads((out/'long-holdout/raw.json').read_text())
    assert len(new['decode'])==12 and all(r['independent_holdout'] is False for r in new['decode'])
    if fault=='early_end':
        assert len(fresh['missing_holdout_points'])==24
        return
    assert len(fresh['decode'])==24 and all(r['independent_holdout'] is True for r in fresh['decode'])
    candidate=json.loads((out/'long-candidate/candidate.json').read_text());frozen_plan=json.loads((out/'long-candidate/holdout-plan.json').read_text())
    audit=lf.audit(candidate,fresh,out/'long-holdout',frozen_plan);assert audit['passed']==(fault is None)
    if fault=='power_error':
        assert all(r['metric']=='power_w' for r in audit['failures'])
        return
    # A restarted same-identity job gets a fresh qualification receipt.  Reuse
    # completed windows under their original receipt and sample only the gap.
    candidate_bytes=(out/'long-candidate/candidate.json').read_bytes()
    saved=copy.deepcopy(fresh['decode'][:-1]);fresh['decode']=copy.deepcopy(saved)
    fresh['identity_sha256']=sha256_value({k:v for k,v in fresh.items() if k!='identity_sha256'})
    atomic_json(out/'long-holdout/raw.json',fresh)
    resumed=lc.run(plan_path=package/'endpoint-training-plan.json',training_raw_path=Path(m['inputs']['raw.json']['path']),
        model_path='/models/'+m['model_id'],gpus=[0,1],base_port=10000,out=out,resident_followup=lf.ResidentFollowup(package))
    assert resumed['complete'] and resumed['extended_calibration_passed'],resumed
    assert events.count('training')==12 and events.count('holdout')==25
    assert (out/'long-candidate/candidate.json').read_bytes()==candidate_bytes
    fresh=json.loads((out/'long-holdout/raw.json').read_text())
    assert fresh['decode'][:-1]==saved and len(fresh['qualification_history'])==2
    assert fresh['decode'][0]['repeats'][0]['qualification_sha256']!=fresh['decode'][-1]['repeats'][0]['qualification_sha256']
    # Fail closed on true observed range, even when mean/prediction remain valid.
    f,b=900,1;endpoint=next(r for r in fresh['decode'] if r['freq_mhz']==f and r['batch']==b and r['context_tokens']>7000)
    rep=endpoint['repeats'][0]
    candidate['nodes'][f'{f}/{b}'][-1]['context']=(rep['effective_context_tokens']+rep['observed_context_max'])/2
    frozen_plan['candidate_sha256']=sha256_value(candidate)
    assert not lf.audit(candidate,fresh,out/'long-holdout',frozen_plan)['passed']
