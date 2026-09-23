import copy
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdblend.profile import power_calibration as pc
from pdblend.profile.model import PerfModel, StaticState
from pdblend.profile.power_table import KIND, PowerCoverageError


def power_model():
    fs=pc.FREQUENCIES
    m=PerfModel(freqs=fs,prefill_time={f:(.01,0.,0.) for f in fs},
        prefill_power={f:(200.,0.) for f in fs},decode_time={f:(.1,0.,0.,0.) for f in fs},
        decode_power={f:(600.,0.) for f in fs},static={f'active_idle@{f}':StaticState(100.) for f in fs},
        model='/models/Qwen2.5-7B-Instruct',tp=4,pp=1,system='pdblend',kv_capacity_tokens=2441936)
    m.decode_power_overrides={f:dict(kind=KIND,batch_interpolation='linear',nodes=[
        dict(batch=b,context_min=c,context_max=c+10,power_w=600.) for b in (1,4,64,128,256) for c in (256,1024,4096)])
        for f in fs}
    return m


class Clock:
    def __init__(self): self.now=1000.; self.live=[]; self.freq=1500; self.watts=600.
    def __call__(self): return self.now
    async def sleep(self,seconds):
        self.now+=seconds
        for r in self.live:
            while r.token_times_s[-1]+.1<=self.now+1e-8:
                r.token_times_s.append(r.token_times_s[-1]+.1)


class Sampler:
    error=None
    def __init__(self,clock): self.clock=clock
    def start(self): self.begin=self.clock()
    def stop(self):
        self.samples=[(self.begin+.5,[self.clock.watts/4]*4),(self.clock(),[self.clock.watts/4]*4)]
        self.frequency_samples=[(self.begin+1,[self.clock.freq]*4)]


def point(freq=1500,batch=4):
    return dict(freq_mhz=freq,batch=batch,context_tokens=1024,max_tokens=500,repeats=3,
                settle_s=2.,measure_s=5.,purpose='independent_power_holdout')


def setup(tmp_path):
    clock=Clock();launches=[]
    profiler=SimpleNamespace(out_dir=tmp_path,raw=dict(kv_capacity_tokens=2441936),
                             meter=SimpleNamespace(sampler=lambda g:Sampler(clock)))
    @asynccontextmanager
    async def background(profiler,client,p,tag):
        launches.append(tag)
        clock.live=[SimpleNamespace(token_times_s=[clock.now-1.6+i*.1 for i in range(17)],error=None) for _ in range(p['batch'])]
        yield clock.live,[]
    return clock,profiler,launches,background


@pytest.mark.asyncio
async def test_shared_power_windows_one_prefill_actual_context_and_checked_resume(tmp_path):
    clock,profiler,launches,background=setup(tmp_path);p=point();binding=dict(candidate_sha256='c',plan_sha256='p')
    checkpoints=[]
    result=await pc.collect_power_point(profiler,None,[0,1,2,3],p,model=power_model(),binding=binding,
        on_window=lambda x:checkpoints.append(x),_clock=clock,_sleep=clock.sleep,_background_factory=background)
    assert len(launches)==1 and len(checkpoints)==3 and result['prefill_runs']==1
    contexts=[r['effective_context_tokens'] for r in result['repeats']]
    assert contexts[0]<contexts[1]<contexts[2] and contexts[0]>p['context_tokens']
    raw=dict(decode=[result]);assert pc.resume_windows(raw,tmp_path,[p],binding)=={pc.point_key(p)}
    # Resume after a completed point must not load or sample again.
    resumed=await pc.collect_power_point(profiler,None,[0,1,2,3],p,model=power_model(),binding=binding,
        previous=result['repeats'],_clock=clock,_sleep=clock.sleep,_background_factory=background)
    assert len(launches)==1 and resumed==result
    corrupt=copy.deepcopy(raw);corrupt['decode'][0]['repeats'][1]=copy.deepcopy(result['repeats'][0])
    with pytest.raises(ValueError,match='reuse'):pc.resume_windows(corrupt,tmp_path,[p],binding)
    path=tmp_path/result['repeats'][0]['samples_file'];path.write_text('{}')
    with pytest.raises(ValueError,match='checksum'):pc.resume_windows(raw,tmp_path,[p],binding)


@pytest.mark.asyncio
async def test_partial_resume_preserves_existing_windows_and_declares_new_prefill(tmp_path):
    clock,profiler,launches,background=setup(tmp_path);p=point();binding={};saved=[]
    def stop_after_first(repeats):
        saved[:]=repeats
        raise RuntimeError('interrupted after complete window')
    with pytest.raises(RuntimeError,match='interrupted'):
        await pc.collect_power_point(profiler,None,[0,1,2,3],p,model=power_model(),binding=binding,on_window=stop_after_first,
            _clock=clock,_sleep=clock.sleep,_background_factory=background)
    original_bytes=(tmp_path/saved[0]['samples_file']).read_bytes()
    row=await pc.collect_power_point(profiler,None,[0,1,2,3],p,model=power_model(),binding=binding,previous=saved,
        _clock=clock,_sleep=clock.sleep,_background_factory=background)
    assert len(launches)==2 and row['prefill_runs']==2 and len(row['repeats'])==3
    assert row['repeats'][0]==saved[0] and (tmp_path/saved[0]['samples_file']).read_bytes()==original_bytes
    pc.resume_windows(dict(decode=[row]),tmp_path,[p],binding)


@pytest.mark.asyncio
async def test_power_error_preserved_without_retry_and_not_folded_into_timing(tmp_path):
    clock,profiler,launches,background=setup(tmp_path);p=point();binding={};clock.watts=800
    row=await pc.collect_power_point(profiler,None,[0,1,2,3],p,model=power_model(),binding=binding,
        _clock=clock,_sleep=clock.sleep,_background_factory=background)
    audit=pc.audit_power(dict(decode=[row]),tmp_path,[p],power_model(),binding)
    errors=[r for r in audit['failures'] if r['metric']=='independent_power_window']
    assert len(errors)==3 and all(r['relative_error']==pytest.approx(.25) for r in errors)
    assert len(launches)==1
    timing=pc.timing_component(dict(passed=False,timing_max=.05,failures=[
        dict(metric='decode_power',maximum=.25),
        dict(metric='shared_window_prediction_failure',details=[dict(metric='decode_power_max',relative_error=.25)])]))
    assert timing['passed'] and timing['original_calibration_passed'] is False
    bad=pc.timing_component(dict(passed=False,failures=[dict(metric='shared_window_prediction_failure',
        details=[dict(metric='decode_power_max'),dict(metric='decode_timing')])]))
    assert not bad['passed'] and bad['failures'][0]['details']==[dict(metric='decode_timing')]
    unknown=pc.timing_component(dict(passed=False,failures=[dict(metric='unrecognized')]))
    assert not unknown['passed']


def test_reservation_uses_observed_prefill_lead_and_stays_inside_full_domain(tmp_path):
    training=dict(decode=[],kv_capacity_tokens=2441936)
    for f in pc.FREQUENCIES:
        for b in (1,64,128,256):
            for c in (256,1024,4096):
                reps=[]
                for i in range(3):
                    p=tmp_path/f'{f}-{b}-{c}-{i}.json';p.write_text(json.dumps(dict(start_token_counts=[100]*b)))
                    reps.append(dict(samples_file=p.name,samples_sha256=pc.digest(p),step_seconds=.1))
                training['decode'].append(dict(freq_mhz=f,batch=b,context_tokens=c,effective_context_tokens=c+150,repeats=reps))
    points=pc.reserve_points(training,power_model(),tmp_path)
    assert len(points)==24
    for p in points:
        assert p['reservation']['measured_start_count_lead']==100
        assert p['max_tokens']>=100+32+21/(.1*.85)
        lo,hi=p['reservation']['power_context_bounds']
        assert lo<=p['context_tokens']<=p['context_tokens']+p['max_tokens']-1<=hi
        assert p['batch']*(p['context_tokens']+p['max_tokens'])<=.9*training['kv_capacity_tokens']


@pytest.mark.asyncio
async def test_outside_power_coverage_keeps_failed_raw_window_and_cancels(tmp_path):
    clock,profiler,launches,background=setup(tmp_path);p=point();m=power_model();saved=[]
    for spec in m.decode_power_overrides.values():
        spec['nodes']=[dict(batch=4,context_min=256,context_max=1050,power_w=600.)]
    with pytest.raises(PowerCoverageError,match='escaped'):
        await pc.collect_power_point(profiler,None,[0,1,2,3],p,model=m,binding={},on_window=lambda r:saved.extend(r),
            _clock=clock,_sleep=clock.sleep,_background_factory=background)
    assert len(saved)==1 and (tmp_path/saved[0]['samples_file']).is_file()
    assert len(launches)==1


@pytest.mark.parametrize('with_timing', [False, True])
def test_runner_loads_once_only_samples_power_and_preserves_failed_original(tmp_path,monkeypatch,with_timing):
    from contextlib import nullcontext
    from pdblend.profile import profiler as profiler_module,wave as wave_module
    from pdblend.engine import launcher,client as client_module
    clock,_,launches,background=setup(tmp_path)
    events=[];m=power_model();base=copy.deepcopy(m);base.decode_power_overrides={}
    package=tmp_path/'package';package.mkdir();original=tmp_path/'original';original.mkdir();out=tmp_path/'out'
    base.save(original/'base.json')
    (original/'raw.json').write_text('{}')
    (original/'manifest.json').write_text(json.dumps(dict(plan=dict(prefill=[],decode=[]))))
    (original/'completion.json').write_text(json.dumps(dict(complete=True,calibration_status='failed')))
    original_before={p.name:p.read_bytes() for p in original.iterdir()}
    env=dict(image_digest='image',vllm='vllm',torch='torch',cuda='cuda',hardware_id='8xL20-lease',source_hash='new-source',gpu_uuids=['g0','g1','g2','g3'])
    inputs={k:dict(path=str(original/v),sha256=pc.digest(original/v)) for k,v in
        [('base_candidate','base.json'),('original_raw','raw.json'),('original_manifest','manifest.json'),('original_completion','completion.json')]}
    manifest=dict(candidate_sha256='candidate',plan_sha256='plan',model_hash='model',tokenizer_hash='tok',
        model_id='Qwen2.5-32B-Instruct',
        original_timing_environment=env,inputs=inputs,implementation_sha256={'file':'hash'})
    plan=dict(points=[point(f,b) for f in pc.FREQUENCIES for b in (1,64,192,256)])
    (package/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(pc,'load_package',lambda p:(manifest,plan,m))
    monkeypatch.setattr(pc,'evaluate_holdout',lambda *a,**kw:dict(passed=False,timing_max=.02,failures=[dict(metric='decode_power')]))
    class Profiler:
        def __init__(self,model,gpus,**kw):
            self.out_dir=kw['out_dir'];self.out_dir.mkdir();self.tp=4
            self.raw=dict(config={},decode=[],prefill=[],mixed=[],environment=env)
            self.model_spec=SimpleNamespace(model_hash='model',tokenizer_hash='tok')
            self.specs=[SimpleNamespace(instance_id='p',base_url='http://unused',gpus=gpus)]
            self.meter=SimpleNamespace(sampler=lambda g:Sampler(clock),reset_all=lambda:events.append('reset'))
        def _checkpoint(self): (self.out_dir/'raw.json').write_text(json.dumps(self.raw))
        def _kv_capacity(self,i):return 2441936
        def _lock(self,f,gpus):clock.freq=f;events.append(('clock',f,tuple(gpus)))
        def _prefill(self,*a,**kw):raise AssertionError('no prefill grid')
        def _mixed(self,*a,**kw):raise AssertionError('no mixed grid')
    class Fleet:
        def __init__(self,specs,logs):self.specs=specs
        def __enter__(self):return self
        def __exit__(self,*a):events.append('stop')
        def start_all(self):events.append('load')
        def __getitem__(self,k):return SimpleNamespace(spec=self.specs[0])
    class Client:
        def __init__(self,*a):pass
        async def __aenter__(self):events.append('client_open');return self
        async def __aexit__(self,*a):events.append('client_close')
    class Wave:
        @classmethod
        def from_environment(cls):return cls()
        async def qualify_external(self,p,f):events.append('qualify')
        @asynccontextmanager
        async def measurement(self):
            events.append('measurement_begin');yield;events.append('measurement_done')
        def write(self,*a):events.append(('wave',a))
    original_collect=pc.collect_power_point
    async def collect(*a,**kw):
        return await original_collect(*a,**kw,_clock=clock,_sleep=clock.sleep,_background_factory=background)
    monkeypatch.setattr(profiler_module,'Profiler',Profiler);monkeypatch.setattr(profiler_module,'_load_flock',nullcontext)
    monkeypatch.setattr(launcher,'Fleet',Fleet);monkeypatch.setattr(client_module,'EngineClient',Client)
    monkeypatch.setattr(wave_module,'ProfileWave',Wave);monkeypatch.setattr(pc,'collect_power_point',collect)
    monkeypatch.setattr(pc.asyncio,'sleep',clock.sleep)
    timing_package=None
    if with_timing:
        from pdblend.profile import timing_calibration
        timing_package=tmp_path/'timing-package';timing_package.mkdir()
        (timing_package/'manifest.json').write_text(json.dumps(dict(model_id='Qwen2.5-32B-Instruct',tp=4,pp=1)))
        for name in ('candidate.json','timing-plan.json'):(timing_package/name).write_text('{}')
        async def timing_collect(*,profiler,client,gpus,package,out):
            assert len(profiler.raw['decode'])==24
            assert 'stop' not in events and 'measurement_done' not in events and 'client_close' not in events
            assert gpus==[0,1,2,3] and package==timing_package
            events.append('timing_panel')
            out.mkdir();value=dict(complete=True,timing_passed=False,formal_eligible=False,energy_comparable=False)
            (out/'completion.json').write_text(json.dumps(value))
            return dict(value,receipt_sha256=pc.digest(out/'completion.json'))
        monkeypatch.setattr(timing_calibration,'collect_existing',timing_collect)
    result=pc.run(package=package,model_path='model',gpus=[0,1,2,3],base_port=12000,out=out,
                  timing_package=timing_package)
    assert result['complete'],result
    assert result['power_passed'] and result['reused_timing_passed'] and result['calibration_components_passed']
    assert result['formal_eligible'] is False and result['measured_decode_points']==24
    assert events.count('load')==1 and events.count('stop')==1 and events.count('qualify')==1
    assert len(launches)==24 and len([e for e in events if isinstance(e,tuple) and e[0]=='clock'])==6
    raw=json.loads((out/'raw.json').read_text())
    assert len(raw['decode'])==24 and sum(len(r['repeats']) for r in raw['decode'])==72
    assert raw['prefill']==raw['mixed']==[]
    assert {p.name:p.read_bytes() for p in original.iterdir()}==original_before
    audit=json.loads((out/'composite-audit.json').read_text())
    assert audit['old_completion_unchanged'] and audit['original_power_retained_as_diagnostic']
    assert audit['timing']['original_calibration_passed'] is False
    if with_timing:
        assert result['timing_overlay']['complete'] and result['timing_overlay']['timing_passed'] is False
        assert events.index('timing_panel')<events.index('client_close')<events.index('measurement_done')<events.index('stop')<events.index('reset')
    else:
        assert result['timing_overlay_requested'] is False and 'timing_overlay' not in result
