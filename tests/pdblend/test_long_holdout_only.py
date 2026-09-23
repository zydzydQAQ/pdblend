import copy
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

# Integration checks require the experiment machine's immutable raw artifacts.
pytestmark = pytest.mark.historical

from pdblend.profile import long_holdout_only as hold, long_context_collect as lc, long_context_followup as lf
from pdblend.profile.profiler import Profiler

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def package(tmp_path):
    report=ROOT/'results/2026-09-23/incremental-wave-closeout-v1/audit.json'
    if not report.exists():pytest.skip('campaign training fixture unavailable')
    training=Path(json.loads(report.read_text())['members']['14b-tp1-longctx']['root'])
    path=tmp_path/'package';hold.prepare(training=training,out=path)
    return path,training


def test_real_training_reused_and_plan_all_24_points(package):
    path,_=package;m,c,p=hold.load_package(path)
    assert m['new_training_points']==0 and m['reused_training_points']==36
    assert len(p['points'])==24 and p['missing_points']==[]
    assert {(x['freq_mhz'],x['batch'],x['long_context_role']) for x in p['points']}=={
        (f,b,role) for f in lf.FREQUENCIES for b in (1,4) for role in ('interior','campaign_endpoint')}
    assert c['holdout_used'] is False
    for b in (2,3):
        with pytest.raises(ValueError):lf.predict(c,'step_seconds',900,b,6144)


def test_holdout_labels_or_wrong_model_cannot_train(package):
    _,training=package;raw=json.loads((training/'raw.json').read_text())
    bad=copy.deepcopy(raw);bad['decode'][0]['independent_holdout']=True
    with pytest.raises(ValueError,match='holdout'):hold.fit_completed_training(bad)
    bad=copy.deepcopy(raw);bad['model_id']='Qwen2.5-7B-Instruct'
    with pytest.raises(ValueError,match='14B'):hold.fit_completed_training(bad)


def test_prepared_candidate_tampering_rejected(package):
    path,_=package;(path/'candidate.json').write_text('{}')
    with pytest.raises(ValueError,match='candidate/plan changed'):hold.load_package(path)


class Clock:
    def __init__(self,candidate):self.now=1000.;self.live=[];self.frequency=900;self.candidate=candidate
    def __call__(self):return self.now
    async def sleep(self,seconds):
        self.now+=seconds
        for r in self.live:
            while r.token_times_s[-1]+self.step<=self.now+1e-8:r.token_times_s.append(r.token_times_s[-1]+self.step)
    @asynccontextmanager
    async def background(self,profiler,client,point,tag):
        self.step=lf.predict(self.candidate,'step_seconds',point['freq_mhz'],point['batch'],point['context_tokens']+120)
        self.power=lf.predict(self.candidate,'power_w',point['freq_mhz'],point['batch'],point['context_tokens']+120)
        self.live=[SimpleNamespace(token_times_s=[self.now-16*self.step+i*self.step for i in range(17)],error=None)
                   for _ in range(point['batch'])]
        try:yield self.live,[]
        finally:self.live=[]


class Sampler:
    error=None
    def __init__(self,clock):self.clock=clock
    def start(self):self.begin=self.clock()
    def stop(self):
        self.samples=[(self.begin+.5,[self.clock.power]),(self.clock(),[self.clock.power])]
        self.frequency_samples=[(self.begin+1,[self.clock.frequency])]


@pytest.mark.asyncio
async def test_resident_callback_real_window_contract_and_resume(package,tmp_path,monkeypatch):
    path,training=package;_,candidate,_=hold.load_package(path);clock=Clock(candidate)
    p=Profiler.__new__(Profiler);p.out_dir=tmp_path/'parent';p.out_dir.mkdir();p.tp=1;p.parallel_layout={};p.specs=[SimpleNamespace()]
    p.raw=json.loads((training/'raw.json').read_text());binding=p.raw['external_interference']
    dest=p.out_dir/binding['samples_file'];dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes((training/binding['samples_file']).read_bytes())
    p.meter=SimpleNamespace(sampler=lambda g:Sampler(clock));p._lock=lambda f,g:setattr(clock,'frequency',f)
    before=copy.deepcopy(p.raw);original=lc.collect_bounded_decode_point;calls=[]
    async def collect(*args,**kwargs):
        calls.append(args[3]);return await original(*args,**kwargs,_clock=clock,_sleep=clock.sleep,_background_factory=clock.background)
    monkeypatch.setattr(lc,'collect_bounded_decode_point',collect);monkeypatch.setattr(hold.asyncio,'sleep',clock.sleep)
    result=await hold.run_existing(package=path,profiler=p,client=None,gpus=[0],out=tmp_path/'holdout')
    assert result['complete'] is True and result['calibration_passed'] is True
    assert result['measured_points']==24 and len(calls)==24 and p.raw==before
    # Replay never gathers another sample; all 72 raw windows remain valid.
    # Use a minimal resume that loads the actual generated archive; production
    # Profiler.resume also checks model/environment identity before this point.
    monkeypatch.setattr(Profiler,'resume',lambda self:setattr(self,'raw',json.loads((self.out_dir/'raw.json').read_text())))
    again=await hold.run_existing(package=path,profiler=p,client=None,gpus=[0],out=tmp_path/'holdout')
    assert again['complete'] is True and len(calls)==24 and p.raw==before
