import copy
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
import pytest

# Integration checks require the experiment machine's immutable raw artifacts.
pytestmark = pytest.mark.historical

from pdblend.profile import resident_long_holdout as resident,long_holdout_only as hold,long_context_collect as lc,long_context_followup as lf
from pdblend.profile.profiler import Profiler

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
async def test_real_plan_every_window_guarded_and_epoch_change_not_accepted(tmp_path,monkeypatch):
    training=Path(json.loads((ROOT/'results/2026-09-23/incremental-wave-closeout-v1/audit.json').read_text())['members']['14b-tp1-longctx']['root'])
    package=tmp_path/'package';hold.prepare(training=training,out=package);_,candidate,plan=hold.load_package(package)
    p=Profiler.__new__(Profiler);p.out_dir=tmp_path/'parent';p.out_dir.mkdir();p.tp=1;p.parallel_layout={};p.specs=[SimpleNamespace()]
    p.raw=json.loads((training/'raw.json').read_text());binding=p.raw['external_interference']
    dest=p.out_dir/binding['samples_file'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((training/binding['samples_file']).read_bytes())
    now=[1000.];state=dict(live=[],epoch='a',count=0,fail=False)
    async def sleep(seconds):
        now[0]+=seconds
        for r in state['live']:
            while r.token_times_s[-1]+state['step']<=now[0]+1e-8:r.token_times_s.append(r.token_times_s[-1]+state['step'])
    @asynccontextmanager
    async def background(profiler,client,point,tag):
        state['step']=lf.predict(candidate,'step_seconds',point['freq_mhz'],point['batch'],point['context_tokens']+120)
        state['power']=lf.predict(candidate,'power_w',point['freq_mhz'],point['batch'],point['context_tokens']+120)
        state['live']=[SimpleNamespace(token_times_s=[now[0]-16*state['step']+i*state['step'] for i in range(17)],error=None) for _ in range(point['batch'])]
        try:yield state['live'],[]
        finally:
            state['live']=[]
            if state['fail'] and state['count']==2:state['epoch']='changed'
    class Sampler:
        error=None
        def start(self):self.start_s=now[0]
        def stop(self):
            self.samples=[(self.start_s+.5,[state['power']]),(now[0],[state['power']])]
            self.frequency_samples=[(self.start_s+1,[state['frequency']])]
    p.meter=SimpleNamespace(sampler=lambda g:Sampler());p._lock=lambda f,g:state.update(frequency=f)
    def qualifier():return dict(epoch_id=state['epoch'],qualification_path=str(dest),qualification_sha256=lc.digest(dest),layout_sha256='layout')
    async def boundary(point,repeat,phase):state['count']+=1
    actual=lc.collect_bounded_decode_point
    async def collect(*args,**kwargs):return await actual(*args,**kwargs,_clock=lambda:now[0],_sleep=sleep)
    monkeypatch.setattr(lc,'_background',background);monkeypatch.setattr(lc,'collect_bounded_decode_point',collect)
    before=copy.deepcopy(p.raw)
    result=await resident.run_existing(package=package,profiler=p,client=None,gpus=[0],out=tmp_path/'good',window_boundary=boundary,qualification_guard=qualifier)
    assert result['complete'] is True and result['calibration_passed'] is True and state['count']==72
    assert p.raw==before
    state.update(count=0,fail=True)
    with pytest.raises(ValueError,match='epoch changed'):
        await resident.run_existing(package=package,profiler=p,client=None,gpus=[0],out=tmp_path/'bad',window_boundary=boundary,qualification_guard=qualifier)
    raw=json.loads((tmp_path/'bad/raw.json').read_text())
    assert len(raw['decode'])==0 and sum(map(len,raw['decode_pending'].values()))==1
    assert p.raw==before
