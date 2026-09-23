import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

# Integration checks require the experiment machine's immutable raw artifacts.
pytestmark = pytest.mark.historical

from pdblend.profile import short_domain as sd,short_domain_collect as collect,sampling_guard as guard

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def package(tmp_path):
    report=json.loads((ROOT/'results/2026-09-23/incremental-wave-closeout-v1/audit.json').read_text())
    training=Path(report['members']['7b-tp1-longctx']['root'])
    path=tmp_path/'package'
    collect.prepare(base_candidate=ROOT/'results/2026-09-22/three-model/calibration-candidates/7b-tp1-4ddc49563cef6321c0b5/candidate.json',
        dataset_manifest=ROOT/'datasets/prepared/2026-09-22-7b-v1/manifest.json',identity_raw=training/'raw.json',out=path)
    return path,training


def test_real_package_matrix_and_source_identity(package):
    path,_=package;m,p=collect.load_package(path)
    assert len(p['training'])==30 and len(p['holdout'])==36
    assert p['experimental_sampling'] is True and m['pure_decode_power_qualified'] is False
    assert {(x['freq_mhz'],x['batch']) for x in p['training']}=={(f,1) for f in sd.FREQUENCIES}
    (path/'plan.json').write_text('{}')
    with pytest.raises(ValueError,match='immutable'):collect.load_package(path)


class Clock:
    def __init__(self):self.now=1000.;self.frequency=900;self.power=100.
    def __call__(self):return self.now


class Client:
    def __init__(self,clock):self.clock=clock;self.calls=0
    async def complete(self,prompt,output,request_id,*,token_diagnostics):
        assert token_diagnostics
        self.calls+=1;c=self.clock;n=len(prompt);submitted=c()
        first=submitted+.01+n*.00001
        times=[first]
        for index in range(1,output):times.append(times[-1]+.02+(n+index)*.000001)
        c.now=times[-1]+.001;c.power=100+n*.01
        return SimpleNamespace(request_id=request_id,submitted_s=submitted,first_token_s=first,finished_s=c(),
            token_times_s=times,completion_tokens=output,prompt_tokens=n,error=None,stream_done=True,usage_received=True)


class Sampler:
    error=None
    def __init__(self,c):self.c=c
    def start(self):self.begin=self.c()
    def stop(self):
        self.samples=[(self.begin+i*.05,[self.c.power]) for i in range(int((self.c()-self.begin)/.05)+1)]
        self.frequency_samples=[(self.begin+.01,[self.c.frequency])]


def profiler(path,training,clock):
    path.mkdir();q=path/'qualifier.json';q.write_text(json.dumps(dict(complete=True,cross_job=True,passed=True)))
    raw=json.loads((training/'raw.json').read_text())
    raw['external_interference']=dict(samples_file=q.name,samples_sha256=collect.digest(q))
    return SimpleNamespace(raw=raw,out_dir=path,meter=SimpleNamespace(sampler=lambda g:Sampler(clock)),
        _lock=lambda f,g:setattr(clock,'frequency',f))


@pytest.mark.asyncio
async def test_live_request_windows_fit_before_holdout_and_resume(package,tmp_path,monkeypatch):
    path,training=package;c=Clock();p=profiler(tmp_path/'parent',training,c);client=Client(c);before=copy.deepcopy(p.raw)
    actual=collect.measure_repeat;boundaries=[]
    async def measure(*args,**kwargs):return await actual(*args,**kwargs,_clock=c)
    async def boundary(point,repeat,phase):boundaries.append((phase,repeat,point['input_tokens']))
    monkeypatch.setattr(collect,'measure_repeat',measure)
    result=await collect.run_existing(package=path,profiler=p,client=client,gpus=[0],out=tmp_path/'out',window_boundary=boundary)
    assert result['complete'] is True and result['experimental_components_passed'] is True
    assert result['pure_decode_power_qualified'] is False and result['formal_eligible'] is False
    assert len(boundaries)==126 and p.raw==before
    raw=json.loads((tmp_path/'out/raw.json').read_text())
    first=next(iter(raw['holdout'].values()))['repeats'][0]
    evidence=json.loads((tmp_path/'out'/first['samples_file']).read_text())
    assert evidence['candidate_sha256']==result['candidate_sha256'] and evidence['warmup_requests']
    calls=client.calls
    again=await collect.run_existing(package=path,profiler=p,client=client,gpus=[0],out=tmp_path/'out')
    assert again['complete'] and client.calls==calls


@pytest.mark.asyncio
async def test_epoch_change_rejects_window_before_checkpoint(package,tmp_path,monkeypatch):
    path,training=package;c=Clock();p=profiler(tmp_path/'parent',training,c);client=Client(c);actual=collect.measure_repeat
    original=guard.static_guard(p);epoch=['a']
    def qualifier():return dict(original(),epoch_id=epoch[0])
    async def measure(*args,**kwargs):
        value=await actual(*args,**kwargs,_clock=c);epoch[0]='b';return value
    monkeypatch.setattr(collect,'measure_repeat',measure)
    with pytest.raises(ValueError,match='epoch changed'):
        await collect.run_existing(package=path,profiler=p,client=client,gpus=[0],out=tmp_path/'out',qualification_guard=qualifier)
    raw=json.loads((tmp_path/'out/raw.json').read_text())
    assert sum(len(x['repeats']) for x in raw['training'].values())==0
    assert json.loads((tmp_path/'out/completion.json').read_text())['complete'] is False
