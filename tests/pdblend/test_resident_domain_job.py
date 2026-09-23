from pathlib import Path
from types import SimpleNamespace

import pytest

from pdblend.profile import resident_domain_job as job
from pdblend.profile import profiler as profiler_module, sampling_epochs as epochs_module
from pdblend.engine import launcher, client as client_module

IDENTITY=dict(system='pdblend',model_id='Qwen2.5-14B-Instruct',model_hash='m',tokenizer_hash='t',tp=1,pp=1)


def install(monkeypatch, events):
    monkeypatch.setattr(job,'package_identity',lambda **k:dict(IDENTITY))
    class Profiler:
        def __init__(self,*args,**kwargs):
            self.raw=dict(IDENTITY);self.out_dir=kwargs['out_dir']
            self.specs=[SimpleNamespace(instance_id='m',base_url='http://localhost')]
            self.meter=SimpleNamespace(reset_all=lambda:events.append('clock_reset'))
        def _kv_capacity(self,instance):return 100000
        def _checkpoint(self):events.append('checkpoint')
    class Fleet:
        def __init__(self,specs,*args):self.instance=SimpleNamespace(spec=specs[0])
        def __enter__(self):events.append('enter');return self
        def start_all(self):events.append('load')
        def __getitem__(self,i):return self.instance
        def __exit__(self,*args):events.append('stop')
    class Epoch:
        def __init__(self,*args):pass
        async def ready(self):events.append('ready')
        async def window_boundary(self,*args):events.append('boundary')
        def qualification_guard(self):return {}
        async def retire(self):events.append('retire')
        def released(self):events.append('released')
        def fail(self,exc):events.append('fail')
    class Client:
        def __init__(self,*args):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):events.append('client_close')
    monkeypatch.setattr(profiler_module,'Profiler',Profiler)
    monkeypatch.setattr(launcher,'Fleet',Fleet)
    monkeypatch.setattr(epochs_module,'SamplingEpochs',Epoch)
    monkeypatch.setattr(client_module,'EngineClient',Client)


def test_same_resident_load_stages_then_retire_cleanup_release(tmp_path,monkeypatch):
    events=[];install(monkeypatch,events)
    async def long(**kwargs):
        assert callable(kwargs['window_boundary']) and callable(kwargs['qualification_guard'])
        events.append('long');return dict(complete=True,calibration_passed=True)
    async def short(**kwargs):
        events.append('short');return dict(complete=True,formal_eligible=False,pure_decode_power_qualified=False)
    monkeypatch.setattr(job.resident_long_holdout,'run_existing',long)
    monkeypatch.setattr(job.short_domain_collect,'run_existing',short)
    r=job.run(model='m',gpus=[0],base_port=1234,out=tmp_path,epochs_root=tmp_path,member='p',
        short_package=Path('s'),long_package=Path('l'))
    assert r['complete'] and not r['full_profile_qualified'] and not r['formal_eligible']
    assert events==['enter','load','ready','long','short','client_close','retire','stop','clock_reset','released','checkpoint']


def test_failure_invalidates_cohort_before_unload_and_never_releases(tmp_path,monkeypatch):
    events=[];install(monkeypatch,events)
    async def failed(**kwargs):raise ValueError('window mismatch')
    monkeypatch.setattr(job.short_domain_collect,'run_existing',failed)
    r=job.run(model='m',gpus=[0],base_port=1234,out=tmp_path,epochs_root=tmp_path,member='p',short_package=Path('s'))
    assert r['status']=='failed' and not r['complete']
    assert events.index('fail')<events.index('stop') and 'released' not in events
    assert events.count('clock_reset')==1


def test_incomplete_measurement_does_not_become_passed(tmp_path,monkeypatch):
    events=[];install(monkeypatch,events)
    async def incomplete(**kwargs):return dict(complete=False)
    monkeypatch.setattr(job.short_domain_collect,'run_existing',incomplete)
    r=job.run(model='m',gpus=[0],base_port=1234,out=tmp_path,epochs_root=tmp_path,member='p',short_package=Path('s'))
    assert r['status']=='inconclusive' and not r['complete'] and r['epoch_cleanup_released']
