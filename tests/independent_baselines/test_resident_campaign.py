import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from pdblend_baselines import resident_campaign as rc
from pdblend.results.power_archive import read_power_archive

MODEL = 'Qwen2.5-7B-Instruct'


def test_specs_enforce_two_symmetric_groups_and_memory_legal_tp():
    specs = rc.build_specs(MODEL, [0, 1], 1, 19000)
    assert [s.base_url for s in specs] == ['http://127.0.0.1:19000', 'http://127.0.0.1:19016']
    assert [s.gpus for s in specs] == [(0,), (1,)]
    for model, gpus, tp in [(MODEL, [0], 1), (MODEL, [-1,0], 1), ('Qwen2.5-32B-Instruct',[0,1],1)]:
        with pytest.raises(ValueError): rc.build_specs(model, gpus, tp, 19000)


@pytest.mark.parametrize('cleanup_failure', [False, True])
def test_owned_pair_starts_once_no_intermediate_unload_and_archives_power(tmp_path, monkeypatch, cleanup_failure):
    events = []
    class Instance:
        def __init__(self, iid): self.iid = iid
        def start(self): events.append(('start', self.iid))
        def wait_ready(self, timeout_s): events.append(('ready', self.iid)); return .1
        def stop(self):
            events.append(('stop', self.iid))
            if cleanup_failure and self.iid == 'resident-P': raise RuntimeError('stop failed')
    class Fleet:
        def __init__(self, specs, logs): self.instances = {s.instance_id:Instance(s.instance_id) for s in specs}
        def __getitem__(self, key): return self.instances[key]
    class Sampler:
        samples = [(1., [100.,100.]), (2., [110.,110.])]
        frequency_samples = [(1., [2520,2520])]
        utilization_samples, power_metadata = [], []
        power_source, error = dict(mode='instant'), None
        def start(self): events.append(('sampler','start'))
        def stop(self): events.append(('sampler','stop'))
        def total_energy_j(self): return 210.
    class Meter:
        def __init__(self, gpus, power_mode): self.gpus = gpus; assert power_mode == 'instant'
        def sampler(self, interval_s): return Sampler()
        def unpark(self, gpu): events.append(('unpark',gpu))
        def reset_clock(self, gpu): events.append(('reset',gpu))
    monkeypatch.setattr(rc,'Fleet',Fleet);monkeypatch.setattr(rc,'Gpus',Meter)
    async def verified(specs): return {}
    async def warm(specs,label): events.append(('warm',label));return []
    async def drained(specs): events.append(('drain',len(specs)));return []
    monkeypatch.setattr(rc,'verify_endpoints',verified);monkeypatch.setattr(rc,'warmup_endpoints',warm)
    monkeypatch.setattr(rc,'drain_endpoints',drained)
    def save(out,system):
        out.mkdir();value=dict(status='passed', complete=True, system=system, cleanup_errors=[])
        (out/'completion.json').write_text(json.dumps(value));return value
    async def dist(args): events.append(('system','dist'));return save(args.out,'distserve')
    async def eco(config,endpoints,trace,out,duration):
        assert all(s['tp']==1 and s['pp']==1 for s in config['instances'])
        assert not any(event[0]=='stop' for event in events)
        events.append(('system','eco'));return save(out,'ecoserve')
    trace=tmp_path/'trace.json';trace.write_text('{}')
    profile=tmp_path/'eco.csv';profile.write_text('placeholder')
    result=asyncio.run(rc.execute(model=MODEL,tp=1,gpus=[0,1],base_port=19000,trace=trace,
        eco_profile=profile,out=tmp_path/'run',duration=.01,own_services=True,
        dist_execute_fn=dist,eco_execute_fn=eco))
    assert result['complete'] is (not cleanup_failure)
    assert result['engine_loads']==2 and result['engine_load_cycles']==1
    for iid in ('resident-P','resident-D'):
        assert events.count(('start',iid))==events.count(('stop',iid))==1
    assert events.index(('system','eco')) < events.index(('stop','resident-P'))
    assert result['energy_samples']==2 and result['energy_j']==210.
    assert len(result['power_artifact_sha256'])==64
    assert read_power_archive(tmp_path/'run/power.json')['frequency_samples']
    if cleanup_failure: assert result['status']=='failed' and result['cleanup_errors']


def test_existing_output_is_never_overwritten_or_started(tmp_path,monkeypatch):
    out=tmp_path/'run';out.mkdir();(out/'completion.json').write_text('original')
    trace=tmp_path/'trace.json';trace.write_text('{}')
    monkeypatch.setattr(rc,'Fleet',lambda *args:pytest.fail('must not start old output'))
    with pytest.raises(FileExistsError):
        asyncio.run(rc.execute(model=MODEL,tp=1,gpus=[0,1],base_port=19000,trace=trace,
            eco_profile=tmp_path/'missing',out=out,own_services=True))
    assert (out/'completion.json').read_text()=='original'
