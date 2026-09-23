"""Real archive validation and runner integration without GPU execution."""
import copy
import json
from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace

import pytest

from pdblend.profile import calibration as c
from pdblend.profile.holdout_inheritance import load_prior_holdout, merge_holdout_rows
from pdblend.profile.identity import sha256_value


def sample(root, name, value):
    path = root/name
    path.write_text(json.dumps(value))
    return dict(samples_file=name, samples_sha256=c.digest(path))


def decode(root, batch):
    reps = [dict(effective_context_tokens=200, step_seconds=.2, power_w=200,
        steady_window_s=5, min_steps=10, power_samples=2, frequency_samples=1,
        **sample(root, f'b{batch}-{i}.json', dict(power=[[1,[200]],[6,[200]]], frequency=[[1,[1500]]]))) for i in range(3)]
    return dict(freq_mhz=1500, batch=batch, context_tokens=64, effective_context_tokens=200,
                step_seconds=.2, power_w=200, repeats=reps)


def publish(root, raw):
    raw['identity_sha256'] = sha256_value({k:v for k,v in raw.items() if k != 'identity_sha256'})
    (root/'raw.json').write_text(json.dumps(raw))
    return c.digest(root/'raw.json')


def fixture(tmp_path):
    root=tmp_path/'old'; root.mkdir()
    manifest=dict(system='pdblend', model_id='Qwen2.5-14B-Instruct', model_hash='m'*64,
        tokenizer_hash='t'*64, tp=4, pp=1, candidate_sha256='c'*64,
        plan=dict(prefill=[dict(freq_mhz=1500,input_tokens=128)],
            decode=[dict(freq_mhz=1500,batch=b,context_tokens=64,unseen_shape=False) for b in (4,8)], repeats=3))
    raw={k:copy.deepcopy(v) for k,v in manifest.items() if k not in ('plan','candidate_sha256')}
    raw.update(profile_key=dict(system='pdblend',tp=4,pp=1,engine_revision='vllm-0.10.1.1'),
        environment=dict(image_digest='sha256:'+'i'*64, vllm='0.10.1.1', torch='2.7.1', cuda='12.8.1',
            hardware_id='8xL20-lease', gpu_uuids=['GPU-'+str(x) for x in range(4)], source_hash='a'*64),
        holdout_candidate_sha256='c'*64, config={},
        prefill=[dict(freq_mhz=1500,input_tokens=128,seconds=.1,**sample(root,'prefill.json',{}))],
        decode=[decode(root,4)], mixed=[])
    (root/'frozen-fit.json').write_text(json.dumps(manifest))
    sha=publish(root,raw)
    current=copy.deepcopy(raw); current['environment']['source_hash']='b'*64
    current.update(prefill=[],decode=[],mixed=[])
    return root,raw,manifest,current,sha


class Model:
    tp=4
    freqs=(1500,)
    def prefill_seconds(self,n,f): return .1
    def step_seconds(self,b,ctx,f): return .2
    def decode_supported(self,b,ctx,f): return f==1500 and ctx<1000
    def decode_power_w(self,b,f,*,ctx=None): return 200


def test_partial_checkpoint_keeps_sources_and_cannot_pass_missing_matrix(tmp_path):
    root,raw,manifest,current,sha=fixture(tmp_path)
    before={p.name:p.read_bytes() for p in root.iterdir()}
    prior,points,receipt=load_prior_holdout(root,sha,manifest,current)
    assert points==dict(prefill={(1500,128)},decode={(1500,4,64)})
    assert receipt['source_revision_changed'] and not receipt['complete'] and 'prior_completion' not in receipt
    view,roots=merge_holdout_rows(current,prior,receipt,expected_plan=manifest['plan'])
    audit=c.evaluate_holdout(view,Model(),tmp_path,expected_plan=manifest['plan'],evidence_roots=roots)
    assert not audit['passed']
    assert any(r['metric']=='decode_matrix_coverage' for r in audit['failures'])
    assert audit['points'][0]['evidence_source']==receipt['binding']['evidence_source']
    with pytest.raises(ValueError,match='incomplete'):
        merge_holdout_rows(current,prior,receipt,expected_plan=manifest['plan'],require_complete=True)
    assert current['decode']==[] and 'evidence_source' not in prior['decode'][0]
    assert {p.name:p.read_bytes() for p in root.iterdir()}==before


@pytest.mark.parametrize('fault', ['raw_sha','internal_sha','sample','candidate','frozen_fit','uuid','model','engine','repeats','duplicate','source'])
def test_inheritance_rejects_changed_or_incomplete_evidence(tmp_path,fault):
    root,raw,manifest,current,sha=fixture(tmp_path)
    if fault=='raw_sha': sha='0'*64
    elif fault=='internal_sha': raw['identity_sha256']='0'*64; (root/'raw.json').write_text(json.dumps(raw));sha=c.digest(root/'raw.json')
    elif fault=='sample': (root/'b4-0.json').write_text('{}')
    elif fault=='candidate': raw['holdout_candidate_sha256']='x'*64;sha=publish(root,raw)
    elif fault=='frozen_fit': (root/'frozen-fit.json').write_text('{}')
    elif fault=='uuid': current['environment']['gpu_uuids'][0]='GPU-other'
    elif fault=='model': current['model_hash']='x'*64
    elif fault=='engine': current['environment']['vllm']='other'
    elif fault=='repeats': raw['decode'][0]['repeats'].pop();sha=publish(root,raw)
    elif fault=='duplicate':raw['decode']*=2;sha=publish(root,raw)
    elif fault=='source':raw['decode'][0]['evidence_source']='other';sha=publish(root,raw)
    with pytest.raises(ValueError):load_prior_holdout(root,sha,manifest,current)


def test_reused_and_new_rows_may_not_overlap_or_relabel(tmp_path):
    root,raw,manifest,current,sha=fixture(tmp_path)
    _,_,receipt=load_prior_holdout(root,sha,manifest,current)
    current['decode']=copy.deepcopy(raw['decode'])
    with pytest.raises(ValueError,match='duplicate'):merge_holdout_rows(current,raw,receipt,expected_plan=manifest['plan'])
    current['decode'][0]['evidence_source']='old'
    with pytest.raises(ValueError,match='only its own'):merge_holdout_rows(current,raw,receipt,expected_plan=manifest['plan'])


def test_full_runner_skips_old_points_and_preserves_immutable_archives(tmp_path,monkeypatch):
    from pdblend.profile import profiler as pm, wave as wm
    from pdblend.engine import launcher, client as cm
    root,raw,manifest,current,sha=fixture(tmp_path)
    candidate=tmp_path/'candidate';candidate.mkdir();(candidate/'candidate.json').write_text('{}')
    manifest['candidate_sha256']=c.digest(candidate/'candidate.json')
    raw['holdout_candidate_sha256']=manifest['candidate_sha256']
    (candidate/'manifest.json').write_text(json.dumps(manifest));(root/'frozen-fit.json').write_text(json.dumps(manifest))
    sha=publish(root,raw); before={p.name:p.read_bytes() for p in root.iterdir()}
    events=[]
    class Profiler:
        def __init__(self,model,gpus,**kw):
            self.out_dir=kw['out_dir'];self.out_dir.mkdir();self.raw=copy.deepcopy(current)
            self.model_spec=SimpleNamespace(model_hash='m'*64,tokenizer_hash='t'*64)
            self.specs=[SimpleNamespace(instance_id='p',base_url='http://unused',gpus=tuple(gpus))]
            self.meter=SimpleNamespace(reset_all=lambda:events.append('reset'))
        def _kv_capacity(self,inst):return 32768
        def _lock(self,f,g):pass
        def _checkpoint(self):publish(self.out_dir,self.raw)
        async def _decode_batch(self,client,gpus,b,ctx,n,tag):
            events.append(('decode',b));assert b==8
            return decode(self.out_dir,b)
        async def _mixed(self,*args,**kwargs):
            events.append('mixed')
            reference=kwargs['reference_raw']
            assert len(reference['decode'])==2 and len(reference['prefill'])==1
            assert len(self.raw['decode'])==1 and not self.raw['prefill']
            self.raw['mixed']=[dict(valid=True,base_step_s=.1,alone_prefill_s=.1,probe_ttft_s=.2,
                **sample(self.out_dir,'mixed.json',{})) for _ in range(12)]
    class Fleet:
        def __init__(self,specs,logs):self.specs=specs
        def __enter__(self):return self
        def __exit__(self,*args):events.append('stop')
        def start_all(self):events.append('load')
        def __getitem__(self,key):return SimpleNamespace(spec=self.specs[0])
    class Client:
        def __init__(self,*a,**kw):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*a):pass
        async def complete(self,*a,**kw):raise AssertionError('old prefill must never rerun')
    class Wave:
        async def qualify_external(self,*a):events.append('qualified')
        @asynccontextmanager
        async def measurement(self):yield
        def write(self,*a):pass
    async def no_sleep(*a):pass
    monkeypatch.setattr(pm,'Profiler',Profiler);monkeypatch.setattr(pm,'_load_flock',nullcontext)
    monkeypatch.setattr(launcher,'Fleet',Fleet);monkeypatch.setattr(cm,'EngineClient',Client)
    monkeypatch.setattr(wm.ProfileWave,'from_environment',lambda:Wave())
    monkeypatch.setattr(c.PerfModel,'load',lambda _:Model());monkeypatch.setattr(c.asyncio,'sleep',no_sleep)
    out=tmp_path/'new'
    result=c.run_holdout(candidate_dir=candidate,model_path='/models/Qwen2.5-14B-Instruct',gpus=[0,1,2,3],
        base_port=15000,out=out,prior_holdout=root,prior_raw_sha256=sha)
    assert result['complete'] and result['calibration_passed'],result
    assert events==['load','qualified',('decode',8),'mixed','stop','reset']
    new=json.loads((out/'raw.json').read_text());combined=json.loads((out/'combined-holdout.json').read_text())
    assert new['prefill']==[] and [r['batch'] for r in new['decode']]==[8]
    assert all('evidence_source' not in r for r in new['decode'])
    assert new['environment']['source_hash']=='b'*64
    assert [r['batch'] for r in combined['decode']]==[4,8]
    assert 'identity_sha256' not in combined
    assert combined['new_raw_identity_sha256']==new['identity_sha256']
    assert combined['decode'][0]['evidence_source']==result['prior_holdout']['binding']['evidence_source']
    assert result['inherited_points']==dict(prefill=1,decode=1)
    assert result['newly_measured_points']==dict(prefill=0,decode=1)
    assert {p.name:p.read_bytes() for p in root.iterdir()}==before
    assert result['raw_sha256']==c.digest(out/'raw.json')
    assert result['combined_holdout_sha256']==c.digest(out/'combined-holdout.json')
    assert not result['formal_eligible'] and not result['energy_comparable']
