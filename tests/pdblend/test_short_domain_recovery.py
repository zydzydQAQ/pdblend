import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdblend.profile import short_domain as sd, short_domain_collect as collect


def evidence(clock=900, *, tokens=29):
    point=dict(purpose='training',role='prefill_timing',freq_mhz=900,input_tokens=tokens,
        output_tokens=1,batch=1,repeats=1,settle_s=2.,measure_s=5.,decode_accumulation_s=0.)
    def request(t):
        return dict(submitted_s=t,finished_s=t+.2,token_times_s=[t+.1],completion_tokens=1,
            error=None,stream_done=True,usage_received=True)
    return dict(point=point,repeat=0,start_s=10.,end_s=16.,settle_start_s=8.,settle_end_s=10.,
        requests=[request(11),request(13),request(15)],warmup_requests=[request(8)],
        power=[[11,[100]],[15,[100]]],frequency=[[12,[clock]]],gpu_count=1)


def qualification(tmp_path):
    p=tmp_path/'qualifier.json';p.write_text(json.dumps(dict(complete=True,cross_job=True,passed=True)))
    return lambda:dict(epoch_id='epoch-0',qualification_path=str(p),
        qualification_sha256=collect.digest(p),layout_sha256='layout-a')


def reject(raw):
    try:sd.summarize(raw)
    except sd.ShortClockMismatch as exc:
        exc.evidence=raw
        raise


def test_clock_mismatch_retains_exact_observation_without_relaxing_gate():
    with pytest.raises(sd.ShortClockMismatch) as error:reject(evidence(885))
    assert error.value.diagnostics==dict(target_mhz=900,mean_mhz=885.,minimum_mhz=885,
        maximum_mhz=885,samples=1,per_gpu_mean_mhz=[885.])
    assert sd.summarize(evidence())['mean_freq_mhz']==900


@pytest.mark.asyncio
async def test_rejected_clock_window_is_preserved_and_retry_is_fully_separate(tmp_path,monkeypatch):
    calls=[];locks=[];rejected=[];boundaries=[]
    async def measure(*args):
        calls.append(1)
        if len(calls)==1:reject(evidence(885))
        raw=evidence();return raw,sd.summarize(raw)
    async def boundary(*args):boundaries.append(args)
    monkeypatch.setattr(collect,'measure_repeat',measure)
    value=await collect.qualified_repeat(profiler=SimpleNamespace(_lock=lambda f,g:locks.append((f,g))),
        client=None,gpus=[0],point=evidence()['point'],index=0,phase='training',out=tmp_path,
        plan_sha='plan',qualifier=qualification(tmp_path),window_boundary=boundary,rejected=rejected)
    assert len(calls)==len(locks)==len(boundaries)==2 and value[1]['mean_freq_mhz']==900
    assert len(rejected)==1
    file=tmp_path/rejected[0]['samples_file']
    assert collect.digest(file)==rejected[0]['samples_sha256']
    assert json.loads(file.read_text())['frequency']==[[12,[885]]]


@pytest.mark.asyncio
async def test_persistent_clock_gap_does_not_abort_other_points(tmp_path,monkeypatch):
    point=evidence()['point'];second=evidence(tokens=64)['point'];plan=dict(training=[point,second],holdout=[])
    identity=dict(system='pdblend',model_id='Qwen2.5-7B-Instruct',model_hash='model',tokenizer_hash='token',tp=1,pp=1)
    env=dict(image_digest='image',vllm='0.10.1.1',torch='2.7.1',cuda='12.8.1',hardware_id='gpu')
    manifest=dict(**identity,environment=env,plan_sha256='plan')
    package=tmp_path/'package';package.mkdir();(package/'manifest.json').write_text('{}')
    monkeypatch.setattr(collect,'load_package',lambda _: (manifest,plan))
    calls=[]
    async def measure(profiler,client,gpus,point,index):
        calls.append(point['input_tokens'])
        if point['input_tokens']==29:reject(evidence(885))
        raw=evidence(tokens=64);return raw,sd.summarize(raw)
    monkeypatch.setattr(collect,'measure_repeat',measure)
    profiler=SimpleNamespace(raw=dict(**identity,environment=env),_lock=lambda *_:None)
    result=await collect.run_existing(package=package,profiler=profiler,client=None,gpus=[0],
        out=tmp_path/'out',qualification_guard=qualification(tmp_path))
    assert result['status']=='inconclusive' and result['complete'] is False
    assert calls==[29,29,29,64]
    raw=json.loads((tmp_path/'out/raw.json').read_text())
    assert len(raw['training'][sd.point_key(second)]['repeats'])==1
    assert len(json.loads((tmp_path/'out/rejections.json').read_text())['windows'])==3


@pytest.mark.asyncio
async def test_epoch_or_unknown_protocol_failure_is_still_fatal(tmp_path,monkeypatch):
    guard=qualification(tmp_path);epoch=['a'];calls=[]
    def qualifier():return dict(guard(),epoch_id=epoch[0])
    async def measure(*args):
        calls.append(1);epoch[0]='b';reject(evidence(885))
    monkeypatch.setattr(collect,'measure_repeat',measure)
    args=dict(profiler=SimpleNamespace(_lock=lambda *_:None),client=None,gpus=[0],point=evidence()['point'],
        index=0,phase='training',out=tmp_path,plan_sha='plan',qualifier=qualifier,window_boundary=None,rejected=[])
    with pytest.raises(ValueError,match='epoch changed'):await collect.qualified_repeat(**args)
    assert len(calls)==1
    async def unknown(*args):raise ValueError('invalid SSE')
    monkeypatch.setattr(collect,'measure_repeat',unknown)
    with pytest.raises(ValueError,match='invalid SSE'):await collect.qualified_repeat(**args)


def archive(tmp_path):
    old=tmp_path/'attempt/short';old.mkdir(parents=True);source=tmp_path/'source';source.mkdir()
    code=source/'pdblend/profile/short_domain.py';code.parent.mkdir(parents=True);code.write_text('# frozen source')
    files={'pdblend/profile/short_domain.py':collect.digest(code)}
    sha=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (source/'manifest.json').write_text(json.dumps(dict(source_sha256=sha,files=files)))
    identity=dict(system='pdblend',model_id='Qwen2.5-7B-Instruct',model_hash='model',tokenizer_hash='token',tp=1,pp=1)
    env=dict(image_digest='image',vllm='0.10.1.1',torch='2.7.1',cuda='12.8.1',hardware_id='gpu',source_hash=sha)
    raw=evidence();point=raw['point'];plan=dict(training=[point],holdout=[])
    package=tmp_path/'old-package';package.mkdir();(package/'plan.json').write_text(json.dumps(plan))
    (package/'manifest.json').write_text(json.dumps(dict(implementation_sha256={'short_domain.py':collect.digest(code)})))
    manifest=dict(**identity,environment=env,plan_sha256=collect.digest(package/'plan.json'))
    binding=dict(identity=identity,environment=env,plan_sha256=manifest['plan_sha256'],package_sha256=collect.digest(package/'manifest.json'))
    (old.parent/'manifest.json').write_text(json.dumps(dict(payload=dict(source_snapshot=str(source),source_sha256=sha,image_digest='image'))))
    (old.parent/'execution.json').write_text(json.dumps(dict(argv=['PDBLEND_SOURCE_SHA256='+sha,'--short-package',str(package)])))
    stamp=qualification(tmp_path)();saved=collect.guard.save_binding(old,stamp)
    raw.update(plan_sha256=binding['plan_sha256'],epoch_binding=stamp)
    sample=old/'samples/window.json';sample.write_text(json.dumps(raw))
    rep=dict(samples_file='samples/window.json',samples_sha256=collect.digest(sample),summary=sd.summarize(raw),qualification=saved)
    archive=dict(kind=sd.KIND,binding=binding,training={sd.point_key(point):dict(point=point,repeats=[rep])},holdout={})
    (old/'raw.json').write_text(json.dumps(archive))
    return old,plan,manifest,code


def test_reuse_requires_original_source_sample_and_qualification_checksums(tmp_path):
    old,plan,manifest,code=archive(tmp_path)
    raw,files=collect.validate_inherited_archive(old,plan,manifest)
    assert 'samples/window.json' in files
    manifest['inherited_archive']=dict(path=str(old),original_binding=raw['binding'],files_sha256=files,
        raw_sha256=collect.digest(old/'raw.json'))
    out=tmp_path/'new';new=dict(training={},holdout={})
    collect.inherit_samples(out,manifest,new)
    assert collect.digest(out/'samples/window.json')==files['samples/window.json']
    assert new['inherited_archive']['original_binding']==raw['binding']
    code.write_text('# altered source')
    with pytest.raises(ValueError,match='frozen source changed'):
        collect.validate_inherited_archive(old,plan,manifest)


def test_reuse_rejects_corrupted_sample(tmp_path):
    old,plan,manifest,_=archive(tmp_path)
    (old/'samples/window.json').write_text('{}')
    with pytest.raises(ValueError,match='raw sample changed'):
        collect.validate_inherited_archive(old,plan,manifest)
