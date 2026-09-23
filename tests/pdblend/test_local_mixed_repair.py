import asyncio
import copy
import json
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdblend.profile import local_power as lp, local_mixed_repair as mr, power_calibration as pc, profiler as pm
from pdblend.profile.identity import sha256_value
from test_local_power import package_fixture
from test_power_calibration import Clock, Sampler


def fixture(root):
    args,env=package_fixture(root);prior=args['prior_holdout'];original=root/'completed';original.mkdir()
    identity=json.loads((prior/'raw.json').read_text())
    rows=dict(prefill=[],decode=[],mixed=[])
    for chunk,batch in ((512,8),(2048,32)):
        p=original/f'p{chunk}.json';p.write_text('{}')
        rows['prefill'].append(dict(freq_mhz=1500,input_tokens=chunk,seconds=.01,samples_file=p.name,samples_sha256=pc.digest(p)))
        reps=[]
        for i in range(3):
            p=original/f'd{batch}-{i}.json';p.write_text('{}')
            reps.append(dict(step_seconds=.1,effective_context_tokens=1034,power_w=600,steady_window_s=5,min_steps=50,
                power_samples=2,frequency_samples=1,samples_file=p.name,samples_sha256=pc.digest(p)))
        rows['decode'].append(dict(freq_mhz=1500,batch=batch,context_tokens=1024,effective_context_tokens=1034,
            step_seconds=.1,power_w=600,repeats=reps))
    for f in (1500,2100,2520):
        for batch,chunk in pm.MIXED_PROBES:
            row=dict(freq_mhz=f,batch=batch,chunk_tokens=chunk,valid=f!=1500)
            if f==1500:row['invalid_reason']='missing_base_step_or_prefill'
            else:
                p=original/f'm{f}-{batch}-{chunk}.json';p.write_text('{}')
                row.update(base_step_s=.1,alone_prefill_s=.01,probe_ttft_s=.11,samples_file=p.name,samples_sha256=pc.digest(p))
            rows['mixed'].append(row)
    inherited=dict(copy.deepcopy(identity),prefill=[copy.deepcopy(rows['prefill'][0])],decode=[copy.deepcopy(rows['decode'][0])],mixed=[])
    for row in (*inherited['prefill'],*inherited['decode'][0]['repeats']):
        (prior/row['samples_file']).write_bytes((original/row['samples_file']).read_bytes())
    inherited['identity_sha256']=sha256_value({k:v for k,v in inherited.items() if k!='identity_sha256'})
    (prior/'raw.json').write_text(json.dumps(inherited))
    combined=dict(copy.deepcopy(identity),**copy.deepcopy(rows))
    for s in ('prefill','decode'):combined[s][0]['evidence_source']='prior'
    combined['evidence_sources']=dict(prior=dict(raw_sha256=pc.digest(prior/'raw.json')))
    current=dict(copy.deepcopy(identity),prefill=rows['prefill'][1:],decode=rows['decode'][1:],mixed=rows['mixed'])
    (original/'raw.json').write_text(json.dumps(current));(original/'combined-holdout.json').write_text(json.dumps(combined))
    plan={s:[{k:r[k] for k in fs} for r in rows[s]] for s,fs in [('prefill',('freq_mhz','input_tokens')),('decode',('freq_mhz','batch','context_tokens'))]}
    (original/'frozen-fit.json').write_text(json.dumps(dict(candidate_sha256=pc.digest(args['base_candidate']),plan=plan)))
    (original/'completion.json').write_text(json.dumps(dict(complete=True,independent_holdout=True,
        candidate_sha256=pc.digest(args['base_candidate']),raw_sha256=pc.digest(original/'raw.json'),
        combined_holdout_sha256=pc.digest(original/'combined-holdout.json'),calibration_passed=False)))
    args.update(out=root/'package2',original_holdout=original);lp.prepare(**args)
    return args,env


def collector(root,args,env,monkeypatch):
    p=pm.Profiler.__new__(pm.Profiler);p.out_dir=root/'measurement';p.out_dir.mkdir()
    ref=json.loads((args['original_holdout']/'combined-holdout.json').read_text())
    p.model='/models/Qwen2.5-14B-Instruct';p.model_spec=SimpleNamespace(model_id='Qwen2.5-14B-Instruct')
    p.tp=4;p.pp=1;p.system='pdblend';p.role='mixed';p.freqs=pc.FREQUENCIES;p.mixed_freqs=(1500,2100,2520)
    p.parallel_layout={};p.decode_settle_s=2
    p.raw=dict(ref,schema=2,model=p.model,role=p.role,config={},freqs=list(p.freqs),environment=env,
        parallel_layout={},prefill=[],decode=[],mixed=[],static={},transfer=[],
        external_interference=dict(samples_file='parent-only.json',samples_sha256='parent'))
    p._checkpoint();clock=Clock();p._test_clock=clock;events=[]
    p._lock=lambda f,g:events.append(('clock',f))
    @asynccontextmanager
    async def background(client,batch,context,tag):
        events.append(('background',batch,context,tag))
        clock.live=[SimpleNamespace(token_times_s=[clock.now-.1,clock.now],error=None) for _ in range(batch)]
        yield clock.live,[]
    p._background=background
    class Client:
        async def complete(self,prompt,max_tokens,request_id):
            events.append(('request',max_tokens,request_id));start=clock.now
            await clock.sleep(.11)
            return SimpleNamespace(error=None,submitted_s=start,first_token_s=clock.now,finished_s=clock.now,ttft_s=.11)
    monkeypatch.setattr(pm.asyncio,'sleep',clock.sleep);monkeypatch.setattr(pm.time,'time',clock)
    return p,Client(),events


def test_package_binds_only_four_missing_reference_points(tmp_path):
    args,_=fixture(tmp_path);m,_,_=lp.load_package(args['out'])
    assert m['mixed_repair']['points']==[list(k) for k in sorted(mr.EXPECTED)]
    old=json.loads((args['out']/'retained-timing-audit.json').read_text())
    assert not old['passed'] and sum(f['metric']=='mixed_evidence' for f in old['failures'])==4
    assert (args['out']/'candidate.json').read_bytes()==args['candidate'].read_bytes()
    assert (args['out']/'power-plan.json').read_bytes()==args['plan'].read_bytes()
    reference=args['original_holdout']/'combined-holdout.json';reference.write_text('{}')
    with pytest.raises(ValueError,match='immutable input'):lp.load_package(args['out'])


@pytest.mark.asyncio
async def test_actual_mixed_collects_only_four_and_composes_new_audit_without_editing_old(tmp_path,monkeypatch):
    args,env=fixture(tmp_path);m,_,_=lp.load_package(args['out']);p,client,events=collector(tmp_path,args,env,monkeypatch)
    parent=copy.deepcopy(p.raw);original={f:f.read_bytes() for f in args['original_holdout'].iterdir()}
    adapter=lp.Adapter('isolated-host');assert adapter.needs_followup(args['out'],p.out_dir)
    result=await adapter.after_samples(profiler=p,client=client,gpus=[0,1,2,3],package=args['out'],out=p.out_dir/'mixed-repair')
    assert result['complete'] and result['valid_points']==4
    assert p.raw==parent and sum(e[0]=='background' for e in events)==4 and sum(e[0]=='request' for e in events)==12
    assert not adapter.needs_followup(args['out'],p.out_dir)
    event_count=len(events)
    resumed=await adapter.after_samples(profiler=p,client=client,gpus=[0,1,2,3],package=args['out'],out=p.out_dir/'mixed-repair')
    assert resumed['complete'] and not any(e[0] in ('background','request') for e in events[event_count:])
    child=json.loads((p.out_dir/'mixed-repair/raw.json').read_text())
    assert child['prefill']==child['decode']==[] and 'external_interference' not in child
    assert all(r['reference_binding']['decode_evidence_source']=='prior' for r in child['mixed'] if r['batch']==8)
    repaired=mr.audit_repair(manifest=m,package=args['out'],out=p.out_dir)
    assert repaired['passed'] and repaired['new_mixed_points']==4 and repaired['retained_valid_mixed_points']==8
    assert repaired['old_original_timing_passed'] is False
    assert {f:f.read_bytes() for f in args['original_holdout'].iterdir()}==original
    assert json.loads((args['out']/'retained-timing-audit.json').read_text())['passed'] is False
    view=json.loads((p.out_dir/'mixed-repair/combined-timing-view.json').read_text())
    assert len(view['mixed'])==12 and 'identity_sha256' not in view
    assert {r['evidence_source'] for r in view['decode']}=={'prior','original-completed'}


@pytest.mark.asyncio
@pytest.mark.parametrize('fault',['reference','sample','identity'])
async def test_completed_repair_rechecks_reference_sample_and_identity(tmp_path,monkeypatch,fault):
    args,env=fixture(tmp_path);m,_,_=lp.load_package(args['out']);p,client,_=collector(tmp_path,args,env,monkeypatch)
    out=p.out_dir/'mixed-repair'
    await mr.collect_existing(profiler=p,client=client,gpus=[0,1,2,3],package=args['out'],out=out,manifest=m)
    if fault=='sample':
        raw=json.loads((out/'raw.json').read_text());(out/raw['mixed'][0]['samples_file']).write_text('{}')
    elif fault=='reference':
        reference=args['original_holdout']/'combined-holdout.json';reference.write_text('{}')
    else:
        raw=json.loads((out/'raw.json').read_text());raw['model_id']='wrong';(out/'raw.json').write_text(json.dumps(raw))
        completion=json.loads((out/'completion.json').read_text());completion['raw_sha256']=pc.digest(out/'raw.json');(out/'completion.json').write_text(json.dumps(completion))
    with pytest.raises((ValueError,KeyError)):
        mr.checked_completed(package=args['out'],out=out,manifest=m)


@pytest.mark.asyncio
async def test_failed_mixed_evidence_remains_inconclusive(tmp_path,monkeypatch):
    args,env=fixture(tmp_path);m,_,_=lp.load_package(args['out']);p,client,_=collector(tmp_path,args,env,monkeypatch)
    @asynccontextmanager
    async def failed(*args):
        raise RuntimeError('actual background ended')
        yield
    p._background=failed
    result=await mr.collect_existing(profiler=p,client=client,gpus=[0,1,2,3],package=args['out'],out=p.out_dir/'mixed-repair',manifest=m)
    assert not result['complete'] and result['status']=='inconclusive'
    audit=mr.audit_repair(manifest=m,package=args['out'],out=p.out_dir)
    assert audit['passed'] is False and audit['complete'] is False
    assert lp.Adapter('isolated-host').needs_followup(args['out'],p.out_dir)


def test_shared_runner_one_load_and_resident_followup_inside_measurement(tmp_path,monkeypatch):
    from pdblend.engine import launcher,client as cm
    args,env=fixture(tmp_path);p,client,events=collector(tmp_path,args,env,monkeypatch)
    # No pre-existing parent checkpoint: this test enters the actual shared runner from scratch.
    (p.out_dir/'raw.json').unlink();p.raw.pop('external_interference')
    p.model_spec.model_hash='model';p.model_spec.tokenizer_hash='tok'
    p.specs=[SimpleNamespace(instance_id='p',base_url='http://unused',gpus=[0,1,2,3])]
    p._kv_capacity=lambda inst:2441936
    clock=p._test_clock
    p.meter=SimpleNamespace(sampler=lambda g:Sampler(clock),reset_all=lambda:events.append(('reset',)))
    p._lock=lambda f,g:setattr(clock,'freq',f)
    class Fleet:
        def __init__(self,*a):pass
        def __enter__(self):return self
        def __exit__(self,*a):events.append(('stop',))
        def start_all(self):events.append(('load',))
        def __getitem__(self,key):return SimpleNamespace(spec=p.specs[0])
    @asynccontextmanager
    async def engine_client(*args):
        events.append(('client_enter',));yield client;events.append(('client_exit',))
    class Gate:
        def __init__(self,profiler):pass
        async def qualify_external(self,*a):events.append(('qualification',))
        @asynccontextmanager
        async def measurement(self):
            events.append(('measurement_enter',));yield
            events.append(('measurement_exit',))
            path=p.out_dir/'isolation.json';path.write_text(json.dumps(dict(passed=True,measurement_complete=True,parallel=False)))
            p.raw['local_power_isolation']=dict(samples_file=path.name,samples_sha256=pc.digest(path))
        def write(self,*args):events.append(('error',))
    @asynccontextmanager
    async def background(profiler,engine,point,tag):
        async with profiler._background(engine,point['batch'],point['context_tokens'],tag) as result:yield result
    actual=pc.collect_power_point
    async def collect(*args,**kw):return await actual(*args,**kw,_clock=clock,_sleep=clock.sleep,_background_factory=background)
    monkeypatch.setattr(pm,'Profiler',lambda *args,**kw:p);monkeypatch.setattr(pm,'_load_flock',nullcontext)
    monkeypatch.setattr(launcher,'Fleet',Fleet);monkeypatch.setattr(cm,'EngineClient',engine_client)
    monkeypatch.setattr(lp,'IsolatedGroup',Gate);monkeypatch.setattr(pc,'collect_power_point',collect)
    result=pc.run(package=args['out'],model_path=p.model,gpus=[0,1,2,3],base_port=10000,out=p.out_dir,panel_adapter=lp.Adapter('isolated-host'))
    assert result['complete'] and result['power_passed'] and result['repaired_timing_passed'],result
    assert not result['reused_timing_passed'] and result['calibration_components_passed']
    assert not result['full_profile_qualified'] and not result['formal_eligible']
    assert len(p.raw['decode'])==12 and p.raw['mixed']==[]
    assert events.count(('load',))==events.count(('stop',))==1
    requests=[i for i,e in enumerate(events) if e[0]=='request']
    assert len(requests)==12 and events.index(('client_enter',))<min(requests)<max(requests)<events.index(('client_exit',))
    assert events.index(('client_exit',))<events.index(('measurement_exit',))<events.index(('stop',))<events.index(('reset',))
