import asyncio
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdblend.bench import longbench_anchor_recovery as module
from pdblend.bench.client import SLOS
from pdblend.bench.resident_session import write_new


def test_failed_tuning_restarts_lower_calibration_and_never_uses_evaluation():
    calls=[];recorded=[]
    answers=iter([True,False,False,True,True])
    async def observe(split,seed,duration,rate,index):
        calls.append((split,seed,duration,rate,index))
        return dict(metrics=dict(passed=next(answers)))
    winner,rows=asyncio.run(module.search([.25,.125,.0625,.03125],observe,recorded.append))
    assert winner['rate_rps']==.0625
    assert [r['status'] for r in rows]==['tuning_failed','calibration_failed','confirmed']
    assert recorded==rows
    assert calls==[('calibration',9701,60.,.25,0),('tuning',9702,120.,.25,0),
                   ('calibration',9701,60.,.125,1),('calibration',9701,60.,.0625,2),
                   ('tuning',9702,120.,.0625,2)]


def test_predeclared_floor_exhaustion_never_promotes_passing_calibration():
    rows=[]
    async def observe(split,*args):return dict(metrics=dict(passed=split=='calibration'))
    winner,candidates=asyncio.run(module.search([.25,.125],observe,rows.append))
    assert winner is None and len(candidates)==2
    assert all(row['status']=='tuning_failed' for row in candidates)
    assert module.candidate_rates(.5,.03125)==[.25,.125,.0625,.03125]
    for floor in (0.,.5,.03,float('nan'),.5/1024):
        with pytest.raises(ValueError):module.candidate_rates(.5,floor)


def test_interrupted_candidate_is_recorded_and_stops_unsafe_reuse():
    rows=[];calls=[]
    async def observe(split,*args):
        calls.append(split)
        if split=='tuning':raise RuntimeError('native drain failed')
        return dict(metrics=dict(passed=True))
    with pytest.raises(RuntimeError,match='drain failed'):
        asyncio.run(module.search([.25,.125],observe,rows.append))
    assert calls==['calibration','tuning']
    assert rows==[dict(index=0,rate_rps=.25,status='measurement_failed',
        calibration=dict(metrics=dict(passed=True)),error='RuntimeError: native drain failed')]


@pytest.fixture
def prior(tmp_path):
    audit=dict(model_id=module.MODEL,model_hash='weight',tokenizer_hash='tokenizer',image_digest='image',
        tp=2,pp=1,corpus_sha256={k:k+'hash' for k in SLOS},corpus_tokenizer_sha256='tok',
        corpus_manifest_sha256='corpus',calibration_seed=9701,tuning_seed=9702,
        evaluation_used_for_selection=False,source_sha256='old-source')
    anchors={}
    for dataset,rate in (('alpaca',8.),('sharegpt',2.),('longbench',.5)):
        root=tmp_path/(dataset+'-tuning-0');root.mkdir()
        write_new(root/'requests.json',dict(seed=9702,duration_s=120,
            requests=[dict(idx=0,arrival_s=0.,prompt=[1,2],max_tokens=2)]))
        measured=dict(system='mixed',dataset=dataset,split='tuning',seed=9702,duration_s=120,
            rate_rps=rate,trace_sha256=module.file_sha(root/'requests.json'),counts_reclaimed=True,
            routing_policy='independent_least_load_fixed_tp',drain=[dict(drain=dict(drained=True))]*4,
            metrics=dict(passed=dataset!='longbench',success_rate=1.,joint_slo_rate=1.,
                offered=1,correct=1,ttft_p99_s=.1,tpot_p99_s=.01,
                slo_ttft_s=SLOS[dataset][0],slo_tpot_s=SLOS[dataset][1]))
        write_new(root/'completion.json',measured)
        if dataset!='longbench':
            anchors[dataset]=dict(model_id=module.MODEL,tp=2,pp=1,replicas=4,clock_mhz=2520,
                base_rate_rps=rate,x05_rate_rps=rate/2,corpus_sha256=audit['corpus_sha256'][dataset],
                scope='highest_tested_and_confirmed_passing_rate',
                confirmation_path='/output/anchor/'+root.name+'/completion.json',
                confirmation_sha256=module.file_sha(root/'completion.json'))
    write_new(tmp_path/'preflight.json',audit)
    write_new(tmp_path/'rate-anchor.json',dict(audit,anchors=anchors))
    write_new(tmp_path/'completion.json',dict(audit,anchors=anchors,status='failed',complete=False,
        selection_splits=['calibration','tuning'],cleanup_errors=[],
        error='RuntimeError: longbench: no independent tuning confirmation passed'))
    return tmp_path,audit,anchors


def test_prior_passes_are_bound_and_returned_unchanged(prior):
    root,audit,anchors=prior
    before={p:module.file_sha(p) for p in root.rglob('*.json')}
    inputs=module.prior_inputs(root)
    recovered,failed=module.validate_prior(inputs,audit)
    assert recovered==anchors and failed==.5
    recovered['alpaca']['base_rate_rps']=999
    assert anchors['alpaca']['base_rate_rps']==8.
    assert before=={p:module.file_sha(p) for p in root.rglob('*.json')}
    path=Path(inputs['inherited']['alpaca']['completion']['path'])
    path.write_text('{}')
    with pytest.raises(ValueError,match='checksum'):module.validate_prior(inputs,audit)


@pytest.mark.parametrize('key,value',[('model_id','Qwen2.5-7B-Instruct'),('tp',1),('image_digest','other'),
                                      ('corpus_manifest_sha256','changed'),('tuning_seed',701)])
def test_prior_identity_cannot_be_relabelled(prior,key,value):
    root,audit,_=prior
    changed=dict(audit,**{key:value})
    with pytest.raises(ValueError,match='identity differs'):
        module.validate_prior(module.prior_inputs(root),changed)


def test_unqualified_prior_dataset_is_not_inherited(prior):
    root,audit,_=prior
    inputs=module.prior_inputs(root)
    path=Path(inputs['inherited']['alpaca']['completion']['path'])
    value=json.loads(path.read_text());value['metrics']['tpot_p99_s']=10.
    path.write_text(json.dumps(value))
    inputs['inherited']['alpaca']['completion']=module.binding(path)
    with pytest.raises(ValueError,match='not qualified'):
        module.validate_prior(inputs,audit)


@pytest.mark.parametrize('cleanup_failed',[False,True])
def test_run_reuses_two_anchors_and_only_runs_longbench_without_changing_prior(prior,monkeypatch,cleanup_failed):
    from pdblend.bench.first_batch import load_anchor
    from pdblend.bench.client import Request
    root,audit,anchors=prior
    refs=module.prior_inputs(root)
    before={p:module.file_sha(p) for p in root.rglob('*.json')}
    prepared=dict(audit,inherited_anchors=anchors,prior_inputs=refs,candidate_rates_rps=[.25,.125],
                  minimum_rate_rps=.125)
    monkeypatch.setattr(module,'preflight',lambda _:prepared)
    calls=[]
    class Sampler:
        samples=[];frequency_samples=[];power_metadata=[];power_source={};error=None
        def start(self):calls.append('sampling')
        def stop(self):pass
    class Meter:
        def __init__(self,gpus,**kwargs):self.gpus=gpus
        def unpark(self,gpu):pass
        def set_clock(self,gpu,value):assert value==2520
        def reset_clock(self,gpu):
            if cleanup_failed and gpu==0:raise RuntimeError('reset failed')
        def sampler(self,**kwargs):return Sampler()
    class Instance:
        def start(self):calls.append('load')
        def wait_ready(self,**kwargs):pass
        def stop(self):calls.append('stop')
    class Fleet:
        def __init__(self,specs,out):
            assert len(specs)==4 and all(s.tp==2 and s.kv_connector is None and s.max_num_seqs==32 for s in specs)
            self.instances={s.instance_id:Instance() for s in specs}
        def __getitem__(self,key):return self.instances[key]
    async def ready(specs):return {'passed':True}
    async def warm(specs,label):return {'label':label}
    async def execute(specs,trace,out,**kwargs):
        calls.append(('execute',kwargs['seed'],kwargs['duration_s']))
        out.mkdir()
        write_new(out/'requests.json',dict(seed=kwargs['seed'],duration_s=kwargs['duration_s']))
        return dict(metrics=dict(passed=True),split='unused')
    monkeypatch.setattr(module.base,'Gpus',Meter)
    monkeypatch.setattr(module.base,'gpu_manifest',lambda *a:list(range(8)))
    monkeypatch.setattr(module.base,'Fleet',Fleet)
    monkeypatch.setattr(module.base,'model_load_lock',nullcontext)
    monkeypatch.setattr(module.base,'verify_endpoints',ready)
    monkeypatch.setattr(module.base,'drain_endpoints',ready)
    monkeypatch.setattr(module.base,'warmup_endpoints',warm)
    monkeypatch.setattr(module.base,'execute',execute)
    def split(corpus,dataset,selected):
        assert dataset=='longbench' and selected in ('calibration','tuning')
        calls.append(('split',selected));return []
    monkeypatch.setattr(module,'load_split',split)
    monkeypatch.setattr(module,'poisson_trace',lambda *a,**kw:[Request(0,0.,[1,2],2,'longbench')])
    monkeypatch.setenv('PDBLEND_GPU_UUIDS',','.join('GPU-'+str(i) for i in range(8)))
    output=root.parent/('new-'+str(cleanup_failed))
    args=SimpleNamespace(out=output,gpus=list(range(8)),base_port=18000,model='/models/'+module.MODEL,corpus=root)
    result=asyncio.run(module.run(args))
    assert result['complete'] is (not cleanup_failed)
    assert result['anchors']['alpaca']==anchors['alpaca'] and result['anchors']['sharegpt']==anchors['sharegpt']
    assert result['anchors']['longbench']['base_rate_rps']==.25
    assert calls.count('load')==calls.count('stop')==4
    assert [c for c in calls if isinstance(c,tuple) and c[0]=='execute']==[('execute',9701,60.),('execute',9702,120.)]
    assert before=={p:module.file_sha(p) for p in root.rglob('*.json')}
    assert (output/'candidate-0.json').is_file() and not (output/'candidate-1.json').exists()
    if not cleanup_failed:
        for dataset in SLOS:
            assert load_anchor(output/'completion.json',module.MODEL,dataset,audit['corpus_sha256'][dataset])['base_rate_rps']>0
    else:
        with pytest.raises(ValueError,match='completed'):
            load_anchor(output/'completion.json',module.MODEL,'longbench',audit['corpus_sha256']['longbench'])
