import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdblend.bench import longbench_mixed_combo as module
from pdblend.bench.comparison_campaign import MODELS, SCALES, SYSTEM_ORDER, group_points
from pdblend.bench.resident_session import digest, write_new


def observation(root, index, split, rate, passed, *, request_failure=False):
    where=root/f'longbench-{split}-recovery-{index}'; where.mkdir(parents=True)
    seed,duration=(9701,60.) if split=='calibration' else (9702,120.)
    write_new(where/'requests.json',dict(seed=seed,duration_s=duration,requests=[{},{}]))
    metrics=dict(passed=passed,offered=2,correct=1 if request_failure else 2,
        success_rate=.5 if request_failure else 1.,joint_slo_rate=1.,ttft_p99_s=1.,
        tpot_p99_s=.1 if passed else .3,slo_ttft_s=15.,slo_tpot_s=.2)
    value=dict(system='mixed',dataset='longbench',split=split,seed=seed,duration_s=duration,
        rate_rps=rate,metrics=metrics,counts_reclaimed=True,drain=[dict(drain=dict(drained=True))]*4,
        trace_sha256=module.file_sha(where/'requests.json'))
    write_new(where/'completion.json',value)
    return dict(path=str((where/'completion.json').relative_to(root)),sha256=module.file_sha(where/'completion.json'),
                split=split,seed=seed,duration_s=duration,rate_rps=rate,metrics=metrics)


def result_fixture(root, *, confirmed=True, request_failure=False):
    rates=[.25,.125]; inherited={k:dict(base_rate_rps=r) for k,r in [('alpaca',8.),('sharegpt',2.)]}
    audit=dict(inherited_anchors=inherited,candidate_rates_rps=rates)
    rows=[]
    for index,rate in enumerate(rates):
        cal=observation(root,index,'calibration',rate,True)
        tuned=confirmed and index==1
        tune=observation(root,index,'tuning',rate,tuned,request_failure=request_failure)
        rows.append(dict(index=index,rate_rps=rate,calibration=cal,tuning=tune,
                         status='confirmed' if tuned else 'tuning_failed'))
    result=dict(status='passed' if confirmed else 'failed',complete=confirmed,hardware_executed=True,
        inherited_anchors=inherited,anchors=deepcopy(inherited),candidates=rows,cleanup_errors=[])
    if confirmed:
        result['anchors']['longbench']=dict(base_rate_rps=rates[-1],confirmation_sha256=tune['sha256'],
            confirmation_path='/output/anchor/'+tune['path'],scope='highest_tested_and_confirmed_passing_rate')
    else:result['error']=module.FLOOR_ERROR
    write_new(root/'completion.json',result)
    return result,audit


@pytest.mark.parametrize('confirmed,expected',[(True,'confirmed'),(False,'slo_exhausted')])
def test_only_verified_confirmation_or_complete_slo_exhaustion_can_continue(tmp_path,confirmed,expected):
    result,audit=result_fixture(tmp_path,confirmed=confirmed)
    assert module.classify_recovery(result,audit,tmp_path)==expected


@pytest.mark.parametrize('mutation',[
    lambda r:r.update(cleanup_errors=['clock reset failed']),
    lambda r:r.update(hardware_executed=False),
    lambda r:r.update(error='RuntimeError: native timeout'),
    lambda r:r['candidates'].pop(),
    lambda r:r['candidates'][0].update(status='measurement_failed',error='HTTP 500'),
    lambda r:r['anchors']['alpaca'].update(base_rate_rps=9),
])
def test_cleanup_infrastructure_and_partial_search_never_fall_back(tmp_path,mutation):
    result,audit=result_fixture(tmp_path,confirmed=False)
    result=deepcopy(result);mutation(result)
    with pytest.raises(ValueError):module.classify_recovery(result,audit,tmp_path)


def test_request_failure_is_not_a_slo_only_failure(tmp_path):
    result,audit=result_fixture(tmp_path,confirmed=False,request_failure=True)
    with pytest.raises(ValueError,match='SLO-only'):
        module.classify_recovery(result,audit,tmp_path)


def test_changed_raw_request_bytes_prevent_fallback(tmp_path):
    result,audit=result_fixture(tmp_path,confirmed=False)
    (tmp_path/'longbench-tuning-recovery-0/requests.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'):
        module.classify_recovery(result,audit,tmp_path)


def test_all_parent_bytes_except_declared_wrappers_are_protected(tmp_path):
    def source(root,files):
        hashes={}
        for name,value in files.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
            hashes[name]=module.file_sha(path)
        write_new(root/'manifest.json',dict(files=hashes,source_sha256=digest(hashes)))
        return module.binding(root/'manifest.json')
    base={'pdblend_runtime/serve.py':'engine','pdblend/bench/client.py':'measurement',
          'pdblend/bench/native_mixed.py':'policy','pdblend/bench/comparison_runtime.py':'adapter'}
    parent=source(tmp_path/'parent',base)
    added=dict(base,**{k:'new wrapper' for k in module.OVERLAYS})
    execution=source(tmp_path/'execution',added)
    assert module.validate_source(parent,execution)['protected_files']==4
    bad=dict(added);bad['pdblend/bench/native_mixed.py']='changed policy'
    with pytest.raises(ValueError,match='implementation changed'):
        module.validate_source(parent,source(tmp_path/'bad',bad))


def campaign_fixture(root):
    identity=dict(model_hash='weight',tokenizer_hash='tokenizer',image_digest='image',
        runtime_source_sha256='engine',measurement_source_sha256='measure',entrypoint='native',
        worker_extension='v1',dtype='bfloat16',environment={},fleet_gpu_uuids=[f'GPU-{i}' for i in range(8)],
        instances=[dict(instance_id=f'mixed{i}',tp=2,pp=1,gpu_uuids=[f'GPU-{2*i}',f'GPU-{2*i+1}'],launch_options={})
                   for i in range(4)])
    points=[];traces=[]
    for model in MODELS:
        for scale in SCALES:
            for system in SYSTEM_ORDER:
                for dataset in ('alpaca','sharegpt','longbench'):
                    prepared=model==module.recovery.MODEL and system=='mixed' and dataset!='longbench'
                    slo=dict(ttft_s=15.,tpot_s=.2)
                    name=f'{model}-{system}-{dataset}-{scale}'
                    trace=None
                    if prepared:
                        path=root/'traces'/f'{name}.json'
                        write_new(path,dict(model_id=model,dataset=dataset,rate_rps=scale,seed=701,duration_s=150,
                                            slo=slo,requests=[dict(arrival_s=1,prompt=[1],max_tokens=1)]))
                        trace=dict(module.binding(path),requests=1);traces.append(trace)
                    blockers=[] if prepared else [module.BLOCKER]
                    if system!='mixed':blockers.append('missing_native_qualification')
                    p=dict(name=name,model_id=model,system=system,dataset=dataset,scale=scale,seed=701,
                        duration_s=150,rate_rps=scale if prepared else None,slo=slo,trace=trace,blockers=blockers,
                        status='prepared' if prepared else 'blocked')
                    if model==module.recovery.MODEL and system=='mixed':p['engine_identity']=identity
                    points.append(p)
    campaign=dict(schema='resident-comparison-campaign/v1',campaign_id='parent',seed=701,duration_s=150,
        scales=list(SCALES),measurement_protocol_version=module.PROTOCOL,points=points,traces=traces,
        groups=group_points(points),summary=dict(points=180,trace_sets=8,prepared_points=8,resident_sessions=1))
    write_new(root/'campaign.json',campaign)
    plan=dict(parent_campaign=module.binding(root/'campaign.json'),source_manifest={'sha256':'source'},_path=str(root/'plan.json'))
    write_new(root/'plan.json',plan)
    return campaign,plan


@pytest.mark.parametrize('confirmed',[True,False])
def test_freeze_retains_all_180_points_and_original_eight_trace_bytes(tmp_path,monkeypatch,confirmed):
    parent,plan=campaign_fixture(tmp_path/'parent');out=tmp_path/'combined'
    result,audit=result_fixture(out/'anchor',confirmed=confirmed)
    corpus=tmp_path/'corpus';corpus.mkdir()
    write_new(corpus/'manifest.json',dict(tokenizer_sha256='tokenizer'))
    write_new(corpus/'longbench.json',{})
    audit.update(corpus_manifest_sha256=module.file_sha(corpus/'manifest.json'),
                 corpus_sha256={'longbench':module.file_sha(corpus/'longbench.json')})
    calls=[]
    def load(*args):calls.append(args);return [dict(prompt=[1,2],output_tokens=2)]
    monkeypatch.setattr(module,'load_split',load)
    from pdblend.bench.client import Request
    monkeypatch.setattr(module,'poisson_trace',lambda *args:[Request(0,1.,[1,2],2)])
    before={p['name']:(digest(p),Path(p['trace']['path']).read_bytes()) for p in parent['groups'][0]['points']}
    outcome='confirmed' if confirmed else 'slo_exhausted'
    group=module.freeze_overlay(plan,result,audit,outcome,corpus,out)
    overlay=json.loads((out/'campaign.json').read_text());freeze=json.loads((out/'freeze.json').read_text())
    assert len(overlay['points'])==180 and len(group['points'])==(12 if confirmed else 8)
    assert len(calls)==int(confirmed) and not freeze['evaluation_has_started']
    for point in group['points']:
        if point['name'] in before:
            assert (digest(point),Path(point['trace']['path']).read_bytes())==before[point['name']]
    lbs=[p for p in overlay['points'] if p['model_id']==module.recovery.MODEL and p['dataset']=='longbench']
    assert len(lbs)==20
    assert all((module.BLOCKER not in p['blockers'])==confirmed for p in lbs)
    assert all('missing_native_qualification' in p['blockers'] for p in lbs if p['system']!='mixed')
    assert len(list((out/'traces').glob('*.json'))) == (4 if confirmed else 0)


def test_infrastructure_failure_stops_before_evaluation_or_second_load(tmp_path,monkeypatch):
    audit=dict(plan={'path':'fake','sha256':'fake'},recovery={'candidate_rates_rps':[.25]})
    plan=dict(recovery_plan={'path':'fake','sha256':'fake'},parent_campaign={})
    args=SimpleNamespace(out=tmp_path/'combined',plan=tmp_path/'plan',base_port=19000,corpus=tmp_path)
    monkeypatch.setattr(module,'preflight',lambda args:audit)
    monkeypatch.setattr(module,'bound',lambda ref:plan)
    async def fail(args):raise RuntimeError('engine failed to start')
    monkeypatch.setattr(module.recovery,'run',fail)
    monkeypatch.setattr(module,'freeze_overlay',lambda *a:pytest.fail('must not freeze evaluation'))
    result=asyncio.run(module.run(args))
    assert not result['complete'] and 'engine failed' in result['error']
    assert result['total_engine_load_cycles']==0 and not (args.out/'session').exists()


@pytest.mark.parametrize('confirmed',[True,False])
def test_combined_order_cleans_and_freezes_before_one_resident_group(tmp_path,monkeypatch,confirmed):
    import shutil
    from pdblend.bench import comparison_runtime
    from pdblend.bench.client import Request
    events=[];parent,plan=campaign_fixture(tmp_path/'parent')
    result,audit=result_fixture(tmp_path/'template',confirmed=confirmed)
    corpus=tmp_path/'corpus';corpus.mkdir()
    write_new(corpus/'manifest.json',dict(tokenizer_sha256='tokenizer'));write_new(corpus/'longbench.json',{})
    audit.update(corpus_manifest_sha256=module.file_sha(corpus/'manifest.json'),
                 corpus_sha256={'longbench':module.file_sha(corpus/'longbench.json')})
    plan['recovery_plan']={'path':'preflight-already-checked','sha256':'preflight-already-checked'}
    plan_path=tmp_path/'combined-plan.json';write_new(plan_path,plan)
    args=SimpleNamespace(out=tmp_path/'combined',plan=plan_path,base_port=19000,corpus=corpus)
    monkeypatch.setattr(module,'preflight',lambda args:dict(plan=module.binding(plan_path),recovery=audit))
    async def recover(args):
        events.extend(['anchor_load','anchor_cleanup'])
        shutil.copytree(tmp_path/'template',args.out)
        return dict(result,engine_loads=4,engine_load_s=2.)
    monkeypatch.setattr(module.recovery,'run',recover)
    async def idle(expected):events.append('physical_idle_verified');return dict(passed=True)
    monkeypatch.setattr(module,'verify_idle',idle)
    def evaluation(*a):events.append('read_evaluation');return []
    monkeypatch.setattr(module,'load_split',evaluation)
    monkeypatch.setattr(module,'poisson_trace',lambda *a:[Request(0,1.,[1,2],2)])
    class Adapter:
        def __init__(self,out,**kwargs):self.out=out
        async def start(self,group):
            freeze=json.loads((self.out.parent/'freeze.json').read_text())
            assert module.bound(freeze['group'])==group
            assert events[:3]==['anchor_load','anchor_cleanup','physical_idle_verified']
            events.append('mixed_load')
            return dict(engine_loads=4,engine_load_cycles=1,engine_load_s=2.)
        async def reset(self,point):return dict(passed=True)
        async def execute(self,point,out):
            assert (self.out.parent/'freeze.json').exists();events.append('window')
            return dict(evidence_valid=True)
        async def drain(self,point):return dict(passed=True)
        async def close(self):events.append('mixed_cleanup');return dict(passed=True)
    monkeypatch.setattr(comparison_runtime,'NativeResidentAdapter',Adapter)
    combined=asyncio.run(module.run(args))
    assert combined['complete'] and combined['measured_points']==(12 if confirmed else 8)
    assert combined['total_engine_load_cycles']==2 and combined['total_engine_loads']==8
    assert not combined['service_energy_includes_extra_load']
    assert events.count('mixed_load')==1 and events[-1]=='mixed_cleanup'
    assert events.count('read_evaluation')==int(confirmed)
    assert combined['longbench_blocker']==(None if confirmed else module.BLOCKER)
