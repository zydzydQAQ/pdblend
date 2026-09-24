"""CPU preparation with actual source hashes, surface refits and simulator search.

Only raw CUDA extraction is replaced by synthetic already-decoded rows. Native
raw extraction is covered by the stage collector tests; no GPU is used here.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from pdblend.bench.comparison_campaign import binding,MODELS
from pdblend.bench.resident_session import digest,file_sha,write_new
from pdblend_baselines.distserve import stage_collect
from pdblend_baselines.distserve.stage_surface import fit_surface
from tests.independent_baselines.test_distserve_stage_surface import fitted


ROOT=Path(__file__).resolve().parents[2]


def load_script(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def replace(path,value):path.write_text(json.dumps(value,sort_keys=True));return binding(path)


@pytest.fixture
def comparison(tmp_path,monkeypatch):
    module=load_script(ROOT/'scripts/2026-09-24_prepare_distserve_comparison.py','distserve_prepare_test')
    project=tmp_path/'project';(project/'scripts').mkdir(parents=True)
    freezer_name='2026-09-22_enqueue_parallel_profiles.py'
    shutil.copyfile(ROOT/'scripts'/freezer_name,project/'scripts'/freezer_name)
    names=set(module.PROTECTED)|set(module.OVERLAYS)|{'pdblend/measure/power.py',*module.PUBLIC_MEASUREMENT}
    for name in names:
        path=project/'src'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('frozen '+name+'\n')
    freezer=load_script(project/'scripts'/freezer_name,'distserve_prepare_freezer')
    source,source_sha=freezer.freeze_source(project/'src',project/'profile-sources')
    files=json.loads((source/'manifest.json').read_text())['files'];runtime,measurement=module.subsets(files)
    points=[];assets=dict(schema='distserve-comparison-assets/v1',models={});identities={}
    monkeypatch.setattr(stage_collect,'rows_from_window',lambda raw:raw['derived_rows'])
    original=fitted()
    for index,model in enumerate(MODELS):
        tp=2 if '32B-' in model else 1
        identity=dict(model_hash=f'weights-{index}',tokenizer_hash=f'tokenizer-{index}',image_digest='image',
            runtime_source_sha256=digest(runtime),measurement_source_sha256=digest(measurement),
            entrypoint='pdblend_runtime.serve',worker_extension='native_v1',dtype='bfloat16',
            environment=dict(VLLM_USE_V1='1'),fleet_gpu_uuids=[f'GPU-{i}' for i in range(8)],
            instances=[dict(instance_id=f'mixed-{i}',tp=tp,pp=1,gpu_uuids=[f'GPU-{g}' for g in range(i*tp,(i+1)*tp)],
                launch_options=dict(max_num_seqs=32,max_model_len=8192,max_num_batched_tokens=8192,
                    gpu_memory_utilization=.85,kv_connector=None,extra_args=['--enforce-eager'])) for i in range(8//tp)])
        identities[model]=identity
        profile_identity=dict(system='distserve',model_id=model,tp=tp,pp=1,capacity_tokens=100000,
            model_hash=identity['model_hash'],tokenizer_hash=identity['tokenizer_hash'],image_digest='image',
            engine_revision='vllm-0.10.1.1',source_revision=source_sha)
        root=project/'profiles'/model;root.mkdir(parents=True);raw_refs=[]
        train=[dict(row,purpose='training') for row in original['training']]
        hold=[dict(row,purpose='holdout') for row in original['holdout']]
        for number,row in enumerate(train+hold):
            path=root/f'raw-{number}.json'
            write_new(path,dict(status='measured',window_id=row['window_id'],capability=profile_identity,derived_rows=[row]))
            raw_refs.append(dict(path=path.name,sha256=file_sha(path)))
        write_new(root/'external.json',dict(passed=True))
        write_new(root/'qualification.json',dict(passed=True,external_interference_path='external.json',
            external_interference_sha256=file_sha(root/'external.json')))
        artifact=fit_surface(train,hold,identity=profile_identity,raw_bindings=raw_refs,measurement_qualification=dict(
            passed=True,receipt_path='qualification.json',receipt_sha256=file_sha(root/'qualification.json')))
        write_new(root/'surface.json',artifact)
        write_new(root/'completion.json',dict(status='passed',complete=True,hardware_executed=True,cleanup_errors=[],
            calibration_passed=True,surface_path='/output/stage/surface.json',surface_sha256=file_sha(root/'surface.json')))
        corpus=project/'corpus'/model/'alpaca.json'
        write_new(corpus,dict(model_name=model,dataset='alpaca',calibration=[dict(input_tokens=512,output_tokens=2)],
            evaluation=[dict(input_tokens=7900,output_tokens=250)]))
        assets['models'][model]=dict(profile_source_manifest=binding(source/'manifest.json'),
            profiles=[dict(surface=binding(root/'surface.json'),completion=binding(root/'completion.json'))],
            calibration={'alpaca':binding(corpus)})
        for system,scale in [('mixed',.5),('distserve',.5),('distserve',.25)]:
            name=f'{index}-{system}-{scale}'
            trace=project/'traces'/(name+'.json')
            point=dict(name=name,model_id=model,dataset='alpaca',scale=scale,system=system,seed=701,duration_s=150.,
                rate_rps=3*scale,slo=dict(ttft_s=5.,tpot_s=.15),revision='original',blockers=[],status='prepared',
                engine_identity=deepcopy(identity),source_manifest=binding(source/'manifest.json'))
            write_new(trace,dict(selection_split='evaluation',**{k:point[k] for k in
                ('model_id','dataset','rate_rps','slo','seed','duration_s')},requests=[]))
            point['trace']=binding(trace);points.append(point)
    # Two valid frozen observations must stay unchanged, including SLO false.
    prior=project/'prior-session'
    for point,slo in [(points[0],True),(points[1],False)]:
        root=prior/'windows'/point['name'];write_new(root/'point.json',point)
        result=dict(evidence_valid=True,formal_eligible=True,slo_pass=slo,energy_service_j=123.)
        write_new(root/'result.json',result)
        write_new(root/'receipt.json',dict(evidence_valid=True,baseline_frozen=True,cleanup_passed=True,
            point_sha256=digest(point),result=result,artifacts={name:file_sha(root/name) for name in ('point.json','result.json')}))
    base=project/'base/campaign.json';write_new(base,dict(points=points,campaign_id='base',execution_campaigns=[]))
    write_new(project/'verification.json',dict(all_pass=True))
    write_new(base.parent/'execution-inputs.json',dict(image_digest='image',model_verification=binding(project/'verification.json')))
    assets_path=project/'assets.json';write_new(assets_path,assets)
    monkeypatch.setattr(module,'ROOT',project)
    for name in module.OVERLAYS:(project/'src'/name).write_text('new comparison-only wrapper '+name+'\n')
    return dict(module=module,project=project,base=base,assets=assets,assets_path=assets_path,points=points,
        source=source,prior=prior,out=project/'prepared',identities=identities)


def run(x,**kwargs):
    return x['module'].prepare(x['base'],x['out'],x['assets_path'],previous=[x['prior']],
        max_per_gpu_rate=1.,epsilon=.5,sample_size=1,**kwargs)


def test_actual_calibration_search_builds_native_pairs_and_preserves_all_frozen_baselines(comparison):
    x=comparison;before={str(p):file_sha(p) for p in x['prior'].rglob('*') if p.is_file()}
    result=run(x)
    assert result['new_distserve_points']==5 and not result['formal_eligible'] and not result['enqueued']
    assert before=={str(p):file_sha(p) for p in x['prior'].rglob('*') if p.is_file()}
    current=json.loads((x['out']/'campaign.json').read_text());points={p['name']:p for p in current['points']}
    assert set(current['preserved_baseline_receipts'])=={x['points'][0]['name'],x['points'][1]['name']}
    assert points[x['points'][0]['name']]==x['points'][0]
    assert points[x['points'][1]['name']]==x['points'][1]
    jobs=json.loads((x['out']/'jobs.json').read_text())
    assert jobs and all(j['payload']['gpu_count']==8 and j['payload']['exclusive'] for j in jobs)
    scheduled=[]
    for path in (x['out']/'groups').glob('*.json'):
        group=json.loads(path.read_text());scheduled.extend(p['name'] for p in group['points'])
        identity=group['engine_identity'];tp=2 if '32B-' in group['model_id'] else 1
        assert identity['metering_execution']=='isolated_process' and all(r['tp']==tp for r in identity['instances'])
        assert len(identity['instances'])%2==0 and len({u for r in identity['instances'] for u in r['gpu_uuids']})==len(identity['instances'])*tp
        for p in group['points']:
            choice=json.loads(Path(p['inputs']['offline_choice']['path']).read_text())
            assert choice['selection_split']=='calibration' and choice['evaluation_used_for_selection'] is False
            assert choice['selected']['replicas']*2==len(identity['instances'])
            assert p['seed']==701 and p['duration_s']==150 and p['qualification_mode']=='distserve_native_bootstrap'
    assert len(scheduled)==len(set(scheduled))==5 and x['points'][1]['name'] not in scheduled
    checked=json.loads((x['out']/'input-preflight.json').read_text())
    assert len(checked)==5 and all(r['preflight_ready'] for r in checked)
    assert all('deployment.simulator_replay' in r['checked_gates'] for r in checked)


@pytest.mark.parametrize('kind',['frozen_result','source','assets_binding','protected_overlay','public_overlay'])
def test_tampering_or_forbidden_implementation_overlay_cannot_create_jobs(comparison,kind):
    x=comparison;kwargs={}
    if kind=='frozen_result':next(x['prior'].glob('windows/*/result.json')).write_text('{}')
    elif kind=='source':(x['source']/'pdblend_runtime/serve.py').write_text('modified')
    elif kind=='assets_binding':
        x['assets']['models'][MODELS[0]]['profiles'][0]['surface']['sha256']='0'*64
        replace(x['assets_path'],x['assets'])
    elif kind=='protected_overlay':kwargs['overlays']=['pdblend_baselines/distserve/deployment.py']
    else:kwargs['overlays']=['pdblend/bench/comparison_metering.py']
    with pytest.raises((ValueError,RuntimeError)):run(x,**kwargs)
    assert not (x['out']/'jobs.json').exists()


@pytest.mark.parametrize('kind',['missing','unqualified','uncovered','wrong_model_calibration'])
def test_unavailable_or_uncovered_profiles_stay_blocked_without_a_fabricated_layout(comparison,kind):
    x=comparison;model=MODELS[1];owned=x['assets']['models'][model]
    if kind=='missing':del x['assets']['models'][model]
    elif kind=='unqualified':
        ref=owned['profiles'][0]['completion'];value=json.loads(Path(ref['path']).read_text());value['calibration_passed']=False
        owned['profiles'][0]['completion']=replace(Path(ref['path']),value)
    else:
        ref=owned['calibration']['alpaca'];value=json.loads(Path(ref['path']).read_text())
        if kind=='uncovered':value['calibration']=[dict(input_tokens=7168,output_tokens=64)]
        else:value['model_name']=MODELS[0]
        owned['calibration']['alpaca']=replace(Path(ref['path']),value)
    replace(x['assets_path'],x['assets']);result=run(x)
    assert result['new_distserve_points']==3
    current=json.loads((x['out']/'campaign.json').read_text())
    blocked=[p for p in current['points'] if p['system']=='distserve' and p['model_id']==model]
    assert len(blocked)==2 and all(p['status']=='blocked' and p['blockers'] and p['engine_identity'] is None for p in blocked)
    assert all(group['model_id']!=model for group in (json.loads(path.read_text()) for path in (x['out']/'groups').glob('*.json')))
