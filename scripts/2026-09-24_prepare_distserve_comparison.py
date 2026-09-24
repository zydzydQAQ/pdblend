#!/usr/bin/env python3
"""Prepare native DistServe comparison jobs from completed own profiles; never enqueue."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True

from pdblend.bench.comparison_campaign import binding,group_points,load_bound,MODELS
from pdblend.bench.comparison_distserve_inputs import validate_distserve_inputs,PROTECTED
from pdblend.bench.comparison_jobs import resident_job
from pdblend.bench.resident_session import digest,file_sha,write_new
from pdblend_baselines.distserve.deployment import search_deployment
from pdblend_baselines.distserve.stage_surface import StageSurface

OVERLAYS=tuple('pdblend/bench/'+name+'.py' for name in (
    'comparison_runtime','comparison_distserve_inputs','comparison_distserve_acceptance',
    'comparison_native_acceptance','comparison_journal','comparison_meter_method',
    'comparison_meter_preflight','isolated_comparison_meter','resident_session'))
PUBLIC_MEASUREMENT=('pdblend/bench/comparison_metrics.py','pdblend/bench/comparison_metering.py','pdblend/bench/client.py')


def require(condition,message):
    if not condition:raise ValueError(message)


def subsets(files):
    return ({k:v for k,v in files.items() if k.startswith(('pdblend_runtime/','pdblend/engine/'))},
            {k:v for k,v in files.items() if k.startswith('pdblend/measure/') or k in PUBLIC_MEASUREMENT})


def frozen_previous(base,previous):
    """Keep evidence-valid baselines, including genuine SLO/request failures."""
    refs=deepcopy(base.get('preserved_baseline_receipts',{}));paths={}
    for name,ref in refs.items():load_bound(ref);paths[Path(ref['path']).resolve()]=name
    for root in previous:
        require(Path(root).is_dir(),'prior session directory missing: '+str(root))
        for path in Path(root).glob('windows/*/receipt.json'):paths[path.resolve()]=None
    points={p['name']:p for p in base['points']}
    require(len(points)==len(base['points']),'campaign point names are not unique')
    for path,expected_name in paths.items():
        receipt=json.loads(path.read_text())
        if not (receipt.get('evidence_valid') is True and receipt.get('baseline_frozen') is True
                and receipt.get('cleanup_passed') is True):
            require(expected_name is None,'parent preserved receipt is no longer a valid frozen baseline')
            continue
        artifacts=receipt.get('artifacts',{})
        require({'point.json','result.json'}<=set(artifacts),'frozen baseline lacks bound point/result artifacts')
        for name,checksum in artifacts.items():
            target=(path.parent/name).resolve()
            require(target.is_relative_to(path.parent) and file_sha(target)==checksum,
                    'frozen baseline bytes changed: '+str(target))
        point=json.loads((path.parent/'point.json').read_text());result=json.loads((path.parent/'result.json').read_text())
        name=point['name'];ref=binding(path)
        require(point.get('system') in ('mixed','distserve','ecoserve','dynamollm')
                and digest(point)==receipt.get('point_sha256') and points.get(name)==point
                and result==receipt.get('result') and result.get('evidence_valid') is True,
                'parent/frozen baseline point or result identity differs')
        require(expected_name in (None,name) and (name not in refs or refs[name]==ref),
                'multiple different frozen observations for one baseline point')
        refs[name]=ref
    return refs


def freezer_module():
    spec=importlib.util.spec_from_file_location('distserve_comparison_freezer',ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def profile_assets(model,assets):
    """Hash failures reject the package; honest missing/failed profiles block points."""
    require(assets.get('profile_source_manifest'),'model-owned immutable profile source is required')
    source=load_bound(assets['profile_source_manifest'])
    require(source.get('files') and digest(source['files'])==source.get('source_sha256'),'profile source inventory differs')
    root=Path(assets['profile_source_manifest']['path']).resolve().parent
    freezer_module().verify_snapshot(root,source['files'])
    refs=[];problems=[];tps=set()
    for pair in assets.get('profiles',[]):
        ref=pair['surface'];artifact=load_bound(ref);complete=load_bound(pair['completion'])
        require(Path(ref['path']).resolve().parent==Path(pair['completion']['path']).resolve().parent
                and Path(ref['path']).name=='surface.json' and Path(complete.get('surface_path','')).name=='surface.json'
                and complete.get('surface_sha256')==ref['sha256'], 'stage completion does not bind its owned surface')
        if not (complete.get('status')=='passed' and complete.get('complete') is True
                and complete.get('hardware_executed') is True and not complete.get('cleanup_errors')
                and complete.get('calibration_passed') is True and artifact.get('qualified') is True):
            problems.append('stage collection or independent holdout is incomplete/unqualified');continue
        try:
            surface=StageSurface.load(ref['path'],frequency=2520)
            identity=surface.identity;tp=identity['tp']
            require(identity.get('system')=='distserve' and identity.get('model_id')==model
                    and identity.get('pp')==1 and identity.get('source_revision')==source['source_sha256']
                    and identity.get('engine_revision')=='vllm-0.10.1.1'
                    and tp in ((2,4) if '32B-' in model else (1,2,4))
                    and type(identity.get('capacity_tokens')) is int and identity['capacity_tokens']>0,
                    'own model/source/native topology or measured capacity differs')
            require(tp not in tps,'duplicate own stage profile for one TP')
        except (ValueError,KeyError,TypeError,OSError) as exc:
            problems.append(str(exc));continue
        refs.append(ref);tps.add(tp)
    if not refs:problems.append('no independently audited own StageSurface is available')
    # A supplied failed alternative is never silently removed from the search.
    return source,refs,problems


def extend_source(out,model,assets,mixed_identity,overlays):
    freezer=freezer_module();old=load_bound(assets['profile_source_manifest'])
    root=Path(assets['profile_source_manifest']['path']).resolve().parent
    staged=out/'staged-source'/model;staged.mkdir(parents=True)
    for name in old['files']:
        target=staged/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target)
    additions={}
    for name in overlays:
        relative=Path(name)
        require(relative.parent==Path('pdblend/bench') and
                (relative.name.startswith('comparison_') or relative.name in ('resident_session.py','isolated_comparison_meter.py'))
                and name not in PUBLIC_MEASUREMENT and name not in PROTECTED,
                'only explicit comparison wrapper overlays are permitted: '+name)
        target=staged/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/'src'/name,target);additions[name]=binding(ROOT/'src'/name)
    source,source_sha=freezer.freeze_source(staged,out/'sources');shutil.rmtree(staged)
    files=load_bound(binding(source/'manifest.json'))['files'];runtime,measurement=subsets(files)
    require(runtime and measurement and mixed_identity['runtime_source_sha256']==digest(runtime)
            and mixed_identity['measurement_source_sha256']==digest(measurement),
            'public engine/measurement differs from frozen Mixed baseline')
    require(all(old['files'].get(name) and files.get(name)==old['files'][name] for name in PROTECTED),
            'protected independent baseline mechanism changed since profiling')
    write_new(out/'source-extensions'/(model+'.json'),dict(base_manifest=assets['profile_source_manifest'],
        overlays=additions,source_manifest=binding(source/'manifest.json'),formal_eligible=False))
    return source,source_sha


def prepare(base_path,out,assets_path,*,execution_inputs=None,previous=(),overlays=OVERLAYS,
            max_per_gpu_rate=5.,epsilon=.25,sample_size=None,priority=385):
    out=Path(out).resolve();require(not out.exists(),'new immutable preparation directory required')
    base_ref=binding(base_path);base=load_bound(base_ref);assets_ref=binding(assets_path);assets=load_bound(assets_ref)
    require(assets.get('schema')=='distserve-comparison-assets/v1' and isinstance(assets.get('models'),dict),
            'explicit model-owned comparison assets manifest required')
    execution_ref=binding(execution_inputs or Path(base_path).parent/'execution-inputs.json')
    execution=load_bound(execution_ref)
    require(load_bound(execution['model_verification']).get('all_pass') is True,
            'model verification receipt is not complete')
    preserved=frozen_previous(base,previous)
    out.mkdir(parents=True);points=deepcopy(base['points']);checks=[];prepared=[];sources={}
    params=dict(max_per_gpu_rate=max_per_gpu_rate,epsilon=epsilon,sample_size=sample_size)
    for model in MODELS:
        pending=[p for p in points if p['system']=='distserve' and p['model_id']==model and p['name'] not in preserved]
        if not pending:continue
        owned=assets['models'].get(model)
        if not owned:
            for point in pending:point.update(status='blocked',blockers=['distserve: missing own completed profile assets'],engine_identity=None)
            continue
        _,profiles,problems=profile_assets(model,owned)
        if problems:
            for point in pending:point.update(status='blocked',blockers=['distserve: '+p for p in problems],engine_identity=None)
            continue
        mixed=next((p for p in base['points'] if p['system']=='mixed' and p['model_id']==model and p.get('engine_identity')),None)
        require(mixed is not None,'frozen public Mixed identity absent for '+model)
        source,revision=extend_source(out,model,owned,mixed['engine_identity'],overlays)
        sources[model]=source;source_ref=binding(source/'manifest.json')
        for point in pending:
            point.update(engine_identity=None,status='blocked',blockers=[],formal_eligible=False)
            if not point.get('trace'):
                point['blockers']=['distserve: immutable evaluation trace unavailable'];continue
            calibration=owned.get('calibration',{}).get(point['dataset'])
            if not calibration:
                point['blockers']=['distserve: own calibration corpus unavailable'];continue
            corpus=load_bound(calibration)
            if not (corpus.get('model_name')==model and corpus.get('dataset')==point['dataset'] and corpus.get('calibration')):
                point['blockers']=['distserve: calibration model/dataset/split differs'];continue
            # Search sees only this bound corpus's calibration split. The
            # evaluation trace is first opened by the later input validator.
            try:
                choice=search_deployment([r['path'] for r in profiles],calibration['path'],model=model,
                    rate_rps=point['rate_rps'],ttft_s=point['slo']['ttft_s'],tpot_s=point['slo']['tpot_s'],
                    gpu_budget=8,frequency=2520,**params)
            except (ValueError,KeyError,TypeError,OSError) as exc:
                point['blockers']=['distserve: offline calibration search failed: '+str(exc)];continue
            choice_path=out/'choices'/(point['name']+'.json');write_new(choice_path,choice)
            point['offline_search_receipt']=binding(choice_path)
            if choice.get('status')!='ready_for_native_execution':
                point['blockers']=['distserve: '+choice.get('status','missing_profile')];continue
            selected=choice['selected'];tp=selected['tp'];pairs=selected['replicas']
            identity=deepcopy(mixed['engine_identity']);fleet=identity['fleet_gpu_uuids']
            require(identity['image_digest']==execution['image_digest'] and all(choice['identity'].get(k)==identity[k]
                    for k in ('model_hash','tokenizer_hash','image_digest')), 'selected profile/public engine identity differs')
            require(selected['config']==[1,tp,1,tp,1] and selected['pp']==1 and type(pairs)is int and pairs>0
                    and selected['total_gpu_count']==2*tp*pairs<=8,'unsupported selected symmetric pair layout')
            options=deepcopy(identity['instances'][0]['launch_options']);options['kv_connector']='P2pNcclConnector'
            identity.update(metering_execution='isolated_process',instances=[dict(instance_id=f'dist-{i}-{role}',tp=tp,pp=1,
                gpu_uuids=fleet[(2*i+j)*tp:(2*i+j+1)*tp],launch_options=deepcopy(options))
                for i in range(pairs) for j,role in enumerate(('P','D'))])
            config=dict(system='distserve',model_id=model,model_path='/models/'+model,max_batch_size=32,
                request_timeout_s=240.,search_parameters=params,offline_choice=binding(choice_path))
            config_path=out/'configs'/(point['name']+'.json');write_new(config_path,config)
            point.update(inputs=dict(system_config=binding(config_path),trace=point['trace'],offline_choice=binding(choice_path),
                profiles=profiles,calibration=calibration,profile_source_manifest=owned['profile_source_manifest'],source_manifest=source_ref),
                engine_identity=identity,source_manifest=source_ref,revision=revision,
                qualification_mode='distserve_native_bootstrap',metering_execution='isolated_process',status='prepared',blockers=[],
                topology=dict(mode='offline_tp',selected=deepcopy(selected),gpu_budget=8,selection_split='calibration',
                    scope='qualified_symmetric_TP_PP1_subset',complete_paper_reproduction=False))
            checked=validate_distserve_inputs(point,identity,source_manifest=source_ref,replay_search=True)
            checks.append(dict(point=point['name'],**checked))
            if not checked['preflight_ready']:
                point.update(status='blocked',blockers=['distserve: '+k+': '+v for k,v in checked['gate_failures'].items()],engine_identity=None)
            else:prepared.append(point['name'])
    groups=group_points(points)
    campaign=dict(base,campaign_id=out.name,parent_campaign=base_ref,distserve_assets=assets_ref,
        execution_campaigns=sorted(set(base.get('execution_campaigns',[])+[str(Path(base_path).resolve())])),
        points=points,groups=groups,preserved_baseline_receipts=preserved,
        summary=dict(points=len(points),prepared_points=sum(not p.get('blockers') for p in points),
            resident_sessions=len(groups),new_distserve_points=len(prepared),pure_service_s=150*len(points)))
    write_new(out/'campaign.json',campaign);write_new(out/'input-preflight.json',checks)
    jobs=[]
    for group in group_points([p for p in points if p['name'] in prepared]):
        path=out/'groups'/(group['session_id']+'.json');write_new(path,group)
        job=resident_job(group,path,root=ROOT,source=sources[group['model_id']],image=execution['image_digest'],
            verification=execution['model_verification']['path'],campaign=out/'campaign.json',priority=priority)
        for prior in previous:job['payload']['argv']+=['--previous',str(Path(prior).resolve())]
        job['payload']['prior_sessions']=[str(Path(p).resolve()) for p in previous]
        jobs.append(job)
    write_new(out/'jobs.json',jobs)
    write_new(out/'execution-inputs.json',dict(execution,per_model_source={model:dict(path=str(path),sha256=path.name)
        for model,path in sources.items()},parent_execution_inputs=execution_ref))
    write_new(out/'preparation.json',dict(schema='distserve-comparison-prepared/v1',enqueued=False,hardware_executed=False,
        formal_eligible=False,script=binding(__file__),assets=assets_ref,campaign=binding(out/'campaign.json'),
        jobs=binding(out/'jobs.json'),pending_native_gates=['per-pair KV/probe/reference qualification',
            'isolated metering method audit','actual 150s request/routing/SLO/power/drain acceptance']))
    return dict(**campaign['summary'],jobs=len(jobs),out=str(out),enqueued=False,formal_eligible=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','out','assets'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--execution-inputs',type=Path)
    p.add_argument('--previous',type=Path,action='append',default=[])
    p.add_argument('--overlay',action='append',help='Explicit comparison wrappers; default is the documented comparison-only set')
    p.add_argument('--max-per-gpu-rate',type=float,default=5.)
    p.add_argument('--epsilon',type=float,default=.25)
    p.add_argument('--sample-size',type=int)
    p.add_argument('--priority',type=int,default=385)
    args=p.parse_args()
    print(json.dumps(prepare(args.base,args.out,args.assets,execution_inputs=args.execution_inputs,previous=args.previous,
        overlays=OVERLAYS if args.overlay is None else args.overlay,max_per_gpu_rate=args.max_per_gpu_rate,
        epsilon=args.epsilon,sample_size=args.sample_size,priority=args.priority),indent=2))


if __name__=='__main__':main()
