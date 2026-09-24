"""Experimental short timing and complete-request power, with fresh holdout.

Diagnostic decode segments never pass the original continuous power gate.
Nothing in this collector changes the existing PerfModel domain.
"""
from __future__ import annotations
import argparse
import asyncio
import copy
import hashlib
import json
import time
import uuid
from pathlib import Path

from pdblend.profile.calibration import short_domain as sd; from pdblend.profile.collection import sampling_guard as guard
from pdblend.profile.identity import sha256_value
from pdblend.profile.collection.long_context_collect import digest
from pdblend.profile.calibration.power_calibration import write_immutable
from pdblend.profile.collection.wave import atomic_json


def hashes():
    from pdblend.source_inventory import implementation_hashes as inventory_hashes
    return inventory_hashes()


def prepare(*,base_candidate,dataset_manifest,identity_raw,out,resume_from=None):
    base_candidate,dataset_manifest,identity_raw,out=map(Path,(base_candidate,dataset_manifest,identity_raw,out))
    if out.exists():raise FileExistsError('new immutable package directory required')
    base=json.loads(base_candidate.read_text());data=json.loads(dataset_manifest.read_text())
    original=json.loads(identity_raw.read_text());identity={k:original[k] for k in sd.IDENTITY}
    if (identity['model_id']!=data['model_name'] or identity['model_id']!=Path(base['model']).name or
        any(identity[k]!=base[k] for k in ('system','tp','pp'))):raise ValueError('own tokenizer/model mismatch')
    plan=sd.make_plan(identity)
    out.mkdir(parents=True);write_immutable(out/'plan.json',plan)
    manifest=dict(schema=1,kind=sd.KIND,**identity,plan_sha256=digest(out/'plan.json'),implementation_sha256=hashes(),
        inputs={name:dict(path=str(path.resolve()),sha256=digest(path)) for name,path in (('base_candidate',base_candidate),('dataset_manifest',dataset_manifest),('identity_raw',identity_raw))},
        environment=original['environment'],
        expected_training_points=len(plan['training']),expected_holdout_points=len(plan['holdout']),
        experimental_sampling=True,formal_eligible=False,energy_comparable=False,pure_decode_power_qualified=False,
        fit_uses_holdout=False,reused_short_profile_unchanged=True)
    if resume_from is not None:
        resumed,files=validate_inherited_archive(resume_from,plan,manifest)
        manifest['inherited_archive']=dict(path=str(Path(resume_from).resolve()),
            raw_sha256=digest(Path(resume_from)/'raw.json'),files_sha256=files,
            original_binding=resumed['binding'],
            accepted_windows=sum(len(r['repeats']) for phase in ('training','holdout') for r in resumed[phase].values()))
    write_immutable(out/'manifest.json',manifest);load_package(out);return manifest


def load_package(package):
    package=Path(package);manifest=json.loads((package/'manifest.json').read_text());plan=json.loads((package/'plan.json').read_text())
    if manifest['kind']!=sd.KIND or manifest['implementation_sha256']!=hashes() or digest(package/'plan.json')!=manifest['plan_sha256']:
        raise ValueError('short panel immutable source or plan differs')
    if plan!=sd.make_plan(manifest):raise ValueError('unexpected short measurement matrix')
    for b in manifest['inputs'].values():
        if digest(b['path'])!=b['sha256']:raise ValueError('short panel bound input changed')
    if 'inherited_archive' in manifest:
        binding=manifest['inherited_archive'];root=Path(binding['path'])
        if digest(root/'raw.json')!=binding['raw_sha256']:raise ValueError('inherited short raw changed')
        raw,files=validate_inherited_archive(root,plan,manifest)
        if files!=binding['files_sha256'] or raw['binding']!=binding['original_binding']:
            raise ValueError('inherited short samples or source binding changed')
    return manifest,plan


def validate_inherited_archive(root,plan,manifest):
    """Audit accepted windows under their original source and qualification."""
    root=Path(root);raw=json.loads((root/'raw.json').read_text());binding=raw['binding']
    if (raw['kind']!=sd.KIND or binding['identity']!={k:manifest[k] for k in sd.IDENTITY}
            or binding['plan_sha256']!=manifest['plan_sha256']):
        raise ValueError('inherited short model, shape plan or identity differs')
    for key in ('image_digest','vllm','torch','cuda','hardware_id'):
        if not binding['environment'].get(key) or binding['environment'][key]!=manifest['environment'][key]:
            raise ValueError('inherited short runtime differs: '+key)
    source=binding['environment'].get('source_hash')
    if not isinstance(source,str) or len(source)!=64:raise ValueError('inherited source hash missing')
    # The parent's immutable source manifest binds every collector implementation.
    parent_manifest=root.parent/'manifest.json'
    if not parent_manifest.is_file():raise ValueError('inherited source manifest missing')
    parent=json.loads(parent_manifest.read_text())
    payload=parent['payload'];snapshot=Path(payload['source_snapshot'])
    source_manifest=snapshot/'manifest.json';frozen=json.loads(source_manifest.read_text())
    canonical=hashlib.sha256(json.dumps(frozen['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (source!=payload['source_sha256'] or source!=frozen['source_sha256'] or source!=canonical
            or payload['image_digest']!=binding['environment']['image_digest']):
        raise ValueError('inherited source/image identity differs')
    for name,sha in frozen['files'].items():
        path=(snapshot/name).resolve()
        if not path.is_relative_to(snapshot.resolve()) or digest(path)!=sha:
            raise ValueError('inherited frozen source changed: '+name)
    execution=root.parent/'execution.json'
    if not execution.is_file():raise ValueError('inherited execution provenance missing')
    invocation=json.loads(execution.read_text())
    argv=invocation['argv']
    if 'PDBLEND_SOURCE_SHA256='+source not in argv:raise ValueError('inherited execution source differs')
    old_package=Path(argv[argv.index('--short-package')+1]);old_manifest=old_package/'manifest.json'
    if digest(old_manifest)!=binding['package_sha256'] or digest(old_package/'plan.json')!=binding['plan_sha256']:
        raise ValueError('inherited collection package changed')
    original=json.loads(old_manifest.read_text())
    for name,sha in original['implementation_sha256'].items():
        if frozen['files'].get('pdblend/profile/'+name)!=sha:
            raise ValueError('inherited package implementation differs from frozen source')
    files={'../manifest.json':digest(parent_manifest),'../execution.json':digest(execution),
        str(source_manifest):digest(source_manifest),str(old_manifest):digest(old_manifest),
        str(old_package/'plan.json'):digest(old_package/'plan.json')}
    for phase in ('training','holdout'):
        expected={sd.point_key(p):p for p in plan[phase]}
        for key,row in raw[phase].items():
            if key not in expected or row['point']!=expected[key] or len(row['repeats'])>row['point']['repeats']:
                raise ValueError('inherited short point differs')
            for index,rep in enumerate(row['repeats']):
                verify_repeat(root,rep,row['point'],index,binding['plan_sha256'])
                for item in (rep,rep['qualification']):files[item['samples_file']]=item['samples_sha256']
    if any(row['repeats'] for row in raw['holdout'].values()):
        raise ValueError('partial short recovery must precede independent holdout fit')
    return raw,files


async def measure_repeat(profiler,client,gpus,point,repeat,*,_clock=time.time):
    from pdblend.bench.gates import random_prompt
    prompt=random_prompt(point['input_tokens'],9701+repeat+point['input_tokens']);requests=[];warm=[]
    def receipt(r):
        return {k:getattr(r,k) for k in ('request_id','submitted_s','first_token_s','finished_s','token_times_s',
            'completion_tokens','prompt_tokens','error','stream_done','usage_received')}
    async def request(index,phase):
        tag=f'short-{point["purpose"]}-{point["role"]}-{point["freq_mhz"]}-{point["input_tokens"]}-{repeat}-{phase}-{index}'
        result=await client.complete(prompt,point['output_tokens'],tag,token_diagnostics=True)
        if result.error:raise RuntimeError('short request failed: '+str(result.error))
        return receipt(result)
    settle_start=_clock()
    while _clock()-settle_start<point['settle_s'] or not warm:warm.append(await request(len(warm),'settle'))
    settle_end=_clock();sampler=profiler.meter.sampler(gpus);sampler.start();start=_clock();decode_s=0.
    try:
        while _clock()-start<point['measure_s'] or len(requests)<3 or decode_s<point['decode_accumulation_s']:
            if _clock()-start>180:raise TimeoutError('short workload could not accumulate bounded decode time')
            row=await request(len(requests),'measure');requests.append(row)
            if len(row['token_times_s'])>1:decode_s+=row['token_times_s'][-1]-row['token_times_s'][0]
        end=_clock()
    finally:sampler.stop()
    if sampler.error:raise RuntimeError('short sampler failed: '+str(sampler.error))
    evidence=dict(schema=1,point=point,repeat=repeat,start_s=start,end_s=end,settle_start_s=settle_start,settle_end_s=settle_end,
        requests=requests,warmup_requests=warm,gpu_count=len(gpus),measured_gpu_ids=list(gpus),
        power=[x for x in sampler.samples if start<=x[0]<=end],frequency=[x for x in sampler.frequency_samples if start<=x[0]<=end],
        sampler_source=getattr(profiler.meter,'power_source',None),power_sensor_lag_qualified=False,
        utilization=[x for x in getattr(sampler,'utilization_samples',[]) if start<=x[0]<=end],
        power_metadata=[x for x in getattr(sampler,'power_metadata',[]) if start<=x['t_s']<=end],
        role='complete_repeated_short_requests',experimental_sampling=True,pure_decode_power_qualified=False)
    try:summary=sd.summarize(evidence)
    except ValueError as exc:
        # Keep rejected raw data available to the caller.  In particular, a
        # clock failure must retain its actual values, not only a generic error.
        exc.evidence=evidence
        raise
    return evidence,summary


def verify_repeat(out,rep,point,index,plan_sha):
    path=(Path(out)/rep['samples_file']).resolve()
    if not path.is_relative_to(Path(out).resolve()) or digest(path)!=rep['samples_sha256']:raise ValueError('short raw sample changed')
    data=json.loads(path.read_text())
    if data['point']!=point or data['repeat']!=index or data['plan_sha256']!=plan_sha or rep['summary']!=sd.summarize(data):
        raise ValueError('short summary/shape/plan differs from immutable raw')
    binding=rep['qualification'];saved=Path(out)/binding['samples_file']
    if digest(saved)!=binding['samples_sha256']:raise ValueError('short qualifier changed')
    from pdblend.profile.collection.long_context_followup import qualification_receipt
    qualification_receipt(Path(out),binding)
    if (data['epoch_binding']['qualification_sha256']!=binding['samples_sha256'] or
        any(data['epoch_binding'][k]!=binding[k] for k in ('epoch_id','layout_sha256'))):
        raise ValueError('short raw window qualification binding changed')


async def qualified_repeat(*,profiler,client,gpus,point,index,phase,out,plan_sha,
                           qualifier,window_boundary,rejected,candidate_sha=None):
    """Retry only clock rejection; all protocol/isolation errors remain fatal."""
    for attempt in range(3):
        if window_boundary is not None:await guard.call(window_boundary,point,index,phase)
        stamp=guard.snapshot(qualifier);saved=guard.save_binding(out,stamp)
        profiler._lock(point['freq_mhz'],gpus)
        try:
            evidence,summary=await measure_repeat(profiler,client,gpus,point,index)
        except ValueError as exc:
            evidence=getattr(exc,'evidence',None)
            if evidence is not None:
                evidence.update(plan_sha256=plan_sha,epoch_binding=stamp,candidate_sha256=candidate_sha)
                path=Path(out)/'rejected'/f'{phase}-{point["role"]}-{point["freq_mhz"]}-{point["input_tokens"]}-{index}-{uuid.uuid4().hex}.json'
                write_immutable(path,evidence)
                rejected.append(dict(point_key=sd.point_key(point),repeat=index,attempt=attempt,
                    samples_file=str(path.relative_to(out)),samples_sha256=digest(path),qualification=saved,
                    error=f'{type(exc).__name__}: {exc}',diagnostics=getattr(exc,'diagnostics',None)))
                atomic_json(Path(out)/'rejections.json',dict(windows=rejected))
            # An invalid epoch must not be downgraded to a recoverable frequency
            # gap, even when both errors occurred in the same window.
            guard.unchanged(qualifier,stamp)
            if not isinstance(exc,sd.ShortClockMismatch):raise
            if attempt==2:return None
            continue
        guard.unchanged(qualifier,stamp)
        evidence.update(plan_sha256=plan_sha,epoch_binding=stamp,candidate_sha256=candidate_sha)
        return evidence,summary,saved


def inherit_samples(out,manifest,raw):
    binding=manifest.get('inherited_archive')
    if not binding:return
    old=Path(binding['path']);archive=json.loads((old/'raw.json').read_text())
    raw['inherited_archive']=copy.deepcopy(binding)
    for phase in ('training','holdout'):
        raw[phase]=copy.deepcopy(archive[phase])
        for row in raw[phase].values():
            for rep in row['repeats']:
                for item in (rep,rep['qualification']):
                    src=old/item['samples_file'];dest=Path(out)/item['samples_file']
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    if dest.exists() and digest(dest)!=item['samples_sha256']:
                        raise ValueError('inherited short destination changed')
                    if not dest.exists():dest.write_bytes(src.read_bytes())


async def run_existing(*,package,profiler,client,gpus,out,window_boundary=None,qualification_guard=None):
    package,out=Path(package),Path(out);manifest,plan=load_package(package);out.mkdir(parents=True,exist_ok=True)
    if (any(profiler.raw.get(k)!=manifest[k] for k in sd.IDENTITY) or len(gpus)!=manifest['tp'] or
            len(set(gpus))!=len(gpus)):
        raise ValueError('short panel resident model/tokenizer/TP identity differs')
    for key in ('image_digest','vllm','torch','cuda','hardware_id'):
        if not manifest['environment'].get(key) or profiler.raw['environment'].get(key)!=manifest['environment'][key]:
            raise ValueError('short panel engine environment mismatch: '+key)
    qualifier=qualification_guard or guard.static_guard(profiler);before=copy.deepcopy(profiler.raw)
    binding=dict(package_sha256=digest(package/'manifest.json'),plan_sha256=manifest['plan_sha256'],
        identity={k:profiler.raw[k] for k in sd.IDENTITY},environment=profiler.raw['environment'])
    raw=dict(schema=1,kind=sd.KIND,binding=binding,training={},holdout={},experimental_sampling=True)
    if (out/'raw.json').exists():
        raw=json.loads((out/'raw.json').read_text())
        if raw['binding']!=binding:raise ValueError('short resume identity/source/environment differs')
    else:inherit_samples(out,manifest,raw)
    rejected=json.loads((out/'rejections.json').read_text())['windows'] if (out/'rejections.json').exists() else []
    missing=[]
    candidate=None;result=dict(status='failed',complete=False,experimental_sampling=True,formal_eligible=False,
        energy_comparable=False,original_continuous_decode_power_protocol_passed=False,pure_decode_power_qualified=False)
    try:
        for phase in ('training','holdout'):
            if phase=='holdout':
                if missing:break
                candidate=sd.fit(plan,raw['training']);candidate['training_rows_sha256']=sha256_value(raw['training'])
                write_immutable(out/'candidate.json',candidate)
            expected={sd.point_key(p):p for p in plan[phase]}
            if any(k not in expected for k in raw[phase]):raise ValueError('foreign point in short archive')
            for point in plan[phase]:
                key=sd.point_key(point);row=raw[phase].setdefault(key,dict(point=point,repeats=[]))
                if row['point']!=point or len(row['repeats'])>point['repeats']:raise ValueError('short resumed row differs')
                for index,rep in enumerate(row['repeats']):verify_repeat(out,rep,point,index,manifest['plan_sha256'])
                for index in range(len(row['repeats']),point['repeats']):
                    measured=await qualified_repeat(profiler=profiler,client=client,gpus=gpus,point=point,index=index,
                        phase=phase,out=out,plan_sha=manifest['plan_sha256'],qualifier=qualifier,
                        window_boundary=window_boundary,rejected=rejected,
                        candidate_sha=digest(out/'candidate.json') if phase=='holdout' else None)
                    if measured is None:
                        missing.append(dict(point_key=key,repeat=index,reason='actual_clock_mismatch_after_three_windows'))
                        atomic_json(out/'raw.json',raw)
                        break
                    evidence,summary,saved=measured
                    path=out/'samples'/f'{phase}-{point["role"]}-{point["freq_mhz"]}-{point["input_tokens"]}-{index}.json'
                    write_immutable(path,evidence)
                    row['repeats'].append(dict(repeat=index,summary=summary,samples_file=str(path.relative_to(out)),samples_sha256=digest(path),qualification=saved))
                    atomic_json(out/'raw.json',raw)
                count=len([r for r in raw[phase].values() if len(r['repeats'])==r['point']['repeats']])
                print(f'short {phase} {count}/{len(plan[phase])} {key}',flush=True)
        if missing:
            result.update(status='inconclusive',complete=False,missing_profile=missing,
                missing_gates=['actual_clock_qualification'],rejected_windows=len(rejected),
                queue_receipt_semantics='sampling_finished_with_missing_points; no_profile_promotion')
            return result
        audit=sd.audit(candidate,plan,raw['holdout']);atomic_json(out/'audit.json',audit)
        result.update(status='passed' if audit['complete'] else 'inconclusive',complete=audit['complete'],
            experimental_components_passed=audit['experimental_components_passed'],audit_sha256=digest(out/'audit.json'),
            candidate_sha256=digest(out/'candidate.json'),queue_receipt_semantics='measurement_complete_only; experimental_qualification_separate',
            missing_gates=audit['missing_gates'])
    except BaseException as exc:result.update(error=f'{type(exc).__name__}: {exc}');raise
    finally:
        atomic_json(out/'raw.json',raw);result['raw_sha256']=digest(out/'raw.json')
        result['rejected_windows']=len(rejected)
        if (out/'rejections.json').exists():result['rejections_sha256']=digest(out/'rejections.json')
        atomic_json(out/'completion.json',result)
        if profiler.raw!=before:raise ValueError('short callback modified parent profile archive')
    return result


def run(*,package,model,gpus,base_port,out):
    from pdblend.profile.collection.profiler import Profiler, _load_flock
    from pdblend.profile.collection.wave import ProfileWave
    from pdblend.engine.launcher import Fleet
    from pdblend.engine.client import EngineClient
    manifest,_=load_package(package);out=Path(out);wave=ProfileWave.from_environment()
    if wave is None:raise ValueError('real profile cohort qualification required')
    profiler=Profiler(model,gpus,tp=manifest['tp'],pp=1,system='pdblend',out_dir=out,hardware_id='8xL20-lease',base_port=base_port,kv_connector='P2pNcclConnector')
    result=dict(status='failed',complete=False,formal_eligible=False,experimental_sampling=True)
    try:
        with Fleet(profiler.specs,out/'logs') as fleet:
            with _load_flock():fleet.start_all()
            instance=fleet[profiler.specs[0].instance_id];profiler.raw['kv_capacity_tokens']=profiler._kv_capacity(instance)
            async def sample():
                await wave.qualify_external(profiler,fleet)
                async with wave.measurement():
                    async with EngineClient(instance.spec.instance_id,instance.spec.base_url) as client:
                        return await run_existing(package=package,profiler=profiler,client=client,gpus=gpus,out=out/'short')
            result=asyncio.run(sample())
    except BaseException as exc:result.update(status='failed',complete=False,error=f'{type(exc).__name__}: {exc}');wave.write('error',dict(error=result['error']))
    finally:
        try:profiler.meter.reset_all()
        except BaseException as exc:result.update(status='failed',complete=False,cleanup_error=str(exc))
        atomic_json(out/'completion.json',dict(result,child_completion=str(out/'short/completion.json')))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--base-candidate',type=Path,required=True);a.add_argument('--dataset-manifest',type=Path,required=True);a.add_argument('--identity-raw',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--resume-from',type=Path)
    a=sub.add_parser('run');a.add_argument('--package',type=Path,required=True);a.add_argument('--model',required=True);a.add_argument('--gpus',nargs='+',type=int,required=True);a.add_argument('--base-port',type=int,required=True);a.add_argument('--out',type=Path,required=True)
    args=vars(p.parse_args());command=args.pop('command');result=prepare(**args) if command=='prepare' else run(**args)
    print(json.dumps(result,indent=2))
    if command=='run':raise SystemExit(0 if result['complete'] else 1)


if __name__=='__main__':main()
