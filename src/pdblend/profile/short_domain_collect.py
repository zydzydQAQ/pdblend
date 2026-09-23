"""Experimental short timing and complete-request power, with fresh holdout.

Diagnostic decode segments never pass the original continuous power gate.
Nothing in this collector changes the existing PerfModel domain.
"""
from __future__ import annotations
import argparse
import asyncio
import copy
import json
import time
from pathlib import Path

from . import short_domain as sd,sampling_guard as guard
from .identity import sha256_value
from .long_context_collect import digest
from .power_calibration import write_immutable
from .wave import atomic_json


def hashes():
    return {n:digest(Path(__file__).with_name(n)) for n in ('short_domain.py','short_domain_collect.py','sampling_guard.py','identity.py')}


def prepare(*,base_candidate,dataset_manifest,identity_raw,out):
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
    write_immutable(out/'manifest.json',manifest);load_package(out);return manifest


def load_package(package):
    package=Path(package);manifest=json.loads((package/'manifest.json').read_text());plan=json.loads((package/'plan.json').read_text())
    if manifest['kind']!=sd.KIND or manifest['implementation_sha256']!=hashes() or digest(package/'plan.json')!=manifest['plan_sha256']:
        raise ValueError('short panel immutable source or plan differs')
    if plan!=sd.make_plan(manifest):raise ValueError('unexpected short measurement matrix')
    for b in manifest['inputs'].values():
        if digest(b['path'])!=b['sha256']:raise ValueError('short panel bound input changed')
    return manifest,plan


async def measure_repeat(profiler,client,gpus,point,repeat,*,_clock=time.time):
    from ..bench.gates import random_prompt
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
        role='complete_repeated_short_requests',experimental_sampling=True,pure_decode_power_qualified=False)
    summary=sd.summarize(evidence)
    return evidence,summary


def verify_repeat(out,rep,point,index,plan_sha):
    path=(Path(out)/rep['samples_file']).resolve()
    if not path.is_relative_to(Path(out).resolve()) or digest(path)!=rep['samples_sha256']:raise ValueError('short raw sample changed')
    data=json.loads(path.read_text())
    if data['point']!=point or data['repeat']!=index or data['plan_sha256']!=plan_sha or rep['summary']!=sd.summarize(data):
        raise ValueError('short summary/shape/plan differs from immutable raw')
    binding=rep['qualification'];saved=Path(out)/binding['samples_file']
    if digest(saved)!=binding['samples_sha256']:raise ValueError('short qualifier changed')
    from .long_context_followup import qualification_receipt
    qualification_receipt(Path(out),binding)
    if (data['epoch_binding']['qualification_sha256']!=binding['samples_sha256'] or
        any(data['epoch_binding'][k]!=binding[k] for k in ('epoch_id','layout_sha256'))):
        raise ValueError('short raw window qualification binding changed')


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
    candidate=None;result=dict(status='failed',complete=False,experimental_sampling=True,formal_eligible=False,
        energy_comparable=False,original_continuous_decode_power_protocol_passed=False,pure_decode_power_qualified=False)
    try:
        for phase in ('training','holdout'):
            if phase=='holdout':
                candidate=sd.fit(plan,raw['training']);candidate['training_rows_sha256']=sha256_value(raw['training'])
                write_immutable(out/'candidate.json',candidate)
            expected={sd.point_key(p):p for p in plan[phase]}
            if any(k not in expected for k in raw[phase]):raise ValueError('foreign point in short archive')
            for point in plan[phase]:
                key=sd.point_key(point);row=raw[phase].setdefault(key,dict(point=point,repeats=[]))
                if row['point']!=point or len(row['repeats'])>point['repeats']:raise ValueError('short resumed row differs')
                for index,rep in enumerate(row['repeats']):verify_repeat(out,rep,point,index,manifest['plan_sha256'])
                for index in range(len(row['repeats']),point['repeats']):
                    if window_boundary is not None:await guard.call(window_boundary,point,index,phase)
                    stamp=guard.snapshot(qualifier);saved=guard.save_binding(out,stamp);profiler._lock(point['freq_mhz'],gpus)
                    evidence,summary=await measure_repeat(profiler,client,gpus,point,index)
                    guard.unchanged(qualifier,stamp)
                    evidence.update(plan_sha256=manifest['plan_sha256'],epoch_binding=stamp,candidate_sha256=digest(out/'candidate.json') if phase=='holdout' else None)
                    path=out/'samples'/f'{phase}-{point["role"]}-{point["freq_mhz"]}-{point["input_tokens"]}-{index}.json'
                    write_immutable(path,evidence)
                    row['repeats'].append(dict(repeat=index,summary=summary,samples_file=str(path.relative_to(out)),samples_sha256=digest(path),qualification=saved))
                    atomic_json(out/'raw.json',raw)
                count=len([r for r in raw[phase].values() if len(r['repeats'])==r['point']['repeats']])
                print(f'short {phase} {count}/{len(plan[phase])} {key}',flush=True)
        audit=sd.audit(candidate,plan,raw['holdout']);atomic_json(out/'audit.json',audit)
        result.update(status='passed' if audit['complete'] else 'inconclusive',complete=audit['complete'],
            experimental_components_passed=audit['experimental_components_passed'],audit_sha256=digest(out/'audit.json'),
            candidate_sha256=digest(out/'candidate.json'),queue_receipt_semantics='measurement_complete_only; experimental_qualification_separate',
            missing_gates=audit['missing_gates'])
    except BaseException as exc:result.update(error=f'{type(exc).__name__}: {exc}');raise
    finally:
        atomic_json(out/'raw.json',raw);result['raw_sha256']=digest(out/'raw.json');atomic_json(out/'completion.json',result)
        if profiler.raw!=before:raise ValueError('short callback modified parent profile archive')
    return result


def run(*,package,model,gpus,base_port,out):
    from .profiler import Profiler,_load_flock
    from .wave import ProfileWave
    from ..engine.launcher import Fleet
    from ..engine.client import EngineClient
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
    a=sub.add_parser('prepare');a.add_argument('--base-candidate',type=Path,required=True);a.add_argument('--dataset-manifest',type=Path,required=True);a.add_argument('--identity-raw',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('run');a.add_argument('--package',type=Path,required=True);a.add_argument('--model',required=True);a.add_argument('--gpus',nargs='+',type=int,required=True);a.add_argument('--base-port',type=int,required=True);a.add_argument('--out',type=Path,required=True)
    args=vars(p.parse_args());command=args.pop('command');result=prepare(**args) if command=='prepare' else run(**args)
    print(json.dumps(result,indent=2))
    if command=='run':raise SystemExit(0 if result['complete'] else 1)


if __name__=='__main__':main()
