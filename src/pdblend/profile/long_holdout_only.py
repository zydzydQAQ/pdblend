"""Independent long-context holdout from already complete 14B TP1 training.

Preparation is CPU-only. Execution loads one instance once, or uses the explicit
resident callback. It never repeats or modifies the 36 existing training points.
"""
from __future__ import annotations
import argparse
import asyncio
import copy
import json
import statistics
import time
from pathlib import Path

from . import long_context_collect as lc, long_context_followup as lf
from .identity import sha256_value
from .power_calibration import write_immutable
from .wave import atomic_json

IDENTITY=('pdblend','Qwen2.5-14B-Instruct',1,1)


def implementation_hashes():
    return {name:lc.digest(Path(__file__).with_name(name)) for name in (
        'long_holdout_only.py','long_context_collect.py','long_context_followup.py',
        'window_sampling.py','profiler.py','wave.py','parallel.py','identity.py')}


def fit_completed_training(raw):
    """The candidate family and exact batches match existing long-domain fits."""
    if tuple(raw.get(k) for k in ('system','model_id','tp','pp'))!=IDENTITY:
        raise ValueError('requires own 14B TP1 PP1 training')
    expected={(f,b,c) for f in lf.FREQUENCIES for b in (1,4) for c in (5120,7168,7680)}
    if (len(raw['decode'])!=36 or {(r['freq_mhz'],r['batch'],r['context_tokens']) for r in raw['decode']}!=expected
            or any(r.get('evidence_class')!='training_extension' or r.get('independent_holdout') is not False for r in raw['decode'])):
        raise ValueError('complete independent-system training anchors required; holdout cannot train')
    nodes={}
    for f in lf.FREQUENCIES:
        for b in (1,4):
            group=sorted((r for r in raw['decode'] if (r['freq_mhz'],r['batch'])==(f,b)),key=lambda r:r['context_tokens'])
            nodes[f'{f}/{b}']=[dict(context=statistics.fmean(p['effective_context_tokens'] for p in r['repeats']),
                step_seconds=statistics.fmean(p['step_seconds'] for p in r['repeats']),
                power_w=statistics.fmean(p['power_w'] for p in r['repeats'])) for r in group]
    return dict(schema=1,kind=lf.KIND,**lf.identity(raw),exact_batches=[1,4],nodes=nodes,
        domain_definition='closed interval of observed training means at each exact frequency/batch',
        batch_interpolation_qualified=False,short_profile_unchanged=True,holdout_used=False,formal_eligible=False)


def prepare(*,training,out):
    training,out=Path(training).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('use a new immutable holdout-only package directory')
    completion=json.loads((training/'completion.json').read_text());raw=json.loads((training/'raw.json').read_text())
    training_plan=json.loads((training/'training-plan.json').read_text())
    if (completion.get('complete') is not True or completion.get('status')!='passed'
            or completion.get('raw_sha256')!=lc.digest(training/'raw.json') or completion.get('fit_performed') is not False
            or completion.get('holdout_points_consumed')!=0):raise ValueError('completed training-only receipt required')
    lf.verify_training(raw,training,training_plan['training'],lc.digest(training/'training-plan.json'))
    candidate=fit_completed_training(raw)
    candidate['training_binding']=dict(raw_sha256=lc.digest(training/'raw.json'),
        training_plan_sha256=lc.digest(training/'training-plan.json'),completion_sha256=lc.digest(training/'completion.json'))
    plan=lf.holdout_plan(candidate,dict(raw,decode=[]),raw,raw['kv_capacity_tokens'])
    if len(plan['points'])!=24 or plan['missing_points']:raise ValueError('all 24 bounded holdout points must be feasible')
    out.mkdir(parents=True)
    write_immutable(out/'candidate.json',candidate);write_immutable(out/'holdout-plan.json',plan)
    manifest=dict(schema=1,kind='completed_training_fresh_long_holdout_v1',**lf.identity(raw),
        candidate_sha256=lc.digest(out/'candidate.json'),plan_sha256=lc.digest(out/'holdout-plan.json'),
        training=str(training),inputs={name:dict(path=str(training/name),sha256=lc.digest(training/name))
            for name in ('raw.json','training-plan.json','completion.json')},implementation_sha256=implementation_hashes(),
        environment=raw['environment'],exact_batches=[1,4],expected_points=24,repeats=3,
        reused_training_points=36,new_training_points=0,candidate_frozen_before_holdout=True,
        batch_interpolation_qualified=False,fit_uses_holdout=False,formal_eligible=False,energy_comparable=False)
    write_immutable(out/'manifest.json',manifest);load_package(out)
    return manifest


def load_package(package):
    package=Path(package);manifest=json.loads((package/'manifest.json').read_text())
    if (tuple(manifest.get(k) for k in ('system','model_id','tp','pp'))!=IDENTITY
            or manifest.get('implementation_sha256')!=implementation_hashes()):raise ValueError('holdout-only identity/implementation differs')
    for name,key in [('candidate.json','candidate_sha256'),('holdout-plan.json','plan_sha256')]:
        if lc.digest(package/name)!=manifest[key]:raise ValueError('immutable candidate/plan changed')
    for binding in manifest['inputs'].values():
        if lc.digest(binding['path'])!=binding['sha256']:raise ValueError('own training input changed')
    training=Path(manifest['training']);raw=json.loads((training/'raw.json').read_text());tplan=json.loads((training/'training-plan.json').read_text())
    lf.verify_training(raw,training,tplan['training'],manifest['inputs']['training-plan.json']['sha256'])
    expected=fit_completed_training(raw)
    expected['training_binding']={k:manifest['inputs'][name]['sha256'] for k,name in (
        ('raw_sha256','raw.json'),('training_plan_sha256','training-plan.json'),('completion_sha256','completion.json'))}
    candidate=json.loads((package/'candidate.json').read_text());plan=json.loads((package/'holdout-plan.json').read_text())
    if candidate!=expected or plan!=lf.holdout_plan(candidate,dict(raw,decode=[]),raw,raw['kv_capacity_tokens']):
        raise ValueError('candidate/plan differs from training-only reconstruction')
    return manifest,candidate,plan


async def run_existing(*,package,profiler,client,gpus,out):
    """Resident callback; caller owns qualification, clocks, Fleet and cleanup."""
    package,out=Path(package),Path(out);manifest,candidate,plan=load_package(package)
    if (lf.identity(profiler.raw)!=lf.identity(manifest) or len(gpus)!=1 or len(profiler.specs)!=1):
        raise ValueError('resident engine differs from 14B TP1 own training')
    for key in ('image_digest','vllm','torch','cuda','hardware_id'):
        if not manifest['environment'].get(key) or profiler.raw['environment'].get(key)!=manifest['environment'][key]:
            raise ValueError('resident engine environment mismatch: '+key)
    if any(lc.point_capacity_error(p,profiler.raw['kv_capacity_tokens']) for p in plan['points']):
        raise ValueError('holdout reservation exceeds actual resident capacity')
    qualifier=profiler.raw.get('external_interference') or {}
    source=lf.qualification_receipt(profiler.out_dir,qualifier)
    before=copy.deepcopy(profiler.raw);child=copy.copy(profiler);child.out_dir=out;out.mkdir(parents=True,exist_ok=True)
    dest=out/'samples'/('qualification-'+qualifier['samples_sha256']+'.json');dest.parent.mkdir(exist_ok=True)
    if dest.exists() and lc.digest(dest)!=qualifier['samples_sha256']:raise ValueError('saved qualifier changed')
    if not dest.exists():dest.write_bytes(source.read_bytes())
    binding=dict(candidate_sha256=manifest['candidate_sha256'],plan_sha256=manifest['plan_sha256'],package_manifest_sha256=lc.digest(package/'manifest.json'))
    child.raw=copy.deepcopy(profiler.raw)
    for key in ('external_interference','parallel_interference','concurrency_environment','identity_sha256','evidence_bindings'):
        child.raw.pop(key,None)
    child.raw.update(prefill=[],decode=[],mixed=[],transfer=[],static={},decode_pending={},independent_holdout=True,
        measurement_plan_sha256=manifest['plan_sha256'],long_holdout_binding=binding,
        evidence_class='independent_long_context_holdout',config=dict(child.raw['config'],independent_long_context_holdout=True))
    if (out/'raw.json').exists():
        child.resume()
        if child.raw.get('long_holdout_binding')!=binding:raise ValueError('resume belongs to different candidate/plan')
    history=child.raw.setdefault('qualification_history',{})
    for old in history.values():lf.qualification_receipt(out,old)
    history[qualifier['samples_sha256']]=dict(samples_file=str(dest.relative_to(out)),samples_sha256=qualifier['samples_sha256'])
    done=lc.resume_points(child.raw,out,plan['points'])
    result=dict(status='running',complete=False,calibration_passed=False,expected_points=24,
        independent_holdout=True,fit_performed=False,new_training_points=0,reused_training_points=36,
        candidate_frozen_before_holdout=True,formal_eligible=False,energy_comparable=False,binding=binding)
    try:
        previous=None
        for point in plan['points']:
            key=lc.point_key(point)
            if key in done:continue
            if previous!=point['freq_mhz']:
                child._lock(point['freq_mhz'],gpus);previous=point['freq_mhz'];await asyncio.sleep(2)
            def checkpoint(repeats):
                for rep in repeats:rep.setdefault('qualification_sha256',qualifier['samples_sha256'])
                child.raw['decode_pending'][key]=repeats;child._checkpoint()
            row=await lc.collect_bounded_decode_point(child,client,gpus,point,purpose='independent_holdout_repair',
                previous=child.raw['decode_pending'].get(key,()),on_window=checkpoint)
            child.raw['decode'].append(row);child.raw['decode_pending'].pop(key,None);done.add(key);child._checkpoint()
            print(f'14B long holdout {len(done)}/24 f={point["freq_mhz"]} B={point["batch"]} {point["long_context_role"]}',flush=True)
        checked=lf.audit(candidate,child.raw,out,plan);atomic_json(out/'holdout-audit.json',checked);load_package(package)
        result.update(status='completed' if checked['complete'] else 'inconclusive',complete=checked['complete'],
            calibration_passed=checked['passed'],audit_sha256=lc.digest(out/'holdout-audit.json'),measured_points=len(done))
    except BaseException as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        child._checkpoint();result['raw_sha256']=lc.digest(out/'raw.json');atomic_json(out/'completion.json',result)
        if profiler.raw!=before:raise ValueError('resident holdout modified caller profile archive')
    return result


def run(*,package,model,gpus,base_port,out):
    from .profiler import Profiler,_load_flock
    from .wave import ProfileWave
    from ..engine.launcher import Fleet
    from ..engine.client import EngineClient
    package,out=Path(package),Path(out);manifest,_,_=load_package(package)
    if len(gpus)!=1 or len(set(gpus))!=1:raise ValueError('14B TP1 holdout owns exactly one GPU')
    wave=ProfileWave.from_environment()
    if wave is None:raise ValueError('fresh coordinated ProfileWave qualification required')
    profiler=Profiler(model,gpus,tp=1,pp=1,system='pdblend',out_dir=out,hardware_id='8xL20-lease',base_port=base_port,kv_connector='P2pNcclConnector')
    if lf.identity(profiler.raw)!=lf.identity(manifest):raise ValueError('live model/tokenizer identity differs')
    result=dict(status='failed',complete=False,calibration_passed=False,formal_eligible=False,energy_comparable=False,
        model_id=manifest['model_id'],tp=1,pp=1,new_training_points=0,reused_training_points=36,started_s=time.time())
    try:
        with Fleet(profiler.specs,out/'logs') as fleet:
            with _load_flock():fleet.start_all()
            instance=fleet[profiler.specs[0].instance_id];profiler.raw['kv_capacity_tokens']=profiler._kv_capacity(instance)
            async def sample():
                await wave.qualify_external(profiler,fleet)
                async with wave.measurement():
                    async with EngineClient(instance.spec.instance_id,instance.spec.base_url) as client:
                        return await run_existing(package=package,profiler=profiler,client=client,gpus=instance.spec.gpus,out=out/'holdout')
            held=asyncio.run(sample())
        result.update(status='passed' if held['complete'] else 'inconclusive',complete=held['complete'],
            calibration_passed=held['calibration_passed'],holdout=held,holdout_completion_sha256=lc.digest(out/'holdout/completion.json'),
            queue_receipt_semantics='measurement_completion_only; calibration_reported_separately')
    except BaseException as exc:
        result.update(error=f'{type(exc).__name__}: {exc}');wave.write('error',dict(error=result['error']))
    finally:
        try:profiler.meter.reset_all()
        except BaseException as exc:result.update(status='failed',complete=False,cleanup_error=repr(exc))
        profiler._checkpoint();result.update(finished_s=time.time(),raw_sha256=lc.digest(out/'raw.json'),
            missing_gates=['full_profile_quality_audit','native_system_mechanisms','campaign_acceptance'])
        atomic_json(out/'completion.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--training',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('run');a.add_argument('--package',type=Path,required=True);a.add_argument('--model',required=True)
    a.add_argument('--gpus',type=int,nargs='+',required=True);a.add_argument('--base-port',type=int,required=True);a.add_argument('--out',type=Path,required=True)
    args=vars(p.parse_args());command=args.pop('command');result=prepare(**args) if command=='prepare' else run(**args)
    print(json.dumps(result,indent=2),flush=True)
    if command=='run':raise SystemExit(0 if result['complete'] else 1)


if __name__=='__main__':main()
