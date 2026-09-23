"""One resident engine for optional long holdout and short experimental panels."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from . import long_holdout_only, resident_long_holdout, short_domain_collect
from .long_context_collect import digest
from .wave import atomic_json


def package_identity(*, short_package=None, long_package=None):
    if not short_package and not long_package:
        raise ValueError('at least one immutable package is required')
    manifests=[]
    if long_package:manifests.append(long_holdout_only.load_package(long_package)[0])
    if short_package:manifests.append(short_domain_collect.load_package(short_package)[0])
    identity={k:manifests[0][k] for k in ('system','model_id','model_hash','tokenizer_hash','tp','pp')}
    if any(any(m[k]!=v for k,v in identity.items()) for m in manifests):
        raise ValueError('resident packages differ in model identity')
    return identity


def preflight(*, model, gpus, epochs_root, member, input_manifest,
              short_package=None, long_package=None, **unused):
    """Validate the actual frozen invocation without constructing a GPU meter."""
    from importlib.metadata import version
    from ..model_registry import ModelRegistry
    identity=package_identity(short_package=short_package,long_package=long_package)
    if len(gpus)!=identity['tp'] or len(set(gpus))!=len(gpus):
        raise ValueError('resident group must equal one exact TP instance')
    expected=json.loads(Path(input_manifest).read_text())
    paths=dict(source_manifest=Path(os.environ['PDBLEND_SOURCE_MANIFEST']),
        model_verification=Path(os.environ['PDBLEND_MODEL_VERIFICATION_RECEIPT']),
        cohort=Path(epochs_root)/'cohort.json')
    if short_package:paths['short_manifest']=Path(short_package)/'manifest.json'
    if long_package:paths['long_manifest']=Path(long_package)/'manifest.json'
    actual={k:digest(p) for k,p in paths.items()}
    if actual!=expected['exact_inputs_sha256']:
        raise ValueError('resident frozen input checksum differs')
    manifest=json.loads(paths['source_manifest'].read_text())
    canonical=hashlib.sha256(json.dumps(manifest['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (canonical!=manifest['source_sha256'] or canonical!=expected['source_sha256']
            or canonical!=os.environ['PDBLEND_SOURCE_SHA256']
            or os.environ['PDBLEND_IMAGE_ID']!=expected['image_digest']):
        raise ValueError('resident frozen source or image differs')
    source=Path(__file__).resolve().parents[2]
    for name,sha in manifest['files'].items():
        if digest(source/name)!=sha:raise ValueError('resident source changed: '+name)
    cohort=json.loads(paths['cohort'].read_text())
    if (member not in cohort['members'] or len(set(cohort['members']))!=len(cohort['members'])
            or cohort['cohort_id']!=expected['sampling_cohort']):
        raise ValueError('resident cohort membership differs')
    spec=ModelRegistry(Path(model).parent,verification_receipt=paths['model_verification']).get(Path(model).name)
    spec.validate_config();spec.validate_topology(identity['tp'],1)
    if any(getattr(spec,k)!=identity[k] for k in ('model_id','model_hash','tokenizer_hash')):
        raise ValueError('resident model verification differs')
    if version('vllm')!='0.10.1.1' or not version('torch').startswith('2.7.1'):
        raise ValueError('resident pinned runtime differs')
    return dict(status='cpu_preflight_passed',hardware_executed=False,formal_eligible=False,
        energy_comparable=False,source_sha256=canonical,image_digest=expected['image_digest'],
        source_files_verified=len(manifest['files']),exact_inputs_sha256=actual,
        sampling_cohort=cohort['cohort_id'],member=member,**identity)


def run(*,model,gpus,base_port,out,epochs_root,member,short_package=None,long_package=None):
    from .profiler import Profiler,_load_flock
    from .sampling_epochs import SamplingEpochs
    from ..engine.launcher import Fleet
    from ..engine.client import EngineClient
    out=Path(out)
    identity=package_identity(short_package=short_package,long_package=long_package)
    if len(gpus)!=identity['tp'] or len(set(gpus))!=len(gpus):raise ValueError('resident group must equal one exact TP instance')
    profiler=Profiler(model,gpus,tp=identity['tp'],pp=1,system='pdblend',out_dir=out,hardware_id='8xL20-lease',
        base_port=base_port,kv_connector='P2pNcclConnector')
    if any(profiler.raw.get(k)!=v for k,v in identity.items()):raise ValueError('live resident model differs from packages')
    epoch=SamplingEpochs(Path(epochs_root),member,profiler)
    result=dict(status='failed',complete=False,formal_eligible=False,energy_comparable=False,stages={},**identity)
    cleanup_permitted=False;clocks_reset=False
    try:
        with Fleet(profiler.specs,out/'logs') as fleet:
            with _load_flock():fleet.start_all()
            instance=fleet[profiler.specs[0].instance_id];profiler.raw['kv_capacity_tokens']=profiler._kv_capacity(instance)
            async def sample():
                nonlocal cleanup_permitted
                await epoch.ready()
                async with EngineClient(instance.spec.instance_id,instance.spec.base_url) as client:
                    shared=dict(profiler=profiler,client=client,gpus=gpus,
                        window_boundary=epoch.window_boundary,qualification_guard=epoch.qualification_guard)
                    if long_package:
                        result['stages']['long_holdout']=await resident_long_holdout.run_existing(package=long_package,out=out/'long-holdout',**shared)
                    if short_package:
                        result['stages']['short_experimental']=await short_domain_collect.run_existing(package=short_package,out=out/'short',**shared)
                await epoch.retire();cleanup_permitted=True
            try:asyncio.run(sample())
            except BaseException as exc:
                # Invalidate peers before Fleet starts unloading this member.
                epoch.fail(exc);raise
        # Clock changes are part of physical cleanup and happen while peers wait.
        profiler.meter.reset_all();clocks_reset=True
        if not cleanup_permitted:raise RuntimeError('resident cleanup did not reach epoch boundary')
        epoch.released()
        complete=all(x['complete'] for x in result['stages'].values())
        result.update(status='passed' if complete else 'inconclusive',complete=complete,
            queue_receipt_semantics='measurement_completion; each component qualification remains separate',
            full_profile_qualified=False,epoch_cleanup_released=True,
            component_receipts={name:dict(path=str(path),sha256=digest(path)) for name,path in (
                ('long_holdout',out/'long-holdout/completion.json'),('short_experimental',out/'short/completion.json')) if path.is_file()})
    except BaseException as exc:
        result.update(status='failed',complete=False,error=f'{type(exc).__name__}: {exc}');epoch.fail(exc)
    finally:
        try:
            if not clocks_reset:profiler.meter.reset_all()
        except BaseException as exc:result.update(status='failed',complete=False,cleanup_error=str(exc));epoch.fail(exc)
        profiler._checkpoint();atomic_json(out/'completion.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--gpus',nargs='+',type=int,required=True)
    p.add_argument('--base-port',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs-root',type=Path,default=os.environ.get('PDBLEND_SAMPLING_EPOCH_ROOT'))
    p.add_argument('--member',default=os.environ.get('PDBLEND_PROFILE_MEMBER'))
    p.add_argument('--short-package',type=Path);p.add_argument('--long-package',type=Path)
    p.add_argument('--input-manifest',type=Path,required=True);p.add_argument('--preflight-only',action='store_true')
    args=vars(p.parse_args());only=args.pop('preflight_only');binding=args.pop('input_manifest')
    if not args['epochs_root'] or not args['member']:
        p.error('sampling epoch root and member are required via CLI or environment')
    result=preflight(**args,input_manifest=binding)
    if only:atomic_json(args['out']/'preflight.json',result)
    else:result=run(**args)
    print(json.dumps(result,indent=2));raise SystemExit(0 if result.get('complete') or result['status']=='cpu_preflight_passed' else 1)


if __name__=='__main__':main()
