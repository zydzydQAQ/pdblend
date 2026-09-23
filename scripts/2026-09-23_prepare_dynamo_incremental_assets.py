#!/usr/bin/env python3
"""Prepare immutable CPU prediction-cache commands; no GPU or queue actions."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

ROOT=Path('/home/pdblend4')
IMAGE='sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc'
BASE=ROOT/'results/2026-09-23/dynamo-functional-sources/05ce62a146b4a02a9ccb0b9ceba07642b4c5273c47f7078a72ee7717b873ab49'
OVERLAY=('history_provenance.py','validation.py','transition_evidence.py','prediction_cache.py')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def prepare(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    helper=load('dynamo_asset_freezer',ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    pins=load('dynamo_original_jobs',ROOT/'scripts/2026-09-23_prepare_dynamo_functional_jobs.py')
    original=json.loads((BASE/'manifest.json').read_text());helper.verify_snapshot(BASE,original['files'])
    with tempfile.TemporaryDirectory(prefix='dynamo-assets-') as directory:
        staging=Path(directory)/'src';shutil.copytree(BASE,staging)
        for name in OVERLAY:
            shutil.copyfile(ROOT/'src/pdblend_baselines/dynamollm'/name,staging/'pdblend_baselines/dynamollm'/name)
        source,digest=helper.freeze_source(staging,ROOT/'results/2026-09-23/dynamo-asset-sources')
    commands={}
    for key,(model,_,_) in pins.MODELS.items():
        target=out/key;target.mkdir()
        modeldir=Path('/home/models')/model
        predictor=pins.PREDICTORS[key]
        corpus=ROOT/f'datasets/prepared/2026-09-22-{key}-v1'
        mounts=[(source,'/source','ro'),(modeldir,modeldir,'ro'),(predictor,predictor,'ro'),
                (corpus,corpus,'ro'),(target,target,'rw')]
        argv=['docker','run','--rm','--network=none','--cpus=4','--entrypoint','/opt/venv/bin/python']
        for host,container,mode in mounts:argv+=['-v',f'{host}:{container}:{mode}']
        argv+=['-e','PYTHONPATH=/source','-e','PYTHONDONTWRITEBYTECODE=1','-e','CUDA_VISIBLE_DEVICES=',
               '-e','NVIDIA_VISIBLE_DEVICES=void','-e','TOKENIZERS_PARALLELISM=false','-e','OMP_NUM_THREADS=4',
               IMAGE,'-B','-m','pdblend_baselines.dynamollm.prediction_cache',
               '--model',str(modeldir),'--predictor',str(predictor),'--corpus',str(corpus),
               '--out',str(target),'--splits','evaluation','calibration']
        commands[key]=dict(argv=argv,model_id=model,gpu_count=0,source_sha256=digest,
            image_digest=IMAGE,output=str(target),arrival_status='missing_rate_anchor',
            fitting_performed=False,formal_eligible=False)
    manifest=dict(schema='dynamo-incremental-cpu-assets-v1',source_snapshot=str(source),
                  source_sha256=digest,image_digest=IMAGE,commands=commands,queue_modified=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    print(json.dumps(prepare(parser.parse_args().out)))
