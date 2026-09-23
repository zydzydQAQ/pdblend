#!/usr/bin/env python3
"""Prepare deferred Docker job specs for the three Dynamo functional runs.

This only writes reviewable specs.  It deliberately never opens queue.json or
starts Docker/GPU work; the root scheduler supplies powerpair dependencies,
lease-local GPU indices, ports, and attempt output directories later.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, shutil, tempfile
from pathlib import Path

ROOT = Path('/home/pdblend4')
SOURCE = ROOT / 'results/2026-09-22/three-model/native-acceptance-sources/51be08d6dcd3f835c775767712aac30f5cff267edfbd5bfb379a895690d4c708'
IMAGE = 'sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc'
VERIFY = ROOT / 'results/2026-09-22/three-model/profile-receipts/model-verification-99fabb0721f21aa50eb2a8518877acdf05cc76df32f0f769900be7b4d4471fc8.json'
PREDICTORS = {
    '7b': ROOT/'results/2026-09-22/three-model/queue-attempts/native-predictor-7b-1c0032a85c1aea8d/attempt-0001-c65efc1925674c688ccc1a8107173174/predictor',
    '14b': ROOT/'results/2026-09-22/three-model/queue-attempts/native-predictor-14b-43501eaf2094bb36/attempt-0001-3d3b8c6e18934ddbb7c2665ed3108a5d/predictor',
    '32b': ROOT/'results/2026-09-22/three-model/queue-attempts/native-predictor-32b-ee21e0de3bbaf1e2/attempt-0001-dec91d0554fa41fca7dd84a14c3c466a/predictor',
}
MODELS = {'7b': ('Qwen2.5-7B-Instruct', 1, [0, 1]), '14b': ('Qwen2.5-14B-Instruct', 1, [0, 1]),
          '32b': ('Qwen2.5-32B-Instruct', 2, [0, 1, 2, 3])}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()

def freeze_execution_source():
    spec = importlib.util.spec_from_file_location('dynamo_source_freezer',
        ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    parent = json.loads((SOURCE/'manifest.json').read_text())
    helper.verify_snapshot(SOURCE, parent['files'])
    with tempfile.TemporaryDirectory(prefix='dynamo-functional-source-') as directory:
        staging = Path(directory)/'src'
        shutil.copytree(SOURCE, staging)
        for name in ('run_v1.py', 'prepare_v1.py', 'portable_profile.py'):
            target = staging/'pdblend_baselines/dynamollm'/name
            shutil.copy2(ROOT/'src/pdblend_baselines/dynamollm'/name, target)
        return helper.freeze_source(staging, ROOT/'results/2026-09-23/dynamo-functional-sources')


def make_spec(key, out, source_overlay, source_sha256, dependencies):
    model, tp, gpus = MODELS[key]
    prepared = ROOT / 'results/2026-09-23/functional-prepared' / f'{key}-run1'
    cfg = json.loads((prepared/'config.json').read_text())
    cfg['model_path'] = f'/models/{model}'
    cfg['tokenizer'] = f'/models/{model}'
    cfg['profiles'] = '/initial-profile.json'
    cfg['dynamo_predictor_dir'] = '/predictor'
    cfg['trace'] = '/trace.json'
    cfg['node_gpus'] = gpus
    cfg['instances'] = [dict(x, gpus=[gpus[i*tp+j] for j in range(tp)], port=19000+2*i,
                              url=f'http://127.0.0.1:{19000+2*i}') for i, x in enumerate(cfg['instances'])]
    specdir = out / key
    specdir.mkdir(parents=True, exist_ok=True)
    profile = ROOT / 'results/2026-09-23/functional-prepared' / f'{key}-run1/profiles.json'
    prediction = ROOT/'results/2026-09-23/independent-baseline-coverage-v7'/f'{key}-predictor-trace-predictions.json'
    prediction_copy = specdir/'prediction-receipt.json'
    prediction_value = json.loads(prediction.read_text())
    trace = Path(prediction_value['trace']).resolve()
    trace_value = json.loads(trace.read_text())
    if (prediction_value.get('model_id') != model or prediction_value.get('seed') != 701
            or trace_value.get('seed') != 701
            or Path(prediction_value['checkpoint']).resolve() != PREDICTORS[key].resolve()):
        raise ValueError('predictor measurement does not match model, checkpoint or seed')
    predictions = prediction_value.get('predictions', [])
    requests = trace_value.get('requests', [])
    if (len(predictions) != len(requests) or not requests
            or any(row.get('request_index') != index
                   or row.get('input_tokens') != len(request['prompt'])
                   or row.get('trace_max_tokens') != request['max_tokens']
                   or type(row.get('predicted_output')) is not int
                   or row['predicted_output'] < 1
                   for index, (row, request) in enumerate(zip(predictions, requests)))):
        raise ValueError('predictor receipt does not bind every trace request')
    outputs = sorted({row['predicted_output'] for row in predictions})
    prediction_value['derived_binding'] = dict(source=str(prediction), source_sha256=digest(prediction),
        trace_hash_scope='file_bytes', seed=701, formal_eligible=False)
    prediction_value['trace_sha256'] = digest(trace)
    prediction_value['predictor_manifest_sha256'] = digest(PREDICTORS[key]/'manifest.json')
    prediction_copy.write_text(json.dumps(prediction_value, indent=2)+'\n')
    cfg['functional_profile_stage'] = {'enabled': True, 'mode': 'functional',
        'frequency_groups': [[900,1200,1500],[1800,2100,2520]],
        'outputs': outputs, 'trace_sha256': digest(trace), 'prediction_receipt': '/prediction-receipt.json',
        'prediction_receipt_sha256':digest(prediction_copy), 'collector_timeout_s':1200,
        'output_dir': 'functional-profile-stage', 'settle_s': 2, 'measure_s': 5}
    cfgpath = specdir/'config.json'; cfgpath.write_text(json.dumps(cfg, indent=2)+'\n')
    mounts = [
        (str(source_overlay), '/opt/pdblend-src', 'ro'), (str(Path('/home/models')), '/models', 'ro'),
        (str(source_overlay/'manifest.json'), '/source-manifest.json', 'ro'),
        (str(VERIFY), '/verification/model-verification.json', 'ro'),
        (str(PREDICTORS[key]), '/predictor', 'ro'), (str(profile), '/initial-profile.json', 'ro'),
        (str(trace), '/trace.json', 'ro'), (str(prediction_copy), '/prediction-receipt.json', 'ro'),
    ]
    profile_value = json.loads(profile.read_text())
    for point in profile_value.get('points', []):
        cell_root = str(Path(point['source_profile_path']).resolve().parent.parent)
        mounts.append((cell_root, cell_root, 'ro'))
    mounts = sorted(set(mounts))
    identity = dict(source_sha256=source_sha256, config_sha256=digest(cfgpath),
                    trace_sha256=digest(trace), predictor_manifest_sha256=digest(PREDICTORS[key]/'manifest.json'))
    suffix = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    job_id = f'dynamo-functional-{key}-seed701-{suffix}'
    argv = ['docker','run','--rm','--name',job_id,
            '--gpus','all','--cap-add','SYS_ADMIN','--ipc=host','--network=host','--shm-size=16g',
            '--ulimit','nofile=65536:65536','--entrypoint','/opt/venv/bin/python']
    for host, target, mode in mounts: argv += ['-v', f'{host}:{target}:{mode}']
    run_output = '/output/dynamo'
    argv += ['-v','{attempt_dir}:/output:rw','-e','PYTHONPATH=/opt/pdblend-src',
             '-e','PDBLEND_MODELS_DIR=/models','-e','PDBLEND_MODEL_VERIFICATION_RECEIPT=/verification/model-verification.json',
             '-e',f'PDBLEND_SOURCE_SHA256={source_sha256}',
             '-e',f'PDBLEND_IMAGE_ID={IMAGE}','-e','PDBLEND_HARDWARE_ID=8xL20-lease',
             '-e','PDBLEND_VLLM_VERSION=0.10.1.1','-e','CUDA_VERSION=12.8.1',
             '-e','PDBLEND_GPU_UUIDS={lease_gpu_uuids}',
             '-e','CUDA_VISIBLE_DEVICES={lease_local_indices}','-e','PDBLEND_LEASE_PORT={lease_port}',
             '-e','PDBLEND_CONCURRENCY_ENVIRONMENT','-e','PDBLEND_CONCURRENCY_ENVIRONMENT_SHA256',
             '-e','PDBLEND_SOURCE_MANIFEST=/source-manifest.json',
             '-e','PYTHONDONTWRITEBYTECODE=1','-e','TOKENIZERS_PARALLELISM=false','-e','OMP_NUM_THREADS=4',
             IMAGE,'-B','-m','pdblend_baselines.dynamollm.run_v1','--config','/spec/config.json',
             '--trace','/trace.json','--out',run_output,'--duration','100','--seed','701','--mode','functional']
    # Config is mounted separately after the command is assembled so every
    # immutable input has an explicit read-only path in the spec.
    argv.insert(argv.index('-v'), '-v'); argv.insert(argv.index('-v') + 1, f'{cfgpath}:/spec/config.json:ro')
    payload = {'schema':'dynamollm-functional-docker-job-v1','prepare_only':True,
            'deferred':dependencies == ['__DEFERRED_POWERPAIR__'],'model_id':model,'tp':tp,'gpu_count':len(gpus),'pp':1,
            'container_name':job_id,
            'depends_on':dependencies,'source_revision':source_sha256,
            'source_sha256':source_sha256,'source_snapshot':str(source_overlay),
            'source_base_revision':SOURCE.name,
            'exclusive':False,'global_lock':False,
            'image_digest':IMAGE,'formal_eligible':False,'energy_comparable':False,
            'hardware_executed':False,'argv':argv,'config_path':str(cfgpath),
            'required_receipts':['dynamo/completion.json'],'timeout_s':1800,
            'mounts':mounts+[ (str(cfgpath),'/spec/config.json','ro') ],
            'exact_inputs_sha256':{'config':digest(cfgpath),'profile':digest(profile),'trace':digest(trace),
                                   'prediction_receipt':digest(prediction_copy),'source_tree':source_sha256},
            'execution':{'duration_s':100,'seed':701,'profile_cells':18*len(outputs),'profile_split':'9+9 per output',
                         'output_dir':run_output,'lease_gpu_placeholder':'{lease_local_indices}',
                         'lease_port_placeholder':'{lease_port}'} }
    return {'job_id':job_id,'payload':payload,'priority':100,'max_attempts':1}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--depends-on',nargs='+',default=['__DEFERRED_POWERPAIR__'])
    args=ap.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    source_overlay, source_sha256 = freeze_execution_source()
    specs={k:make_spec(k,args.out,source_overlay,source_sha256,args.depends_on) for k in MODELS}
    for k,s in specs.items(): (args.out/f'{k}.json').write_text(json.dumps(s,indent=2)+'\n')
    (args.out/'jobs.json').write_text(json.dumps(list(specs.values()),indent=2)+'\n')
    (args.out/'manifest.json').write_text(json.dumps({'schema':'dynamollm-functional-job-manifest-v1',
        'prepare_only':True,'source_snapshot':str(source_overlay),'source_sha256':source_sha256,
        'jobs':{k:dict(job_id=s['job_id'],spec_sha256=digest(args.out/f'{k}.json')) for k,s in specs.items()}},indent=2)+'\n')

if __name__ == '__main__': main()
