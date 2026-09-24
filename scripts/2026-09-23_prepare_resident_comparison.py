#!/usr/bin/env python3
"""Freeze 180 points and enqueue only bound, lease-scoped resident sessions."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

from pdblend.bench.comparison_campaign import prepare, export, MODELS, SCALES, DATASETS
from pdblend.bench.resident_session import digest, file_sha, write_new
from pdblend.experimentation.lease import GPULeaseQueue, gpu_snapshot
from pdblend.model_registry import ModelRegistry

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT/'results/2026-09-22/three-model/profile-receipts/model-verification-99fabb0721f21aa50eb2a8518877acdf05cc76df32f0f769900be7b4d4471fc8.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--first-spec', type=Path, default=ROOT/'results/2026-09-23/first-five-system-batch-v3/spec.json')
    p.add_argument('--queue', type=Path, default=ROOT/'results/2026-09-22/three-model/queue.json')
    p.add_argument('--enqueue', action='store_true')
    p.add_argument('--allow-mixed-qualification', action='store_true',
                   help='Prepare real native Mixed qualification; formal acceptance is independently audited afterwards')
    p.add_argument('--inputs', type=Path, help='Additional prequalified per-point system inputs')
    args = p.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    module = importlib.util.spec_from_file_location('source_freezer', ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    freezer = importlib.util.module_from_spec(module); module.loader.exec_module(freezer)
    source, source_sha = freezer.freeze_source(ROOT/'src', out/'sources')
    files = json.loads((source/'manifest.json').read_text())['files']
    runtime = {k: v for k, v in files.items() if k.startswith(('pdblend_runtime/', 'pdblend/engine/'))}
    measurement = {k: v for k, v in files.items() if k.startswith('pdblend/measure/') or k in (
        'pdblend/bench/comparison_metrics.py', 'pdblend/bench/comparison_metering.py',
        'pdblend/bench/client.py')}
    image = subprocess.check_output(['docker', 'image', 'inspect', 'pdblend:l20-cu128-vllm-v1',
                                     '--format', '{{.Id}}'], text=True).strip()
    inventory = gpu_snapshot()
    if len(inventory) != 8:
        raise RuntimeError('comparison requires the physical eight-GPU host')
    uuids = [inventory[i]['uuid'] for i in sorted(inventory)]
    registry = ModelRegistry('/models', verification_receipt=VERIFY)
    inputs = json.loads(args.inputs.read_text()) if args.inputs else {}
    env = dict(VLLM_USE_V1='1', VLLM_WORKER_MULTIPROC_METHOD='spawn', TOKENIZERS_PARALLELISM='false',
               NCCL_P2P_DISABLE='0', NCCL_IB_DISABLE='1', NCCL_CUMEM_ENABLE='0')
    if args.allow_mixed_qualification:
        for model in MODELS:
            spec = registry.get(model)
            tp = 2 if '32B' in model else 1
            identity = dict(model_hash=spec.model_hash, tokenizer_hash=spec.tokenizer_hash,
                image_digest=image, runtime_source_sha256=digest(runtime),
                measurement_source_sha256=digest(measurement), entrypoint='pdblend_runtime.serve',
                worker_extension='native_v1', dtype='bfloat16', fleet_gpu_uuids=uuids, environment=env,
                instances=[dict(instance_id=f'mixed{i}', tp=tp, pp=1, gpu_uuids=uuids[i*tp:(i+1)*tp],
                    launch_options=dict(max_model_len=8192, gpu_memory_utilization=.85,
                        max_num_seqs=32, max_num_batched_tokens=8192, kv_connector=None,
                        extra_args=['--enforce-eager'])) for i in range(8//tp)])
            size = model.split('-')[1].lower()
            for ds in DATASETS:
                for scale in SCALES:
                    name = f'{size}-mixed-{ds}-x{scale:g}-seed701'
                    inputs[name] = dict(engine_identity=identity, qualification_mode='mixed_native_bootstrap',
                        blockers=[], revision=source_sha, source_manifest=dict(
                            path=str(source/'manifest.json'), sha256=file_sha(source/'manifest.json')))
    campaign = prepare(args.first_spec, out, ROOT/'datasets/prepared', prepared_inputs=inputs)
    jobs = []
    for index, group in enumerate(campaign['groups']):
        path = out/'groups'/(group['session_id']+'.json'); write_new(path, group)
        job_id = 'comparison-'+group['model_id'].split('-')[1].lower()+'-'+digest(group)[:16]
        argv = ['docker','run','--rm','--name',job_id,'--gpus','all','--cap-add','SYS_ADMIN',
                '--ipc=host','--network=host','--shm-size=16g','--ulimit','nofile=65536:65536',
                '--entrypoint','/opt/venv/bin/python']
        for src, dst, mode in [(str(source), '/opt/pdblend-src','ro'),
                               (str(ROOT),str(ROOT),'ro'),('/home/models','/models','ro'),
                               ('/tmp/pdblend-physical-clock-owners','/tmp/pdblend-physical-clock-owners','rw'),
                               ('{attempt_dir}','{attempt_dir}','rw'),
                               ('{attempt_dir}','/output','rw')]:
            argv += ['-v',f'{src}:{dst}:{mode}']
        config_env = dict(env, PYTHONPATH='/opt/pdblend-src', PYTHONDONTWRITEBYTECODE='1',
            PDBLEND_SOURCE_MANIFEST=str(source/'manifest.json'), PDBLEND_SOURCE_SHA256=source_sha,
            PDBLEND_IMAGE_ID=image, PDBLEND_MODELS_DIR='/models', PDBLEND_MODEL_VERIFICATION_RECEIPT=str(VERIFY),
            PDBLEND_GPU_UUIDS='{lease_gpu_uuids}', PDBLEND_CLOCK_LOCK_DIR='/tmp/pdblend-physical-clock-owners',
            PDBLEND_CONCURRENCY_ENVIRONMENT='/output/concurrency-environment.json')
        for key, value in config_env.items():
            argv += ['-e',key+'='+value]
        argv += [image,'-B','-m','pdblend.bench.comparison_runtime','--group',str(path),
                 '--out','{attempt_dir}/session','--base-port','{lease_port}']
        jobs.append(dict(job_id=job_id, priority=800-index, max_attempts=1,
            payload=dict(argv=argv, container_name=job_id, gpu_count=8, exclusive=True,
                         reserve_host=True, timeout_s=18000, required_receipts=['session/completion.json'],
                         cwd=str(ROOT), source_sha256=source_sha, image_digest=image,
                         comparison_campaign=str(out/'campaign.json'), session_id=group['session_id'])))
    write_new(out/'jobs.json',jobs)
    write_new(out/'execution-inputs.json',dict(source_sha256=source_sha,source=str(source),image_digest=image,
               model_verification=dict(path=str(VERIFY),sha256=file_sha(VERIFY)),gpu_uuids=uuids))
    export(out/'campaign.json',ROOT/'results/compare.csv')
    if args.enqueue:
        queue = GPULeaseQueue(args.queue)
        for job in jobs:
            queue.enqueue(**job)
    print(json.dumps(dict(**campaign['summary'], jobs=len(jobs), enqueued=args.enqueue,
                          campaign=str(out/'campaign.json')),indent=2))


if __name__ == '__main__':
    main()
