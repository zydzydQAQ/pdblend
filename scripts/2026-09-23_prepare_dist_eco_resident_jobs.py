#!/usr/bin/env python3
"""Freeze three executable DistServe -> EcoServe resident jobs; never enqueue.

Each job owns one lease-local pair, profiles remain system-specific, and the
shared /coord lock serializes only model loading. Service windows run in
parallel after each pair is ready.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pdblend_baselines.native_profile import audit

IMAGE = 'sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc'
BASE = ROOT/'results/2026-09-22/three-model/native-acceptance-sources/6130cf13ca38ed0839b7d5d9f3d39e4c0fae3593e9a560bf128ef6ffb4d3d64e'
VERIFY = ROOT/'results/2026-09-22/three-model/profile-receipts/model-verification-99fabb0721f21aa50eb2a8518877acdf05cc76df32f0f769900be7b4d4471fc8.json'
OVERLAYS = ('pdblend_baselines/resident_campaign.py',
            'pdblend_baselines/distserve/run_native.py',
            'pdblend_baselines/ecoserve/run_native.py',
            'pdblend_baselines/ecoserve/runtime.py',
            'pdblend_baselines/ecoserve/mechanism_four.py',
            'pdblend_baselines/native_profile.py')
MODELS = {'7b': ('Qwen2.5-7B-Instruct', 1), '14b': ('Qwen2.5-14B-Instruct', 1),
          '32b': ('Qwen2.5-32B-Instruct', 2)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freezer():
    spec = importlib.util.spec_from_file_location('resident_source_freezer', ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze_execution_source(out):
    helper = freezer()
    manifest = json.loads((BASE/'manifest.json').read_text())
    helper.verify_snapshot(BASE, manifest['files'])
    with tempfile.TemporaryDirectory(prefix='dist-eco-resident-source-') as directory:
        staging = Path(directory)/'src'
        shutil.copytree(BASE, staging)
        for name in OVERLAYS:
            destination = staging/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/'src'/name, destination)
        snapshot, digest = helper.freeze_source(staging, out/'sources')
    helper.verify_snapshot(snapshot, json.loads((snapshot/'manifest.json').read_text())['files'])
    return snapshot, digest


def prepare(out, dynamo_specs, queue_path):
    out = out.resolve()
    if out.exists():
        raise FileExistsError('refusing to overwrite a resident job package')
    prior = json.loads(dynamo_specs.read_text())
    by_model = {entry['payload']['model_id']:entry for entry in prior}
    if set(by_model) != {value[0] for value in MODELS.values()} or len(prior) != 3:
        raise ValueError('exactly three model-bound Dynamo dependencies required')
    queue = json.loads(queue_path.read_text())  # Read-only; this script has no queue mutation API.
    dependencies = sorted(entry['job_id'] for entry in prior)
    if any(job not in queue['jobs'] or queue['jobs'][job]['status'] not in ('queued', 'running', 'completed')
           for job in dependencies):
        raise ValueError('all three declared Dynamo dependencies must already be queued/running/completed')
    verification = json.loads(VERIFY.read_text())
    if verification.get('all_pass') is not True:
        raise ValueError('verified three-model inventory is incomplete')
    out.mkdir(parents=True)
    coord = out/'coord'; coord.mkdir()
    source, source_sha = freeze_execution_source(out)
    jobs, input_manifest = [], {}
    shared_trace_sha = None
    for key, (model, tp) in MODELS.items():
        parent = by_model[model]['payload']
        trace = Path(next(host for host, target, mode in parent['mounts'] if target == '/trace.json')).resolve()
        profile_root = Path(next(host for host, target, mode in parent['mounts']
                                 if host.endswith('/dynamollm-profile') and '/native-probe-'+key+'-' in host)).resolve().parent
        profile = profile_root/'ecoserve-prefill.csv'
        profile_manifest = Path(str(profile)+'.manifest.json')
        own = audit(profile)
        raw = json.loads(profile_manifest.read_text())
        if (own['system'] != 'ecoserve' or raw.get('model') != model
                or own['metadata'].get('tp') != tp or own['metadata'].get('pp') != 1
                or own['metadata'].get('image_digest') != IMAGE):
            raise ValueError('native-probe EcoServe profile identity/topology/image differs')
        trace_sha = sha(trace)
        if shared_trace_sha is not None and trace_sha != shared_trace_sha:
            raise ValueError('three resident jobs must use identical immutable request traces')
        shared_trace_sha = trace_sha
        requests = json.loads(trace.read_text())
        if requests.get('seed') != 701 or not requests.get('requests'):
            raise ValueError('explicit nonempty seed-701 trace required')
        if verification['models'][key]['model_path'] != '/models/'+model:
            raise ValueError('verified model path differs from Docker read-only model path')
        inputs = dict(model_id=model, tp=tp, pp=1, source_sha256=source_sha, image_digest=IMAGE,
                      trace_sha256=trace_sha, csv_sha256=sha(profile),
                      csv_manifest_sha256=sha(profile_manifest), verification_sha256=sha(VERIFY),
                      dependencies=dependencies, load_coord_dir=str(coord), duration_per_system_s=100,
                      seed=701, warmup_seed=9701)
        suffix = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]
        job_id = f'dist-eco-resident-{key}-seed701-{suffix}'
        mounts = [(str(source), '/opt/pdblend-src', 'ro'),
                  (str(source/'manifest.json'), '/source-manifest.json', 'ro'),
                  ('/home/models', '/models', 'ro'), (str(VERIFY), '/verification/model-verification.json', 'ro'),
                  (str(profile_root), str(profile_root), 'ro'), (str(trace), '/trace.json', 'ro'),
                  (str(coord), '/coord', 'rw')]
        argv = ['docker', 'run', '--rm', '--name', job_id, '--gpus', 'all', '--cap-add', 'SYS_ADMIN',
                '--ipc=host', '--network=host', '--shm-size=16g', '--ulimit', 'nofile=65536:65536',
                '--entrypoint', '/opt/venv/bin/python']
        for host, target, mode in mounts:
            argv += ['-v', f'{host}:{target}:{mode}']
        argv += ['-v', '{attempt_dir}:/output:rw']
        env = dict(PYTHONPATH='/opt/pdblend-src', PYTHONDONTWRITEBYTECODE='1', PDBLEND_MODELS_DIR='/models',
                   PDBLEND_MODEL_VERIFICATION_RECEIPT='/verification/model-verification.json',
                   PDBLEND_SOURCE_SHA256=source_sha, PDBLEND_SOURCE_MANIFEST='/source-manifest.json',
                   PDBLEND_IMAGE_ID=IMAGE, PDBLEND_HARDWARE_ID='8xL20-lease', PDBLEND_VLLM_VERSION='0.10.1.1',
                   CUDA_VERSION='12.8.1', PDBLEND_GPU_UUIDS='{lease_gpu_uuids}',
                   CUDA_VISIBLE_DEVICES='{lease_local_indices}', PDBLEND_LEASE_PORT='{lease_port}',
                   PDBLEND_RESIDENT_LOAD_LOCK='/coord/model-load.lock', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
        for name, value in env.items():
            argv += ['-e', name+'='+value]
        argv += ['-e', 'PDBLEND_CONCURRENCY_ENVIRONMENT', '-e', 'PDBLEND_CONCURRENCY_ENVIRONMENT_SHA256',
                 IMAGE, '-B', '-m', 'pdblend_baselines.resident_campaign', '--model', '/models/'+model,
                 '--tp', str(tp), '--gpus', '{lease_local_indices}', '--base-port', '{lease_port}',
                 '--trace', '/trace.json', '--eco-profile', str(profile), '--out', '/output/campaign', '--duration', '100']
        payload = dict(schema='dist-eco-resident-docker-job-v1', prepare_only=False, execution_ready=True,
                       deferred=False, model_id=model, tp=tp, pp=1, gpu_count=2*tp,
                       exclusive=False, global_lock=False, container_name=job_id, depends_on=dependencies,
                       source_revision=source_sha, source_sha256=source_sha, source_snapshot=str(source),
                       source_base_revision=BASE.name, image_digest=IMAGE, hardware_executed=False,
                       formal_eligible=False, energy_comparable=False, argv=argv, mounts=mounts,
                       required_receipts=['campaign/completion.json'], timeout_s=3600,
                       exact_inputs_sha256=inputs,
                       execution=dict(scope='functional', systems=['distserve','ecoserve'], duration_per_system_s=100,
                                      seed=701, warmup_seed=9701, model_load_cycles=1, resident_instances=2,
                                      load_lock='/coord/model-load.lock', load_only_serialized=True,
                                      output_dir='/output/campaign', formal_eligible=False))
        job = dict(job_id=job_id, payload=payload, priority=90, max_attempts=1)
        jobs.append(job)
        destination = out/(key+'.json');destination.write_text(json.dumps(job, indent=2)+'\n')
        input_manifest[key] = dict(job_id=job_id, job_spec_sha256=sha(destination), profile=str(profile),
                                   profile_manifest=str(profile_manifest), profile_raw_root=str(profile_root),
                                   trace=str(trace), hashes=inputs)
    (out/'jobs.json').write_text(json.dumps(jobs, indent=2)+'\n')
    result = dict(schema='dist-eco-resident-job-package-v1', enqueued=False, hardware_executed=False,
                  source_snapshot=str(source), source_sha256=source_sha, source_base=str(BASE),
                  overlay_sha256={name:sha(ROOT/'src'/name) for name in OVERLAYS},
                  builder_sha256=sha(Path(__file__)), dynamo_specs=str(dynamo_specs.resolve()),
                  dynamo_specs_sha256=sha(dynamo_specs), dependencies=dependencies,
                  jobs_sha256=sha(out/'jobs.json'), models=input_manifest, formal_eligible=False,
                  energy_comparable=False, coordinator=dict(path=str(coord), scope='model load only; service windows parallel'))
    (out/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    freezer().verify_snapshot(source, json.loads((source/'manifest.json').read_text())['files'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/2026-09-23/dist-eco-resident-v1')
    parser.add_argument('--dynamo-specs', type=Path, default=ROOT/'results/2026-09-23/dynamo-functional-job-specs-final-v2/jobs.json')
    parser.add_argument('--queue', type=Path, default=ROOT/'results/2026-09-22/three-model/queue.json')
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.dynamo_specs, args.queue), indent=2))


if __name__ == '__main__':
    main()
