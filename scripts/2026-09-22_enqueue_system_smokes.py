#!/usr/bin/env python3
"""Queue real development PDBlend/Mixed smokes ahead of pending profiles.

DistServe/EcoServe/DynamoLLM are deliberately absent: their independent V1 GPU
runners are not integrated yet. Never substitute the legacy shared planners.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pdblend.experimentation.lease import GPULeaseQueue


def load_profile_enqueuer():
    spec = importlib.util.spec_from_file_location('profile_enqueuer', ROOT / 'scripts/2026-09-22_enqueue_parallel_profiles.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def completed_pdblend_profiles(queue_path: Path) -> dict:
    state = json.loads(queue_path.read_text())
    result = {}
    for lease in state['leases'].values():
        job = state['jobs'][lease['job_id']]
        if lease['status'] != 'succeeded' or job['status'] != 'succeeded':
            continue
        directory = Path(lease['attempt_dir'])
        if not directory.is_absolute():
            directory = ROOT / directory
        raw_path, profile = directory / 'raw.json', directory / 'profile.json'
        if not raw_path.is_file() or not profile.is_file():
            continue
        raw = json.loads(raw_path.read_text())
        if raw.get('system') != 'pdblend' or raw.get('pp') != 1:
            continue
        key = (raw.get('model_id'), raw.get('tp'))
        result[key] = profile.resolve()
    return result


def build_specs(*, snapshot: Path, source_hash: str, image: str, verification: Path,
                profiles: dict, duration: float = 100.0) -> list[dict]:
    jobs = []
    for model, tp in (('7B', 1), ('14B', 1), ('32B', 2)):
        model_id = f'Qwen2.5-{model}-Instruct'
        profile = profiles.get((model_id, tp))
        if profile is None:
            raise ValueError(f'missing own measured PDBlend profile: {model_id} TP{tp}')
        for system in ('pdblend', 'mixed'):
            profile_hash = hashlib.sha256(profile.read_bytes()).hexdigest() if system == 'pdblend' else None
            identity = dict(system=system, model_id=model_id, tp=tp, pp=1, seed=701,
                            source_sha256=source_hash, image_digest=image,
                            profile_sha256=profile_hash, duration_s=duration, rate_rps=.2)
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
            name = f'smoke-independent-{system}-{model.lower()}-tp{tp}-{digest}'
            argv = [
                'docker', 'run', '--rm', '--name', name, '--gpus', 'all',
                '--cap-add', 'SYS_ADMIN', '--ipc=host', '--network', 'host', '--shm-size', '16g',
                '--ulimit', 'nofile=65536:65536', '--entrypoint', '/opt/venv/bin/python',
                '-v', f'{snapshot}:/opt/pdblend-src:ro', '-v', '/home/models:/models:ro',
                '-v', '{attempt_dir}:/output:rw',
                '-v', f'{verification}:/verification/model-verification.json:ro',
                '-e', 'PYTHONPATH=/opt/pdblend-src', '-e', 'PDBLEND_MODELS_DIR=/models',
                '-e', 'PDBLEND_MODEL_VERIFICATION_RECEIPT=/verification/model-verification.json',
                '-e', 'PDBLEND_GPU_UUIDS={lease_gpu_uuids}',
                '-e', f'PDBLEND_SOURCE_SHA256={source_hash}', '-e', f'PDBLEND_IMAGE_ID={image}',
                '-e', 'PDBLEND_HARDWARE_ID=8xL20-lease', '-e', 'CUDA_VERSION=12.8.1',
            ]
            if system == 'pdblend':
                argv += ['-v', f'{profile}:/profile/profile.json:ro']
            argv += [image, '-B', '-m', f'pdblend.bench.{system}_smoke',
                     '--model', f'/models/{model_id}', '--gpus', '{lease_local_indices}',
                     '--tp', str(tp), '--base-port', '{lease_port}', '--duration', str(duration),
                     '--seed', '701', '--out', '/output']
            if system == 'pdblend':
                argv += ['--profile', '/profile/profile.json', '--rate', '0.2']
            payload = dict(identity, argv=argv, gpu_count=2*tp, exclusive=False, global_lock=False,
                           container_name=name, required_receipts=['completion.json'], timeout_s=1800,
                           evidence_class='system_development_smoke', formal_eligible=False,
                           energy_comparable=False, calibration_qualified=False,
                           topology=dict(model_id=model_id, tp=tp, pp=1, gpu_count=2*tp),
                           profile_source=str(profile) if system == 'pdblend' else None)
            jobs.append(dict(job_id=name, payload=payload, priority=100, max_attempts=1))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'results/2026-09-22/three-model')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    out = args.root.resolve()
    helper = load_profile_enqueuer()
    image = subprocess.check_output(['docker','image','inspect','pdblend:l20-cu128-vllm-v1','--format','{{.Id}}'], text=True).strip()
    snapshot, source_hash = helper.freeze_source(ROOT/'src', out/'smoke-system-sources', dry_run=args.dry_run)
    receipt, _ = helper.freeze_receipt(out/'model-verification.json',out/'profile-receipts',dry_run=args.dry_run)
    jobs = build_specs(snapshot=snapshot, source_hash=source_hash, image=image, verification=receipt,
                       profiles=completed_pdblend_profiles(out/'queue.json'))
    if not args.dry_run:
        (out/'system-smoke-spec.json').write_text(json.dumps(jobs, indent=2)+'\n')
        q = GPULeaseQueue(out/'queue.json')
        for job in jobs:
            q.enqueue(**job)
    print(json.dumps(dict(dry_run=args.dry_run, source_sha256=source_hash,
                         jobs=[j['job_id'] for j in jobs],
                         unavailable_systems=['distserve','ecoserve','dynamollm']), indent=2))


if __name__ == '__main__':
    main()
