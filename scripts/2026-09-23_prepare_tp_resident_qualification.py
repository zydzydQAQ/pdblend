#!/usr/bin/env python3
"""Prepare, freeze and CPU-preflight the three-GPU resident job; never enqueue."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.dont_write_bytecode = True
from pdblend.bench.tp_qualification import canonical_sha

IMAGE = 'sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/2026-09-23/tp-resident-qualification-v1')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    old = json.loads((ROOT / 'results/2026-09-23/pdblend-tp-resident-enqueued-v1/jobs.json').read_text())[0]['payload']
    spec = importlib.util.spec_from_file_location('tp_source_freezer', ROOT / 'scripts/2026-09-22_enqueue_parallel_profiles.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    source, source_sha = helper.freeze_source(ROOT / 'src', out / 'sources')
    helper.verify_snapshot(source, json.loads((source / 'manifest.json').read_text())['files'])
    rng = random.Random(701)
    rows = []
    for arrival, count in [(0, 96), (30, 96), (60, 1), (63, 1), (66, 1)]:
        for _ in range(count):
            i = len(rows)
            length = (512, 1024, 2048)[i % 3]
            rows.append(dict(idx=i, arrival_s=arrival, prompt=[rng.randrange(100, 5000) for _ in range(length)],
                             max_tokens=128, source='resident-dual-tp-capacity-burst-seed701'))
    write(out / 'trace.json', dict(seed=701, requests=rows, trace_sha256=canonical_sha(rows)))
    layout = [dict(pool_id='small', tp=1, replicas=1), dict(pool_id='large', tp=2, replicas=1)]
    write(out / 'resident-layout.json', layout)
    paths = {key: Path(value['path']) for key, value in old['profile_inputs'].items()}
    checksums = dict(source_manifest=sha(source / 'manifest.json'), resident_layout=sha(out / 'resident-layout.json'),
                     tp1_profile=sha(paths['tp1']), tp2_profile=sha(paths['tp2']),
                     trace=sha(out / 'trace.json'), model_verification=sha(paths['model_verification']))
    bindings = dict(source_sha256=source_sha, image_digest=IMAGE, exact_inputs_sha256=checksums)
    write(out / 'input-manifest.json', bindings)
    job_id = 'pdblend-tp-resident-7b-three-gpu-dual-pool-qualification-v1'
    mounts = [(source, '/opt/pdblend-src', 'ro'), (source / 'manifest.json', '/source-manifest.json', 'ro'),
              (Path('/home/models'), '/models', 'ro'),
              (paths['model_verification'], '/verification/model-verification.json', 'ro'),
              (paths['tp1'], '/profiles/7b-tp1.json', 'ro'), (paths['tp2'], '/profiles/7b-tp2.json', 'ro'),
              (out / 'resident-layout.json', '/spec/resident-layout.json', 'ro'),
              (out / 'trace.json', '/spec/trace.json', 'ro'),
              (out / 'input-manifest.json', '/spec/input-manifest.json', 'ro'),
              ('{attempt_dir}', '/output', 'rw')]
    environment = dict(PYTHONPATH='/opt/pdblend-src', PYTHONDONTWRITEBYTECODE='1', PDBLEND_MODELS_DIR='/models',
                       PDBLEND_MODEL_VERIFICATION_RECEIPT='/verification/model-verification.json',
                       PDBLEND_SOURCE_SHA256=source_sha, PDBLEND_SOURCE_MANIFEST='/source-manifest.json',
                       PDBLEND_IMAGE_ID=IMAGE, PDBLEND_VLLM_VERSION='0.10.1.1', CUDA_VERSION='12.8.1',
                       PDBLEND_GPU_UUIDS='{lease_gpu_uuids}', CUDA_VISIBLE_DEVICES='{lease_local_indices}',
                       PDBLEND_LEASE_PORT='{lease_port}')
    argv = ['docker', 'run', '--rm', '--name', job_id, '--gpus', 'all', '--cap-add', 'SYS_ADMIN',
            '--ipc=host', '--network=host', '--shm-size=16g', '--ulimit', 'nofile=65536:65536',
            '--entrypoint', '/opt/venv/bin/python']
    for host, target, mode in mounts:
        argv += ['-v', f'{host}:{target}:{mode}']
    for key, value in environment.items():
        argv += ['-e', f'{key}={value}']
    argv += ['-e', 'PDBLEND_CONCURRENCY_ENVIRONMENT', '-e', 'PDBLEND_CONCURRENCY_ENVIRONMENT_SHA256']
    image_token = 'pdblend:l20-cu128-vllm-v1@' + IMAGE
    argv += [image_token, '-B', '-m', 'pdblend.bench.tp_qualification', '--model', 'Qwen2.5-7B-Instruct',
             '--gpus', '{lease_local_indices}', '--profile', '/profiles/7b-tp1.json',
             '--topology-profile', '1=/profiles/7b-tp1.json', '--topology-profile', '2=/profiles/7b-tp2.json',
             '--resident-layout', '/spec/resident-layout.json', '--trace', '/spec/trace.json',
             '--input-manifest', '/spec/input-manifest.json', '--base-port', '{lease_port}',
             '--out', '/output/tp-qualification']
    payload = dict(schema=1, system='pdblend', model_id='Qwen2.5-7B-Instruct', tp=1, pp=1, gpu_count=3,
                   source_snapshot=str(source), argv=argv, container_name=job_id, after_terminal=[], depends_on=[],
                   exclusive=False, global_lock=False, formal_eligible=False, energy_comparable=False,
                   completion_is_formal_qualification=False, required_receipts=['tp-qualification/completion.json'],
                   timeout_s=1800, **bindings,
                   profile_inputs={key:dict(path=str(path), sha256=sha(path)) for key, path in paths.items()},
                   input_manifest=dict(path=str(out / 'input-manifest.json'), sha256=sha(out / 'input-manifest.json')),
                   trace=dict(path=str(out / 'trace.json'), sha256=checksums['trace'],
                              canonical_requests_sha256=canonical_sha(rows), seed=701, requests=len(rows),
                              duration_s=66, inputs=[512, 1024, 2048], output_tokens=128,
                              burst_arrivals_s=[0, 30], burst_size=96),
                   resident_layout=dict(path=str(out / 'resident-layout.json'), sha256=checksums['resident_layout'], pools=layout))
    job = dict(job_id=job_id, priority=450, max_attempts=1, payload=payload)
    write(out / 'jobs.json', [job])
    # Run the exact frozen CLI with GPUs unavailable; only output directory and
    # lease placeholders differ. No interception or fabricated GPU artifacts.
    cpu_out = out / 'cpu-preflight'
    cpu_out.mkdir()
    cpu = []
    i = 0
    while i < len(argv):
        value = argv[i]
        if value == '--gpus' and i < argv.index(image_token):
            i += 2
            continue
        if value == '--cap-add':
            i += 2
            continue
        value = value.replace(job_id, job_id + '-cpu-preflight').replace('{attempt_dir}', str(cpu_out))
        value = value.replace('{lease_local_indices}', '0,1,2').replace('{lease_port}', '19000')
        value = value.replace('{lease_gpu_uuids}', 'cpu-preflight-no-gpu')
        cpu.append(value)
        i += 1
    cpu[cpu.index(image_token):cpu.index(image_token)] = ['-e', 'NVIDIA_VISIBLE_DEVICES=void']
    cpu += ['--preflight-only']
    write(out / 'preflight-command.json', dict(argv=cpu, hardware_executed=False))
    process = subprocess.run(cpu, capture_output=True, text=True, timeout=120)
    (out / 'preflight.stdout').write_text(process.stdout)
    (out / 'preflight.stderr').write_text(process.stderr)
    receipt_path = cpu_out / 'tp-qualification/preflight.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    review = dict(schema='tp-resident-qualification-preparation-v1',
                  status='prepare_only_cpu_container_passed' if process.returncode == 0 else 'preflight_failed',
                  hardware_executed=False, queue_modified=False, job_id=job_id, priority=450, max_attempts=1,
                  gpu_count=3, after_terminal=[], depends_on=[], jobs_sha256=sha(out / 'jobs.json'),
                  source_snapshot=str(source), **bindings,
                  cpu_container_preflight=dict(returncode=process.returncode, receipt=str(receipt_path),
                                               receipt_sha256=sha(receipt_path) if receipt else None,
                                               argv=str(out / 'preflight-command.json'),
                                               status=receipt.get('status')),
                  required_gpu_gate='Every request completed; both TP1 and TP2 each complete at least two requests; '
                                    'route/profile/generation identities agree; native startup and power metering present.',
                  limitations=['CPU admission counts are not GPU results.', 'No native KV equivalence, golden output, '
                               'dynamic reshard, or formal energy qualification is claimed.'])
    write(out / 'review.json', review)
    print(json.dumps(dict(out=str(out), status=review['status'], returncode=process.returncode), indent=2))
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
