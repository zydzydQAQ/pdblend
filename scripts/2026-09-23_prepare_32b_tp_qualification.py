#!/usr/bin/env python3
"""Prepare fixed/P-D or resident TP2/TP4 functional jobs without touching a queue."""
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
PROFILE_ROOT = ROOT / 'results/2026-09-22/three-model/calibration-candidates'
PROFILES = {2: PROFILE_ROOT / '32b-tp2-f31ed0be1462e4ef9564/candidate.json',
            4: PROFILE_ROOT / '32b-tp4-5d5ae4a37aed7c851cf7/candidate.json'}
VERIFY = ROOT / 'results/2026-09-22/three-model/profile-receipts/model-verification-99fabb0721f21aa50eb2a8518877acdf05cc76df32f0f769900be7b4d4471fc8.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', required=True, choices=('fixed-pd', 'resident'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--revision', default='v1')
    args = parser.parse_args()
    resident = args.kind == 'resident'
    out = (args.out or ROOT / f'results/2026-09-23/32b-tp-{args.kind}-qualification-v1').resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('tp_source_freezer', ROOT / 'scripts/2026-09-22_enqueue_parallel_profiles.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    source, source_sha = helper.freeze_source(ROOT / 'src', out / 'sources')
    helper.verify_snapshot(source, json.loads((source / 'manifest.json').read_text())['files'])
    profile_paths = PROFILES if resident else {2: PROFILES[2]}
    gpu_count = 6 if resident else 4
    layout = ([dict(pool_id='small', tp=2, replicas=1), dict(pool_id='large', tp=4, replicas=1)] if resident
              else [dict(pool_id='tp2', tp=2, replicas=2)])
    write(out / 'resident-layout.json', layout)
    rng = random.Random(701)
    arrivals = ([0] * 96 + [30] * 96 + [60, 63, 66] if resident else [0, 0, 8, 8, 16, 16])
    rows = []
    for i, arrival in enumerate(arrivals):
        length = (512, 1024, 2048)[i % 3]
        rows.append(dict(idx=i, arrival_s=arrival, prompt=[rng.randrange(100, 5000) for _ in range(length)],
                         max_tokens=128 if resident else 16,
                         source='32b-resident-capacity-burst-seed701' if resident else '32b-fixed-offline-pd-seed701'))
    write(out / 'trace.json', dict(seed=701, requests=rows, trace_sha256=canonical_sha(rows)))
    checksums = dict(source_manifest=sha(source / 'manifest.json'), resident_layout=sha(out / 'resident-layout.json'),
                     trace=sha(out / 'trace.json'), model_verification=sha(VERIFY),
                     **{f'tp{tp}_profile': sha(path) for tp, path in profile_paths.items()})
    bindings = dict(source_sha256=source_sha, image_digest=IMAGE, exact_inputs_sha256=checksums)
    write(out / 'input-manifest.json', bindings)
    job_id = f'pdblend-32b-tp-{args.kind}-qualification-{args.revision}'
    mounts = [(source, '/opt/pdblend-src', 'ro'), (source / 'manifest.json', '/source-manifest.json', 'ro'),
              (Path('/home/models'), '/models', 'ro'), (VERIFY, '/verification/model-verification.json', 'ro'),
              *((path, f'/profiles/32b-tp{tp}.json', 'ro') for tp, path in profile_paths.items()),
              (out / 'resident-layout.json', '/spec/resident-layout.json', 'ro'),
              (out / 'trace.json', '/spec/trace.json', 'ro'), (out / 'input-manifest.json', '/spec/input-manifest.json', 'ro'),
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
    module = 'pdblend.bench.tp_qualification' if resident else 'pdblend.bench.tp_fixed_pd_qualification'
    argv += [image_token, '-B', '-m', module, '--model', 'Qwen2.5-32B-Instruct', '--tp', '2',
             '--gpus', '{lease_local_indices}', '--profile', '/profiles/32b-tp2.json']
    if resident:
        argv += ['--higher-tp', '4']
    for tp in profile_paths:
        argv += ['--topology-profile', f'{tp}=/profiles/32b-tp{tp}.json']
    argv += ['--resident-layout', '/spec/resident-layout.json', '--trace', '/spec/trace.json',
             '--input-manifest', '/spec/input-manifest.json', '--base-port', '{lease_port}', '--out', '/output/tp-qualification']
    payload = dict(schema=1, system='pdblend', model_id='Qwen2.5-32B-Instruct', tp=2, pp=1, gpu_count=gpu_count,
                   source_snapshot=str(source), argv=argv, container_name=job_id, after_terminal=[], depends_on=[],
                   exclusive=False, global_lock=False, formal_eligible=False, energy_comparable=False,
                   completion_is_formal_qualification=False, required_receipts=['tp-qualification/completion.json'],
                   timeout_s=2100, **bindings,
                   profile_inputs={f'tp{tp}':dict(path=str(path), sha256=sha(path)) for tp, path in profile_paths.items()},
                   input_manifest=dict(path=str(out / 'input-manifest.json'), sha256=sha(out / 'input-manifest.json')),
                   trace=dict(path=str(out / 'trace.json'), sha256=checksums['trace'],
                              canonical_requests_sha256=canonical_sha(rows), seed=701, requests=len(rows),
                              input_tokens=[512, 1024, 2048], output_tokens=128 if resident else 16),
                   resident_layout=dict(path=str(out / 'resident-layout.json'), sha256=checksums['resident_layout'], pools=layout),
                   windows=['resident_hetero_tp'] if resident else ['fixed_tp', 'offline_tp', 'mechanism_pd'],
                   shared_load=True)
    job = dict(job_id=job_id, priority=440, max_attempts=1, payload=payload)
    write(out / 'jobs.json', [job])
    cpu_out = out / 'cpu-preflight'
    cpu_out.mkdir()
    cpu, i = [], 0
    while i < len(argv):
        value = argv[i]
        if value == '--cap-add' or value == '--gpus' and i < argv.index(image_token):
            i += 2
            continue
        value = value.replace(job_id, job_id + '-cpu-preflight').replace('{attempt_dir}', str(cpu_out))
        value = value.replace('{lease_local_indices}', ','.join(map(str, range(gpu_count))))
        value = value.replace('{lease_port}', '19200').replace('{lease_gpu_uuids}', 'cpu-preflight-no-gpu')
        cpu.append(value)
        i += 1
    index = cpu.index(image_token)
    cpu[index:index] = ['-e', 'NVIDIA_VISIBLE_DEVICES=void']
    cpu += ['--preflight-only']
    write(out / 'preflight-command.json', dict(argv=cpu, hardware_executed=False))
    process = subprocess.run(cpu, capture_output=True, text=True, timeout=120)
    (out / 'preflight.stdout').write_text(process.stdout)
    (out / 'preflight.stderr').write_text(process.stderr)
    receipt_path = cpu_out / 'tp-qualification/preflight.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    review = dict(schema='32b-tp-functional-preparation-v1', kind=args.kind,
                  status='prepare_only_cpu_container_passed' if process.returncode == 0 else 'preflight_failed',
                  hardware_executed=False, queue_modified=False, job_id=job_id, priority=440, max_attempts=1,
                  gpu_count=gpu_count, after_terminal=[], depends_on=[], jobs_sha256=sha(out / 'jobs.json'),
                  source_snapshot=str(source), **bindings,
                  cpu_container_preflight=dict(returncode=process.returncode, receipt=str(receipt_path),
                                               receipt_sha256=sha(receipt_path) if receipt else None,
                                               argv=str(out / 'preflight-command.json'), status=receipt.get('status')),
                  shared_load=True, required_windows=payload['windows'], formal_eligible=False, energy_comparable=False,
                  scope='resident two-TP native route coverage' if resident else 'fixed/offline selection and separately labeled symmetric TP2 P-D mechanism',
                  limitations=['CPU preflight is not GPU evidence.', 'No native KV equivalence, golden output, dynamic '
                               'reshard or formal energy qualification is claimed.'])
    write(out / 'review.json', review)
    print(json.dumps(dict(out=str(out), status=review['status'], returncode=process.returncode), indent=2))
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
