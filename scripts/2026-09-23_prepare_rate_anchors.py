#!/usr/bin/env python3
"""Freeze and CPU-preflight eight-GPU Mixed rate calibration jobs; no enqueue."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc'
VERIFY = ROOT/'results/2026-09-22/three-model/profile-receipts/model-verification-99fabb0721f21aa50eb2a8518877acdf05cc76df32f0f769900be7b4d4471fc8.json'


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args(); out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('freeze_rates', ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    source, source_sha = helper.freeze_source(ROOT/'src', out/'sources')
    jobs, reviews = [], []
    for size in ('7B', '14B', '32B'):
        model = f'Qwen2.5-{size}-Instruct'; tp = 2 if size == '32B' else 1
        corpus = ROOT/f'datasets/prepared/2026-09-22-{size.lower()}-v1'
        job_id = f'mixed-rate-anchor-{size.lower()}-{source_sha[:12]}'
        argv = ['docker', 'run', '--rm', '--name', job_id, '--gpus', 'all', '--cap-add', 'SYS_ADMIN',
            '--ipc=host', '--network=host', '--shm-size=16g', '--ulimit', 'nofile=65536:65536',
            '--entrypoint', '/opt/venv/bin/python']
        for host, target, mode in [(source, '/opt/pdblend-src', 'ro'), ('/home/models', '/models', 'ro'),
                (VERIFY, '/verification/model-verification.json', 'ro'),
                (corpus, '/corpus', 'ro'), ('{attempt_dir}', '/output', 'rw')]:
            argv += ['-v', f'{host}:{target}:{mode}']
        env = dict(PYTHONPATH='/opt/pdblend-src', PYTHONDONTWRITEBYTECODE='1',
            PDBLEND_MODELS_DIR='/models', PDBLEND_MODEL_VERIFICATION_RECEIPT='/verification/model-verification.json',
            PDBLEND_SOURCE_MANIFEST='/opt/pdblend-src/manifest.json', PDBLEND_SOURCE_SHA256=source_sha,
            PDBLEND_IMAGE_ID=IMAGE, PDBLEND_GPU_UUIDS='{lease_gpu_uuids}',
            CUDA_VISIBLE_DEVICES='{lease_local_indices}', CUDA_VERSION='12.8.1',
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
        for key, value in env.items(): argv += ['-e', f'{key}={value}']
        argv += ['pdblend:l20-cu128-vllm-v1@'+IMAGE, '-B', '-m', 'pdblend.bench.rate_anchor',
            '--model', '/models/'+model, '--tp', str(tp), '--gpus', '{lease_local_indices}',
            '--corpus', '/corpus', '--out', '/output/anchor', '--base-port', '{lease_port}']
        check_dir = out/f'preflight-{size.lower()}'; check_dir.mkdir()
        check = [value.replace('{attempt_dir}', str(check_dir)).replace('{lease_gpu_uuids}', '')
            .replace('{lease_local_indices}', '0,1,2,3,4,5,6,7').replace('{lease_port}', '12000') for value in argv]
        check[check.index('--name')+1] += '-preflight'
        index = check.index('--gpus'); del check[index:index+2]
        check += ['--preflight-only']
        write(check_dir/'argv.json', check)
        proc = subprocess.run(check, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (check_dir/'stdout.log').write_text(proc.stdout)
        if proc.returncode:
            raise RuntimeError(f'{model} CPU preflight failed: {proc.stdout[-2000:]}')
        receipt = check_dir/'anchor/preflight.json'
        reviews.append(dict(model=model, returncode=proc.returncode, receipt=str(receipt),
            receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest()))
        jobs.append(dict(job_id=job_id, priority=310, max_attempts=1, payload=dict(
            system='mixed', model_id=model, tp=tp, pp=1, gpu_count=8, exclusive=True, global_lock=True,
            source_snapshot=str(source), source_sha256=source_sha, image_digest=IMAGE,
            container_name=job_id, argv=argv, depends_on=[], after_terminal=[],
            timeout_s=10800, required_receipts=['anchor/completion.json'],
            formal_eligible=False, energy_comparable=False,
            scope='model_owned_calibration_and_tuning_rate_anchor', reserve_host=True,
            input_sha256={'corpus_manifest':hashlib.sha256((corpus/'manifest.json').read_bytes()).hexdigest(),
                          'model_verification':hashlib.sha256(VERIFY.read_bytes()).hexdigest()})))
    write(out/'jobs.json', jobs)
    write(out/'review.json', dict(status='cpu_preflight_passed', queue_modified=False,
        hardware_executed=False, source_snapshot=str(source), source_sha256=source_sha,
        jobs_sha256=hashlib.sha256((out/'jobs.json').read_bytes()).hexdigest(), reviews=reviews))
    print(out/'jobs.json')


if __name__ == '__main__': main()
