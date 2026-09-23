#!/usr/bin/env python3
"""Reuse completed training profiles and queue independent frozen-fit holdouts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pdblend.profile.calibration import prepare_candidate
from pdblend.experimentation.lease import GPULeaseQueue


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'results/2026-09-22/three-model')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args(); root = a.root.resolve()
    spec = importlib.util.spec_from_file_location('freeze_profile', ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    state = json.loads((root/'queue.json').read_text())
    inputs = {}
    for lease in state['leases'].values():
        path = Path(lease['attempt_dir'])
        if lease['status'] != 'succeeded' or not (path/'raw.json').is_file() or not (path/'profile.json').is_file():
            continue
        raw = json.loads((path/'raw.json').read_text())
        if raw.get('system') == 'pdblend' and len(raw.get('decode', [])) >= 60:
            inputs[(raw['model_id'], raw['tp'])] = path.resolve()
    image = subprocess.check_output(['docker','image','inspect','pdblend:l20-cu128-vllm-v1','--format','{{.Id}}'],text=True).strip()
    snapshot, source_hash = helper.freeze_source(ROOT/'src', root/'calibration-sources')
    verification, _ = helper.freeze_receipt(root/'model-verification.json', root/'profile-receipts')
    waves = [(('7B',1),('14B',1),('7B',2),('14B',2),('32B',2)),
             (('7B',4),('32B',4)), (('14B',4),)]
    jobs, dependencies = [], []
    for index, wave in enumerate(waves):
        members = [f'{size.lower()}-tp{tp}' for size,tp in wave]
        wave_id = f'holdout-{source_hash[:16]}-wave{index+1}'
        wave_dir = root/'calibration-waves'/wave_id
        wave_dir.mkdir(parents=True,exist_ok=True)
        wave_payload = dict(cohort_id=wave_id, coordinator=True, members=members)
        path = wave_dir/'wave.json'
        if path.exists() and json.loads(path.read_text()) != wave_payload:
            raise ValueError('cohort identity collision')
        path.write_text(json.dumps(wave_payload,indent=2)+'\n')
        wave_jobs=[]
        for (size,tp), member in zip(wave,members):
            model = f'Qwen2.5-{size}-Instruct'; source = inputs[(model,tp)]
            identity = dict(model_id=model,tp=tp,training_raw_sha256=hashlib.sha256((source/'raw.json').read_bytes()).hexdigest(),
                            source_sha256=source_hash,image_digest=image)
            suffix = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
            candidate_dir = root/'calibration-candidates'/f'{member}-{suffix}'
            if not candidate_dir.exists():
                manifest = prepare_candidate(source/'raw.json',source/'profile.json',candidate_dir)
            else:
                manifest = json.loads((candidate_dir/'manifest.json').read_text())
                if hashlib.sha256((candidate_dir/'candidate.json').read_bytes()).hexdigest() != manifest['candidate_sha256']:
                    raise ValueError('candidate checksum mismatch')
            name=f'holdout-{member}-{suffix}'
            argv=['docker','run','--rm','--name',name,'--gpus','all','--cap-add','SYS_ADMIN',
                  '--ipc=host','--network','host','--shm-size','16g','--ulimit','nofile=65536:65536',
                  '--entrypoint','/opt/venv/bin/python',
                  '-v',f'{snapshot}:/opt/pdblend-src:ro','-v','/home/models:/models:ro',
                  '-v',f'{verification}:/verification/model-verification.json:ro',
                  '-v',f'{candidate_dir}:/candidate:ro','-v','{attempt_dir}:/output:rw',
                  '-v',f'{root}/profile-wave:/coord:rw','-v',f'{wave_dir}:/wave:rw',
                  '-e','PYTHONPATH=/opt/pdblend-src','-e','PDBLEND_MODELS_DIR=/models',
                  '-e','PDBLEND_MODEL_VERIFICATION_RECEIPT=/verification/model-verification.json',
                  '-e',f'PDBLEND_SOURCE_SHA256={source_hash}','-e',f'PDBLEND_IMAGE_ID={image}',
                  '-e','PDBLEND_HARDWARE_ID=8xL20-lease','-e','CUDA_VERSION=12.8.1',
                  '-e','PDBLEND_GPU_UUIDS={lease_gpu_uuids}','-e','PDBLEND_COORD_DIR=/coord',
                  '-e','PDBLEND_PROFILE_WAVE=/wave','-e',f'PDBLEND_PROFILE_MEMBER={member}',
                  image,'-B','-m','pdblend.profile.calibration','--candidate-dir','/candidate',
                  '--model',f'/models/{model}','--gpus',*[str(i) for i in range(tp)],
                  '--base-port','{lease_port}','--out','/output']
            payload=dict(identity,argv=argv,gpu_count=tp,exclusive=False,global_lock=False,
                         container_name=name,timeout_s=5400,required_receipts=['completion.json'],
                         depends_on=list(dependencies),evidence_class='independent_calibration_holdout',
                         formal_eligible=False,energy_comparable=False,candidate_dir=str(candidate_dir),
                         topology=dict(model_id=model,tp=tp,pp=1,gpu_count=tp))
            jobs.append(dict(job_id=name,payload=payload,priority=300-index,max_attempts=1))
            wave_jobs.append(name)
            print(json.dumps(dict(job=name,gpus=tp,points=len(manifest['plan']['decode']),
                                  training_max=manifest['decode_training_max_after'])),flush=True)
        dependencies = wave_jobs
    output=root/'calibration-holdout-spec.json'
    output.write_text(json.dumps(jobs,indent=2)+'\n')
    if not a.prepare_only:
        q=GPULeaseQueue(root/'queue.json')
        for job in jobs:q.enqueue(**job)
    print(json.dumps(dict(prepared_only=a.prepare_only,spec=str(output),jobs=len(jobs))),flush=True)


if __name__=='__main__':main()
