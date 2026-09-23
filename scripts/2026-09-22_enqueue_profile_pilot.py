#!/usr/bin/env python3
"""Enqueue a first real new-stack profile after a successful topology smoke."""
import hashlib,json,shutil,subprocess
from pathlib import Path
from pdblend.experimentation.lease import GPULeaseQueue
root=Path(__file__).resolve().parents[1]; out=root/'results/2026-09-22/three-model'; snapshot=out/'profile-source'
if not snapshot.exists():
    shutil.copytree(root/'src',snapshot,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
files={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(snapshot.rglob('*')) if p.is_file()}
source=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(); image=subprocess.check_output(['docker','image','inspect','pdblend:l20-cu128-vllm-v1','--format','{{.Id}}'],text=True).strip()
(snapshot/'manifest.json').write_text(json.dumps({'files':files,'source_sha256':source,'image_digest':image},indent=2))
q=GPULeaseQueue(out/'queue.json'); dependency='three-model-smoke-7b-tp1-pp1'
job='profile-pilot-7b-tp1-pdblend-mixed'
argv=['docker','run','--rm','--name',job,'--gpus','all','--cap-add','SYS_ADMIN','--ipc=host','--network','host','--shm-size','16g','--ulimit','nofile=65536:65536','--entrypoint','/opt/venv/bin/python','-v',f'{snapshot}:/opt/pdblend-src:ro','-v','/home/models:/models:ro','-v','{attempt_dir}:/output:rw','-e','PYTHONPATH=/opt/pdblend-src','-e','PDBLEND_MODELS_DIR=/models','-e',f'PDBLEND_SOURCE_SHA256={source}','-e',f'PDBLEND_IMAGE_ID={image}','-e','PDBLEND_HARDWARE_ID=8xL20-lease','-e','PDBLEND_VLLM_VERSION=0.10.1.1','-e','PDBLEND_TORCH_VERSION=2.7.0','-e','CUDA_VERSION=12.8.1',image,'-B','-m','pdblend.cli','profile','--model','/models/Qwen2.5-7B-Instruct','--gpus','0,1','--tp','1','--pp','1','--system','pdblend','--role','mixed','--hardware-id','8xL20-lease','--engine-revision','vllm-0.10.1.1','--out','/output','--decode-repeats','3','--decode-settle','2','--decode-measure','5']
q.enqueue(job,{'argv':argv,'gpu_count':8,'exclusive':True,'container_name':job,'required_receipts':['completion.json'],'timeout_s':7200,'evidence_class':'profile'},depends_on=[dependency],max_attempts=1)
print(json.dumps({'job':job,'depends_on':dependency,'source_sha256':source,'image':image}))
