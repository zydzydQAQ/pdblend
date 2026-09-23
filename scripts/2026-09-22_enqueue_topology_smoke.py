#!/usr/bin/env python3
"""Freeze the current source and enqueue exclusive three-model smoke jobs."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from pdblend.experimentation.lease import GPULeaseQueue

root = Path(__file__).resolve().parents[1]
out = root / 'results/2026-09-22/three-model'
snapshot = out / 'smoke-source'
snapshot.mkdir(parents=True, exist_ok=False)
shutil.copytree(root / 'src', snapshot / 'src', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
files = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(snapshot.rglob('*')) if p.is_file()}
source_hash = hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
image = subprocess.check_output(['docker','image','inspect','pdblend:l20-cu128-vllm-v1','--format','{{.Id}}'],text=True).strip()
(snapshot/'manifest.json').write_text(json.dumps({'files':files,'source_sha256':source_hash,'image_digest':image},indent=2))
q = GPULeaseQueue(out / 'queue.json')
# PP smoke can execute a standalone pipeline, but has no qualified PD connector.
for model,tp,pp in [('7b',1,1),('14b',1,1),('32b',2,1),('7b',2,1),('7b',4,1),('14b',2,1),('14b',4,1),('32b',4,1),('14b',8,1),('32b',8,1),('7b',1,2),('14b',1,2),('32b',1,2)]:
    name=f'three-model-smoke-{model}-tp{tp}-pp{pp}'
    argv=['docker','run','--rm','--name',name,'--gpus','all','--cap-add','SYS_ADMIN','--ipc=host','--network','host','--shm-size','16g','--ulimit','nofile=65536:65536',
          '--entrypoint','/opt/venv/bin/python','-v',f'{snapshot}/src:/opt/pdblend-src:ro','-v','/home/models:/models:ro','-v','{attempt_dir}:/output:rw',
          '-e','PYTHONPATH=/opt/pdblend-src','-e','PDBLEND_MODELS_DIR=/models','-e',f'PDBLEND_SOURCE_SHA256={source_hash}','-e',f'PDBLEND_IMAGE_ID={image}',image,
          '-B','-m','pdblend.bench.topology_smoke','--model',model,'--tp',str(tp),'--pp',str(pp),'--out','/output']
    q.enqueue(name,{'argv':argv,'gpu_count':8,'exclusive':True,'container_name':name,'required_receipts':['completion.json'],'timeout_s':2400,'evidence_class':'screening'},max_attempts=1)
print(json.dumps({'jobs':len(q.list_jobs()),'source_sha256':source_hash,'image':image}))
