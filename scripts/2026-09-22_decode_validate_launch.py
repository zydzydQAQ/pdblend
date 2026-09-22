#!/usr/bin/env python3
"""Launch disjoint single-GPU validation workers with immutable source snapshots."""
import argparse,fcntl,json,subprocess,shutil,time
from pathlib import Path
from pdblend.profile.merge import sha256

p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[1];a.candidate=a.candidate.resolve();a.out=a.out.resolve()
lease=open('/tmp/pdblend4-gpu-experiment.lock','a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
if a.out.exists():raise FileExistsError(a.out)
a.out.mkdir(parents=True);shutil.copytree(root/'src',a.out/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
image=subprocess.check_output(['docker','image','inspect','pdblend:l20-cu128-vllm-v1','--format','{{.Id}}'],text=True).strip()
source={str(x.relative_to(a.out/'source')):sha256(x) for x in sorted((a.out/'source').rglob('*.py'))}
import hashlib
source_hash=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
points=[dict(batch=b,context_tokens=c) for c in (256,1024,2048,4096) for b in (1,2,3,4,6,24,48,64,80,96,160)]
parts=[points[i::8] for i in range(8)]
for i in (0,2):
 for b,c in ((1,256),(64,256),(1,1024)):
  if not any(x['batch']==b and x['context_tokens']==c for x in parts[i]):parts[i].append(dict(batch=b,context_tokens=c,label='cross_gpu_repeat'))
procs=[];meta=dict(candidate=str(a.candidate),candidate_sha256=sha256(a.candidate),started_s=time.time(),image=image,source=source,source_hash=source_hash)
(a.out/'manifest.json').write_text(json.dumps(meta,indent=1))
for i,part in enumerate(parts):
 folder=a.out/f'gpu-{i}';folder.mkdir();plan=dict(candidate=str(a.candidate),candidate_sha256=meta['candidate_sha256'],gpu=i,points=part)
 planpath=folder/'plan.json';planpath.write_text(json.dumps(plan,indent=1))
 cmd=['docker','run','--rm','--name',f'pdb4-validate900-{i}','--gpus',f'device={i}','--cap-add','SYS_ADMIN','--ipc=host','--shm-size=16g','--network','host','--ulimit','nofile=65536:65536',
 '-v',f'{root}:{root}:ro','-v',f'{a.out}:{a.out}','-v',f'{a.out}/source:/opt/pdblend-src:ro','-v','/home/models:/models:ro',
 '-e','PYTHONPATH=/opt/pdblend-src','-e','PDBLEND_MODELS_DIR=/models','-e',f'PDBLEND_IMAGE_DIGEST={image}','-e',f'PDBLEND_SOURCE_HASH={source_hash}',
 '-w',str(root),image,'python','-B',str(root/'scripts/2026-09-22_decode_validate_worker.py'),str(planpath),str(folder/'measurement'),'--port',str(8500+i*100)]
 log=(folder/'run.log').open('w');procs.append((subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log))
codes=[]
for proc,log in procs:codes.append(proc.wait());log.close()
meta.update(returncodes=codes,finished_s=time.time(),candidate_unchanged=sha256(a.candidate)==meta['candidate_sha256'])
(a.out/'manifest.json').write_text(json.dumps(meta,indent=1));print(json.dumps(dict(returncodes=codes)))
raise SystemExit(0 if not any(codes) and meta['candidate_unchanged'] else 1)
