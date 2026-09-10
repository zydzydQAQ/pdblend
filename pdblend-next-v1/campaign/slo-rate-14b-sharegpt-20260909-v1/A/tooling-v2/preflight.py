"""Read-only physical/model preflight; writes only its new campaign evidence."""
from pathlib import Path
import hashlib,json,os,socket,subprocess,time,fcntl
import power_selftest as p
import bootstrap as b
from inputs import E,N
out=N/p.NODE/'preparation';out.mkdir(parents=True,exist_ok=True)
assert socket.gethostname()==p.EXPECTED_HOSTNAME
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True)
gpus=[]
for line in raw.splitlines():
 a=line.split(',');gpus.append(dict(index=int(a[0]),uuid=a[1].strip(),name=a[2].strip(),memory_mib=int(a[3])))
assert len(gpus)==8 and all(g['name']=='NVIDIA L20' for g in gpus)
b.save(p.IDENTITY,dict(node=p.NODE,actual_hostname=p.EXPECTED_HOSTNAME,GPUs=gpus,captured_s=time.time()))
model=b.checked(p.ref(E/'model-manifest.json'));checks=[]
for f in model['files']:
 path=Path(model['model_root'])/f['name'];h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 checks.append(dict(name=f['name'],bytes=path.stat().st_size,sha256=h.hexdigest(),passed=path.stat().st_size==f['bytes'] and h.hexdigest()==f['sha256']))
 b.save(out/'model-sha-progress.json',checks)
assert all(c['passed'] for c in checks)
for image in ['sha256:0bb51d143b7fcaaea2e794dd6e207cf4165a4f21522a2e932a4bd4a117074bc2']:
 assert subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True).strip()==image
# A snapshot only; deployment will obtain and keep the common node lock.
ids=subprocess.check_output(['docker','ps','-q'],text=True).split();containers=json.loads(subprocess.check_output(['docker','inspect',*ids],text=True)) if ids else []
b.save(out/'containers.before.json',containers)
b.save(out/'preflight.json',dict(passed=True,node=p.NODE,hostname=p.EXPECTED_HOSTNAME,GPUs=gpus,model_manifest=p.ref(E/'model-manifest.json'),model_all_shards_verified=True,model_checks=checks,captured_s=time.time(),running_containers=[x['Id'] for x in containers]))
from inputs import bootstrap_spec
print(json.dumps(bootstrap_spec()))
