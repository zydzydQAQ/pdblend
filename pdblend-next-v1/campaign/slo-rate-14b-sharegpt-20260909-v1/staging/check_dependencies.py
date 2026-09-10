from pathlib import Path
import json,hashlib,sys,socket
root=Path('/root/workspace/pdblend-next-v1/campaign/slo-rate-14b-sharegpt-20260909-v1');manifest=root/'staging/baseline-dependencies.json';d=json.loads(manifest.read_text());missing=[];conflicts=[];same=[]
for path,digest in d['files'].items():
 p=Path(path)
 if not p.exists():missing.append(path)
 elif p.is_file():
  h=hashlib.sha256(p.read_bytes()).hexdigest()
  if h==digest:same.append(path)
  else:conflicts.append(dict(path=path,expected=digest,actual=h))
 else:conflicts.append(dict(path=path,error='exists-not-file'))
result=dict(hostname=socket.gethostname(),missing=missing,conflicts=conflicts,same_count=len(same),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest());(root/'staging'/('dependency-check-'+sys.argv[1]+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(missing=len(missing),conflicts=conflicts,same=len(same))))
