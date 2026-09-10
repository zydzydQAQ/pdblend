from pathlib import Path
import json,hashlib,tarfile,sys,os,time
root=Path('/root/workspace/pdblend-next-v1/campaign/slo-rate-14b-sharegpt-20260909-v1');m=json.loads((root/'staging/baseline-dependencies.json').read_text());node=sys.argv[1];receipt=dict(node=node,started_s=time.time(),created=[],existing_identical=[])
with tarfile.open(root/'staging'/('baseline-missing-'+node+'.tar.gz'),'r:gz') as tar:
 for item in tar:
  assert item.isfile() and not item.name.startswith('/') and '..' not in Path(item.name).parts
  path=Path('/')/item.name;assert str(path) in m['files'];data=tar.extractfile(item).read();digest=hashlib.sha256(data).hexdigest();assert digest==m['files'][str(path)]
  if path.exists():
   assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest;receipt['existing_identical'].append(str(path));continue
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as f:f.write(data)
  os.chmod(path,0o444);receipt['created'].append(str(path))
receipt['finished_s']=time.time();(root/'staging'/('installed-'+node+'.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(node=node,created=len(receipt['created']),existing_identical=len(receipt['existing_identical']))))
