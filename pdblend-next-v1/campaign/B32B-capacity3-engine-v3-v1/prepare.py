"""Read-only B identity and full model-source hash freeze; no control or GPU work."""
import hashlib,json,pathlib,socket,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def out(args):return subprocess.check_output(args,text=True)
def main():
 assert not (ROOT/'deployment-status.json').exists()
 ids=out(['docker','ps','-aq']).split();inventory=json.loads(out(['docker','inspect',*ids]))
 assert {r['Name'].lstrip('/') for r in inventory if r['State']['Running']}=={'pdb-v2-nextv3b0','pdb-v2-nextv3b1'}
 for r in inventory:
  assert r['Name']!='/pdb-v2-nextv3b2'
  args=r.get('Args',[])
  if '--config' in args:
   p=pathlib.Path(args[args.index('--config')+1]);c=json.loads(p.read_text())
   assert c.get('id')!='nextv3b2'
   used=[c.get('port')]+list(range(c.get('kv_port',0),c.get('kv_port',0)+32))
   assert not set(used)&({33502}|set(range(33764,33796)))
 for port in [33502,*range(33764,33796)]:
  with socket.socket() as sock:sock.bind(('127.0.0.1',port))
 model=pathlib.Path('/root/workspace/models/Qwen2.5-32B-Instruct');index=json.loads((model/'model.safetensors.index.json').read_text());weights=sorted(set(index['weight_map'].values()));paths=[model/x for x in weights]+[model/'config.json',model/'model.safetensors.index.json',model/'generation_config.json',model/'tokenizer_config.json',model/'tokenizer.json']
 files={};stats={};started=time.time()
 for p in paths:
  s=p.stat();files[str(p)]=sha(p);stats[str(p)]=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ino=s.st_ino)
 d=dict(schema_version=1,model=str(model),model_files=files,model_file_stats=stats,weight_shards=len(weights),model_hash_started_s=started,model_hash_finished_s=time.time(),inventory=inventory,port_checks=[33502,*range(33764,33796)],note='Full model hashing warms host page cache. Subsequent deploy observation is fresh process, not disk-cold.')
 (ROOT/'prepare-receipt.json').write_text(json.dumps(d,indent=2)+'\n'); print(json.dumps(dict(weight_shards=len(weights),hashed_files=len(files),elapsed_s=time.time()-started)))
if __name__=='__main__':main()
