"""Passive terminal binding mirror for live independent report verification."""
import base64,json,time
from pathlib import Path
from operate import remote,sha,write_new
HERE=Path(__file__).resolve().parent
seen=set()
while True:
    code="""import pathlib,json
root=pathlib.Path(ROOT);ready=[]
for p in (root/'boundary-baseline-bindings-p4v2').glob('*/binding.json'):
 b=json.loads(p.read_text())
 if b.get('output_correctness_verified') is True:ready.append(str(p.parent))
print(json.dumps(ready))"""
    for name in json.loads(remote('ROOT='+repr(str(HERE))+'\n'+code)):
        if name in seen:continue
        root=Path(name);known={str(p):sha(p) for p in root.rglob('*') if p.is_file()}
        code="""import pathlib,json,sys,base64,hashlib
root=pathlib.Path(ROOT);known=json.load(sys.stdin);b=json.loads((root/'binding.json').read_text());rows=[]
for p,h in b['files'].items():assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h
for p in root.rglob('*'):
 if not p.is_file():continue
 raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();assert str(p) not in known or known[str(p)]==h
 if known.get(str(p))!=h:rows.append(dict(path=str(p),data=base64.b64encode(raw).decode(),sha256=h))
print(json.dumps(rows))"""
        rows=json.loads(remote('ROOT='+repr(name)+'\n'+code,json.dumps(known).encode()))
        for r in rows:
            import hashlib
            raw=base64.b64decode(r['data']);assert hashlib.sha256(raw).hexdigest()==r['sha256']
            write_new(Path(r['path']),raw)
        seen.add(name);print(json.dumps(dict(captured_s=time.time(),binding=name,new_files=len(rows))),flush=True)
    state=json.loads((HERE/'boundary-local-handoff-p4v2/status.json').read_text())
    if state.get('complete') or state.get('failed'):break
    time.sleep(15)
