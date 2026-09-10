"""Publish a qualified environment handoff only after independent raw replay."""
from pathlib import Path
import hashlib,json,sys,time
import verify_pdb
ROOT=Path(__file__).resolve().parent.parent
node=sys.argv[1];assert node in ('A','C')
def ref(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
p=ROOT/node/'pdb-ready.json';old=json.loads(p.read_text());assert old['complete'] and old['node']==node
q=ref(ROOT/node/'pdb-qualification/qualified.json');v=verify_pdb.verify(q);assert v['binding']==old['binding']
old['native_qualification_validator']=old['qualification_validator'];old['qualification_validator']=ref(ROOT/'env/verify_pdb.py');old['source_audit']=ref(ROOT/'env/source-audit.json');old['independent_handoff_verified_s']=time.time();old['slo_scale_assignment']=.5 if node=='A' else 2.;old['formal_rate_measurements_started']=False
old['runtime_pythonpath']=[str(ROOT/'env/runtime/src'),str(ROOT/'env/runtime'),str(ROOT/'env/meter'),str(ROOT/node/'tooling-v2'),'/root/workspace/pdblend/.runtime-deps']
old['qualification']=q
# The result evidence is immutable; only this environment handoff pointer is updated.
t=p.with_suffix('.tmp');t.write_text(json.dumps(old,indent=2)+'\n');t.replace(p)
print(json.dumps(old,indent=2))
