"""Complete membership replay of C's six original failed Eco main cells on B CPU."""
import json,os,socket,time
from pathlib import Path
import audit_original_AB_Eco60_window_v1 as w
B=Path(__file__).resolve().parent;INPUT=B/'original-C-Eco6-inputs-v1.json';OUT=B/'original-C-Eco6-window-audit-v1.json'
d=w.read(INPUT)
for p,h in d['files'].items():assert w.sha(p)==h,p
assert len(d['points'])==6
rows=[w.audit(p) for p in d['points']]
result=dict(schema='original-C-Eco6-full-window-replay-readonly-audit-v1',created_s=time.time(),host=socket.gethostname(),pid=os.getpid(),source=w.ref(Path(__file__).resolve()),replay_source=w.ref(Path(w.__file__)),inputs=w.ref(INPUT),points=rows,point_count=6,original_720_unchanged=True,quarantined_cells=[p['cell_id'] for p in rows if p['scientific_quarantine_recommended']],no_GPU_work=True)
assert not OUT.exists()
with OUT.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(dict(output=w.ref(OUT),host=socket.gethostname(),pid=os.getpid(),points=[{k:r[k] for k in ('cell_id','failed_requests','demonstrated_window_stranding_count')} for r in rows])))
