#!/usr/bin/env python3
import csv,json,sys
from pathlib import Path
root,out=Path(sys.argv[1]),Path(sys.argv[2]); rows=[]
audit=root/'profile-v2-parallel'/'audit.json'
if audit.exists():
 d=json.loads(audit.read_text()); q=d.get('quality',{}); mx=[v.get('max',0) for k,v in q.items() if k.startswith('decode_time@')]
 dp=[v for k,v in q.items() if k.startswith('decode_power@')]
 m=d.get('mixed',{})
 rows.append(dict(name='profile-v2-audit',j_per_token='',mean_power_w='',window_mean_power_w='',joint_slo_rate='',
  ttft_p90='',ttft_p99='',tpot_p99='',plans='',wakes='',parks='',policy='profile-audit',profile=d.get('profile',''),
  audit_status=d.get('status',''),decode_time_max=max(mx or [0]),decode_power_mape=max((x.get('mape',0) for x in dp),default=0),
  decode_power_max=max((x.get('max',0) for x in dp),default=0),mixed_median=m.get('median',''),mixed_max=m.get('max',''),
  pending=len(d.get('pending',[]))))
for p in sorted(root.glob('profile-v2-*/measurement/summary.json')):
 d=json.loads(p.read_text());s=d['slo'];c=d.get('controller',{}).get('events',{})
 rows.append(dict(name=p.parent.parent.name,j_per_token=d['j_per_token'],mean_power_w=d['mean_power_w'],window_mean_power_w=d['window_mean_power_w'],joint_slo_rate=s['joint_slo_rate'],ttft_p90=s['ttft_p90'],ttft_p99=s['ttft_p99'],tpot_p99=s['tpot_p99'],plans=c.get('plan',0),wakes=c.get('wake',0),parks=c.get('park',0),policy=d['policy']['name'],profile=d['profile']))
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',newline='') as f:
 fields=list(rows[0]) if rows else ['name']
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
print(f'wrote {out} rows={len(rows)}')
