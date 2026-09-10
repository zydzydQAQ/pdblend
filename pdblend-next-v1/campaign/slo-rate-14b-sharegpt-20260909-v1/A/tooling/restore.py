"""Independent audit of fresh-node cold bootstrap; no retained process is claimed."""
from pathlib import Path
import json,sys
import bootstrap as b
import power_selftest as p

def audit(reference):
 boot=b.checked(reference);assert boot['schema']=='slo14-cold-bootstrap-proof-v1'
 assert boot['hostname']==p.EXPECTED_HOSTNAME and boot['node']==p.NODE and boot['model']=='14b'
 assert boot['complete'] and boot['ordinary_passed'] and not boot['node_lease_held'] and boot['old_node_qualification_inherited'] is False
 assert all(p.sha(x)==v for x,v in boot['files'].items())
 s=b.checked(boot['status']);spec=b.checked(boot['spec'])
 from inputs import validate_bootstrap
 validate_bootstrap(spec)
 assert s['complete'] and s['ordinary_passed'] and s['finished_s'] and not s.get('error') and s['setup_measurement']['measurement_valid'] and not s['node_lease_held']
 assert boot['instances']==s['instances'] and boot['ordinary']==s['ordinary']
 assert len(boot['instances'])==2 and [i['gpus'] for i in boot['instances']]==[[6],[7]]
 oracle=b.checked(spec['numerical_reference']);replies=b.checked(s['ordinary'])
 assert len(replies)==2*len(oracle['cases'])
 for i in boot['instances']:
  assert i['tp']==1 and i['host_pid']>0 and i['container']['id'] and i['container']['StartedAt']
  expected=dict(spec['expected_provenance'],instance_id=i['id'],cuda_visible_devices=str(i['gpus'][0]))
  assert all(i['provenance'].get(k)==v for k,v in expected.items())
  for c in oracle['cases']:
   rs=[r for r in replies if r['instance_id']==i['id'] and r['prompt_length']==c['prompt_length']];assert len(rs)==1
   a=rs[0]['response'];assert a['token_ids']==c['token_ids'] and a['usage']['completion_tokens']==c['max_tokens'] and a['usage']['prompt_tokens']==c['prompt_length']
 restorations=b.checked(s['restoration']);assert len(restorations)==2 and all(r['complete'] for r in restorations)
 sys.path[:0]=[str(p.METER),str(p.HOST/'src')]
 from capacity_certificate import raw_measurement,close
 raw=raw_measurement(s['setup_measurement']['receipt']);assert close(raw['energy_j'],s['setup_measurement']['energy_j'])
 assert s['started_s']<=raw['measurement_start_s']<raw['measurement_end_s']<=s['finished_s']
 return dict(passed=True,independently_recomputed=True,energy_j=raw['energy_j'])
