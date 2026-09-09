"""Recompute every A12 layout, transition and matched energy bound from raw evidence."""
from pathlib import Path
import json,sys,time,hashlib
A=Path(__file__).resolve().parent
sys.path.insert(0,str(A/'load-p6-code-001'))
from capacity_executor import fixed,require,durable,sha
from capacity_certificate import ref,raw_measurement,derive_group,build,validate

def main():
 src=A/'load-p6-full-002';out=A/'alpaca-capacity-certificate-p6-001'
 require(not out.exists(),'new certificate output required')
 state=json.loads((src/'status.json').read_text());require(state.get('complete') and state.get('cleanup_complete') and not state.get('error') and not state.get('cleanup_errors') and state.get('finished_s'),'complete measured calibration required')
 require(not Path('/proc',str(state['pid'])).exists(),'physical owner must exit before certificate audit')
 inventory=json.loads((src/'inventory.json').read_text());require(inventory['complete'] and not inventory['transition_inflight'] and {i['id'] for i in inventory['active_instances']}=={'nextv3a6','nextv3a7'},'actual return to initial2 required')
 require(not any(e['kind'] in ['transition_failed','rollback_failed'] for e in inventory['events']),'failed physical transaction prevents qualification')
 require(len(state['completed'])==27,'all27 declared observations required')
 raw_measurement(state['full_operation_measurement'])
 sp=fixed(fixed(ref(src/'spec-reference.json')));capref=sp['capacity_binding'];cap=fixed(capref);identity=cap['identity']
 out.mkdir();groups=[]
 def save(name,kind,members,**fields):
  p=out/(name+'.json');durable(p,dict(schema='capacity-evidence-group-v1',identity=identity,kind=kind,capacity_binding=capref,members=members,**fields));return ref(p)
 for layout,key in [(2,'high2'),(3,'high3')]:
  group=save(f'layout{layout}','layout',[ref(src/f'cycle-{n}-{key}-layout{layout}'/'result.json') for n in (1,2,3)]);derive_group(group,identity);groups.append(group)
 savings=[]
 for key in ['idle','low','low40']:
  kind='idle_savings' if key=='idle' else 'savings';members=[dict(source=ref(src/f'cycle-{n}-{key}-layout3'/'result.json'),target=ref(src/f'cycle-{n}-{key}-layout2'/'result.json')) for n in (1,2,3)]
  group=save('saving-'+key,kind,members,**({'inventory':ref(src/'inventory.json')} if key=='idle' else {}));derive_group(group,identity);savings.append(group)
 group=save('savings-grid','savings_grid',savings,demand_domain_sha256=sp['demand_domain_sha256']);derive_group(group,identity);groups.append(group)
 for operation in ['restore_cold','remove']:
  paths=[src/f'cycle-{n}-under_load-layout2to3'/'result.json' if operation=='restore_cold' else src/f'cycle-{n}-remove.json' for n in (1,2,3)]
  group=save(operation,'transition',[dict(result=ref(p),inventory=ref(src/'inventory.json')) for p in paths],operation=operation,gpus=[5]);derive_group(group,identity);groups.append(group)
 reference=build(identity,groups,out/'certificate.json');certificate=fixed(reference);validate(certificate,identity)
 durable(out/'qualification.json',dict(passed=True,created_s=time.time(),certificate=reference,full_operation=state['full_operation_measurement'],source_status=ref(src/'status.json'),source_inventory=ref(src/'inventory.json'),requires_actual_900s_p6_validation=True,production_ready=False,source_scope='measured P6 manual capacity layout; same P6 source with automatic capacity must complete actual900s'))
 print(json.dumps(dict(certificate=reference,layouts=certificate['layouts'],transitions=certificate['transitions'],savings=certificate['savings'])))
if __name__=='__main__':main()
