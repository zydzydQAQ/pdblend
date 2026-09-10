"""Replay real terminal owners and combine only qualified, same-identity evidence."""
import json,sys,copy
from pathlib import Path
H=Path(__file__).resolve().parent
U=H.parents[1]/'uniform-rate-20260909-v1'
sys.path.insert(0,str(U/'dynamic-producer'))
import fresh_support as f
from capacity_certificate import derive_group,build,validate,idle_result
import audit_support as audit

def union_inventory(references):
 inventories=[f.checked(r) for r in references];first=inventories[0];known={}
 for value in inventories:
  f.need(value['identity']==first['identity'] and value['initial_ids']==first['initial_ids'] and value['active_instances']==first['active_instances'],'terminal original membership differs')
  f.need(value['complete'] and not value['transition_inflight'],'source terminal inventory incomplete')
  for key,instance in value['known_instances'].items():
   f.need(instance['verified'] is True,'unverified union member')
   if key not in value['initial_ids']:
    f.need(instance['state']=='stopped' and instance.get('physical_proof'),'union member lacks actual creation and stopped evidence')
    for operation in ('restore','remove'):
     f.need(any(e['kind']=='physical_commit' and e.get('instance_id')==key and e.get('operation')==operation and e.get('execution_verified') for e in value['events']),'missing physical commit for union member')
   if key in known:
    f.need(key in value['initial_ids'] and {k:v for k,v in known[key].items() if k!='changed_s'}=={k:v for k,v in instance.items() if k!='changed_s'},'same physical id has conflicting immutable membership')
    f.need(type(known[key].get('changed_s')) in (int,float) and type(instance.get('changed_s')) in (int,float),'initial member lifecycle timestamp missing')
   else:
    known[key]=instance
 return dict(schema='capacity-terminal-membership-evidence-union-v1',not_runtime_inventory=True,
   source_inventories=references,identity=first['identity'],complete=True,transition_inflight=False,
   initial_ids=first['initial_ids'],active_instances=first['active_instances'],known_instances=known,
   meaning='Exact union of independently audited terminal member proofs; not an inventory at any instant.')

def verify_membership(reference):
 actual=f.checked(reference);expected=union_inventory(actual['source_inventories'])
 f.need(actual==expected,'derived membership union differs from actual source records')
 return actual

def compose(old_out,new_root,destination):
 old_out,new_root,destination=map(Path,(old_out,new_root,destination));new_out=new_root/'layout_calibration'
 old_spec=f.ref(old_out.parent/'layout_calibration-inputs/spec.json');new_spec=f.ref(new_root/'supplement-spec.json')
 proofs=[]
 for label,out,spec,extra in [('original',old_out,old_spec,dict(exclude_gap=True)),('supplement',new_out,new_spec,dict(supplement=True))]:
  proof=audit.audit_stage(out,spec,'layout_calibration',**extra)
  proofs.append(dict(label=label,output=str(out),spec=spec,options=extra,audit=f.save(destination/f'{label}-terminal-audit.json',proof)))
 old_cap=f.checked(f.checked(old_spec)['capacity_binding']);new_cap=f.checked(f.checked(new_spec)['capacity_binding'])
 f.need(old_cap['identity']==new_cap['identity'],'new development endpoint changed model/node/source identity')
 identity=old_cap['identity'];old_inventory=f.ref(old_out/'inventory.json');new_inventory=f.ref(new_out/'inventory.json')
 inventory=f.save(destination/'idle-membership-evidence.json',union_inventory([old_inventory,new_inventory]))
 groups=[]
 def group(name,kind,members,**fields):
  ref=f.save(destination/(name+'.json'),dict(schema='capacity-evidence-group-v1',identity=identity,capacity_binding=f.checked(new_spec)['capacity_binding'],kind=kind,members=members,**fields))
  derive_group(ref,identity);return ref
 for layout,key in ((2,'high2'),(3,'high3')):
  groups.append(group('layout'+str(layout),'layout',[f.ref(old_out/f'cycle-{n}-{key}-layout{layout}/result.json') for n in (1,2,3)]))
 old_pairs=[dict(source=f.ref(old_out/f'cycle-{n}-idle-layout3/result.json'),target=f.ref(old_out/f'cycle-{n}-idle-layout2/result.json')) for n in (1,2)]
 new_pair=dict(source=f.ref(new_out/'cycle-1-idle-layout3/result.json'),target=f.ref(new_out/'cycle-1-idle-layout2/result.json'))
 # Each reused/native pair must also pass with its real source owner inventory.
 for pair,inv in [(p,old_inventory) for p in old_pairs]+[(new_pair,new_inventory)]:
  for ref in pair.values():idle_result(ref,identity,f.checked(inv))
 savings=[group('saving-idle','idle_savings',old_pairs+[new_pair],inventory=inventory)]
 for key,out in [('low',new_out),('low40',old_out)]:
  savings.append(group('saving-'+key,'savings',[dict(source=f.ref(out/f'cycle-{n}-{key}-layout3/result.json'),target=f.ref(out/f'cycle-{n}-{key}-layout2/result.json')) for n in (1,2,3)]))
 groups.append(group('savings-grid','savings_grid',savings,demand_domain_sha256=f.checked(new_spec)['demand_domain_sha256']))
 for operation in ('restore_cold','remove'):
  paths=[old_out/(f'cycle-{n}-under_load-layout2to3/result.json' if operation=='restore_cold' else f'cycle-{n}-remove.json') for n in (1,2,3)]
  groups.append(group(operation,'transition',[dict(result=f.ref(p),inventory=old_inventory) for p in paths],operation=operation,gpus=[5]))
 certificate=build(identity,groups,destination/'certificate.json');validate(f.checked(certificate),identity)
 report=dict(schema='new-A-mixed-development-certificate-replay-v2',passed=True,independently_recomputed=True,
   identity=identity,calibrations=proofs,certificate=certificate,membership_evidence=inventory,
   idle_pair_source_inventories=[old_inventory,old_inventory,new_inventory],
   low_rate_rps=1.5,low_window_s=60,low_pairs=3,new_idle_windows=2,
   excluded_results=[f.ref(old_out/'cycle-1-low-layout3/result.json'),f.ref(old_out/'cycle-3-idle-layout2/result.json')],
   limitations=['The failed development low .3 result remains a negative; .3 sustainability is not established.',
                'The unchanged certificate savings grid starts at zero; that is an empirical envelope, not proof all rates meet SLO.',
                'Historical profile cost values were not recalibrated; actual new-A domain is 900/1500/2100.'])
 return f.save(destination/'composition-audit.json',report)

def verify(reference):
 report=f.checked(reference);identity=report['identity']
 for stage in report['calibrations']:
  f.need(audit.audit_stage(stage['output'],stage['spec'],'layout_calibration',**stage['options'])==f.checked(stage['audit']),'terminal evidence replay differs')
 inv=verify_membership(report['membership_evidence'])
 certificate=f.checked(report['certificate']);validate(certificate,identity)
 idle_group=f.read(Path(report['certificate']['path']).parent/'saving-idle.json')
 f.need(idle_group['inventory']==report['membership_evidence'] and len(idle_group['members'])==3,'idle membership reference changed')
 for pair,real_inv in zip(idle_group['members'],report['idle_pair_source_inventories']):
  for ref in pair.values():idle_result(ref,identity,f.checked(real_inv))
 f.need(report['passed'] and report['independently_recomputed'],'composition not audited')
 return report
