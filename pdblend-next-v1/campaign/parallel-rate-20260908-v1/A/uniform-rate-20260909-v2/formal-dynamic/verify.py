"""Verify a fresh per-cell inventory and the evidence-only formal collector."""
from pathlib import Path
import copy,sys
F=Path(__file__).resolve().parent;H=F.parent/'calibration-supplement';U=F.parents[1]/'uniform-rate-20260909-v1'
sys.path.insert(0,str(U/'dynamic-producer'))
import fresh_support as f

def verify(reference):
 q=f.checked(reference)
 if q['schema']=='new-A-mixed-fresh-dynamic-capacity-qualification-v2':
  sys.path.insert(0,str(H));return f.load(H/'verify_v2.py','newA_v2_parent_qualification').verify(reference)
 f.need(q['schema']=='new-A-dynamic-formal-cell-qualification-v2','one declared formal wrapper required')
 f.need(q['parent_validator']==f.ref(H/'verify_v2.py'),'unrecognized parent verifier')
 sys.path.insert(0,str(H));parent=f.load(H/'verify_v2.py','newA_v2_parent_qualification').verify(q['parent_qualification'])
 old=f.checked(parent['binding']);binding=f.checked(q['binding']);config=f.checked(q['config']);cap=f.checked(q['capacity'])
 expected=copy.deepcopy(old);expected.update(configs=dict(alpaca=q['config']['path']),files=q['files'],output=q['measurement_output'],token_evidence_collector=q['collector'],formal_source_equivalence=q['source_equivalence'])
 f.need(binding==expected and q['node']=='Anew20260909' and q['model']=='14b' and q['dataset']=='alpaca','operational wrapper changed qualified owner/config')
 original_config=f.read(old['configs']['alpaca']);original_cap=f.read(original_config['capacity_binding_path'])
 desired_cap=copy.deepcopy(original_cap);desired_cap.update(runtime_dir=q['runtime_dir'],owner_id=q['owner_id'],max_creations=8)
 f.need(cap==desired_cap and cap['calibration']==f.checked(q['parent_qualification'])['certificate'],'operational capacity changed source/control')
 desired=dict(original_config,journal=q['journal'],capacity_binding_path=q['capacity']['path'],capacity_binding_sha256=q['capacity']['sha256'],capacity_inventory_path=q['inventory_path'],measurement_window_protocol='per-dataset-slo-five-system-fixed-window-v1',arrival_window_s=100.)
 f.need(config==desired and config['capacity_integration_v1'] is True,'formal wrapper changed control/SLO/profile')
 f.need(all(f.sha(path)==sha for path,sha in q['files'].items()),'cell source/evidence closure changed')
 manifest=f.checked(q['collector_manifest']);f.need(manifest['collector']==q['collector'] and manifest['legacy_fields_unchanged'] and manifest['control_source_unchanged'],'formal collector proof differs')
 equivalence=f.checked(q['source_equivalence'])
 for name,item in equivalence['files'].items():
  f.need(f.sha(item['original'])==item['original_sha256'] and f.sha(F/name)==item['actual_sha256'],'formal executor source equivalence changed')
  if item.get('byte_identical'):f.need(item['actual_sha256']==item['original_sha256'],'unchanged executor differs')
 return dict(passed=True,independently_recomputed=True,qualification=reference,binding=q['binding'],node='Anew20260909',model='14b',datasets=['alpaca'],dynamic_capacity_qualification=True,collector=q['collector'])
