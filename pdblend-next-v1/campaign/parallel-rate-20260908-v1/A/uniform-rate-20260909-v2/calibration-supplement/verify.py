"""Independently replay the mixed fresh-A certificate and original 752/900 gates."""
from pathlib import Path
import copy,sys
H=Path(__file__).resolve().parent;U=H.parents[1]/'uniform-rate-20260909-v1'
sys.path.insert(0,str(U/'dynamic-producer'))
import fresh_support as f
sys.path.insert(0,str(H))
import combine
import audit as original_audit
from capacity_certificate import validate

def fixed_verify(reference,validator):
 f.need(f.sha(validator['path'])==validator['sha256'],'fixed verifier source changed')
 module=f.load(validator['path'],'mixedA_actual_fixed_verifier')
 result=module.verify(reference)
 f.need(result['passed'] and result['independently_recomputed'],'fixed native qualification failed')
 return result

def verify(reference):
 q=f.checked(reference)
 f.need(q['schema']=='new-A-mixed-fresh-dynamic-capacity-qualification-v2','unknown fresh capacity qualification')
 for key in ('files','source_files'):
  f.need(q[key] and all(f.sha(path)==sha for path,sha in q[key].items()),'qualification closure changed')
 fixed=fixed_verify(q['fixed_qualification'],q['fixed_validator'])
 state=f.checked(q['status'])
 f.need(state['complete'] and not state.get('error') and state['finished_s'] and not state['node_lease_held'] and f.no_live_pid(state['pid'],state['startticks']),'producer must be terminal')
 f.need(state['stages']==q['stages'] and state['binding']==q['binding'] and state['composition']==q['composition'],'published state differs')
 composition=combine.verify(q['composition'])
 f.need(composition['certificate']==q['certificate'],'mixed certificate reference differs')
 f.need([s['mode'] for s in q['stages']]==['automatic_underload_gate','qualification900'],'both original autonomous gates required')
 for stage in q['stages']:
  f.need(original_audit.audit_stage(stage['output'],stage['spec'],stage['mode'])==f.checked(stage['audit']),'original autonomous stage replay differs')
  cap=f.checked(f.checked(stage['spec'])['capacity_binding'])
  f.need(cap['identity']==composition['identity'] and cap['calibration']==q['certificate'],'gate uses different measured identity/certificate')
 validate(f.checked(q['certificate']),composition['identity'])
 binding=f.checked(q['binding']);f.need(binding['instances']==f.checked(fixed['binding'])['instances'] and binding['independent_capacity_qualification_granted'] is True and binding['host_release']==str(f.HOST) and binding['model']=='14b' and binding['system']=='pdblend','qualified source or original instances differ')
 return dict(passed=True,independently_recomputed=True,qualification=reference,binding=q['binding'],node='Anew20260909',model='14b',datasets=['alpaca'],dynamic_capacity_qualification=True,fresh_physical_calibration=True,old_A_results_inherited=False,limitations=composition['limitations'])
