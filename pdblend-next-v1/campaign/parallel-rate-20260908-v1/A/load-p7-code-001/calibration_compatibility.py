"""Use the frozen narrow compatibility verifier; never relabel P6 measurements."""
from pathlib import Path
import sys
from capacity_executor import fixed,require,sha
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R))
from capacity_calibration_compatibility_v1 import verify

def validate_compatibility(spec,capacity):
 reference=spec['controller_calibration_compatibility']
 require(capacity['controller_calibration_compatibility']==reference,'capacity/spec compatibility differs')
 path=Path(spec['host_release'])/'manifest.json'
 proof=verify(reference,dict(path=str(path),sha256=sha(path)),capacity)
 original=fixed(proof['measured_capacity_binding'])
 locations={'owner_id','http_port_base','kv_port_base','runtime_dir','files','controller_calibration_compatibility'}
 require({k:v for k,v in capacity.items() if k not in locations}
     =={k:v for k,v in original.items() if k not in locations},
     'reuse cannot change measured physical policy, budgets, domain, or limits')
 require(spec['profiles']==original['calibrated_source_semantics']['profile'],
         'actual serving profile differs from reused numerical evidence')
 require(spec['files'].get(reference['path'])==reference['sha256'] and all(
     spec['files'].get(p)==h for p,h in proof['files'].items()),'new spec must freeze compatibility dependencies')
 return proof
