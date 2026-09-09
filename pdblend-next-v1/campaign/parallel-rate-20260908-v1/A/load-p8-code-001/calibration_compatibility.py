"""Use explicit P6 numerical / P8 controller proof and restrict bootstrap scope."""
from pathlib import Path
import sys
from capacity_executor import fixed, require, sha
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
from capacity_calibration_compatibility_v3 import verify


def validate_compatibility(spec, capacity):
    reference = spec['controller_calibration_compatibility']
    require(capacity['controller_calibration_compatibility'] == reference, 'capacity/spec compatibility differs')
    path = Path(spec['host_release']) / 'manifest.json'
    proof = verify(reference, dict(path=str(path), sha256=sha(path)), capacity)
    original = fixed(proof['measured_capacity_binding'])
    require(spec['profiles'] == original['calibrated_source_semantics']['profile'],
            'actual serving profile differs from original numerical evidence')
    require(spec['files'].get(reference['path']) == reference['sha256'] and all(
            spec['files'].get(p) == h for p, h in proof['files'].items()), 'new spec must freeze compatibility dependencies')
    if proof['certificate_scope'] == 'original_P6_for_development_recalibration_only':
        require(spec['mode'] == 'underload_gate' and fixed(spec['config'])['capacity_integration_v1'] is False,
                'old transition bounds may bootstrap only explicit new physical calibration')
    else:
        require(spec['mode'] in ('automatic_underload_gate', 'qualification900'),
                'new transition certificate only serves declared autonomous qualification here')
    return proof
