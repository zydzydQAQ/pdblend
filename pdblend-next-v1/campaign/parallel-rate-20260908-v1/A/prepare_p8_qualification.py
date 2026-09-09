"""Bind P8 autonomous qualification to its fresh three-transaction cost certificate."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

A = Path(__file__).resolve().parent
R = A.parent
CODE = A/'load-p8-code-001'
HOST = R/'hosts/14b-capacity-p8'
sys.path[:0] = [str(CODE), str(HOST/'src'), str(HOST), str(R), '/root/workspace/pdblend/.runtime-deps']
from capacity_executor import fixed, require, sha
from capacity_certificate import ref
from calibration_compatibility import validate_compatibility


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def prepare(stage):
    require(stage in ('gate', '900'), 'only explicitly declared autonomous qualification stages')
    registration = fixed(ref(A/'alpaca-capacity-certificate-p8-001/registration.json'))
    require(registration['passed'] and registration['requires_actual_P8_autonomous_gate_and900'], 'fresh P8 costs not registered')
    compatibility = registration['compatibility']; proof = fixed(compatibility)
    require(proof['certificate_scope']=='original_P6_layout_savings_with_actual_P8_transition_groups'
            and proof['new_transition_qualification_complete'], 'bootstrap evidence cannot run autonomous control')
    cpu = fixed(ref(CODE/'cpu-validation.json'))
    require(cpu['passed'] and not cpu['gpu_qualified'] and all(sha(p)==h for p,h in cpu['files'].items()), 'frozen driver CPU/source differs')
    p8cpu = ref(R/'cpu-validation-p8.json')
    require(fixed(p8cpu)['passed'], 'common P8 source CPU verification required')
    oldspec = fixed(ref(A/'p7-qualification900-inputs-001/spec.json'))
    out = A/('p8-autonomous-gate-001' if stage=='gate' else 'p8-qualification900-dynamic-001')
    inputs = A/('p8-autonomous-gate-inputs-001' if stage=='gate' else 'p8-qualification900-inputs-001')
    require(not out.exists() and not inputs.exists(), 'new immutable P8 invocation required')
    gate = None
    if stage=='900':
        from p8_qualification_audit import audit_gate
        gate = audit_gate()
        require(gate['passed'], 'actual P8 autonomous first gate must complete and clean')
    cap = fixed(proof['measured_capacity_binding'])
    files = {**cap['files'], **proof['files'], **cpu['files']}
    manifest = fixed(ref(HOST/'manifest.json'))
    files.update({str(HOST/name):h for name,h in manifest['files'].items()})
    references = [compatibility, proof['verifier'], registration['certificate'],
        ref(A/'alpaca-capacity-certificate-p8-001/registration.json'), p8cpu,
        ref(HOST/'manifest.json'), ref(CODE/'manifest.json'), ref(CODE/'cpu-validation.json'),
        ref(__file__), ref(A/'p8_qualification_audit.py'), ref(A/'diagnosis-p7-transition-bound-001/terminal-negative.json')]
    if gate is not None:
        write(inputs/'autonomous-gate-audit.json',gate)
        references.append(ref(inputs/'autonomous-gate-audit.json'));files.update(gate['files'])
    for rr in references: files[rr['path']]=rr['sha256']
    cap.update(owner_id='ap8gate' if stage=='gate' else 'ap8q900',
        http_port_base=32100 if stage=='gate' else 32200, kv_port_base=62000,
        runtime_dir=str(out/'runtime'), calibration=registration['certificate'],
        physical_operation_timeout_s=120, controller_calibration_compatibility=compatibility, files=files)
    write(inputs/'capacity-binding.json',cap); capref=ref(inputs/'capacity-binding.json')
    config = fixed(oldspec['config'])
    config.update(port=32150 if stage=='gate' else 32250, capacity_integration_v1=True,
        capacity_binding_path=capref['path'],capacity_binding_sha256=capref['sha256'])
    write(inputs/'config.json',config); configref=ref(inputs/'config.json')
    trace = oldspec['trace'] if stage=='900' else fixed(ref(A/'p7-autonomous-gate-001/automatic_underload_gate/result.json'))['trace']
    work = dict(schema='P8-autonomous-qualification-work-v1',authorized=True,stage=stage,created_s=time.time(),
        source=ref(HOST/'manifest.json'),profile=oldspec['profiles'],trace=trace,certificate=registration['certificate'],
        capacity_binding=capref,controller_calibration_compatibility=compatibility,
        original_layout_controller_manifest=proof['measured_controller_manifest'],
        actual_transition_controller_manifest=proof['actual_controller_manifest'],
        initial_instances=2,automatic_retries=False,manual_restore_used=False,
        arrival_duration_s=60 if stage=='gate' else 900,request_timeout_s=120,physical_operation_timeout_s=120,
        all8gpu_energy=True,formal_performance_not_inferred=True,
        empirical_cost_prediction_misses_retained=True,zero_empirical_prediction_miss_not_required=True,
        empirical_cost_estimate_is_not_energy_savings_proof=True)
    write(inputs/'work-declaration.json',work)
    spec = copy.deepcopy(oldspec)
    for name in ('cold_start_after_s','cycles','required_autonomous_gate_audit','paired_failed_result','paired_trace_result'):
        spec.pop(name,None)
    spec.update(mode='automatic_underload_gate' if stage=='gate' else 'qualification900',arm='dynamic',
        trace=trace,config=configref,capacity_binding=capref,host_release=str(HOST),
        api_base='http://127.0.0.1:'+str(config['port']),controller_calibration_compatibility=compatibility,
        input_declaration=ref(inputs/'work-declaration.json'),cpu_evidence=p8cpu,
        actual_certificate=registration['certificate'],stop_path=str(A/'STOP_P8_QUALIFICATION'),files=dict(files))
    if stage=='gate': spec['paired_trace_result']=ref(A/'load-p6-gate-002/underload_gate/result.json')
    else: spec['required_autonomous_gate_audit']=ref(inputs/'autonomous-gate-audit.json')
    for rr in [configref,capref,trace,spec['input_declaration'],
               spec['paired_trace_result'] if stage=='gate' else spec['required_autonomous_gate_audit']]:
        spec['files'][rr['path']]=rr['sha256']
    validate_compatibility(spec,cap)
    write(inputs/'spec.json',spec);sr=ref(inputs/'spec.json')
    declaration=dict(work,specs=dict(dynamic=sr),output=str(out),
        parent_original900_declaration=ref(A/'p6-qualification900-inputs-002/declaration.json'))
    write(inputs/'declaration.json',declaration)
    print(json.dumps(dict(stage=stage,spec=sr,out=str(out),declaration=ref(inputs/'declaration.json'))))
    return sr,out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['gate','900']);prepare(parser.parse_args().stage)
