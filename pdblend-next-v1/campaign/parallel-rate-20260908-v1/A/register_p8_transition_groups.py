"""Reuse the whole P6 layout/savings groups and register all three fresh P8 costs."""
import importlib.util
import json
from pathlib import Path
import sys
import time

A = Path(__file__).resolve().parent
R = A.parent
HOST = R / 'hosts/14b-capacity-p8'
CODE = A / 'load-p8-code-001'
OUT = A / 'alpaca-capacity-certificate-p8-001'
sys.path[:0] = [str(CODE), str(HOST / 'src'), str(HOST), str(A), str(R), '/root/workspace/pdblend/.runtime-deps']
from capacity_certificate import ref, raw_measurement, build, validate, derive_group
from capacity_executor import fixed, durable, require, sha
from run_p8_cold_recalibration import audit


def main():
    require(not OUT.exists(), 'new immutable P8 cost certificate required')
    declaration_ref = ref(A / 'p8-cold-recalibration-inputs-001/declaration.json')
    declaration = fixed(declaration_ref)
    queue = fixed(ref(A / 'p8-cold-recalibration-queue-001/status.json'))
    import continue_p6_qualification_v3 as prior
    require(queue['complete'] and queue['phase'] == 'complete' and not prior.alive(queue['pid'])
            and len(queue['steps']) == 3, 'all declared physical repetitions must terminate successfully')
    bootstrap_ref = ref(R / 'common/P6-numerical-P8-recalibration-bootstrap-v2.json')
    bootstrap = fixed(bootstrap_ref)
    original_ref = bootstrap['original_P6_certificate']
    original = fixed(original_ref)
    identity = original['identity']
    hostref = ref(HOST / 'manifest.json')
    require(hostref == declaration['source'] == bootstrap['actual_controller_manifest'], 'actual P8 source differs')
    s = importlib.util.spec_from_file_location('p8_reused_exact_row_verifier', A / 'p7_qualification_audit.py')
    rows_auditor = importlib.util.module_from_spec(s); s.loader.exec_module(rows_auditor)
    verified = []
    cold, remove = [], []
    for run in declaration['runs']:
        evidence = audit(run)
        spec = fixed(run['spec'])
        require(spec['mode'] == 'underload_gate' and Path(spec['host_release']) == HOST,
                'actual P8 explicit physical calibration required')
        result = fixed(evidence['result'])
        config = fixed(spec['config']); trace = fixed(result['trace'])
        require(result['trace'] == declaration['original_752_trace'] and config['capacity_integration_v1'] is False,
                'original declared752 workload and manual transition path required')
        rows_path = Path(run['output']) / 'underload_gate/requests.json'
        rows_auditor.audit_rows(result, trace, config, raw_measurement(result['raw_measurement']),
                               json.loads(rows_path.read_text()), 60)
        inventory_ref = evidence['inventory']
        inventory = fixed(inventory_ref)
        require(inventory['identity'] == identity and inventory['complete'] and not inventory['transition_inflight'],
                'new measured inventory identity or cleanup differs')
        require(fixed(evidence['remove'])['instance_id'] == result['physical']['instance_id'],
                'remove must be the same actually created physical instance')
        verified.append(dict(**run, status=evidence['status'], result=evidence['result'],
                             remove=evidence['remove'], inventory=inventory_ref))
        cold.append(dict(result=evidence['result'], inventory=inventory_ref))
        remove.append(dict(result=evidence['remove'], inventory=inventory_ref))
    OUT.mkdir()
    selection = dict(schema='P8-actual-cold3-remove3-source-selection-v1', created_s=time.time(),
        declaration=declaration_ref, actual_controller_manifest=hostref,
        original_P6_certificate=original_ref, original_P6_layout_savings_source=bootstrap['measured_controller_manifest'],
        original_P6_identity_retained=identity, runs=verified, cold_members=cold, remove_members=remove,
        all_three_declared_repetitions_selected=True, original_P6_transitions_preserved_not_selected=True,
        old_P7_failure_preserved=ref(A / 'diagnosis-p7-transition-bound-001/terminal-negative.json'),
        registration_source=ref(__file__), row_verifier=ref(A / 'p7_qualification_audit.py'))
    durable(OUT / 'transition-source-selection.json', selection)
    selection_ref = ref(OUT / 'transition-source-selection.json')
    groups = []
    for reference in original['evidence_groups']:
        group = fixed(reference)
        if group['kind'] != 'transition':
            groups.append(reference)
    require(len(groups) == 3, 'entire original2 layout groups and savings grid required')
    capref = fixed(declaration['runs'][0]['spec'])['capacity_binding']
    for operation, members in [('restore_cold', cold), ('remove', remove)]:
        path = OUT / (operation+'.json')
        durable(path, dict(schema='capacity-evidence-group-v1', identity=identity,
            kind='transition', operation=operation, gpus=[5], capacity_binding=capref, members=members,
            terminal_source_selection=selection_ref, actual_controller_manifest=hostref,
            original_numerical_identity_not_relabelled=True, registration_source=ref(__file__)))
        reference = ref(path); derive_group(reference, identity); groups.append(reference)
    certificate_ref = build(identity, groups, OUT / 'certificate.json')
    certificate = fixed(certificate_ref); validate(certificate, identity)
    require(certificate['layouts'] == original['layouts'] and certificate['savings'] == original['savings'],
            'P6 layout/savings groups must remain byte-equivalent in bounds')
    proof = dict(bootstrap, created_s=time.time(), certificate=certificate_ref,
        certificate_scope='original_P6_layout_savings_with_actual_P8_transition_groups',
        new_transition_qualification_complete=True, transition_source_selection=selection_ref,
        actual_transition_controller_manifest=hostref, files=dict(bootstrap['files']))
    for run in verified:
        spec = fixed(run['spec']); proof['files'].update(spec['files'])
        for path in Path(run['output']).rglob('*'):
            if path.is_file(): proof['files'][str(path)] = sha(path)
        proof['files'][run['spec']['path']] = run['spec']['sha256']
    for path in list(OUT.glob('*.json')) + [Path(__file__), A/'p7_qualification_audit.py', A/'run_p8_cold_recalibration.py']:
        proof['files'][str(path)] = sha(path)
    path = OUT/'controller-calibration-compatibility.json'
    durable(path, proof)
    # Validate with an explicit actual source module in its isolated interpreter.
    from capacity_calibration_compatibility_v3 import verify
    cap = fixed(bootstrap['measured_capacity_binding'])
    cap.update(physical_operation_timeout_s=120, calibration=certificate_ref,
               controller_calibration_compatibility=ref(path), files=dict(proof['files']))
    cap['files'][str(path)] = sha(path)
    verify(ref(path), hostref, cap)
    durable(OUT/'registration.json', dict(passed=True, certificate=certificate_ref,
        compatibility=ref(path), transition_source_selection=selection_ref,
        requires_actual_P8_autonomous_gate_and900=True, formal_eligible=False))
    print(json.dumps(dict(certificate=certificate_ref, compatibility=ref(path), transitions=certificate['transitions'])))


if __name__ == '__main__': main()
