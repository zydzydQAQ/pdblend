"""Explicit P6 numerical / P8 controller reuse, with newly measured transition groups."""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from capacity_calibration_compatibility_v1 import checked, checked_host, need, sha, verify as verify_p7
from build_capacity_p8 import physical_equivalence

MODULES = {'capacity_runtime.py', 'capacity_executor.py', 'capacity_backend.py',
           'capacity_certificate.py', 'capacity_planner.py'}
RUNTIME = 'src/ecopadg/serving/runtime.py'


def validate_certificate_bound(certificate_reference, identity, manifest_reference):
    """Validate in an isolated interpreter with both source modules pinned."""
    manifest, root = checked_host(manifest_reference)
    payload = dict(root=str(root), manifest=manifest_reference, certificate=certificate_reference,
                   identity=identity, modules={name:manifest['files'][name] for name in
                                              ('capacity_certificate.py','capacity_executor.py')})
    script = r"""
import hashlib,importlib,json,pathlib,sys
p=json.load(sys.stdin)
root=pathlib.Path(p['root'])
sys.path[:0]=[str(root),str(root/'src'),'/root/workspace/pdblend/.runtime-deps']
for filename,digest in p['modules'].items():
    module=importlib.import_module(filename[:-3])
    path=pathlib.Path(module.__file__).resolve()
    assert path==(root/filename).resolve()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
from capacity_certificate import validate
reference=p['certificate'];path=pathlib.Path(reference['path'])
assert hashlib.sha256(path.read_bytes()).hexdigest()==reference['sha256']
certificate=json.loads(path.read_text())
evidence=validate(certificate,p['identity'])
print(json.dumps(dict(passed=True,evidence_count=len(evidence),modules=p['modules'])))
"""
    result=subprocess.run([sys.executable,'-I','-c',script],input=json.dumps(payload),
                          text=True,capture_output=True,timeout=120)
    need(result.returncode==0,'explicit actual-source certificate validator rejected: '+result.stderr[-1000:])
    value=json.loads(result.stdout)
    need(value['passed'] is True and value['modules']==payload['modules'],
         'actual-source numerical validation result differs')
    return value


def verify(reference, actual_controller_manifest, capacity):
    proof = checked(reference)
    need(proof['verifier'] == dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
         'proof must bind this exact explicit-source verifier')
    need(capacity['files'].get(reference['path']) == reference['sha256'],
         'capacity binding must freeze its compatibility proof')
    need(proof['schema'] == 'P6-numerical-P8-bounded-physical-controller-compatibility-v1'
         and proof['authorized'] is True and proof['original_measurements_not_relabelled'] is True
         and proof['requires_actual_P8_cold3_and_remove3'] is True
         and proof['requires_actual_P8_autonomous_gate_and900'] is True,
         'explicit P8 source/physical qualification contract required')
    need(proof['actual_controller_manifest'] == actual_controller_manifest, 'actual controller outside declared proof')
    need(proof['files'] and all(sha(p) == h for p, h in proof['files'].items()), 'compatibility dependencies changed')
    old_capacity = checked(proof['measured_capacity_binding'])
    p7proof = verify_p7(proof['P6_to_P7_proof'], proof['P7_controller_manifest'], old_capacity)
    need(p7proof['measured_controller_manifest'] == proof['measured_controller_manifest']
         and p7proof['certificate'] == proof['original_P6_certificate'], 'original numerical provenance changed')
    previous, previous_root = checked_host(proof['P7_controller_manifest'])
    actual, actual_root = checked_host(actual_controller_manifest)
    need(actual['implementation_series'] == 'parallel-p8' and actual['model'] == previous['model'] == '14b',
         'only A measured P6/P7 to actual A P8 is declared')
    need(set(actual['files']) == set(previous['files'])
         and {p for p in actual['files'] if actual['files'][p] != previous['files'][p]}
         == {'capacity_executor.py', 'capacity_runtime.py'}, 'P8 source delta exceeds the two reviewed modules')
    for name, reference_source in proof['approved_patch_sources'].items():
        need(name in {'capacity_executor.py', 'capacity_runtime.py'}
             and sha(reference_source['path']) == reference_source['sha256'] == actual['files'][name],
             'P8 implementation differs from reviewed optional budget patch')
    need(set(proof['approved_patch_sources']) == {'capacity_executor.py', 'capacity_runtime.py'},
         'both changed module pins required')
    equality = physical_equivalence(previous_root, actual_root)
    need(equality == proof['physical_calibration_methods_ast_equal'] and all(equality.values()),
         'manually invoked numerical calibration path changed')
    need(actual['files'][RUNTIME] == previous['files'][RUNTIME], 'P7 admission/clock/strategy source must be unchanged')
    for name in MODULES - {'capacity_executor.py', 'capacity_runtime.py'}:
        need(actual['files'][name] == previous['files'][name] == p7proof['capacity_module_hashes'][name],
             'native verification, numerical validator or planner differs')
    need(capacity['identity'] == old_capacity['identity'] == proof['calibration_source_identity']
         and capacity['calibrated_source_semantics'] == old_capacity['calibrated_source_semantics'],
         'original P6 numerical identity and semantics must remain explicit')
    locations = {'owner_id', 'http_port_base', 'kv_port_base', 'runtime_dir', 'files',
                 'controller_calibration_compatibility', 'physical_operation_timeout_s', 'calibration'}
    need({k: v for k, v in capacity.items() if k not in locations}
         == {k: v for k, v in old_capacity.items() if k not in locations},
         'new budget cannot alter memory, shape, planner policy, creation limit or empirical domains')
    need(type(capacity['physical_operation_timeout_s']) in (int, float)
         and capacity['physical_operation_timeout_s'] == 120,
         'this declaration allows only the explicit finite120 physical budget')
    need(capacity['calibration'] == proof['certificate'], 'capacity numerical certificate differs from declared selection')
    certificate = checked(proof['certificate'])
    need(certificate['identity'] == old_capacity['identity'] and certificate['measurement_verified'] is True
         and certificate['bounds_are_empirical_not_hard_guarantees'] is True,
         'true empirical certificate required')
    validate_certificate_bound(proof['certificate'], capacity['identity'], actual_controller_manifest)
    original = checked(proof['original_P6_certificate'])
    if proof['certificate_scope'] == 'original_P6_for_development_recalibration_only':
        need(proof['certificate'] == proof['original_P6_certificate']
             and proof['new_transition_qualification_complete'] is False,
             'old transitions only bootstrap explicitly declared development calibration')
    else:
        need(proof['certificate_scope'] == 'original_P6_layout_savings_with_actual_P8_transition_groups'
             and proof['new_transition_qualification_complete'] is True,
             'new physical transition qualification scope required')
        need(certificate['layouts'] == original['layouts'] and certificate['savings'] == original['savings'],
             'unchanged P6 layout and savings groups must be reused without selection or relabelling')
        selection = checked(proof['transition_source_selection'])
        need(selection['schema'] == 'P8-actual-cold3-remove3-source-selection-v1'
             and selection['actual_controller_manifest'] == actual_controller_manifest
             and selection['original_P6_certificate'] == proof['original_P6_certificate'],
             'explicit old-layout/new-transition selection required')
        need(len(selection['runs']) == 3 and len({r['spec']['path'] for r in selection['runs']}) == 3,
             'three distinct declared physical invocations required')
        for run in selection['runs']:
            spec = checked(run['spec']); status = checked(run['status'])
            need(Path(spec['host_release']) == actual_root and spec['mode'] == 'underload_gate'
                 and checked(spec['config'])['capacity_integration_v1'] is False,
                 'new numerical transactions must be actual P8 explicit calibration')
            need(status['complete'] and status['cleanup_complete'] and not status.get('cleanup_errors')
                 and not status.get('error'), 'all three real calibration invocations must complete and clean')
            need(all(sha(p) == h for p, h in spec['files'].items()), 'new calibration source/raw pins changed')
        groups = [checked(r) for r in certificate['evidence_groups']]
        transitions = [g for g in groups if g['kind'] == 'transition']
        need(len(transitions) == 2 and {g['operation'] for g in transitions} == {'restore_cold', 'remove'},
             'new actual cold and removal groups required')
        for group in transitions:
            need(group['terminal_source_selection'] == proof['transition_source_selection']
                 and group['actual_controller_manifest'] == actual_controller_manifest,
                 'every new transition group must retain actual P8 measurement provenance')
            expected = selection['cold_members'] if group['operation'] == 'restore_cold' else selection['remove_members']
            need(group['members'] == expected, 'new certificate may not select or replace declared physical repetitions')
    need(all(t['duration_upper_s'] <= 120 for t in certificate['transitions']),
         'physical safety budget must cover every empirical planning duration')
    return proof
