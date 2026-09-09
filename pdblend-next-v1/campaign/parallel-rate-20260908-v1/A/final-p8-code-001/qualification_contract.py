"""Recompute actual P8 lifecycle qualification; preserve measured P6 provenance."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import subprocess

HERE = Path(__file__).resolve().parent
A = HERE.parent
ROOT = A.parent
HOST = ROOT / 'hosts/14b-capacity-p8'


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(value, message):
    if not value:
        raise RuntimeError(message)


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked(reference):
    need(sha(reference['path']) == reference['sha256'], 'changed qualification reference')
    return json.loads(Path(reference['path']).read_text())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checked_stage_spec(audit, stage_module):
    spec = checked(audit['spec'])
    need(spec['files'] and all(sha(p) == h for p, h in spec['files'].items()),
         'actual executed stage source/raw files changed')
    config = checked(spec['config'])
    need(config['capacity_binding_path'] == spec['capacity_binding']['path']
         and config['capacity_binding_sha256'] == spec['capacity_binding']['sha256']
         and config['profiles'] == spec['profiles']['path'],
         'executed config capacity/profile differs from stage specification')
    # The frozen driver resolves its own helper in an isolated interpreter.
    # Do not depend on the report process's sys.path or sys.modules.
    driver = A / 'load-p8-code-001/capacity_load_calibrate.py'
    need(spec['files'].get(str(driver)) == sha(driver), 'actual driver must be frozen')
    result = subprocess.run([sys.executable, '-I', str(driver), '--spec', audit['spec']['path'],
        '--spec-sha256', audit['spec']['sha256'], '--out', audit['output']],
        capture_output=True, text=True, timeout=120, check=False)
    need(result.returncode == 0, 'isolated original stage validator rejected: ' + result.stderr[-1000:])
    need(json.loads(result.stdout)['cpu_only'] is True, 'stage validation must be read-only')
    need(audit['source'] == ref(HOST / 'manifest.json') and audit['profile'] == spec['profiles'],
         'only actual P8 controller and original profile qualify')
    need(audit['full_work'] and audit['failed_requests'] == 0 and audit['request_timeouts'] == 0,
         'candidate stage must complete every declared request')
    return spec


def main_work_evidence(audit, stage_module):
    out = Path(audit['output'])
    inventory = json.loads((out / 'inventory.json').read_text())
    control = [json.loads(line) for line in (out / 'control.jsonl').open()]
    dispatch = [json.loads(line) for line in (out / 'engine-dispatch.jsonl').open()]
    # The same actual gate filter requires a declared client ID, a main
    # admission route, native execution on GPU5, and a queued-work proposal.
    return stage_module.autonomous_gate_evidence(inventory, dispatch, control, audit['n_expected'])


def transition_prediction_errors(audit, stage_module):
    spec = checked(audit['spec'])
    cap = checked(spec['capacity_binding'])
    certificate = checked(cap['calibration'])
    inv = checked(ref(Path(audit['output']) / 'inventory.json'))
    errors = []
    commits = [e for e in inv['events'] if e['kind'] == 'physical_commit']
    for commit in commits:
        transaction = commit['transaction']
        begins = [e for e in inv['events'] if e['kind'] == 'transition_begin' and e['transaction'] == transaction]
        measurements = [e for e in inv['events'] if e['kind'] == 'transition_measurement' and e['transaction'] == transaction]
        need(len(begins) == len(measurements) == 1, 'each actual transaction requires unique original start and raw power')
        proposal = begins[0].get('proposal')
        operation = 'restore_cold' if commit['operation'] == 'restore' else 'remove'
        empirical = [e for e in certificate['transitions'] if e['operation'] == operation and e['gpus'] == [5]]
        need(len(empirical) == 1, 'exact empirical transaction domain required')
        duration_estimate = proposal['duration_upper_s'] if proposal else empirical[0]['duration_upper_s']
        energy_estimate = proposal['action_energy_upper_j'] if proposal else empirical[0]['energy_upper_j']
        raw_ref = measurements[0]['receipt']
        raw = stage_module.prior.raw_measurement(raw_ref)
        duration = max(commit['finished_s'] - commit['started_s'], raw['duration_s'])
        errors.append(dict(transaction=transaction, operation=operation, scope=commit['scope'],
            estimate_source='actual_policy_proposal' if proposal else 'frozen_empirical_cleanup_reference',
            certificate=cap['calibration'], raw_measurement=raw_ref,
            planned_duration_s=duration_estimate, measured_duration_s=duration,
            duration_error_s=duration-duration_estimate,
            planned_energy_j=energy_estimate, measured_whole_node_energy_j=raw['energy_j'],
            energy_error_j=raw['energy_j']-energy_estimate,
            duration_exceeded=duration>duration_estimate, energy_exceeded=raw['energy_j']>energy_estimate,
            bounds_are_not_guarantees=True, overlapping_energy_not_added_to_service=True))
    need(len(errors) == audit['physical_commits'], 'all physical transactions must retain prediction errors')
    return errors


def build_qualification():
    old_paths = list(sys.path)
    try:
        sys.path[:0] = [str(A), str(ROOT), str(HOST), str(HOST / 'src'),
                       '/root/workspace/pdblend/.runtime-deps']
        auditor = load(A / 'p8_qualification_audit_v2.py', 'formal_p8_actual_stage_auditor')
        gate, dynamic = auditor.audit_gate(), auditor.audit900()
        gate_spec = checked_stage_spec(gate, auditor)
        dynamic_spec = checked_stage_spec(dynamic, auditor)
        gate_main = main_work_evidence(gate, auditor)
        dynamic_main = main_work_evidence(dynamic, auditor)
        proof_ref = dynamic_spec['controller_calibration_compatibility']
        need(proof_ref == gate_spec['controller_calibration_compatibility'], 'same narrow source compatibility required')
        proof = checked(proof_ref)
        need(proof['certificate_scope']=='original_P6_layout_savings_with_actual_P8_transition_groups' and proof['new_transition_qualification_complete'], 'actual P8 three-transition scope required')
        cap = checked(dynamic_spec['capacity_binding'])
        verifier_ref = proof['verifier']
        need(cap['files'].get(verifier_ref['path']) == verifier_ref['sha256'] == sha(verifier_ref['path']), 'exact shared verifier must be frozen')
        verifier = load(Path(verifier_ref['path']), 'formal_p8_exact_compatibility_verifier')
        verifier.verify(proof_ref, ref(HOST / 'manifest.json'), cap)
        need(dynamic_spec['trace'] == checked(ref(A / 'p6-qualification900-inputs-002/declaration.json'))['trace'],
             'original900 trace cannot change')
        fixed_ref = ref(ROOT / 'C/A-fixed900-negative-independent-001.json')
        negative = load(A / 'fixed900_negative_audit_v1.py', 'formal_p8_reference_negative_auditor')
        negative.validate_evidence(fixed_ref)
        fixed = checked(fixed_ref)
        need(fixed['passed'] and fixed['measurement_valid'] and not fixed['full_work']
             and not fixed['equal_work_energy_comparison_eligible'], 'fixed P6 reference remains a negative observation')
        prior_ref = ref(A / 'p6-qualification-continuation-003/status.json')
        prior = checked(prior_ref)
        need(prior['phase'] == 'stopped_failure' and not prior['complete'] and not auditor.prior.alive(prior['pid']),
             'original strict paired qualification failure must remain terminal')
        failed_p6 = ref(A / 'p6-qualification900-dynamic-002/status.json')
        failed = checked(failed_p6)
        need(not failed['complete'] and failed['cleanup_complete'] and not failed.get('cleanup_errors')
             and not auditor.prior.alive(failed['pid']), 'failed P6 live-controller attempt must remain terminal and cleaned')
        failed_p7 = ref(A / 'diagnosis-p7-transition-bound-001/terminal-negative.json')
        recovered_p7 = ref(A / 'p7-failed900-recovery-001/status.json')
        need(checked(recovered_p7)['complete'], 'historical P7 retained-native restoration must be verified')
        files = {**gate['files'], **dynamic['files'], **fixed['files'], **proof['files']}
        references = [ref(__file__), ref(A / 'p8_qualification_audit_v2.py'), fixed_ref, prior_ref,
                      failed_p6, failed_p7, recovered_p7, proof['verifier'], proof_ref, dynamic_spec['capacity_binding'], cap['calibration'],
                      ref(HOST / 'manifest.json'), gate['spec'], dynamic['spec']]
        for reference in references:
            checked(reference) if Path(reference['path']).suffix == '.json' else None
            files[reference['path']] = reference['sha256']
        for p in (A / 'p6-qualification900-dynamic-002').rglob('*'):
            if p.is_file():
                files[str(p)] = sha(p)
        need(all(sha(p) == h for p, h in files.items()), 'qualification source/raw artifacts changed')
        return dict(schema='p8-candidate900-qualified-P6-calibration-v1', passed=True,
            source=ref(HOST / 'manifest.json'), profile=dynamic_spec['profiles'],
            certificate=cap['calibration'], capacity_binding=dynamic_spec['capacity_binding'],
            trace=dynamic_spec['trace'], controller_calibration_compatibility=proof_ref,
            measured_controller_manifest=proof['measured_controller_manifest'],
            actual_controller_manifest=proof['actual_controller_manifest'],
            actual_transition_controller_manifest=proof['actual_transition_controller_manifest'],
            empirical_cost_prediction_misses=dict(gate=gate['empirical_cost_prediction_misses'], dynamic=dynamic['empirical_cost_prediction_misses']),
            empirical_estimates_are_not_guarantees=True, zero_empirical_prediction_miss_not_required=True,
            transition_prediction_errors=dict(gate=transition_prediction_errors(gate,auditor), dynamic=transition_prediction_errors(dynamic,auditor)),
            arms=dict(fixed2=fixed, dynamic=dynamic), autonomous_gate=gate,
            main_work_evidence=dict(gate=gate_main, dynamic=dynamic_main),
            fixed_reference_audit=fixed_ref, prior_all_full_gate=prior_ref, failed_P6_dynamic_status=failed_p6, failed_P7_transition=failed_p7, historical_P7_recovery=recovered_p7,
            qualification_auditor=ref(__file__), stage_auditor=ref(A / 'p8_qualification_audit_v2.py'),
            candidate_requires_complete_zero_failure=True, reference_observation_only=True,
            equal_work_energy_comparison_eligible=False, initial_instances=2,
            actual_growth_and_return=True, formal_100s_performance_not_inferred=True,
            P6_numerical_measurements_not_relabelled=True, development_only=True, files=files)
    finally:
        sys.path[:] = old_paths


def validate_qualification(reference):
    actual = build_qualification()
    need(checked(reference) == actual, 'qualification does not match independently recomputed actual evidence')
    return actual


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    value = build_qualification()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(ref(args.out)))
