"""Preserve static fingerprints and normalize explicitly frozen capacity contracts."""
from pathlib import Path
import source_identity as static
import capacity_calibration_compatibility_v1 as compatibility
import capacity_calibration_compatibility_v3 as compatibility_p8
import capacity_calibration_compatibility_v4 as compatibility_p9

sha, read, digest = static.sha, static.read, static.digest
P8_VERIFIER_SHA = '8417881aa82109f4a33720c3bc8ba09913c713f050cc2d9625200937ce9a497d'
P9_VERIFIER_SHA = 'b5e720e771e6ec239411a12e571208b1e3d4067dd28252e4e8d35404fe34d6fd'


def normalized_references(value):
    if isinstance(value, list):
        return [normalized_references(x) for x in value]
    if isinstance(value, dict):
        return {k: normalized_references(v) for k, v in value.items()
                if not (k == 'path' and 'sha256' in value)}
    return value


def identity(binding, dataset):
    result = static.identity(binding, dataset)
    config = read(binding['configs'][dataset])
    enabled = config.get('capacity_integration_v1') is True
    result['capacity_integration_enabled'] = enabled
    if not enabled:
        return result
    path = Path(config['capacity_binding_path'])
    expected = config['capacity_binding_sha256']
    if sha(path) != expected or binding['files'].get(str(path)) != expected:
        raise ValueError('actual capacity binding is not frozen in execution binding')
    capacity = read(path)
    for name, expected_file in capacity['files'].items():
        if sha(name) != expected_file:
            raise ValueError('capacity dependency changed: ' + name)
    if capacity['schema'] != 'capacity-runtime-binding-v1':
        raise ValueError('unrecognized capacity binding')
    certificate_ref = capacity['calibration']
    if sha(certificate_ref['path']) != certificate_ref['sha256']:
        raise ValueError('capacity certificate changed')
    certificate = read(certificate_ref['path'])
    if certificate['identity'] != capacity['identity'] or certificate.get('measurement_verified') is not True:
        raise ValueError('capacity certificate does not cover actual identity')
    semantics = capacity['calibrated_source_semantics']
    if semantics['profile']['sha256'] != result['profile_sha256']:
        raise ValueError('capacity calibration profile differs from serving profile')
    actual_manifest = dict(path=str(Path(binding['host_release']) / 'manifest.json'),
                           sha256=result['host_manifest_sha256'])
    reuse = capacity.get('controller_calibration_compatibility')
    reuse_proof = None
    is_p8_reuse = False
    is_p9_reuse = False
    if semantics['candidate_manifest'] != actual_manifest or reuse is not None:
        if not reuse or any(files.get(reuse['path']) != reuse['sha256']
                            for files in (binding['files'], capacity['files'])):
            raise ValueError('different actual and measured controllers require a frozen compatibility proof')
        declared_proof = read(reuse['path'])
        is_p8_reuse = declared_proof.get('schema') == 'P6-numerical-P8-bounded-physical-controller-compatibility-v1'
        is_p9_reuse = declared_proof.get('schema') == 'P6-layout-P8-transition-P9-fresh-clock-controller-compatibility-v1'
        if is_p9_reuse:
            verifier = dict(path=str(Path(compatibility_p9.__file__).resolve()), sha256=sha(compatibility_p9.__file__))
            if verifier['sha256'] != P9_VERIFIER_SHA or declared_proof.get('verifier') != verifier or any(
                    files.get(verifier['path']) != verifier['sha256']
                    for files in (binding['files'], capacity['files'])):
                raise ValueError('actual execution must freeze the exact reviewed P9 compatibility verifier')
            if config.get('clock_failure_fresh_confirmation_v1') is not True:
                raise ValueError('A P9 qualified policy must explicitly enable fresh failure confirmation')
            reuse_proof = compatibility_p9.verify(reuse, actual_manifest, capacity)
        elif is_p8_reuse:
            verifier = dict(path=str(Path(compatibility_p8.__file__).resolve()), sha256=sha(compatibility_p8.__file__))
            if verifier['sha256'] != P8_VERIFIER_SHA or declared_proof.get('verifier') != verifier or any(
                    files.get(verifier['path']) != verifier['sha256']
                    for files in (binding['files'], capacity['files'])):
                raise ValueError('actual execution must freeze the exact approved P8 compatibility verifier')
            if declared_proof.get('certificate_scope') != 'original_P6_layout_savings_with_actual_P8_transition_groups' or declared_proof.get('new_transition_qualification_complete') is not True:
                raise ValueError('development bootstrap calibration cannot authorize capacity-enabled measurements')
            reuse_proof = compatibility_p8.verify(reuse, actual_manifest, capacity)
        else:
            reuse_proof = compatibility.verify(reuse, actual_manifest, capacity)
        if config.get('controller_calibration_compatibility', reuse) != reuse:
            raise ValueError('serving and calibrated compatibility references differ')
    result.update(capacity_measured_controller_manifest=semantics['candidate_manifest'],
        capacity_actual_controller_manifest=actual_manifest,
        controller_calibration_compatibility=reuse,
        numerical_calibration_reused_across_controller_versions=reuse is not None,
        numerical_reuse_does_not_prove_autonomous_qualification=True)
    host = read(Path(binding['host_release']) / 'manifest.json')
    actual_module_hashes = dict(semantics['capacity_modules'])
    if is_p8_reuse or is_p9_reuse:
        numerical_proof = (read(reuse_proof['predecessor_P8_compatibility']['path']) if is_p9_reuse else reuse_proof)
        for name, reference in numerical_proof['approved_patch_sources'].items():
            actual_module_hashes[name] = reference['sha256']
    if any(host['files'].get(name) != expected_file
           for name, expected_file in actual_module_hashes.items()):
        raise ValueError('capacity module differs from explicitly qualified actual source')
    result.update(capacity_layout_measured_controller_manifest=semantics['candidate_manifest'],
        capacity_transition_measured_controller_manifest=(reuse_proof['actual_transition_controller_manifest'] if is_p9_reuse else actual_manifest if is_p8_reuse else semantics['candidate_manifest']),
        capacity_transition_source_selection=(numerical_proof.get('transition_source_selection') if is_p8_reuse or is_p9_reuse else reuse_proof.get('transition_source_selection') if reuse_proof else None),
        clock_failure_fresh_confirmation_enabled=config.get('clock_failure_fresh_confirmation_v1') is True,
        capacity_physical_operation_timeout_s=capacity.get('physical_operation_timeout_s'),
        capacity_bounds_are_empirical_not_hard_guarantees=certificate.get('bounds_are_empirical_not_hard_guarantees') is True)

    # Port and owner names identify this operation. The identity, policy,
    # numerical references, planner, domain, capacity bounds and measured
    # certificate remain part of the version even if only one changes.
    operational_capacity = {'owner_id', 'runtime_dir', 'http_port_base', 'kv_port_base', 'files'}
    capacity_policy = normalized_references({k: v for k, v in capacity.items()
                                            if k not in operational_capacity})
    excluded = {'journal', 'port', 'profiles', 'host_source_release', 'controller_source_release',
        'engine_source_release', 'candidate_label', 'profile_compatibility', 'transfer_evidence',
        'frequency_evidence', 'interconnect', 'capacity_inventory_path', 'capacity_job_path',
        'capacity_lease_authority', 'capacity_binding_path', 'capacity_binding_sha256'}
    policy = {k: v for k, v in config.items() if k not in excluded and k != 'instances'}
    policy['instances'] = sorted([dict(tp=i['tp'], gpus=sorted(i['gpus']), role=i['role'])
                                 for i in config['instances']], key=lambda i: i['gpus'])
    policy['capacity_runtime_contract'] = capacity_policy
    result['policy_sha256'] = digest(policy)
    result['version_id'] = digest(dict(source=result['controller_source_sha256'],
        profile=result['profile_sha256'], policy=result['policy_sha256']))
    result.update(capacity_binding=dict(path=str(path), sha256=sha(path)),
        capacity_contract_sha256=digest(capacity_policy), capacity_certificate=certificate_ref,
        capacity_identity=capacity['identity'],
        capacity_calibrated_resident_gpu_counts=sorted({sum(len(g) for g in x['resident_groups'])
                                                      for x in certificate['layouts']}))
    return result


def original_identity(point):
    source = point['executed_source']
    if sha(source['binding_path']) != source['binding_sha256']:
        raise ValueError('original executed binding changed')
    return identity(read(source['binding_path']), point['dataset'])
