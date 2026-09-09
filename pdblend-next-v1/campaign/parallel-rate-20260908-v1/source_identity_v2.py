"""Preserve static fingerprints and normalize explicitly frozen capacity contracts."""
from pathlib import Path
import source_identity as static

sha, read, digest = static.sha, static.read, static.digest


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
    if (semantics['candidate_manifest']['sha256'] != result['host_manifest_sha256']
            or semantics['profile']['sha256'] != result['profile_sha256']):
        raise ValueError('capacity calibration source/profile differs from serving source')
    host = read(Path(binding['host_release']) / 'manifest.json')
    if any(host['files'].get(name) != expected_file
           for name, expected_file in semantics['capacity_modules'].items()):
        raise ValueError('capacity module differs from real calibrated module')
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
