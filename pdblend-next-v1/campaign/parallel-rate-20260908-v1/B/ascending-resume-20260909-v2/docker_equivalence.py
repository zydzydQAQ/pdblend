"""One approved Docker representation equivalence: explicit Dns null or [].

No original inspection is modified. Missing Dns, custom DNS, any changed other
key, and absent/added HostConfig keys are rejected in both independent checks.
"""
import copy
import hashlib
import json


def hostconfig_equivalence(historical, actual):
    if not isinstance(historical, dict) or not isinstance(actual, dict):
        raise RuntimeError('HostConfig must be two objects')
    if 'Dns' not in historical or 'Dns' not in actual:
        raise RuntimeError('explicit Dns fields required')
    empty_dns = lambda value: value is None or (type(value) is list and not value)
    if not empty_dns(historical['Dns']) or not empty_dns(actual['Dns']):
        raise RuntimeError('nonempty or invalid DNS configuration is not authorized')
    left, right = copy.deepcopy(historical), copy.deepcopy(actual)
    left['Dns'] = right['Dns'] = []
    if left != right:
        raise RuntimeError('retained Docker HostConfig differs outside Dns null/empty representation')
    canonical_sha = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return dict(schema='Docker-HostConfig-explicit-empty-Dns-equivalence-v1',
                historical_Dns=historical['Dns'], actual_Dns=actual['Dns'],
                differing_fields=[] if historical == actual else ['Dns'],
                historical_HostConfig_sha256=canonical_sha(historical),
                actual_HostConfig_sha256=canonical_sha(actual),
                all_other_fields_exact=True, equivalent=True)
