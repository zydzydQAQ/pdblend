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


def mounts_equivalence(historical, actual):
    """Only ordering of two unique, unchanged C bind-mount dictionaries differs."""
    expected={'/models','/root/workspace'}
    for rows in (historical,actual):
        if not isinstance(rows,list) or len(rows)!=2 or any(not isinstance(x,dict) for x in rows):
            raise RuntimeError('two explicit mount dictionaries required')
        destinations=[x.get('Destination') for x in rows]
        if len(set(destinations))!=2 or set(destinations)!=expected:
            raise RuntimeError('unique original nonoverlapping mount destinations required')
    left=sorted(historical,key=lambda x:x['Destination'])
    right=sorted(actual,key=lambda x:x['Destination'])
    if left!=right:
        raise RuntimeError('mount contents differ beyond array ordering')
    return dict(schema='C-Docker-unique-Mounts-order-equivalence-v1',equivalent=True,
                historical=historical,actual=actual,only_array_order_changed=historical!=actual,
                all_mount_fields_exact=True,unique_nonoverlapping_destinations=sorted(expected))
