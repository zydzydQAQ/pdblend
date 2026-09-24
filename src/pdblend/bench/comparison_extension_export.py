"""Discover authorized boundary manifests without modifying a campaign."""
from __future__ import annotations

import json
from pathlib import Path


def extension_watch_paths(campaign_path, *, extension_manifests=(), session_roots=()):
    campaign = json.loads(Path(campaign_path).read_text())
    paths = {Path(ref['path']) for ref in campaign.get('extension_policy_refs', [])}
    inventory = campaign.get('baseline_comparison_policy', {}).get('frozen_baselines')
    if inventory:
        paths.add(Path(inventory['path']))
    paths.update(Path(ref['path'] if isinstance(ref, dict) else ref) for ref in extension_manifests)
    for root in session_roots:
        paths.update(Path(root).glob('**/extensions/latest.json'))
    return sorted(paths)


def load_extensions(campaign, *, extension_manifests=(), session_roots=(), load_bound):
    from .single_observation_slo_boundary import read_extension_manifest

    allowed = {ref['sha256']: ref for ref in campaign.get('extension_policy_refs', [])}
    candidates = [(reference, True) for reference in extension_manifests]
    for root in session_roots:
        candidates.extend((path, False) for path in sorted(Path(root).glob('**/extensions/latest.json')))
    verified = {}

    def descends(item, ancestor):
        reference = item['manifest'].get('previous_manifest')
        visited = set()
        while reference:
            if reference == ancestor:
                return True
            key = (reference['path'], reference['sha256'])
            if key in visited:
                raise ValueError('cyclic boundary manifest history')
            visited.add(key)
            value = load_bound(reference)
            if (value.get('schema') != item['manifest']['schema'] or value.get('mode') != 'manifest'
                    or value.get('policy') != item['manifest']['policy']):
                raise ValueError('boundary manifest predecessor changed policy')
            reference = value.get('previous_manifest')
        return False

    for reference, explicit in candidates:
        item = read_extension_manifest(reference, load_bound=load_bound)
        policy_ref = item['manifest']['policy']
        if allowed.get(policy_ref['sha256']) != policy_ref:
            if explicit:
                raise ValueError('extension policy is not authorized by the comparison campaign')
            continue  # Historical roots can contain another run's extensions.
        key = policy_ref['sha256']
        old = verified.get(key)
        if old is not None:
            if old['manifest_ref'] == item['manifest_ref']:
                continue
            previous, current = old['manifest']['decisions'], item['manifest']['decisions']
            short, long = sorted((previous, current), key=len)
            if long[:len(short)] != short:
                raise ValueError('conflicting boundary decision histories under one policy')
            # Same decision count may describe a newly completed pending
            # window. Its immutable predecessor chain must identify the update.
            old_ref, new_ref = old['manifest_ref'], item['manifest_ref']
            if descends(item, old_ref):
                verified[key] = item
            elif descends(old, new_ref):
                continue
            else:
                raise ValueError('ambiguous boundary manifests without a shared predecessor chain')
        else:
            verified[key] = item
    return list(verified.values())
