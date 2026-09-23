"""Read-only inheritance of complete holdout points across source revisions.

Old measurements retain their directory, UUIDs, source revision and checksums.
The new raw archive never incorporates old rows under its own provenance.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from pdblend.profile.calibration.core import _checkpoint_points, digest
from pdblend.profile.identity import sha256_value

POINT_FIELDS = dict(prefill=('freq_mhz', 'input_tokens'), decode=('freq_mhz', 'batch', 'context_tokens'))


def load_prior_holdout(root, expected_raw_sha256, manifest, current_raw):
    root = Path(root).resolve()
    path, frozen_path = root/'raw.json', root/'frozen-fit.json'
    payload = path.read_bytes()
    if (not isinstance(expected_raw_sha256, str) or len(expected_raw_sha256) != 64
            or hashlib.sha256(payload).hexdigest() != expected_raw_sha256):
        raise ValueError('prior holdout raw checksum missing or changed')
    raw = json.loads(payload)
    if json.loads(frozen_path.read_text()) != manifest:
        raise ValueError('prior holdout frozen candidate/plan manifest differs')
    if raw.get('holdout_candidate_sha256') != manifest['candidate_sha256']:
        raise ValueError('prior holdout is not bound to this frozen candidate')
    if raw.get('identity_sha256') != sha256_value({k:v for k,v in raw.items() if k != 'identity_sha256'}):
        raise ValueError('prior holdout internal raw identity checksum differs')
    for key in ('system', 'model_id', 'model_hash', 'tokenizer_hash', 'tp', 'pp'):
        if raw.get(key) != manifest.get(key) or raw.get(key) != current_raw.get(key):
            raise ValueError('prior holdout model/topology differs: '+key)
    if raw.get('profile_key') != current_raw.get('profile_key'):
        raise ValueError('prior holdout engine/profile key differs')
    before, after = raw.get('environment', {}), current_raw.get('environment', {})
    for key in ('image_digest', 'vllm', 'torch', 'cuda', 'hardware_id', 'gpu_uuids'):
        if not before.get(key) or before[key] != after.get(key):
            raise ValueError('prior holdout execution environment differs: '+key)
    for env in (before, after):
        if not isinstance(env.get('source_hash'), str) or len(env['source_hash']) != 64:
            raise ValueError('prior/current source revision is missing')
    if len(before['gpu_uuids']) != manifest['tp'] or len(set(before['gpu_uuids'])) != manifest['tp']:
        raise ValueError('prior holdout physical UUID topology is incomplete')
    if raw.get('prior_holdout_binding'):
        raise ValueError('nested holdout inheritance is not supported; declare original sources explicitly')
    points = _checkpoint_points(raw, root)
    for section, fields in POINT_FIELDS.items():
        wanted = {tuple(row[k] for k in fields) for row in manifest['plan'][section]}
        if not points[section] <= wanted:
            raise ValueError('prior holdout has unexpected point: '+section)
        for row in raw.get(section, []):
            if 'evidence_source' in row:
                raise ValueError('prior raw cannot relabel another evidence source')
            if section == 'decode' and len(row.get('repeats', [])) != manifest['plan']['repeats']:
                raise ValueError('prior holdout decode point has incomplete/extra repeats')
    # This narrow restart path inherits only prefill/decode. Mixed is measured
    # afresh, so a partial mixed phase can never suppress a complete new phase.
    source = 'prior-'+expected_raw_sha256[:20]
    binding = dict(raw_sha256=expected_raw_sha256, candidate_sha256=manifest['candidate_sha256'],
                   evidence_source=source, root=str(root), frozen_fit_sha256=digest(frozen_path))
    receipt = dict(schema=1, binding=binding, complete=False, scope='verified_partial_holdout_points',
        formal_eligible=False, energy_comparable=False, original_raw_unchanged=True,
        original_environment=copy.deepcopy(before), new_environment=copy.deepcopy(after),
        original_profile_key=copy.deepcopy(raw['profile_key']),
        original_concurrency_environment=copy.deepcopy(raw.get('concurrency_environment')),
        original_external_interference=copy.deepcopy(raw.get('external_interference')),
        inherited_points={key:[list(point) for point in sorted(value)] for key,value in points.items()},
        old_source_relabelled=False, source_revision_changed=before['source_hash'] != after['source_hash'],
        ignored_prior_mixed_points=len(raw.get('mixed', [])))
    if (root/'completion.json').is_file():
        old = json.loads((root/'completion.json').read_text())
        receipt['prior_completion'] = dict(path=str(root/'completion.json'), sha256=digest(root/'completion.json'),
                                          status=old.get('status'), complete=old.get('complete'))
    if digest(path) != expected_raw_sha256:
        raise ValueError('prior holdout changed while its checkpoint was validated')
    return raw, points, receipt


def merge_holdout_rows(current_raw, prior_raw, receipt, *, expected_plan, require_complete=False):
    """Build a derived audit view; both input archives stay untouched."""
    result = copy.deepcopy(current_raw)
    # This digest identifies the unmerged measurement archive. The derived
    # file is bound separately by completion.combined_holdout_sha256.
    if 'identity_sha256' in result:
        result['new_raw_identity_sha256'] = result.pop('identity_sha256')
    source = receipt['binding']['evidence_source']
    for section, fields in POINT_FIELDS.items():
        current = current_raw.get(section, [])
        if any('evidence_source' in row for row in current):
            raise ValueError('new raw must contain only its own source measurements')
        combined = [dict(copy.deepcopy(row), evidence_source=source) for row in prior_raw.get(section, [])]
        combined += copy.deepcopy(current)
        keys = [tuple(row[k] for k in fields) for row in combined]
        wanted = {tuple(row[k] for k in fields) for row in expected_plan[section]}
        if len(keys) != len(set(keys)):
            raise ValueError('duplicate inherited/new holdout point: '+section)
        if not set(keys) <= wanted or require_complete and set(keys) != wanted:
            raise ValueError('combined holdout matrix is incomplete or unexpected: '+section)
        result[section] = combined
    result['evidence_sources'] = {source:copy.deepcopy(receipt['binding'])}
    result['measurement_archive'] = False
    result['scope'] = 'derived_holdout_audit_view_with_original_sources'
    result['formal_eligible'] = False
    return result, {source:Path(receipt['binding']['root'])}
