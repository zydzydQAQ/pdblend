"""CPU reference fixtures: metadata partitioning, never synthetic performance data."""
import json
from pathlib import Path
import pytest
import source_identity_v2 as identity


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return dict(path=str(path), sha256=identity.sha(path))


def fixture(tmp_path, owner, *, margin=2, certificate_revision=1, wrong_source=False):
    host = tmp_path / 'host'
    module = write(host / 'capacity_runtime.py', {'fixture': 'reference identity only'})
    manifest = write(host / 'manifest.json', {'files': {'capacity_runtime.py': module['sha256']}})
    profile = write(tmp_path / 'profile.json', {'fixture': 'reference identity only'})
    certificate = write(tmp_path / f'certificate-{certificate_revision}.json', dict(identity={'test': 'same'},
        measurement_verified=True, fixture_revision=certificate_revision,
        layouts=[{'resident_groups': [[6], [7]]}, {'resident_groups': [[5], [6], [7]]}]))
    cap = dict(schema='capacity-runtime-binding-v1', identity={'test': 'same'}, calibration=certificate,
        owner_id=owner, runtime_dir=str(tmp_path / owner), http_port_base=1000+len(owner),
        kv_port_base=2000+len(owner), files={profile['path']: profile['sha256']},
        policy={'margin': margin}, calibrated_source_semantics=dict(
            candidate_manifest=dict(manifest, sha256='0'*64) if wrong_source else manifest,
            profile=profile, capacity_modules={'capacity_runtime.py': module['sha256']}))
    capacity = write(tmp_path / owner / 'capacity.json', cap)
    config = write(tmp_path / owner / 'config.json', dict(strategy='pdblend-joint', profiles=profile['path'],
        instances=[dict(tp=1, gpus=[6], role='mixed'), dict(tp=1, gpus=[7], role='mixed')],
        capacity_integration_v1=True, capacity_binding_path=capacity['path'],
        capacity_binding_sha256=capacity['sha256'], capacity_inventory_path=str(tmp_path / owner / 'inventory.json'),
        capacity_job_path=str(tmp_path / owner / 'job.json'), capacity_lease_authority={'pid': len(owner)}))
    return dict(host_release=str(host), configs={'alpaca': config['path']},
        files={ref['path']: ref['sha256'] for ref in (manifest, profile, capacity, config)})


def test_operation_owner_ports_and_paths_do_not_split_one_policy(tmp_path):
    a = identity.identity(fixture(tmp_path, 'first'), 'alpaca')
    b = identity.identity(fixture(tmp_path, 'longer_second'), 'alpaca')
    assert a['capacity_binding']['sha256'] != b['capacity_binding']['sha256']
    assert a['version_id'] == b['version_id']


def test_policy_or_certificate_changes_are_distinct_versions(tmp_path):
    a = identity.identity(fixture(tmp_path, 'first'), 'alpaca')
    b = identity.identity(fixture(tmp_path, 'second', margin=3), 'alpaca')
    c = identity.identity(fixture(tmp_path, 'third', certificate_revision=2), 'alpaca')
    assert len({a['version_id'], b['version_id'], c['version_id']}) == 3


def test_uncovered_actual_serving_source_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='source/profile differs'):
        identity.identity(fixture(tmp_path, 'wrong', wrong_source=True), 'alpaca')
