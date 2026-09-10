import json
import copy

import pytest

from ecopadg.scalability.artifacts import sha256, object_hash
from ecopadg.scalability.preflight import reference_errors, verify_freeze, quiescent, engine_identity
from ecopadg.scalability import preflight as pf


def test_missing_and_changed_certification_sources_remain_blockers(tmp_path):
    source=tmp_path/'raw.json';source.write_text('{}')
    valid={'certification_artifacts':{str(source):sha256(source)}}
    assert not reference_errors(valid)
    source.write_text('{"changed":true}')
    assert reference_errors(valid)==['changed evidence: '+str(source)]
    source.unlink()
    assert reference_errors(valid)==['missing evidence: '+str(source)]


def test_freeze_checks_bytes_not_just_presence(tmp_path):
    p=tmp_path/'source.py';p.write_text('pass\n')
    freeze=dict(schema='pdblend-scalability-freeze-v1',files={str(p):sha256(p)})
    assert any('qualification_path' in error for error in verify_freeze(freeze,require_local_host=False))
    p.write_text('raise RuntimeError\n')
    assert verify_freeze(freeze,require_local_host=False)


def test_drain_requires_pending_imports_and_engine_health():
    assert quiescent(dict(running=0,waiting=0,active=0))
    assert not quiescent(dict(running=0,transfer_buffered_tensors=1))
    assert not quiescent(dict(running=0,transfer_inflight_receives=1))
    assert not quiescent(dict(runtime_error='failed'))


def test_engine_identity_binds_source_image_and_physical_device():
    original=dict(instance_id='a',tp=1,dtype='bfloat16',model='m',cuda_visible_devices='0',image_id='sha256:a',
                  source_files_at_import={'engine.py':'abc'},pid=7)
    assert engine_identity(original)==engine_identity(dict(original,pid=8))
    assert engine_identity(original)!=engine_identity(dict(original,cuda_visible_devices='1'))


def source_inputs(tmp_path, *, source_field='source_files_at_import'):
    live_source = tmp_path / 'deployment' / 'engine.py'
    live_source.parent.mkdir()
    live_source.write_text('real engine source\n')
    source_hash = sha256(live_source)
    live = dict(instance_id='current', image_id='same-image', tp=1,
                source_files_at_import={str(live_source): source_hash})
    original_path = '/old/root/src/ecopadg/serving/engine.py'
    value = source_hash if source_field == 'engine_source' else {original_path: source_hash}
    measured = dict(instance_id='original', image_id='same-image', **{source_field: value})
    raw = tmp_path / 'measurement.json'
    raw.write_text(json.dumps(dict(complete=True, engine_provenance=[measured])))
    profiles = tmp_path / 'profiles.json'
    transfers = tmp_path / 'transfers.json'
    references = {str(raw): sha256(raw)}
    profiles.write_text(json.dumps(dict(certification_artifacts=references,
        points=[dict(tp=1, source_sha256=sha256(raw))])))
    transfers.write_text(json.dumps(dict(certification_artifacts=references,
        links=[dict(source_tp=1, target_tp=1, source_sha256=sha256(raw))])))
    return dict(profiles=str(profiles), transfer_evidence=str(transfers)), live, raw


@pytest.mark.parametrize('source_field', ['source_files_at_import', 'source_files', 'engine_source'])
def test_original_measurement_source_matches_live_even_after_mount_relocation(tmp_path, source_field):
    config, live, _ = source_inputs(tmp_path, source_field=source_field)
    assert pf.measured_source_errors(config, [live]) == []


def test_same_docker_image_cannot_hide_different_mounted_engine_source(tmp_path):
    config, live, _ = source_inputs(tmp_path)
    live['source_files_at_import'] = {next(iter(live['source_files_at_import'])): 'a' * 64}
    errors = pf.measured_source_errors(config, [live])
    assert sum('engine-source mismatch' in e for e in errors) == 2
    assert all('serving/engine.py' in e for e in errors)


def test_live_additional_source_is_not_covered_by_a_partial_historical_map(tmp_path):
    config, live, _ = source_inputs(tmp_path)
    live['source_files_at_import']['/new/src/ecopadg/serving/custom_executor.py'] = 'e' * 64
    errors = pf.measured_source_errors(config, [live])
    assert errors and any('custom_executor.py' in e for e in errors)


def test_restored_raw_without_original_source_proof_still_fails(tmp_path):
    config, live, raw = source_inputs(tmp_path)
    raw.write_text(json.dumps(dict(complete=True, engine_provenance=[dict(image_id='same-image')])))
    for key in ('profiles', 'transfer_evidence'):
        document = json.loads(open(config[key]).read())
        document['certification_artifacts'] = {str(raw): sha256(raw)}
        for row in document.get('points', document.get('links', [])):
            row['source_sha256'] = sha256(raw)
        with open(config[key], 'w') as handle:
            json.dump(document, handle)
    assert any('missing actual engine.py' in e for e in pf.measured_source_errors(config, [live]))


def test_a_fresh_mechanism_decoy_does_not_certify_unlinked_old_measurements(tmp_path):
    config, live, raw = source_inputs(tmp_path)
    document = json.loads(open(config['profiles']).read())
    document['points'][0]['source_sha256'] = 'f' * 64
    with open(config['profiles'], 'w') as handle:
        json.dump(document, handle)
    errors = pf.measured_source_errors(config, [live])
    assert any('producer hashes absent from artifact index' in e for e in errors)
    assert any('lack matching original raw source evidence' in e for e in errors)


def test_topology_compares_physical_matrix_and_binds_transfer_metadata(tmp_path):
    text = 'GPU0 X PIX\nGPU1 PIX X\n'
    path = tmp_path / 'interconnect.txt'
    path.write_text(text)
    config = dict(interconnect=str(path), transfers=[dict(source_gpus=[0], target_gpus=[1],
        interconnect_class='PIX', topology_sha256=sha256(path))])
    assert pf.topology_errors(config, '\x1b[4mGPU0\x1b[0m  X  PIX\nGPU1 PIX X\n') == []
    assert pf.topology_errors(config, text.replace('PIX', 'SYS')) == [
        'live GPU topology differs from configured interconnect matrix']
    config['transfers'][0]['source_gpus'] = [1]
    assert pf.topology_errors(config, text)


def valid_freeze(tmp_path):
    config, live, measurement = source_inputs(tmp_path)
    topo = tmp_path / 'topology.txt'
    topo.write_text('GPU0 X\n')
    config['interconnect'] = str(topo)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    checks = dict.fromkeys(pf.REQUIRED_CHECKS, True)
    identities = {live['instance_id']: live}
    raw = tmp_path / 'qualification-raw.json'
    raw.write_text(json.dumps(dict(scope='diagnostic_raw', measurement='hardware', passed=True,
        smoke=False, cleanup_complete=True, checks=checks, engine_identities=identities)))
    proof = tmp_path / 'qualification.json'
    proof.write_text(json.dumps(dict(scope='hardware_qualification', passed=True, checks=checks,
        cleanup_complete=True, engine_identities=identities, profile_sha256=sha256(config['profiles']),
        artifacts={str(raw): sha256(raw)})))
    pools = {}
    for dataset in ('sharegpt', 'longbench'):
        path = tmp_path / (dataset + '.json')
        path.write_text(json.dumps(dict(dataset=dataset, records=[])))
        pools[dataset] = str(path)
    paths = [config_path, config['profiles'], config['transfer_evidence'], topo,
             measurement, raw, proof, *pools.values(), *live['source_files_at_import']]
    return dict(schema='pdblend-scalability-freeze-v1',
        files={str(path): sha256(path) for path in paths}, config_path=str(config_path),
        source_config_sha256=object_hash(config), qualification_path=str(proof),
        profile_sha256=sha256(config['profiles']), engine_identities=identities,
        pools=pools, pool_paths=list(pools.values()))


def test_complete_freeze_checks_qualification_raw_source_identity_and_all_artifacts(tmp_path):
    freeze = valid_freeze(tmp_path)
    assert pf.verify_freeze(freeze, require_local_host=False) == []
    missing = copy.deepcopy(freeze)
    missing.pop('qualification_path')
    assert any('qualification_path' in e for e in pf.verify_freeze(missing, require_local_host=False))
    missing = copy.deepcopy(freeze)
    missing['files'].pop(missing['pools']['sharegpt'])
    assert any('sharegpt pool' in e for e in pf.verify_freeze(missing, require_local_host=False))


@pytest.mark.parametrize('change', ['failed-proof', 'missing-check', 'changed-identities', 'cpu-raw', 'missing-raw-freeze'])
def test_rehashed_metadata_cannot_bypass_qualification_contents(tmp_path, change):
    freeze = valid_freeze(tmp_path)
    proof_path = freeze['qualification_path']
    proof = json.loads(open(proof_path).read())
    raw_path = next(iter(proof['artifacts']))
    if change == 'failed-proof':
        proof['passed'] = False
    elif change == 'missing-check':
        proof['checks'].pop('profile_coverage')
    elif change == 'changed-identities':
        proof['engine_identities']['current']['image_id'] = 'different-image'
    elif change == 'cpu-raw':
        raw = json.loads(open(raw_path).read())
        raw['measurement'] = 'cpu_simulation'
        with open(raw_path, 'w') as handle:
            json.dump(raw, handle)
        freeze['files'][raw_path] = proof['artifacts'][raw_path] = sha256(raw_path)
    else:
        freeze['files'].pop(raw_path)
    with open(proof_path, 'w') as handle:
        json.dump(proof, handle)
    freeze['files'][proof_path] = sha256(proof_path)
    assert pf.verify_freeze(freeze, require_local_host=False)
