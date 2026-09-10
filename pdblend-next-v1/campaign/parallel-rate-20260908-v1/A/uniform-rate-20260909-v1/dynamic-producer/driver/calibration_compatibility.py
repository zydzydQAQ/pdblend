"""Fresh-node identity closure for unchanged physical calibration and control."""
from pathlib import Path
from capacity_executor import fixed, require, sha


def validate_compatibility(spec, capacity):
    proof_ref = spec['controller_calibration_compatibility']
    require(capacity['controller_calibration_compatibility'] == proof_ref, 'fresh identity proof differs')
    proof = fixed(proof_ref)
    require(proof['schema'] == 'new-A-fresh-capacity-source-identity-v1'
            and proof['node'] == 'Anew20260909' and proof['old_capacity_evidence_inherited'] is False,
            'fresh new-A calibration identity required')
    host = Path(spec['host_release']) / 'manifest.json'
    require(proof['host_manifest'] == dict(path=str(host), sha256=sha(host)), 'fresh source differs')
    require(proof['profile'] == spec['profiles'], 'fresh frequency/profile domain differs')
    require(proof['files'] and all(sha(p) == h for p, h in proof['files'].items()), 'fresh source closure changed')
    require(spec['files'].get(proof_ref['path']) == proof_ref['sha256']
            and all(spec['files'].get(p) == h for p, h in proof['files'].items()), 'unbound identity proof')
    bootstrap = fixed(proof['bootstrap'])
    require(bootstrap['complete'] and bootstrap['ordinary_passed'] and not bootstrap.get('error')
            and not bootstrap['node_lease_held'] and bootstrap['hostname'] == 'iZwz9274emxme9019d2sjgZ',
            'new node native bootstrap not complete')
    original = fixed(spec['original_binding'])
    require(original['instances'] == bootstrap['instances'] and original['hostname'] == bootstrap['hostname'],
            'original two physical instances changed')
    expected = dict(node_sha256=proof['node_identity']['sha256'], model_sha256=proof['model_manifest']['sha256'],
                    engine_image=proof['engine_image'], tp=1, source_sha256=proof['semantics']['sha256'])
    require(capacity['identity'] == expected, 'capacity source/model/node identity differs')
    semantic = fixed(proof['semantics'])
    require(semantic['profile'] == proof['profile'] and semantic['host_manifest'] == proof['host_manifest']
            and semantic['max_service_frequency_mhz'] == 2100, 'unsupported service frequency domain')
    config = fixed(spec['config'])
    require(config['profiles'] == proof['profile']['path'] and config['max_service_frequency_mhz'] == 2100
            and config.get('idle_domain_reacquire_v1') is True and semantic['idle_domain_reacquire_v1'] is True,
            'runtime exceeds new-node qualified frequency domain')
    require(capacity['calibrated_source_semantics'] == semantic, 'numerical source semantics differs')
    oracle = fixed(capacity['correctness_oracles'])
    require(oracle['model_source_identity'] == expected and oracle['source'] == bootstrap['ordinary']
            and oracle['bootstrap'] == proof['bootstrap'] and oracle['measured'] is True,
            'new-node oracle must use actual new-node replies')
    replies = fixed(bootstrap['ordinary'])
    for case in oracle['cases']:
        matched = [r for r in replies if r['prompt_length'] == case['prompt_length']]
        require(len(matched) == 2 and {r['instance_id'] for r in matched} == {i['id'] for i in bootstrap['instances']}
                and all(r['response']['token_ids'] == case['token_ids']
                        and r['response']['usage']['completion_tokens'] == case['max_tokens'] == 64
                        and r['response']['usage']['prompt_tokens'] == case['prompt_length'] for r in matched),
                'new-node numerical replies differ')
    require(spec['mode'] in ('layout_calibration', 'automatic_underload_gate', 'qualification900'),
            'unknown fresh calibration stage')
    if spec['mode'] == 'layout_calibration':
        require(capacity['calibration_only'] is True and not capacity.get('calibration'),
                'fresh layout calibration cannot inherit an old certificate')
    else:
        from capacity_certificate import validate
        certificate = fixed(capacity['calibration'])
        require(spec['actual_certificate'] == capacity['calibration'], 'stage certificate differs')
        validate(certificate, expected)
    return proof
