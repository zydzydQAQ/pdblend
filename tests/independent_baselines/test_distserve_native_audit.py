import json

import pytest

from pdblend_baselines.distserve.native_audit import ExactNativeLatency, main, read_profile, run
from pdblend_baselines.native_profile import DIST_FREQS, _select_samples, _sha, measured_points


def artifacts(tmp_path, *, model='Qwen2.5-7B-Instruct', tp=1):
    meta = dict(model_hash='a'*64, tokenizer_hash='b'*64, engine_version='vllm-0.10.1.1',
                image_digest='sha256:'+'c'*64, source_revision='d'*64,
                gpu_uuids=['GPU-test-'+str(rank) for rank in range(tp)], tp=tp, pp=1)
    rows = []
    for frequency in DIST_FREQS:
        for role in ('prefill', 'decode'):
            sample = dict(ranks=[])
            for rank in range(tp):
                values = [dict(role=role, system='distserve', rank=rank, tp_rank=rank, pp_rank=0,
                    tp=tp, pp=1, batch=1, max_input_tokens=128, input_tokens=128 if role=='prefill' else 1,
                    context_tokens=context, max_context_tokens=context, request_ids=['native-fixture'],
                    measurement_scope='runner', gpu_elapsed_ms=1.+rank*.1)
                    for context in ([128] if role=='prefill' else range(129, 160))]
                sample['ranks'].append(dict(rank=rank, tp=tp, pp=1, samples=values))
            selected = _select_samples(sample['ranks'], role=role, context=128, batch=1, scope='runner', tp=tp)
            rows.append(dict(frequency_mhz=frequency, role=role, context_tokens=128, batch_size=1,
                sample=sample, sample_sha256=_sha(sample), points=measured_points(selected, tp=tp, raw_sha256=_sha(sample))))
    artifact = dict(schema='pdblend-baseline-profile-v1', system='distserve', model=model,
                    metadata=meta, rows=rows, complete=True, formal_eligible=False)
    profile = tmp_path/'distserve.json'
    profile.write_text(json.dumps(artifact))
    cap = dict(**meta, model_id=model, supported=True, native_evidence_complete=True,
               state=dict(tp=tp, pp=1, total_kv_tokens=16384))
    evidence = tmp_path/'completion.json'
    evidence.write_text(json.dumps(dict(capabilities=[cap])))
    return profile, evidence


@pytest.mark.parametrize('model,tp', [('Qwen2.5-7B-Instruct', 1), ('Qwen2.5-14B-Instruct', 1),
                                     ('Qwen2.5-32B-Instruct', 2)])
def test_real_official_simpy_and_partial_search_keep_full_space_missing_or_unsupported(tmp_path, model, tp):
    pytest.importorskip('simpy')
    path, evidence = artifacts(tmp_path, model=model, tp=tp)
    report = run([path], [evidence], model=model, requests=3, maximum_rate=.1, epsilon=.025)
    assert report['status'] == 'diagnostic_complete'
    assert not report['formal_eligible'] and not report['complete_offline_choice']
    assert not report['complete_search_space'] and report['seed'] == 701
    assert report['history']['kind'] == 'synthetic_fixed_shape_functional_probe'
    assert len(report['frequencies']) == 6
    for frequency in report['frequencies']:
        assert frequency['status_counts']['missing_profile'] > 0
        assert frequency['status_counts']['unsupported_engine'] > 0
        assert all(row['gpu_count'] <= 8 for row in frequency['matrix'])
        assert frequency['local_diagnostic_candidate']['config'] == (1, tp, 1, tp, 1)
        actual = frequency['searches'][0]['native_simpy_observations'][0]
        assert len(actual['request_events']) == 3 and actual['worker_events']
        assert actual['upstream_revision'] == '82831f1604cc6b10bebd360f6c437a07790dde9f'
        assert all(events['events'][-1][1] == 'exit_system' for events in actual['request_events'])
    if tp == 1 and model == 'Qwen2.5-7B-Instruct':
        assert len(report['frequencies'][0]['matrix']) == 38


def test_native_capacity_must_be_bound_to_exact_profile_group(tmp_path):
    path, evidence = artifacts(tmp_path)
    value = json.loads(evidence.read_text())
    value['capabilities'][0]['gpu_uuids'] = ['GPU-foreign']
    evidence.write_text(json.dumps(value))
    report = run([path], [evidence], model='Qwen2.5-7B-Instruct', requests=1)
    assert report['status'] == 'inconclusive' and not report['native_capacity_receipts']
    assert all(not f['searches'] for f in report['frequencies'])


@pytest.mark.parametrize('mutation', ['foreign_system', 'foreign_model', 'native_owner', 'tp', 'pp', 'sha', 'point', 'frequency'])
def test_strict_own_native_identity_and_hash_validation(tmp_path, mutation):
    path, _ = artifacts(tmp_path)
    value = json.loads(path.read_text())
    if mutation == 'foreign_system': value['system'] = 'dynamollm'
    elif mutation == 'foreign_model': value['model'] = 'Qwen2.5-14B-Instruct'
    elif mutation == 'native_owner': value['rows'][0]['sample']['ranks'][0]['samples'][0]['system'] = 'pdblend'
    elif mutation == 'tp': value['rows'][0]['sample']['ranks'][0]['tp'] = 2
    elif mutation == 'pp': value['metadata']['pp'] = 2
    elif mutation == 'sha': value['rows'][0]['sample_sha256'] = 'e'*64
    elif mutation == 'point': value['rows'][0]['points'][0]['stage_latency_ms'] += 1
    elif mutation == 'frequency': value['rows'][0]['frequency_mhz'] = 901
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        read_profile(path, 'Qwen2.5-7B-Instruct')


def test_exact_provider_translates_decode_token_count_and_never_extrapolates(tmp_path):
    path, _ = artifacts(tmp_path)
    artifact = read_profile(path, 'Qwen2.5-7B-Instruct')
    points = [p for row in artifact['rows'] if row['frequency_mhz'] == 900 for p in row['points']]
    latency = ExactNativeLatency(points)
    assert latency.stage_latency('decode', 1, 1, 0, 1, (), [128]) == 1.
    for call in [('decode', 1, 1, 0, 2, (), [128, 128]),
                 ('decode', 1, 1, 0, 1, (), [159]),
                 ('prefill', 1, 1, 0, 1, [127], [127])]:
        with pytest.raises(ValueError, match='missing_profile'):
            latency.stage_latency(*call)
    with pytest.raises(ValueError, match='unsupported_engine'):
        latency.stage_latency('prefill', 1, 2, 0, 1, [128], [128])


def test_real_simpy_unmeasured_intermediate_batch_remains_missing_profile(tmp_path):
    pytest.importorskip('simpy')
    path, evidence = artifacts(tmp_path, model='Qwen2.5-32B-Instruct', tp=2)
    result = run([path], [evidence], model='Qwen2.5-32B-Instruct', requests=3)
    assert result['status'] == 'inconclusive'
    search = result['frequencies'][0]['searches'][0]
    assert search['trials'] and search['trials'][0]['passed']
    assert 'batch=2' in search['error'] and 'missing_profile' in search['error']
    assert result['frequencies'][0]['local_diagnostic_candidate'] is None


def test_cli_writes_reviewable_simpy_report_without_promoting_partial_choice(tmp_path):
    pytest.importorskip('simpy')
    path, evidence = artifacts(tmp_path)
    out = tmp_path/'audit.json'
    assert main(['--profile', str(path), '--native-evidence', str(evidence), '--model',
                 'Qwen2.5-7B-Instruct', '--requests', '3', '--out', str(out)]) == 0
    result = json.loads(out.read_text())
    assert result['audit_valid'] and not result['formal_eligible']
    assert result['native_capacity_receipts'][0]['evidence_path'] == str(evidence)
