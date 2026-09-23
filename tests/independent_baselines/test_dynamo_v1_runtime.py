"""CPU contracts only: these tests never instantiate CUDA, NVML, or vLLM."""
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdblend_baselines.dynamollm import predictor
from pdblend_baselines.dynamollm.deployment import SubprocessLifecycle, gpu_devices
from pdblend_baselines.dynamollm.gpu_weights import DynamoWorkerExtension, transfer_pieces
from pdblend_baselines.dynamollm.native_hooks import aggregate_drain
from pdblend_baselines.dynamollm.run_v1 import load_trace, qualify
from pdblend_baselines.dynamollm.transport import validate_rank_ack
from pdblend_baselines.dynamollm.validation import preflight


def test_schema3_predictor_identity_bridge_verifies_files_not_aggregate(tmp_path):
    files = {}
    for name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json'):
        (tmp_path/name).write_text(name)
        files[name] = predictor.digest(tmp_path/name)
    config_sha = 'c'*64
    identity = dict(model='Qwen2.5-7B-Instruct', tokenizer_sha256='old-format',
                    tokenizer_files={'config.json': config_sha})
    manifest = dict(schema=3, model_name=identity['model'], tokenizer_files_sha256=files,
                    tokenizer_sha256=hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                    model_config_sha256=config_sha)
    assert predictor.verify_corpus_identity(manifest, identity, tmp_path).startswith('corpus-schema3')
    with pytest.raises(ValueError, match='model differs'):
        predictor.verify_corpus_identity(dict(manifest, model_name='Qwen2.5-14B-Instruct'), identity, tmp_path)
    (tmp_path/'tokenizer.json').write_text('changed')
    with pytest.raises(ValueError, match='file differs'):
        predictor.verify_corpus_identity(manifest, identity, tmp_path)


@pytest.mark.parametrize('hidden,heads,kv,intermediate,source_tp,target_tp', [
    (3584, 28, 4, 18944, 1, 2), (3584, 28, 4, 18944, 2, 4),
    (5120, 40, 8, 13824, 1, 4), (5120, 40, 8, 27648, 2, 4),
    (5120, 40, 8, 27648, 4, 2)])
def test_native_qwen_qkv_shards_cover_each_target_once(hidden, heads, kv, intermediate, source_tp, target_tp):
    geometry = dict(hidden_size=hidden, num_attention_heads=heads,
                    num_key_value_heads=kv, intermediate_size=intermediate)
    width = hidden + 2*kv*(hidden//heads)
    for target_rank in range(target_tp):
        covered = []
        for source_rank in range(source_tp):
            pieces = transfer_pieces('model.layers.0.self_attn.qkv_proj.weight',
                (width//source_tp, hidden), (width//target_tp, hidden),
                source_tp, source_rank, target_tp, target_rank, geometry)
            for piece in pieces:
                assert piece['axis'] == 0
                covered.extend(range(piece['target_offset'], piece['target_offset']+piece['length']))
        assert sorted(covered) == list(range(width//target_tp))


def test_worker_generation_is_checked_before_torch_or_cuda(monkeypatch):
    worker = DynamoWorkerExtension()
    worker.parallel_config = SimpleNamespace(pipeline_parallel_size=1, data_parallel_size=1)
    monkeypatch.setenv('DYNAMO_GENERATION', '7')
    with pytest.raises(ValueError, match='generation mismatch'):
        worker.dynamo_operation('describe', {'expected_generation': 6})
    with pytest.raises(ValueError, match='transaction_id'):
        worker.dynamo_operation('open', {'expected_generation': 7})
    monkeypatch.setattr('pdblend_baselines.dynamollm.gpu_weights.worker_operation',
                        lambda *args, **kwargs: dict(ok=True, rank=0))
    row = worker.dynamo_operation('describe', {'expected_generation': 7})
    assert row['generation'] == 7 and row['engine_revision'] == 'vllm-0.10.1.1'
    worker._native_generation = 8  # Public NativeWorker generation RPC advanced it.
    with pytest.raises(ValueError, match='generation mismatch'):
        worker.dynamo_operation('describe', {'expected_generation': 7})
    assert worker.dynamo_operation('describe', {'expected_generation': 8})['generation'] == 8


def test_all_rank_ack_rejects_missing_duplicate_stale_and_wrong_transaction():
    spec = dict(tp=2, generation=3)
    rows = [dict(rank=i, generation=3, ok=True, transaction_id='tx') for i in range(2)]
    assert validate_rank_ack({'ranks': rows}, spec, transaction_id='tx')
    for bad in [rows[:1], [rows[0], rows[0]], [rows[0], dict(rows[1], generation=2)]]:
        with pytest.raises(RuntimeError, match='rank/generation'):
            validate_rank_ack({'ranks': bad}, spec)
    with pytest.raises(RuntimeError, match='transaction'):
        validate_rank_ack({'ranks': rows}, spec, transaction_id='another')


def test_drain_respects_v1_null_block_and_all_rank_barrier(monkeypatch):
    monkeypatch.setattr('pdblend_baselines.dynamollm.native_hooks.time.time', lambda: 10.)
    state = dict(timestamp=10., generation=1, acknowledged_generation=1, evidence_complete=True,
                 transport_healthy=True, accepting=False, active=0, running=0, waiting=0,
                 kv_allocations={}, transfer_allocations={}, total_kv_tokens=144, free_kv_tokens=144,
                 num_gpu_blocks=10, free_blocks=9, reserved_blocks=1)
    ranks = [dict(rank=0, generation=1, ok=True, drained=True, cuda_synchronized=True,
                  active_weight_sessions=0)]
    assert aggregate_drain(state, ranks, tp=1, generation=1)['owner_ack']
    with pytest.raises(RuntimeError, match='free block'):
        aggregate_drain(dict(state, free_blocks=10), ranks, tp=1, generation=1)
    with pytest.raises(RuntimeError, match='drain proof'):
        aggregate_drain(dict(state, kv_allocations={'live': [1]}), ranks, tp=1, generation=1)
    with pytest.raises(RuntimeError, match='all-rank'):
        aggregate_drain(state, [dict(ranks[0], active_weight_sessions=1)], tp=1, generation=1)


def test_lease_mapping_and_lifecycle_command_are_model_bound(monkeypatch, tmp_path):
    monkeypatch.setenv('PDBLEND_GPU_UUIDS', 'GPU-a,GPU-b,GPU-c,GPU-d')
    assert gpu_devices([2, 0]) == ['GPU-c', 'GPU-a']
    with pytest.raises(ValueError, match='outside'):
        gpu_devices([4])
    lifecycle = SubprocessLifecycle(dict(model_path='/models/Qwen2.5-32B-Instruct',
        model_id='Qwen2.5-32B-Instruct', legal_tp=[2, 4], node_gpus=[0, 1, 2, 3]),
        None, lambda *a, **k: None, tmp_path)
    command = lifecycle.command(dict(tp=2, gpus=[0, 1], port=18000), dummy=True)
    assert 'pdblend_runtime.serve' in command
    assert command[command.index('--pipeline-parallel-size')+1] == '1'
    assert command[command.index('--load-format')+1] == 'dummy'
    assert '--enforce-eager' in command
    assert 'pdblend_baselines.dynamollm.gpu_weights.DynamoWorkerExtension' in command
    with pytest.raises(ValueError, match='legal TP'):
        lifecycle.command(dict(tp=1, gpus=[0], port=18000))


def test_preflight_never_borrows_7b_predictor_or_pdblend_profile(tmp_path, monkeypatch):
    model = 'Qwen2.5-14B-Instruct'
    monkeypatch.setattr(predictor, 'model_identity', lambda _: dict(model=model, tokenizer_sha256='a'*64))
    def check(_directory, **kwargs):
        assert kwargs['expected_model'] == model
        raise ValueError('checkpoint targets 7B')
    monkeypatch.setattr(predictor, 'verify_checkpoint', check)
    profile = tmp_path/'profile.json'
    profile.write_text(json.dumps(dict(system='pdblend', model=model, engine_revision='vllm-0.10.1.1')))
    result = preflight(dict(model_id=model, model_path='/models/'+model, node_gpus=[0, 1],
        instances=[dict(id='a', gpus=[0], tp=1), dict(id='b', gpus=[1], tp=1)],
        profiles=str(profile), dynamo_predictor_dir='/models/7b-predictor'))
    assert set(result['missing_evidence']) >= {'missing_predictor', 'missing_profile'}
    assert not result['ready'] and not result['hardware_actions_started']


def test_full_preflight_does_not_shorten_original_control_periods():
    result = preflight({}, mode='full', duration_s=100)
    assert result['periods_s'] == {'ScaleInst': 1800., 'ScaleShard': 300., 'ScaleFreq': 5.}
    assert 'duration' in result['missing_evidence']
    assert 'missing_history' in result['missing_evidence']


def test_trace_uses_shared_schedule_without_retokenizing_or_seed_changes(tmp_path):
    path = tmp_path/'trace.json'
    request = dict(idx=0, arrival_s=.5, prompt=[10, 20], max_tokens=16, source='smoke-seed701')
    path.write_text(json.dumps(dict(seed=701, requests=[request])))
    row = load_trace(path, 100)[0]
    assert row['arrival_s'] == .5 and row['prompt'] == request['prompt']
    path.write_text(json.dumps(dict(seed=1701, requests=[request])))
    with pytest.raises(ValueError, match='701'):
        load_trace(path, 100)


def test_functional_and_primitive_never_become_full_or_formal():
    outcomes = [dict(request_id='r', ok=True)]
    rows = [dict(event='dynamo_route', request_id='r')]
    result = qualify(rows, outcomes, mode='functional', duration_s=100)
    assert result['status'] == 'passed' and not result['formal_eligible'] and not result['energy_comparable']
    assert qualify(rows, outcomes, mode='primitive', duration_s=100)['status'] == 'inconclusive'
    full = qualify(rows, outcomes, mode='full', duration_s=1890)
    assert full['status'] == 'inconclusive'
    assert 'real_ScaleInst_action_missing' in full['failures']


@pytest.mark.asyncio
async def test_real_controller_dispatches_independent_policy_and_releases_reservation(tmp_path):
    from pdblend_baselines.dynamollm.runtime import DynamoController
    profile = tmp_path/'own-profile.json'
    profile.write_text(json.dumps(dict(schema=2, measurement='hardware',
        coordinate_system='input_output_batch', points=[dict(role='mixed', tp=1,
            frequency_mhz=900, input_tokens=128, context_tokens=144, batch=1,
            prefill_s=.001, iteration_s=.001, power_w=100, samples=3, source_sha256='a'*64)])))
    rows = []
    class Transport:
        async def state(self, iid):
            return dict(role='mixed', accepting=True, free_kv_tokens=10000, kv_allocations={}, generation=0)
        async def clock(self, gpus, frequency):
            assert frequency == 900
        async def stream(self, iid, payload):
            assert iid == 'own-a' and payload['request_id'] == 'r'
            yield dict(token_ids=list(range(16)), choices=[dict(finish_reason='length')])
        async def cancel(self, *args):
            raise AssertionError('completed request should not cancel')
    config = dict(profiles=str(profile), instances=[dict(id='own-a', gpus=[0], tp=1, shape='LL')],
        development_allow_predictor_injection=True,
        development_predictor=SimpleNamespace(predict_text=lambda text: 16),
        development_tokenizer=SimpleNamespace(decode=lambda *a, **kw: 'visible prompt'),
        slo_ttft_s=10, slo_tpot_s=1)
    controller = DynamoController(config, Transport(), lambda event, **kw: rows.append(dict(event=event, **kw)))
    try:
        await controller.startup()
        output = [row async for row in controller.handle(dict(prompt=[1000]*128, max_tokens=16), 'r')]
        assert output[0]['token_ids'] == list(range(16))
        assert any(row['event'] == 'dynamo_route' for row in rows)
        assert not controller.kv_reservations and not controller.replicas['own-a'].requests
    finally:
        await controller.close()
