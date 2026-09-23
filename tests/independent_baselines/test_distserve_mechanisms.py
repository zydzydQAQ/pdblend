"""Decision contracts derived from the pinned DistServe artifact, not GPU tests."""
import hashlib
import json
from pathlib import Path

import pytest

from pdblend_baselines.distserve.policy import Request, PrefillScheduler, DecodeScheduler
from pdblend_baselines.distserve.planning import (
    enumerate_configs, capability_matrix, binary_goodput, best_config,
)


def req(name, n=16, output=16):
    return Request(name, n, output, {})


def test_prefill_fcfs_cannot_skip_a_head_that_exceeds_token_budget():
    scheduler = PrefillScheduler(max_batch_size=4, max_tokens_per_batch=32, num_gpu_blocks=64)
    scheduler.add(req('long', 48)); scheduler.add(req('short', 16))
    assert scheduler.next_batch() == ()
    assert [r.request_id for r in scheduler.waiting] == ['long', 'short']


def test_prefill_batches_by_requests_tokens_and_retained_gpu_blocks():
    scheduler = PrefillScheduler(max_batch_size=2, max_tokens_per_batch=48, num_gpu_blocks=4)
    for name in ('a', 'b', 'c'): scheduler.add(req(name, 32))
    batch = scheduler.next_batch()
    assert [r.request_id for r in batch] == ['a']
    scheduler.complete(batch)
    assert scheduler.retained_blocks == 2
    batch = scheduler.next_batch(); scheduler.complete(batch)
    assert [r.request_id for r in batch] == ['b']
    assert scheduler.next_batch() == ()
    scheduler.release('a')
    assert [r.request_id for r in scheduler.next_batch()] == ['c']


def test_decode_bridge_leaves_request_on_source_until_accepted():
    scheduler = DecodeScheduler(max_batch_size=2, max_tokens_per_batch=64, num_gpu_blocks=8,
                                waiting_block_prop_threshold=.25)
    a, b = req('a', 32), req('b', 16)
    scheduler.add_bridge(a); scheduler.add_bridge(b)
    assert scheduler.accept_next(free_gpu_blocks=8) is a
    # Upstream uses a strict '<' waiting-block threshold before accepting the next request.
    assert scheduler.accept_next(free_gpu_blocks=6) is None
    assert [r.request_id for r in scheduler.bridge] == ['b']
    assert scheduler.next_batch() == (a,)
    assert scheduler.accept_next(free_gpu_blocks=6) is b


def test_decode_fcfs_and_actual_context_budget():
    scheduler = DecodeScheduler(max_batch_size=2, max_tokens_per_batch=48, num_gpu_blocks=32)
    a, b = req('a', 32), req('b', 32)
    scheduler.waiting.extend((a,b))
    assert scheduler.next_batch() == (a,)
    assert list(scheduler.waiting) == [b]
    scheduler.finish('a')
    assert scheduler.next_batch() == (b,)


def test_eight_card_configs_include_pp7_and_explicit_missing_profiles():
    configs = enumerate_configs(layers=28, attention_heads=28, num_nodes=1,
                                gpus_per_node=8, allowed_tps=(1,2,4))
    assert (1,1,7,1,1) in configs
    assert all(c[0]*(c[1]*c[2]+c[3]*c[4]) <= 8 for c in configs)
    rows = capability_matrix(configs, supported_pairs={(1,1),(2,1),(4,1)},
                             measured_pairs={(1,1)})
    assert next(r for r in rows if r['config']==(1,2,1,1,1))['status']=='missing_profile'
    assert next(r for r in rows if r['config']==(1,1,7,1,1))['status']=='unsupported_engine'


def test_binary_search_preserves_separate_quantiles_not_joint_attainment():
    seen=[]
    def simulate(config, rate):
        seen.append(rate)
        # 90% marginal success each, but 80% joint success.
        return {'ttft_s':[2]+[.1]*9, 'tpot_s':[.1,2]+[.1]*8}
    report = binary_goodput((1,1,1,1,1), simulate, ttft_s=1.9, tpot_s=1.9,
                            ttft_percentage=90, tpot_percentage=90, max_per_gpu_rate=2, epsilon=.1)
    assert report['best_per_gpu_rate'] > 1.8
    assert report['trials'][0]['joint_attainment'] == .8
    assert report['predicate']=='separate_ttft_tpot_quantiles_strict_less'
    assert seen[0] == 2.0  # rate = per_gpu_rate * P+D GPU count


def test_binary_search_records_failure_without_labeling_it_low_goodput():
    def fail(config, rate): raise ValueError('missing profile')
    result=binary_goodput((1,1,1,1,1),fail,ttft_s=1,tpot_s=.1)
    assert result['status']=='simulation_failed'
    assert result['best_per_gpu_rate'] is None


def test_best_config_maximizes_per_gpu_goodput_then_fewer_cards():
    a,b=(1,1,1,1,1),(1,2,1,2,1)
    assert best_config({a:1.,b:1.}) == (a,1.)


def test_pinned_upstream_files_are_unmodified_and_licensed():
    root=Path(__file__).resolve().parent/'fixtures'/'baselines/distserve/references'
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['revision']=='82831f1604cc6b10bebd360f6c437a07790dde9f'
    for item in manifest['files']:
        assert hashlib.sha256((root/'upstream'/item['path']).read_bytes()).hexdigest()==item['sha256']
    assert 'Apache License' in (root/'upstream/LICENSE').read_text()
