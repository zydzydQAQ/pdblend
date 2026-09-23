import json
from dataclasses import replace

import pytest

from pdblend.bench.client import Request
from pdblend.bench.run import run_point
from pdblend.bench.tp_runtime import prepare_tp_runtime, mechanism_pd_plan
from pdblend.control.planner import SLO, PlannerConfig, PoolPlanner
from pdblend.control.policies import get_policy
from pdblend.control.topology import ResidentPool, Topology
from pdblend.proxy.router import ResidentRouter, Router
from synthetic import fc, synthetic_model


def profiled_model(tp=1, model_id='Qwen2.5-7B-Instruct'):
    model = synthetic_model()
    model.model, model.tp = model_id, tp
    model.profile_key = dict(model_id=model_id, system='pdblend', tp=tp, pp=1,
                             engine_revision='0.10.1.1', hardware_id='test-only')
    model.bounded_coverage = dict(prefill_tokens=[128, 7168])
    for f in model.freqs:
        a, b, c, d = model.decode_time[f]
        model.decode_overrides[f] = dict(kind='quadratic_relative', coefficients=[a, b * 64, c * 65536, d * 4096],
                                          domain=dict(batch=[1, 256], context=[256, 8192], max_batch_context=225000))
    return model


def profiles(tmp_path, tps=(1, 2, 4)):
    result = {}
    for tp in tps:
        path = tmp_path / f'tp{tp}.json'
        profiled_model(tp).save(path)
        result[tp, 1] = path
    return result


def prepare(tmp_path, mode='offline_tp', **kwargs):
    return prepare_tp_runtime(model_name='Qwen2.5-7B-Instruct', gpus=list(range(8)), fixed_tp=1,
                              mode=mode, profiles=profiles(tmp_path), policy=get_policy('pdblend'),
                              forecast=fc(.5), slo=SLO(5, .15), **kwargs)


def test_offline_mode_selects_one_real_tp_without_weakening_policy_floor(tmp_path):
    runtime = prepare(tmp_path)
    assert len({s.tp for s in runtime.specs}) == 1
    assert sum(len(s.gpus) for s in runtime.specs) == 8
    assert runtime.selected_plan.counts['M'] >= min(4, len(runtime.specs))
    assert runtime.metadata['policy_m_floor'] == 4
    assert not runtime.metadata['policy_floor_override']
    assert all(s.profile_key and s.pp == 1 for s in runtime.specs)


def test_bounded_profile_uses_b1_for_sub_one_average_occupancy():
    model = profiled_model()
    planner = PoolPlanner(model, PlannerConfig(slots=8, slo=SLO(5, .15)))
    demand = fc(.1)
    pool = planner._mixed_pool(.1, demand, 512, 512, 8, 900)
    assert 0 < pool['batch'] < 1
    assert pool['tpot_s'] >= model.step_seconds(1, 576, 900)
    assert pool['power_w'] > 0


def test_resident_specs_use_distinct_gpu_groups_and_their_own_profiles(tmp_path):
    runtime = prepare(tmp_path, mode='resident_hetero_tp', resident_pools=(
        ResidentPool('small', Topology(1), 2), ResidentPool('large', Topology(2), 3)))
    assert [s.tp for s in runtime.specs] == [1, 1, 2, 2, 2]
    allocated = [g for spec in runtime.specs for g in spec.gpus]
    assert sorted(allocated) == list(range(8))
    assert runtime.pool_models['small'].tp == 1 and runtime.pool_models['large'].tp == 2
    assert len({s.port for s in runtime.specs}) == 5


def test_resident_mixed_tp_does_not_require_symmetric_pd_pair_budget(tmp_path):
    runtime = prepare_tp_runtime(
        model_name='Qwen2.5-7B-Instruct', gpus=[0, 1, 2], fixed_tp=1,
        mode='resident_hetero_tp', profiles=profiles(tmp_path, (1, 2)),
        policy=get_policy('pdblend'), forecast=fc(.1), slo=SLO(5, .15),
        resident_pools=(ResidentPool('small', Topology(1), 1),
                        ResidentPool('large', Topology(2), 1)))
    assert [s.tp for s in runtime.specs] == [1, 2]
    assert sorted(g for spec in runtime.specs for g in spec.gpus) == [0, 1, 2]


def test_resident_tp2_tp4_reserves_all_kv_rank_ports(tmp_path):
    paths = {}
    for tp in (2, 4):
        path = tmp_path/f'tp{tp}.json'
        profiled_model(tp, 'Qwen2.5-32B-Instruct').save(path)
        paths[tp, 1] = path
    runtime = prepare_tp_runtime(model_name='Qwen2.5-32B-Instruct', gpus=list(range(6)),
        fixed_tp=2, mode='resident_hetero_tp', profiles=paths, policy=get_policy('pdblend'),
        forecast=fc(.1), slo=SLO(5, .15), resident_pools=(
            ResidentPool('small', Topology(2), 1), ResidentPool('large', Topology(4), 1)))
    ports = [s.port+20000+rank for s in runtime.specs for rank in range(s.tp)]
    assert len(ports) == len(set(ports)) == 6
    assert runtime.specs[1].port == runtime.specs[0].port+2


def test_resident_rejects_overlap_before_launch(tmp_path):
    with pytest.raises(ValueError, match='overlapping'):
        prepare(tmp_path, mode='resident_hetero_tp', resident_pools=(
            ResidentPool('one', Topology(1, gpus=(0, 1)), 2),
            ResidentPool('two', Topology(2, gpus=(1, 2)), 1)))


def test_tp_mode_rejects_wrong_profile_identity_and_missing_coverage(tmp_path):
    paths = profiles(tmp_path, (1,))
    model = profiled_model()
    model.system = 'dynamollm'
    model.save(paths[1, 1])
    args = dict(model_name='Qwen2.5-7B-Instruct', gpus=list(range(8)), fixed_tp=1,
                mode='fixed_tp', profiles=paths, policy=get_policy('pdblend'), forecast=fc(.5), slo=SLO(5, .15))
    with pytest.raises(ValueError, match='independent profile'):
        prepare_tp_runtime(**args)
    model = profiled_model()
    model.bounded_coverage = {}
    model.save(paths[1, 1])
    with pytest.raises(ValueError, match='bounded measured coverage'):
        prepare_tp_runtime(**args)


def test_slow_mode_reports_unsupported_without_gpu_or_profile_access(monkeypatch, tmp_path):
    monkeypatch.setattr('pdblend.bench.run.Gpus', lambda *_: pytest.fail('GPU access must not occur'))
    result = run_point('Qwen2.5-7B-Instruct', list(range(8)), 1, 'pdblend', tmp_path / 'missing.json',
                       [], SLO(5, .15), tmp_path / 'out', tp_mode='slow_reshard_tp')
    assert result['status'] == 'unsupported_engine' and not result['hardware_executed']
    assert json.loads((tmp_path / 'out/summary.json').read_text())['tp_mode'] == 'slow_reshard_tp'


def test_forced_pd_probe_is_explicitly_not_a_policy_decision():
    plan = mechanism_pd_plan(4)
    assert plan.counts == {'P': 1, 'D': 1, 'M': 2}
    assert plan.detail['mechanism_forced_roles'] and not plan.detail['policy_decision']


def test_resident_proxy_router_uses_actual_pool_controller_views():
    small, large = profiled_model(1), profiled_model(2)
    # The TP2 profile is independently faster in this synthetic routing test.
    large.prefill_time = {f: tuple(v / 2 for v in row) for f, row in large.prefill_time.items()}
    large.decode_overrides = {f: {**row, 'coefficients': [v / 2 for v in row['coefficients']]}
                              for f, row in large.decode_overrides.items()}
    a = Router(['a'], instance_metadata={'a': dict(tp=1, pool_id='small', generation=1,
                                                  model_id=small.model, profile_key='small-key')})
    b = Router(['p', 'd'], instance_metadata={iid: dict(tp=2, pool_id='large', generation=2,
                                                       model_id=large.model, profile_key='large-key') for iid in ('p', 'd')})
    b.set_roles({'p': 'P', 'd': 'D'}, pd_threshold_tokens=1024)
    router = ResidentRouter({'small': a, 'large': b}, {'small': small, 'large': large})
    record = router.dispatch('one', 2048, 16)
    assert record.path == 'PD' and record.generation == 2 and record.tp == 2
    router.first_token(record)
    router.finish(record, 16)
    assert not router.selector.active and b.inflight() == {'p': 0, 'd': 0}
    record = router.dispatch('two', 2048, 16)
    router.finish(record, 0, 'native completion unknown')
    assert router.quarantined == {'p', 'd'} and 'two' in router.selector.active
    fallback = router.dispatch('three', 2048, 16)
    assert fallback.decode_instance == 'a' and fallback.tp == 1
