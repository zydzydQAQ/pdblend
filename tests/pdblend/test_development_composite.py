"""Development provenance, interpolation and routing guards; no GPU required."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from pdblend.profile.query.development_composite import DevelopmentCompositeModel, KIND
from pdblend.profile.query.model import PerfModel
from pdblend.profile.query.versions import load_profile
from test_slo_routing_recovery import router, configure


ROOT = Path(__file__).resolve().parents[2]


def base():
    model = PerfModel.load(Path(__file__).parent/'fixtures/legacy_profile.json')
    model.system, model.model, model.tp, model.pp = 'pdblend', 'fixture', 1, 1
    model.kv_capacity_tokens = 100000
    return model


def compiled():
    timing = []
    for f, latency in ((1500, 10.), (2100, 20.)):
        for role in ('prefill', 'decode'):
            timing.append(dict(frequency_mhz=f, role=role, coefficients=[latency, 1., 2.],
                coverage_vertices=([[.05], [.5]] if role == 'prefill'
                                   else [[1., .05], [4., .05], [4., .5], [1., .5]])))
    powers = {f'decode/{f}/{b}/0/0.0': [dict(context_min=c, context_max=c, power_w=w+b+c/1000)]
              for f, w in ((1500, 100.), (2520, 200.)) for b in (2, 3) for c in (1000,)}
    for nodes in powers.values():
        nodes.append(dict(context_min=2000, context_max=2000, power_w=nodes[0]['power_w']+1))
    return dict(identity=dict(system='pdblend', model_id='fixture', tp=1, pp=1), timing_models=timing,
        power_nodes=powers, summary={'test_only': True}, source_bindings=[],
        capacity=dict(actual_total_kv_tokens=100000, max_num_seqs=32, block_size=16),
        handoff_nodes=[dict(input_tokens=n, output_tokens=16, requested_f_P_mhz=2520,
            requested_f_D_mhz=2520, observed_training_max_first_gap_s=.08+n/100000)
            for n in (512, 2048, 7168)])


def model():
    return DevelopmentCompositeModel(base(), compiled(), {'profile_sha256': 'test'})


def test_native_and_interpolated_timing_never_extrapolate():
    m = model()
    x = 1024/8192
    assert m.prefill_seconds(1024, 1500) == pytest.approx((10+x+2*x*x)/1000)
    assert m.prefill_seconds(1024, 1800) == pytest.approx((15+x+2*x*x)/1000)
    assert m.prefill_seconds(7000, 1800) == m.base.prefill_seconds(7000, 1800)
    assert m.prefill_seconds(1024, 2520) == m.base.prefill_seconds(1024, 2520)
    assert m.token_energy_j(2, 1024, 1500) == pytest.approx(
        m.step_seconds(2, 1024, 1500)*m.decode_power_w(2, 1500, ctx=1024)/2)
    assert m.query_provenance_summary()['query_counts']['prefill_timing:inherited'] == 2


def test_power_interpolates_only_inside_both_panels_and_batches():
    m = model()
    assert m.decode_power_w(2.5, 2010, ctx=1500) == pytest.approx(154.)
    assert m.decode_power_w(1, 2010, ctx=1500) == m.base.decode_power_w(1, 2010, ctx=1500)
    assert m.decode_power_w(2, 1500, ctx=5000) == m.base.decode_power_w(2, 1500, ctx=5000)
    assert m.decode_power_w(2, 900, ctx=1500) == m.base.decode_power_w(2, 900, ctx=1500)


def test_capacity_refusal_is_preserved_outside_native_timing_hull():
    m = model()
    assert not m.decode_supported(32, 5000, 2520)
    with pytest.raises(ValueError, match='capacity'):
        m.step_seconds(32, 5000, 2520)
    assert not m.decode_supported(33, 100, 1500)


def test_planner_peak_includes_small_batches_with_real_kv_limits():
    from pdblend.planner.pool import PlannerConfig, PoolPlanner, SLO
    m = model()
    m.base.kv_capacity_tokens = 48016
    planner = PoolPlanner(m, PlannerConfig(4, SLO(15., .2), max_num_seqs=32))
    assert m.decode_supported(6, 7000, 2520)
    assert not m.decode_supported(7, 7000, 2520)
    assert planner._peak_decode_tps(7000, 2520) == pytest.approx(max(
        b/m.step_seconds(b, 7000, 2520) for b in range(1, 7)))
    assert planner._peak_decode_tps(7000, 2520) > 0


def test_endpoint_does_not_expand_output_or_clock_coverage():
    m = model()
    query = dict(input_tokens=1024, output_tokens=16, f_P_mhz=2520, f_D_mhz=2520,
        batch=1, context_tokens=1040, prefill_instance='p', decode_instance='d')
    assert m.pd_first_gap_seconds(**query) == pytest.approx(.09024)
    for changes in (dict(output_tokens=2), dict(f_D_mhz=1500), dict(batch=2), dict(input_tokens=10)):
        with pytest.raises(ValueError, match='domain'):
            m.pd_first_gap_seconds(**dict(query, **changes))
    assert m.query_provenance_summary()['query_counts']['first_gap:unsupported_domain'] == 4


def test_router_uses_first_gap_once_and_preserves_uncovered_protection():
    m = model()
    r = router()
    configure(r, m)
    predicted = r._slo_route_prediction(('PD', 'p', 'd'), 1024, 16)
    assert predicted['tpot_s'] == pytest.approx((.09024+14*predicted['step_s'])/15)
    assert predicted['transfer_s'] is None
    assert predicted['handoff_source'] == m.pd_first_gap_source
    with pytest.raises(ValueError, match='domain'):
        r._slo_route_prediction(('PD', 'p', 'd'), 1024, 2)
    predicted = r._slo_route_prediction(('PD', 'p', 'd'), 1024, 8)
    assert predicted['handoff_source'] == 'legacy_transfer_plus_step'
    assert predicted['handoff_s'] == pytest.approx(m.base.transfer_seconds(1024)+predicted['step_s'])


def bound(path, data):
    path.write_text(json.dumps(data))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_loader_binds_components_identity_and_rejects_formal_or_tampering(tmp_path):
    b = tmp_path/'base.json';base().save(b)
    b_ref = dict(path=str(b), sha256=hashlib.sha256(b.read_bytes()).hexdigest())
    components = tmp_path/'components.json'
    c_ref = bound(components, compiled())
    descriptor = tmp_path/'profile.json'
    bound(descriptor, dict(kind=KIND, system='pdblend', model_id='fixture', tp=1, pp=1,
        base_profile=b_ref, compiled=c_ref))
    kwargs = dict(system='pdblend', model_id='fixture', tp=1)
    loaded = load_profile(descriptor, **kwargs)
    assert loaded.profile_key['base_profile_sha256'] == b_ref['sha256']
    with pytest.raises(ValueError, match='formally'):
        load_profile(descriptor, **kwargs, usage='formal')
    with pytest.raises(ValueError, match='mismatch'):
        load_profile(descriptor, **dict(kwargs, tp=2))
    components.write_text(components.read_text()+' ')
    with pytest.raises(ValueError, match='checksum'):
        load_profile(descriptor, **kwargs)


def test_partial_fit_never_uses_holdout_and_affine_rank_is_explicit():
    spec = importlib.util.spec_from_file_location('development_builder', ROOT/'scripts/2026-09-24_build_development_profiles.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows = [dict(role='prefill', frequency_mhz=1500, purpose='training', prompt_tokens=n,
                 latency_ms=t, window_id=str(n)) for n, t in ((100, 10.), (1000, 20.))]
    held = dict(rows[0], purpose='holdout', prompt_tokens=500, latency_ms=1e9, window_id='holdout')
    first = module.fit_partial(rows)[0]
    second = module.fit_partial(rows+[held])[0]
    assert first['coefficients'] == second['coefficients']
    assert first['coefficients'][2] == 0
    assert second['holdout_events'] == 1
    assert module.fit_partial([held]) == []


def test_unique_query_log_counts_identical_calls_without_repetition():
    m = model()
    m.prefill_seconds(1024, 1500);m.prefill_seconds(1024, 1500)
    assert len(m.query_provenance_log()) == 1
    assert m.query_provenance_log()[0]['count'] == 2
