from types import SimpleNamespace

import pytest

from pdblend.online.router import ResidentRouter, Router
from pdblend.planner.pool import SLO


def pools():
    routers, models = {}, {}
    for iid, tp, latency in [('fast', 2, .01), ('efficient', 1, .02)]:
        routers[iid] = Router([iid], instance_metadata={iid: dict(
            tp=tp, pool_id=iid, model_id='same-model', generation=1, profile_key=iid)})
        models[iid] = SimpleNamespace(freqs=(900, 1500), kv_capacity_tokens=10000,
            prefill_seconds=lambda n, f: n / 10000,
            step_seconds=lambda b, c, f, latency=latency: latency,
            transfer_seconds=lambda n: .005)
    return ResidentRouter(routers, models)


def estimate(choice, context):
    return dict(qualified=True, incremental_energy_j=100 if choice[2] == 'fast' else 10,
                profile_keys=context['profile_keys'], coverage={'cpu_fixture_only': True})


def test_opt_in_qualified_energy_minimizes_joules_inside_slo_domain():
    router = pools()
    router.configure_energy_routing(slo=SLO(1, .1), estimator=estimate)
    record = router.dispatch('r', 100, 10)
    assert record.decode_instance == 'efficient'
    assert record.route_estimate['energy']['incremental_energy_j'] == 10
    assert record.route_estimate['frequencies'] == {'efficient': 1500}


def test_energy_never_outweighs_tpot_and_ttft_constraints():
    router = pools()
    router.configure_energy_routing(slo=SLO(1, .015), estimator=estimate)
    assert router.dispatch('r', 100, 10).decode_instance == 'fast'
    router.configure_energy_routing(slo=SLO(.001, .1), estimator=estimate)
    assert router.dispatch('impossible', 100, 10) is None


def test_missing_energy_qualification_uses_one_consistent_latency_scale():
    router = pools()
    router.configure_energy_routing(slo=SLO(1, .1), estimator=lambda choice, context:
                                    None if choice[2] == 'fast' else estimate(choice, context))
    assert router.dispatch('r', 100, 10).decode_instance == 'fast'


def test_missing_power_coverage_falls_back_without_breaking_admission():
    router = pools()

    def missing(choice, context):
        raise ValueError('power query outside measured coverage')

    router.configure_energy_routing(slo=SLO(1, .1), estimator=missing)
    assert router.dispatch('r', 100, 10).decode_instance == 'fast'


def test_actual_clock_queued_work_and_end_context_are_passed_to_energy_estimator():
    router = pools()
    observed = []
    router.frequency_provider = lambda iid: 900
    router.configure_energy_routing(slo=SLO(1, .1), estimator=lambda choice, context:
                                    observed.append(context) or estimate(choice, context))
    first = router.dispatch('first', 100, 20)
    second = router.dispatch('second', 100, 20)
    assert second.route_estimate['queued_prefill_s'] == .01
    assert second.route_estimate['context_tokens'] == 120
    assert second.route_estimate['batch'] == 2
    assert all(all(f == 900 for f in e['frequencies'].values()) for e in observed)
    assert first.decode_instance == second.decode_instance == 'efficient'


def test_explicit_outer_shares_are_followed_and_report_actual_admission():
    router = pools()
    router.set_target_shares({'fast': .3, 'efficient': .7})
    for i in range(10):
        record = router.dispatch(str(i), 100, 10)
        router.finish(record, 10)
    feedback = router.dispatch_feedback(reset=True)
    assert feedback['counts'] == {'fast': 3, 'efficient': 7}
    assert feedback['shares'] == feedback['target_shares']
    assert router.dispatch_feedback()['total'] == 0


def test_infeasible_target_pool_falls_back_to_available_capacity():
    router = pools()
    router.set_target_shares({'fast': 0., 'efficient': 1.})
    router.pools['efficient'].set_accepting('efficient', False)
    assert router.dispatch('r', 100, 10).decode_instance == 'fast'


@pytest.mark.parametrize('shares', [{'fast': 1}, {'fast': -1, 'efficient': 2},
                                    {'fast': .4, 'efficient': .4}])
def test_invalid_target_shares_rejected(shares):
    with pytest.raises(ValueError):
        pools().set_target_shares(shares)
