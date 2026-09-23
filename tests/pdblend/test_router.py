from pdblend.proxy.router import Router


def make(roles, tau=0):
    r = Router(list(roles), pd_threshold_tokens=tau)
    r.set_roles(roles)
    return r


def test_mixed_only_uses_jsq():
    r = make({"a": "M", "b": "M"})
    rec = r.dispatch("r1", 100, 10)
    assert rec.path == "M" and rec.prefill_instance == rec.decode_instance
    rec2 = r.dispatch("r2", 100, 10)
    assert rec2.decode_instance != rec.decode_instance
    r.finish(rec, 10)
    assert r.loads[rec.decode_instance].inflight_seqs == 0
    assert r.loads[rec.prefill_instance].inflight_prefill_tokens == 0


def test_threshold_splits_between_mixed_and_pd():
    r = make({"m": "M", "p": "P", "d": "D"}, tau=1000)
    short = r.dispatch("s", 200, 10)
    long_ = r.dispatch("l", 4000, 10)
    assert short.path == "M" and short.decode_instance == "m"
    assert long_.path == "PD" and (long_.prefill_instance, long_.decode_instance) == ("p", "d")
    assert r.loads["p"].inflight_prefill_tokens == 4000
    r.first_token(long_)
    assert r.loads["p"].inflight_prefill_tokens == 0
    assert r.loads["d"].inflight_seqs == 1


def test_pd_only_takes_everything_and_parked_is_skipped():
    r = make({"p": "P", "d1": "D", "d2": "D", "x": "parked"}, tau=10_000)
    a = r.dispatch("a", 10, 10)
    b = r.dispatch("b", 10, 10)
    assert a.path == b.path == "PD"
    assert {a.decode_instance, b.decode_instance} == {"d1", "d2"}


def test_nothing_accepting_rejects():
    r = make({"x": "parked"})
    assert r.dispatch("a", 10, 10) is None and r.rejected == 1


def test_lone_prefill_pool_falls_back_to_mixed():
    r = make({"p": "P", "m": "M"}, tau=0)
    assert r.dispatch("a", 10, 10).path == "M"


def test_pressure_gate_routes_long_prompts_with_hysteresis():
    r = Router(["m", "p", "d"], pd_threshold_tokens=4096,
               pd_pressure_enter=.75, pd_pressure_exit=.55,
               pd_route_hold_s=30.0, pd_route_stable_windows=2,
               pd_min_input_tokens=1024)
    r.set_roles({"m": "M", "p": "P", "d": "D"})
    r.configure_pressure_gate(enter=.75, exit=.55, hold_s=30.0, stable_windows=2,
                              min_input_tokens=1024)
    assert r.dispatch("short", 512, 10).path == "M"
    assert r.set_pressure_state(m_pressure=.8, now=100.0)
    assert r.dispatch("short2", 512, 10).path == "M"
    assert r.choose(2048)[0] == 'M'  # pressure alone cannot change the evaluated load split
    r.set_roles({}, 1024)  # controller commits the matching feasible plan
    assert r.dispatch("long", 2048, 10).path == "PD"
    r.set_pressure_state(m_pressure=.2, now=131.0, stable_window=True)
    assert r.pressure_state()["pd_active"]
    r.set_pressure_state(m_pressure=.2, now=146.0, stable_window=True)
    assert not r.pressure_state()["pd_active"]
    r.set_roles({}, 4096)
    assert r.dispatch("long2", 2048, 10).path == "M"


def test_shield_pressure_enters_long_prompt_pd_mode_immediately():
    r = Router(["m", "p", "d"], pd_threshold_tokens=1024)
    r.set_roles({"m": "M", "p": "P", "d": "D"})
    r.configure_pressure_gate(enter=.75, exit=.55, hold_s=30, stable_windows=2, min_input_tokens=1024)
    r.set_pressure_state(shield_active=True, now=10)
    assert r.dispatch("short", 512, 8).path == "M"
    assert r.dispatch("long", 1024, 8).path == "PD"


def test_pd_selects_one_compatible_pair_instead_of_independent_projection():
    router = Router(['p1', 'd1', 'p2', 'd2'], instance_metadata={
        'p1': {'tp': 1, 'pool_id': 'one', 'generation': 1},
        'd1': {'tp': 1, 'pool_id': 'one', 'generation': 1},
        'p2': {'tp': 2, 'pool_id': 'two', 'generation': 2},
        'd2': {'tp': 2, 'pool_id': 'two', 'generation': 2},
    })
    router.set_roles({'p1': 'P', 'd1': 'D', 'p2': 'P', 'd2': 'D'})
    router.loads['p2'].inflight_prefill_tokens = 1000
    router.loads['d1'].inflight_seqs = 5
    route = router.dispatch('one', 2048, 16)
    assert (route.prefill_instance, route.decode_instance) == ('p1', 'd1')
    assert route.generation == 1
    router.loads['d1'].generation = 4
    assert router.choose(2048) == ('PD', 'p2', 'd2')


def test_live_request_prevents_topology_metadata_rewrite():
    import pytest
    router = make({'m': 'M'})
    record = router.dispatch('r', 100, 16)
    with pytest.raises(RuntimeError, match='in flight'):
        router.set_instance_metadata('m', tp=2, generation=1)
    router.finish(record, 16)
    router.set_instance_metadata('m', tp=2, generation=1)
    assert router.loads['m'].generation == 1
