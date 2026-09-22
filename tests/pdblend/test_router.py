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
