import random

from pdblend.bench.client import (Outcome, azure_trace, poisson_trace, slo_attainment, staged_trace,
                                  trace_summary)


def records(n=50):
    rng = random.Random(1)
    return [dict(prompt=[rng.randint(1, 1000) for _ in range(rng.randint(50, 800))],
                 output_tokens=rng.randint(2, 300)) for _ in range(n)]


def test_poisson_trace_rate_and_determinism():
    a = poisson_trace(records(), 10.0, 100.0, 701)
    b = poisson_trace(records(), 10.0, 100.0, 701)
    assert [r.arrival_s for r in a] == [r.arrival_s for r in b]
    assert 800 < len(a) < 1200
    assert all(r.max_tokens <= 512 for r in a)


def test_staged_trace_follows_stage_scales():
    stages = [(100.0, 0.5), (100.0, 2.0)]
    t = staged_trace(records(), 4.0, stages, cv=1.5, seed=701)
    first = sum(1 for r in t if r.arrival_s < 100)
    second = len(t) - first
    assert second > 2 * first
    s = trace_summary(t)
    assert s["requests"] == len(t) and s["input_p95"] >= s["input_p50"]


def test_azure_trace_thins_to_peak_and_clips_lengths():
    rng = random.Random(3)
    rows = []
    t = 0.0
    for _ in range(3000):
        t += rng.expovariate(20.0)
        rows.append((t, rng.randint(10, 9000), rng.randint(0, 700)))
    reqs, meta = azure_trace(rows, records(), peak_rps=5.0, seed=1, bin_s=10.0)
    assert 0 < meta["thinning"] < 1 and meta["kept"] == len(reqs)
    assert all(1 <= r.input_tokens <= 7168 and 2 <= r.max_tokens <= 512 for r in reqs)
    assert all(r.input_tokens + r.max_tokens <= 8192 for r in reqs)
    assert len(reqs) < len(rows) * 0.5


def test_slo_attainment():
    outs = [Outcome(i, 0, 100, 10, 0.0, first_token_s=0.5 + i * 0.2, finished_s=2.0 + i * 0.2, completion_tokens=10, path="M")
            for i in range(5)]
    outs.append(Outcome(9, 0, 100, 10, 0.0, error="503"))
    s = slo_attainment(outs, ttft_slo=1.0, tpot_slo=0.2)
    assert s["offered"] == 6 and s["succeeded"] == 5
    assert s["joint_slo"] == 3            # ttft 0.5, 0.7, 0.9 pass; 1.1, 1.3 fail
    assert abs(s["joint_slo_rate"] - 0.5) < 1e-9
    assert s["paths"]["M"] == 5
