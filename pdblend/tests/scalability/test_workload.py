import copy
import json
import random

import pytest

from ecopadg.scalability import protocol as p
from ecopadg.scalability.workload import build_trace, load_pool, trace_hash


POOL = [dict(prompt_token_ids=[i + 1] * (i + 2), prompt_len=i + 2, output_len=i + 3) for i in range(11)]


def trace(**kwargs):
    defaults = dict(dataset="sharegpt", n_gpus=4, seed=701, rate_rps=1.0)
    defaults.update(kwargs)
    return build_trace(POOL, **defaults)


def test_paired_systems_get_identical_trace_bytes_and_original_caps():
    results = [trace() for _ in p.SYSTEMS]
    assert len({trace_hash(x) for x in results}) == 1
    first = results[0]
    assert "system" not in first
    assert first["duration_s"] == first["arrival_window_s"] == 600
    assert all(r["ignore_eos"] is True for r in first["requests"])
    for request, prompt, index in zip(first["requests"], first["prompts"], first["source_pool_indices"]):
        assert prompt == POOL[index]["prompt_token_ids"]
        assert request["prompt_len"] == POOL[index]["prompt_len"]
        assert request["output_len"] == POOL[index]["output_len"]


def test_true_poisson_arrivals_and_same_content_prefix_when_rate_changes():
    low, high = trace(rate_rps=.5), trace(rate_rps=1)
    count = low["n_requests"]
    assert low["source_pool_indices"] == high["source_pool_indices"][:count]
    assert low["prompts"] == high["prompts"][:count]
    assert low["requests"][0]["arrival_s"] == random.Random(701).expovariate(.5)
    assert low["requests"][0]["arrival_s"] > 0
    for a, b in zip(low["requests"], high["requests"]):
        assert a["arrival_s"] == pytest.approx(2 * b["arrival_s"])
    assert all(0 < r["arrival_s"] < 600 for r in high["requests"])


def test_content_is_unbiased_by_scale_and_has_independent_rng():
    a, b, c = trace(n_gpus=3), trace(n_gpus=8), trace(content_seed=99, stage="diagnostic")
    assert a["requests"] == b["requests"] and a["prompts"] == b["prompts"]
    assert [r["arrival_s"] for r in a["requests"]] == [r["arrival_s"] for r in c["requests"]]
    assert a["source_pool_indices"] != c["source_pool_indices"]
    counts = [a["source_pool_indices"].count(i) for i in range(len(POOL))]
    assert min(counts) > 0
    assert trace(seed=1701)["requests"] != a["requests"]


def test_per_request_deadline_includes_long_outputs_and_last_arrival():
    result = build_trace([dict(prompt=[1, 2], prompt_len=2, output_len=1024)],
                         dataset="longbench", n_gpus=4, seed=701, rate_rps=.1)
    assert result["requests"][0]["timeout_s"] == pytest.approx(249.6)
    assert result["last_request_deadline_offset_s"] == pytest.approx(
        result["requests"][-1]["arrival_s"] + 249.6)


def test_pool_reader_and_generation_do_not_mutate_input(tmp_path):
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(dict(records=POOL)))
    pool = load_pool(path)
    before = copy.deepcopy(pool)
    result = build_trace(pool, dataset="sharegpt", n_gpus=4, seed=701, rate_rps=.25)
    assert pool == before
    assert result["n_requests"] == len(result["prompts"])
    assert trace_hash(result) == p.digest(result)


@pytest.mark.parametrize("bad", [[], [dict(prompt="", prompt_len=1, output_len=2)],
                                  [dict(prompt=[1], prompt_len=2, output_len=3)],
                                  [dict(prompt=[1], prompt_len=1, output_len=1)]])
def test_invalid_or_fabricated_shape_pool_rejected(bad):
    with pytest.raises(ValueError):
        build_trace(bad, dataset="sharegpt", n_gpus=4, seed=701, rate_rps=.5)


def test_warmup_is_separate_and_duration_cannot_silently_shorten():
    warm = trace(stage="warmup")
    assert warm["duration_s"] == 120
    assert warm["prompts"] != trace()["prompts"][:warm["n_requests"]]
    with pytest.raises(ValueError, match="frozen stage"):
        trace(duration_s=10)
    with pytest.raises(ValueError, match="frozen protocol"):
        trace(content_seed=99)
