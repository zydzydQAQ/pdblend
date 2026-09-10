import copy

import pytest

from ecopadg.scalability import protocol as p


def base_config():
    return dict(model_name=p.MODEL, strategy="pdblend-joint", allow_pd=False,
                dynamic_pools=True, slow_topology=True, max_service_frequency_mhz=2520,
                distserve_prefill_batch=1, distserve_decode_batch=8,
                instances=[dict(id=f"engine-{g}", gpus=[g], tp=1, role="mixed",
                                url=f"http://127.0.0.1:{30000+g}", provenance={"stale": True},
                                host_pid=9, container={"StartedAt": "old"}) for g in range(8)])


@pytest.mark.parametrize("n,counts", [(3, (1, 1, 1)), (4, (2, 1, 1)), (6, (2, 2, 2)), (8, (4, 2, 2))])
def test_config_creates_real_three_role_layout_from_explicit_inventory(n, counts):
    original = base_config()
    before = copy.deepcopy(original)
    config = p.build_config(original, system="pdblend", dataset="sharegpt", allocated_gpu_ids=list(range(n)))
    roles = [i["role"] for i in config["instances"]]
    assert tuple(roles.count(role) for role in ("mixed", "prefill", "decode")) == counts
    assert config["node_gpus"] == list(range(n))
    assert config["allocated_gpu_ids"] == list(range(n))
    assert config["raw_power_gpu_ids"] == list(range(8))
    assert config["allow_pd"] is True
    assert config["dynamic_pools"] is config["slow_topology"] is False
    assert config["formal_eligible"] is False
    assert all("provenance" not in i and "host_pid" not in i and "container" not in i for i in config["instances"])
    assert original == before


def test_fixed_baselines_keep_explicit_mapping_and_frequency():
    mixed = p.build_config(base_config(), system="mixed", dataset="sharegpt", allocated_gpu_ids=[6, 7, 0])
    assert mixed["strategy"] == "mixed"
    assert mixed["fixed_frequency_mhz"] == 2520 and mixed["dvfs"] is False
    assert [i["gpus"] for i in mixed["instances"]] == [[6], [7], [0]]
    assert [i["role"] for i in mixed["instances"]] == ["mixed"] * 3
    pd = p.build_config(base_config(), system="fixed_pd", dataset="longbench", allocated_gpu_ids=[0, 1, 2, 3], fixed_pd_p_count=1)
    assert pd["strategy"] == "distserve"
    assert [i["role"] for i in pd["instances"]] == ["prefill", "decode", "decode", "decode"]
    assert pd["distserve_decode_batch"] == 8 and pd["allow_pd"] is True


@pytest.mark.parametrize("change", ["missing", "duplicate", "tp2", "frequency"])
def test_builder_rejects_unmeasured_or_ambiguous_inventory(change):
    base = base_config()
    if change == "missing":
        base["instances"] = base["instances"][:2]
    elif change == "duplicate":
        base["instances"][1]["gpus"] = [0]
    elif change == "tp2":
        base["instances"][0]["tp"] = 2
    else:
        base["max_service_frequency_mhz"] = 2100
    with pytest.raises(ValueError):
        p.build_config(base, system="pdblend", dataset="sharegpt", allocated_gpu_ids=[0, 1, 2])


def test_pd_pilot_choice_and_stage_batch_limits_cannot_be_invented():
    with pytest.raises(ValueError, match="pilot-selected"):
        p.build_config(base_config(), system="fixed_pd", dataset="sharegpt", allocated_gpu_ids=[0, 1, 2])
    base = base_config()
    del base["distserve_decode_batch"]
    with pytest.raises(ValueError, match="stage batch"):
        p.build_config(base, system="fixed_pd", dataset="sharegpt", allocated_gpu_ids=[0, 1, 2], fixed_pd_p_count=1)


def test_protocol_copy_hash_and_timeout_are_frozen():
    first = p.protocol_dict()
    frozen_hash = p.protocol_hash()
    first["formal_seeds"].append(1)
    assert p.protocol_hash() == frozen_hash
    assert p.request_timeout_s("sharegpt", 2) == 120
    assert p.request_timeout_s("longbench", 1024) == pytest.approx(249.6)
    with pytest.raises(ValueError):
        p.request_timeout_s("sharegpt", 1)


def test_formal_matrix_and_pilot_seed_domains():
    row = dict(system="pdblend", dataset="longbench", n_gpus=3, seed=p.PILOT_SEED, stage="pilot", rate_rps=.5)
    assert p.validate_row(row)["stage"] == "pilot"
    for stage, seed in [("formal", 701), ("weak", 701), ("capacity", 701)]:
        with pytest.raises(ValueError, match="formal dataset matrix"):
            p.validate_row(dict(row, stage=stage, seed=seed))
    with pytest.raises(ValueError, match="seed domains"):
        p.validate_row(dict(row, dataset="sharegpt", stage="capacity"))
    assert p.validate_row(dict(row, stage="diagnostic"))["stage"] == "diagnostic"


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"), True])
def test_invalid_rates_rejected(rate):
    with pytest.raises(ValueError):
        p.validate_row(dict(system="mixed", dataset="sharegpt", n_gpus=3, seed=701, stage="weak", rate_rps=rate))
