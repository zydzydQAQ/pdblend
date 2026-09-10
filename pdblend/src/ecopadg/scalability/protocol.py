"""Frozen, CPU-only declarations for the 14B resource-scaling experiment.

Configuration construction is not physical qualification. Every generated
configuration must be matched to fresh engine/profile/transfer evidence by the
experiment runner before a formal measurement can be accepted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

PROTOCOL_ID = "pdblend-resource-scalability-v1"
MODEL = "Qwen2.5-14B-Instruct"
GPU_COUNTS = (3, 4, 6, 8)
POOL_COUNTS = {3: (1, 1, 1), 4: (2, 1, 1), 6: (2, 2, 2), 8: (4, 2, 2)}
SYSTEMS = ("pdblend", "mixed", "fixed_pd")
FORMAL_SEEDS = (701, 1701, 2701, 3701, 4701)
PILOT_SEED = 9701
CONTENT_SEED = 20260910
ARRIVAL_WINDOW_S = 600.0
WARMUP_S = 120.0
SLOS = {"sharegpt": (5.0, 0.15), "longbench": (15.0, 0.20)}
FORMAL_SCALES = {"sharegpt": GPU_COUNTS, "longbench": (4, 8)}
STAGES = ("pilot", "formal", "capacity", "weak", "warmup", "diagnostic")


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def protocol_dict():
    return dict(
        schema=1, protocol_id=PROTOCOL_ID, model=MODEL, tensor_parallel_size=1,
        gpu_counts=list(GPU_COUNTS),
        pool_counts={str(n): dict(zip(("mixed", "prefill", "decode"), counts))
                     for n, counts in POOL_COUNTS.items()},
        systems=list(SYSTEMS),
        formal_scales={name: list(ns) for name, ns in FORMAL_SCALES.items()},
        formal_seeds=list(FORMAL_SEEDS), pilot_seed=PILOT_SEED,
        content_seed=CONTENT_SEED, arrival_window_s=ARRIVAL_WINDOW_S,
        warmup_s=WARMUP_S, arrival_process="unconditioned Poisson",
        slos={name: dict(ttft_s=ttft, tpot_s=tpot) for name, (ttft, tpot) in SLOS.items()},
        request_deadline="max(120, ttft_s + (output_len - 1) * tpot_s + 30)",
        slo_definition="complete AND TTFT < ttft_s AND mean TPOT < tpot_s",
        offered_denominator="all offered requests including rejected, failed and timed-out requests",
        mean_tpot_definition="(last_token_s - first_token_s) / (output_tokens - 1)",
        minimum_output_tokens=2, ignore_eos=True, attainment_target=0.90,
        capacity_relative_width=0.05, capacity_method="pilot doubling and bisection; independent formal confirmation",
        weak_q="0.7 * min(pilot capacity / N) across all systems at N=3,4, separately per dataset",
        longbench_n3="pilot only for the LongBench-specific weak-load normalization",
        energy=dict(raw_gpu_ids=list(range(8)), primary="allocated_gpu_ids, including idle and parked GPUs",
                    secondary="all eight GPUs on the node", sample_interval_s=0.02,
                    failure_energy="retained", overlapping_windows="never added"),
        formal_evidence_required=["frozen_sources", "actual_model_and_engine_identity",
                                  "fresh_native_correctness", "actual_role_and_tp_layout",
                                  "matching_profile_coverage", "matching_transfer_coverage",
                                  "effective_slo", "offered_request_accounting",
                                  "per_gpu_power", "native_terminal_cleanup"],
        configuration_certificate_inherited=False,
    )


def protocol_hash():
    return digest(protocol_dict())


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return float(value)


def request_timeout_s(dataset, output_len):
    if dataset not in SLOS or type(output_len) is not int or output_len < 2:
        raise ValueError("known dataset and output_len >= 2 required")
    ttft, tpot = SLOS[dataset]
    return max(120.0, ttft + (output_len - 1) * tpot + 30.0)


def validate_row(row):
    row = copy.deepcopy(row)
    if row.get("system") not in SYSTEMS or row.get("dataset") not in SLOS:
        raise ValueError("unknown system or dataset")
    stage = row.get("stage", "formal")
    n = row.get("n_gpus")
    if stage not in STAGES or type(n) is not int or n not in GPU_COUNTS:
        raise ValueError("unknown stage or GPU count")
    if stage not in ("pilot", "warmup", "diagnostic") and n not in FORMAL_SCALES[row["dataset"]]:
        raise ValueError("GPU count is not part of the formal dataset matrix")
    seed = row.get("seed")
    allowed = (PILOT_SEED,) if stage == "pilot" else FORMAL_SEEDS
    if stage in ("warmup", "diagnostic"):
        allowed = (*FORMAL_SEEDS, PILOT_SEED)
    if type(seed) is not int or seed not in allowed:
        raise ValueError("pilot and formal seed domains must be separate")
    row["rate_rps"] = _positive_number(row.get("rate_rps"), "rate_rps")
    row["stage"] = stage
    allocated = row.get("allocated_gpu_ids")
    if allocated is not None:
        _validate_gpu_ids(allocated, n)
    return row


def _validate_gpu_ids(gpus, n=None):
    if not isinstance(gpus, (list, tuple)) or any(type(g) is not int or not 0 <= g < 8 for g in gpus):
        raise ValueError("allocated_gpu_ids must contain physical indices 0..7")
    if len(set(gpus)) != len(gpus) or len(gpus) not in GPU_COUNTS or (n is not None and len(gpus) != n):
        raise ValueError("allocated_gpu_ids must contain the declared number of unique GPUs")


def build_config(base_config, *, system, dataset, allocated_gpu_ids,
                 fixed_pd_p_count=None, max_frequency_mhz=None):
    """Build an unqualified config from explicit TP1 endpoint declarations.

    The input GPU order defines placement. No ports, engine endpoints, profiles,
    transfer certificates or current process identities are invented.
    """
    if system not in SYSTEMS or dataset not in SLOS:
        raise ValueError("unknown system or dataset")
    _validate_gpu_ids(allocated_gpu_ids)
    if base_config.get("model_name") != MODEL:
        raise ValueError("base config must explicitly declare Qwen2.5-14B-Instruct")
    n = len(allocated_gpu_ids)
    inventory = {}
    for instance in base_config.get("instances", []):
        gpus = instance.get("gpus", [])
        if instance.get("tp") != 1 or len(gpus) != 1:
            raise ValueError("explicit base inventory must contain only TP1 instances")
        if gpus[0] in inventory or not instance.get("id") or not instance.get("url"):
            raise ValueError("base inventory needs unique physical GPUs and explicit IDs/URLs")
        inventory[gpus[0]] = instance
    if any(g not in inventory for g in allocated_gpu_ids):
        raise ValueError("base inventory lacks an allocated GPU endpoint; prepare it explicitly")
    if len({inventory[g]["id"] for g in allocated_gpu_ids}) != n or len({inventory[g]["url"] for g in allocated_gpu_ids}) != n:
        raise ValueError("allocated engines must have distinct IDs and URLs")
    if system == "pdblend":
        m, p, d = POOL_COUNTS[n]
        roles = ["mixed"] * m + ["prefill"] * p + ["decode"] * d
    elif system == "mixed":
        roles = ["mixed"] * n
    else:
        if type(fixed_pd_p_count) is not int or not 1 <= fixed_pd_p_count < n:
            raise ValueError("fixed PD requires an explicit pilot-selected prefill count")
        roles = ["prefill"] * fixed_pd_p_count + ["decode"] * (n - fixed_pd_p_count)
        if any(type(base_config.get(k)) is not int or base_config[k] <= 0
               for k in ("distserve_prefill_batch", "distserve_decode_batch")):
            raise ValueError("fixed PD requires explicit positive independent stage batch limits")
    config = copy.deepcopy(base_config)
    instances = []
    for gpu, role in zip(allocated_gpu_ids, roles):
        instance = copy.deepcopy(inventory[gpu])
        for key in ("provenance", "host_pid", "container", "scheduler_cache_observed", "scheduler_cache_count"):
            instance.pop(key, None)
        instance.update(role=role, tp=1, gpus=[gpu])
        instances.append(instance)
    ttft, tpot = SLOS[dataset]
    config.update(strategy={"pdblend": "pdblend-joint", "mixed": "mixed", "fixed_pd": "distserve"}[system],
                  comparison_system=system, instances=instances, node_gpus=list(allocated_gpu_ids),
                  allocated_gpu_ids=list(allocated_gpu_ids), raw_power_gpu_ids=list(range(8)),
                  allow_pd=system != "mixed", dynamic_pools=False, slow_topology=False,
                  dvfs=system == "pdblend", slo_ttft_s=ttft, slo_tpot_s=tpot,
                  slo_attainment_target=0.9, arrival_window_s=ARRIVAL_WINDOW_S,
                  request_timeout_policy=protocol_dict()["request_deadline"],
                  protocol_id=PROTOCOL_ID, protocol_sha256=protocol_hash(),
                  formal_eligible=False, physical_qualification_required=True,
                  fixed_pd_p_count=fixed_pd_p_count if system == "fixed_pd" else None)
    # The shared serving baselines use 2520 MHz internally. Reject a different
    # maximum instead of emitting metadata that the real implementation ignores.
    maximum = max_frequency_mhz if max_frequency_mhz is not None else base_config.get("max_service_frequency_mhz")
    if type(maximum) is not int or maximum != 2520:
        raise ValueError("this baseline implementation requires explicit measured max_frequency_mhz=2520")
    config["max_service_frequency_mhz"] = maximum
    config["fixed_frequency_mhz"] = maximum if system != "pdblend" else None
    config["baseline_frequency_adapter_required"] = False
    config["park_idle"] = system == "pdblend"
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = dict(protocol=protocol_dict(), protocol_sha256=protocol_hash())
    if args.out:
        with args.out.open("xb") as handle:
            handle.write(canonical_bytes(result))
    else:
        print(canonical_bytes(result).decode(), end="")


if __name__ == "__main__":
    main()
