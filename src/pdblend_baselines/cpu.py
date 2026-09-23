"""Executable CPU contracts for the migrated independent baseline cores.

The probe uses explicitly synthetic observations. Passing it establishes an
import/algorithm contract, never a calibrated profile or a hardware receipt.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace


# Geometry is an input to DistServe's own search, not a shared performance model.
# TP choices are the initial eight-L20 campaign choices; TP1/PP1 is excluded for
# 32B because BF16 weights alone exceed one card's available memory.
MODEL_GEOMETRY = {
    "7b": {"layers": 28, "attention_heads": 28, "allowed_tps": (1, 2, 4)},
    "14b": {"layers": 48, "attention_heads": 40, "allowed_tps": (1, 2, 4, 8)},
    "32b": {"layers": 64, "attention_heads": 40, "allowed_tps": (2, 4, 8)},
}


def verify_migration() -> dict:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "migration-manifest.json").read_text())
    adapted = {row['path']: row for row in manifest.get('adaptations', [])}
    failed = []
    for row in manifest.get('core_files', []) + manifest.get('adapter_files', []):
        adaptation = adapted.get(row['path'])
        expected = row['sha256']
        if adaptation:
            original = (root / adaptation['original_artifact']).resolve()
            if (not original.is_relative_to(root.resolve()) or
                    adaptation.get('original_sha256') != expected or
                    hashlib.sha256(original.read_bytes()).hexdigest() != expected or
                    not adaptation.get('reason')):
                failed.append(row['path'] + ': original lineage')
            expected = adaptation['sha256']
        if hashlib.sha256((root / row['path']).read_bytes()).hexdigest() != expected:
            failed.append(row['path'])
    if failed:
        raise ValueError("migrated core checksum mismatch: " + ", ".join(failed))
    return {"verified_files": len(manifest.get("core_files", [])) + len(manifest.get("adapter_files", [])),
            "source_identity_verified": True, "explicit_adaptations": sorted(adapted),
            "hardware_qualified": False}


def topology_report(model: str) -> dict:
    from .distserve.planning import enumerate_configs

    geometry = MODEL_GEOMETRY[model]
    configs = enumerate_configs(**geometry, num_nodes=1, gpus_per_node=8)
    return {"model": model, "geometry": geometry, "configurations": configs,
            "configuration_count": len(configs), "measurement": "structural_search_only",
            "hardware_qualified": False}


def run_contracts(models: tuple[str, ...] = ("7b", "14b", "32b")) -> dict:
    from .distserve.simulator import MeasuredLatency, OfficialSimulator
    from .dynamollm.policy import Configuration, Epochs, shard_milp
    from .dynamollm.profiles import PaperProfiles
    from .dynamollm.reconfiguration import weight_transfer_plan
    from .ecoserve.policy import OfficialMacro, PrefillProfile

    integrity = verify_migration()
    latency = MeasuredLatency([
        dict(role=role, tp=1, pp=1, batch=4, max_input_tokens=64,
             max_context_tokens=128, stage_latency_ms=delay, source_sha256="0" * 64)
        for role, delay in (("prefill", 10.), ("decode", 2.))
    ])
    simulator = OfficialSimulator([(16, 3), (32, 4), (48, 2)], latency=latency,
                                  capacities={(1, 1): 8192}, seed=701)
    # Preserve author simulator stdout behavior without corrupting JSON output.
    with contextlib.redirect_stdout(io.StringIO()):
        simulated = simulator((1, 1, 1, 1, 1), 1.)
    profiles = PaperProfiles([
        dict(role="mixed", tp=2, frequency_mhz=2520, input_tokens=n,
             context_tokens=c, batch=b, prefill_s=n / 10000,
             iteration_s=.01, power_w=100., samples=1, source_sha256="0" * 64)
        for n in (16, 32) for c in (64, 128) for b in (1, 4)
    ])
    estimate = profiles.query(2, 2520, 24, 96, 2)
    allocation = shard_milp([Configuration(1, 1, 1., 100.),
                             Configuration(2, 1, 3., 150.)], 4, 5.)
    transfer = weight_transfer_plan(((0, 1, 2, 3),), ((0, 1), (2, 3)), 8)
    macro = OfficialMacro(("a", "b", "c"), PrefillProfile({16: 2., 4096: 400.}),
                          1000, 100, now_ms=lambda: 10000.)
    for instance in macro.instance_states:
        instance.free_blocks = 1024
    selected = macro.schedule(SimpleNamespace(request_id="contract", prompt_len=16))
    if len(simulated["ttft_s"]) != 3 or any(t <= 0 for t in simulated["ttft_s"]):
        raise AssertionError("DistServe request simulation did not complete")
    if abs(estimate.prefill_s - .0024) > 1e-12:
        raise AssertionError("Dynamo independent profile interpolation mismatch")
    if sum(c.tp * n for c, n in allocation) != 4 or sum(c.power_w * n for c, n in allocation) != 300:
        raise AssertionError("Dynamo whole-pool MILP mismatch")
    if selected != 0 or macro.instance_states[0].requests[0].prefill_blocks != 2:
        raise AssertionError("EcoServe author admission or block rounding mismatch")
    return {"schema": 1, "status": "passed", "measurement": "synthetic_cpu_contract",
            "hardware_qualified": False, "energy_comparable": False,
            "integrity": integrity, "models": [topology_report(m) for m in models],
            "contracts": {
                "distserve": {"offered": 3, "completed": len(simulated["ttft_s"]),
                              "request_and_worker_events": True},
                "dynamollm": {"profile_interpolation": True, "milp_exact_budget": True,
                              "periods_due_at_1800": Epochs(0).due(1800),
                              "retained_abstract_units": transfer["retained_units"]},
                "ecoserve": {"selected_instance": selected, "author_block_rounding": True}},
            "remaining_hardware_adapter": {
                "distserve": "V1 stage timing, pipeline execution, KV pull/release and cancellation",
                "dynamollm": "V1 physical weight transfer, rank/generation ACK and lifecycle transport",
                "ecoserve": "V1 native scheduler state, macro rotation, split/merge and live KV continuity"}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", *MODEL_GEOMETRY), default="all")
    args = parser.parse_args(argv)
    models = tuple(MODEL_GEOMETRY) if args.model == "all" else (args.model,)
    print(json.dumps(run_contracts(models), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
