# Measured transition costs

`PlannerConfig.transition_estimator` accepts `TransitionCatalog.load(path, model=model)`.
Each entry binds one independent profile identity (including its version key), exact source and
target role counts, active clocks, split threshold, TP/PP, pool and generation. An uncovered transition
uses the existing wake/clock estimate and records that fallback in `Plan.detail.transition_cost`.
With `qualified_only=True`, uncovered transitions are skipped; a still-feasible current plan is retained.
This does not enable native TP resharding.

The catalog JSON has `kind: "pdblend_transition_catalog_v1"`, `identity`, and `entries`.
Use `build_catalog_entry(receipt_path, source_plan, target_plan, model, counterfactual_path=None)`
to construct an entry from the actual `online.transition_measurement.measure_transitions` receipt.
The producer retains incomplete evidence as **unqualified**. Loading reproduces the audit and verifies
all SHA-256 hashes instead of trusting the stored `qualified` flag or energy value.

For a qualified entry, the raw receipt must additionally record:

- `binding: {identity, source, target}`, exactly as returned by `identity(model)` and `signature(plan)`;
- `target_verified: true`, from native verification of the resulting state;
- successful, metered action phases. Role/frequency changes require validation and route publication;
  parking requires proxy and native drain; wake requires readiness; cold start/restart additionally
  require load/start, warmup, verification and publication. `REQUIRED_PHASES` defines phase aliases;
- an independently metered counterfactual receipt covering the same per-GPU interval geometry, source
  and target binding, workload and power source. Both receipts carry
  `pairing: {pair_id, workload_sha256, role, uncertainty_j}`. Roles are `transition` and
  `counterfactual`; `uncertainty_j` is the measured absolute energy uncertainty bound.

The critical path is `max(finished_s) - min(started_s)`, including parallel overlap and intervening gaps.
Measured energy uses the union of actual per-GPU phase intervals, with the counterfactual measured over
exactly that same union. The decision penalty is
`max(0, measured_union_energy_j - counterfactual_union_energy_j + both_uncertainty_bounds)`.
Gross phase energy, summed parallel durations, and idle watts times startup latency cannot establish
this incremental penalty. The current raw collector emits `incremental_energy_j: null`; without the
additional paired evidence its receipts remain useful timing/energy observations, not qualified costs.

Example producer (source/target files contain serialized `Plan` fields):

```bash
python -m pdblend.planner.transitions \
  --receipt /absolute/transition.json \
  --counterfactual /absolute/matched-control.json \
  --source-plan /absolute/source.json --target-plan /absolute/target.json \
  --profile /absolute/profile.json --model-id Qwen2.5-7B-Instruct --tp 1 \
  --output /absolute/transition-catalog.json
```

Omitting `--counterfactual` is supported and produces an explicitly unqualified entry. Adding an entry
or changing a bound profile requires regenerating/re-auditing the catalog; no automatic latest-version
lookup or synthesized native verification is performed.

The `bench` CLI and matrix point dictionaries expose `transition_catalog_path`,
`transition_qualified_only`, `capacity_floor_path`, `joint_resident`, and `incremental_energy_path`.
CLI names use dashes. These controls reject independent baseline policies. For heterogeneous TP,
any artifact path may point to an index with
`kind: "pdblend_optimization_artifact_set_v1"` and
`profiles: {"tp1-pp1": "relative/tp1.json", "tp2-pp1": "relative/tp2.json"}`.
Missing topology entries fail before engine startup.

A capacity-floor file uses `kind: "pdblend_capacity_floor_v1"`, the exact `identity(model)`,
`stage`, `acceptance_manifest: {path, sha256}`, and `floors`. Each floor supplies
`min_m_instances`, `rate_range`, `input_range`, `output_range`, and
`slo: {ttft_s, tpot_s}`. The loader reruns the complete controlled-stage acceptance,
checks actual fixed-M candidate summaries and exact profile keys, and requires both rate endpoints
and the requested length bounds to have been measured. Trace summaries now include min/max lengths;
older summaries without these observations cannot relax the floor. A tighter runtime SLO or a workload
outside the accepted domain restores the original policy floor.
