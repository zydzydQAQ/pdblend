# Controlled optimization-stage acceptance

Run the standalone checker after collecting genuine, exclusive GPU evidence:

```bash
PYTHONPATH=src /home/pdblend/.venv/bin/python -m pdblend.bench.optimization_acceptance stage.json --out acceptance.json
```

Exit code 0 means the named stage passed. Exit code 1 means its comparison is
invalid or its improvement remains unproven. The checker never synthesizes
missing measurements or upgrades CPU tests to hardware qualification.

The manifest uses `schema: "pdblend.optimization-ablation/v1"`, a nonempty
`stage` string, `traces`, `independent_baselines`, and `pairs`. Every file
reference has exactly `{"path": "relative/path.json", "sha256": "<file digest>"}`.
Paths are relative to the manifest and digests cover exact file bytes.

`traces` binds the frozen evaluation trace files, each carrying `seed: 701` and
a nonempty `requests` array. `independent_baselines` binds separate accepted
profile receipts for Mixed, DistServe, DynamoLLM and EcoServe at every tested
model/TP combination. A profile receipt contains `system`, `model_id`, `tp`,
`pp: 1`, and `accepted: true`; baseline system names are `mixed`, `distserve`,
`dynamollm`, and `ecoserve`. Different systems cannot share a profile receipt.

Each pair has `repeat` (a nonnegative integer) and `control` / `candidate`
objects. Each arm contains:

- `conditions`: exact `model_id`, `dataset`, `tp`, `pp`, `seed`, `trace_sha256`,
  `rate_rps`, `slo: {ttft_s, tpot_s}`, `engine_sha256`, and `weights_sha256`.
- `summary`: a reference to the measured run summary, using the existing
  benchmark fields `energy_j`, `window_s`, `goodput_request_s`,
  `goodput_token_s`, `j_per_goodput_token`, `quarantined_instances`, `metering`,
  and `slo: {offered, joint_slo_rate, joint_slo_requests, joint_output_tokens}`.
- `execution`: a reference to the scheduler's stage/lease receipt. It binds
  `system: "pdblend"`, `stage`, `arm`, exact `conditions`, `summary_sha256`,
  `profile_sha256`, `start_s`, `end_s`, eight distinct `gpu_uuids`,
  `exclusive: true`, and `lease_verified: true`.
- `profile`: a reference to an accepted PDBlend profile receipt with the
  matching model/TP/PP identity. Control and candidate may use different
  approved calibration versions when that is the named optimization stage.
- `qualification`: a reference to a PDBlend GPU functional receipt binding
  `system`, `model_id`, `tp`, `pp`, `functional_passed: true`,
  `hardware_qualified: true`, and `native_cleanup_complete: true`.
- `uncertainty`: a reference to an instrument uncertainty estimate containing
  `summary_sha256`, a positive `absolute_energy_j` bound and its `method`.
  This bound must come from actual meter characterization; zero or missing
  uncertainty does not qualify a run.

The test publisher in `tests/pdblend/test_optimization_acceptance.py::campaign`
is a complete executable schema example. Its synthetic data is for CPU
contract tests only and must not be reused in a campaign.

Acceptance requires all Qwen2.5 7B, 14B and 32B models crossed with `alpaca`,
`sharegpt`, and `longbench`, seed 701, PP1, and at least three independent paired
repetitions for each exact condition. Additional rates need their own three
repetitions. Both arms use identical traces, rates, SLOs, models, TP, engine,
weights, physical GPUs, and load-window duration. All exclusive windows must
be disjoint on any shared GPU. Immutable summary evidence cannot be reused as
another repetition. Offered-request counts must match the frozen trace.

Each run must meet joint SLO ≥ 0.9, contain consistent measured token/energy/
goodput totals, have no quarantined instances or metering error, and use the
exact instantaneous NVML field-186 identity or an explicit NVML cumulative
energy counter. Legacy averaged and unidentified power sources fail.

For every exact condition, mean candidate request goodput **and** token
goodput must be at least their respective control means. No regression
tolerance is silently applied. Energy comparison uses the mean J per
SLO-qualified output token. Each arm's uncertainty is its largest absolute
repeat deviation from that mean plus its largest per-token instrument bound.
The candidate's reduction must exceed the **sum** of both arms' uncertainty
bounds. This conservative bound does not assert a confidence level.

The report includes every condition's metrics, uncertainty, and acceptance
verdict. All conditions must pass to accept that stage. The report keeps
`formal_eligible: false`: accepting a named ablation does not establish an
unmeasured combined optimization, a baseline ranking, or online TP resharding.
