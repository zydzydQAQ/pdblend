# Explicit profile consumption and bounded optimization measurements

`query.versions.load_profile(path=None, *, registry=None, version_id=None,
system, model_id, tp, pp=1, usage="development")` returns a `LoadedVersion`:
`.model`, `.identity`, `.coverage`, `.qualification`, `.profile_key`, and
`.manifest_fields()`. The model also carries `calibration_identity`,
`calibration_coverage`, and `calibration_qualification` for decision logs.
Legacy JSON remains explicit and keeps its numerical behavior. Formal use is
rejected without formal qualification. Registry selection always names one
version; `latest` is rejected.

A JSON selection has `kind: "pdblend_profile_selection_v1"` and either
`profile_path` or `registry` plus `version_id`. Paths resolve beside the file.
Qualified timing/power versions alone lack runtime capacity/static/transition
qualification. Optional `runtime_base` binds `profile`, `raw`, and `audit`,
each as `{path, sha256}`. Generate the audit with
`calibration.runtime_components.audit_runtime(base_profile, raw_path, out=...)`.
The consumer recomputes this audit, checks every raw sample checksum, and
compares measured capacity, static states, clock repeats, and transfer fit.
Historical summaries remain labelled as such: no independent runtime holdout,
native correctness, full transition energy, or formal eligibility is inferred.

The dated script `scripts/2026-09-23_optimization_profiles.py --registry REGISTRY
--out NEW_DIRECTORY` publishes explicit selections and CPU query receipts from
existing versions without changing any original registry or evidence.

## New low-batch and Mixed measurements

Use the existing engine environment, `PYTHONPATH=src`, and a **new** coordinated
`SamplingEpochs` cohort. Preparation is CPU-only:

```sh
python -m pdblend.profile.collection.optimization_profiles prepare \
  --base-candidate PROFILE --identity-raw RAW --out NEW_PACKAGE --frequencies 1500
python -m pdblend.profile.collection.optimization_profiles run \
  --package NEW_PACKAGE --model MODEL --gpus 0 --base-port 8200 \
  --out NEW_OUTPUT --epochs-root NEW_COHORT --member MEMBER --preflight-only
```

Remove `--preflight-only` only inside the scheduled GPU allocation. Group size
must equal TP. The initial plan has frequency 1500, exact decode batches 2/3,
Mixed batch 8, 512-token prefill probes at 1 request/second, training contexts
512/2048, and separate holdout context 1280. Every point has three fresh runs,
two seconds of settle and five seconds of measurement. All 27 windows reuse
one resident engine; the fit freezes before holdout. Contexts are reconstructed
from actual decode events, never replaced with nominal prompts.

This new panel explicitly launches `pdblend_runtime.serve` with
`InstanceSpec.native_control=True`. CPU preflight requires and verifies
`PDBLEND_MODEL_VERIFICATION_RECEIPT`, and records the actual launcher command.
The entrypoint is bound in the plan, raw window, component, union, and consumer
qualification. Historical plain `vllm serve` timings remain historical inputs;
they still require a native timing crosscheck. No native timing/formal approval
is inferred from the new power holdout. Existing profilers and frozen packages
keep their original serving entrypoints.

Each window snapshots `/baseline/events` before power sampling starts and after
it stops. Native event sequences must have no gaps/overflow; schedule timestamps
are clipped to the same interval. Evidence records counts and time-weighted
batch distributions, including background decode batch and `prefill_mode`.
Pure B2/B3 requires at least 99% of the interval at the offered exact batch,
without foreign requests or prefill. A mismatch is archived and fails the
component. These are scheduler residency intervals, not CUDA kernel durations.
Actual distributions still do not qualify fractional-batch interpolation.

Mixed probes and continuous decode share the same integrated power interval.
The entire interval, including gaps between probes, contributes energy. Raw
decode timing and arithmetic mean power remain recorded separately. Mixed
power never becomes pure-decode power. Missed probe cadence, early decode end,
clock mismatch, epoch change, insufficient samples, or sample reuse fail closed.
Independent holdout gates retain maximum/mean error and repeat-CV thresholds
of 15%/10%/10%. Raw and audit failures are preserved.

Only explicitly qualified exact B2/B3 points become available. Fractional low
batches remain unsupported; legacy B1 and B>=4 separation remains unchanged.
The optional `bounded_context_power_exact_low_batch_v2` table schema also
requires explicit independent low-batch qualification and forbids bridging.

Split complementary frequencies across the two three-model groups: for
example group A at 1500 and group B at 2520. After both components independently
pass, combine without refitting:

```sh
python -m pdblend.profile.collection.optimization_profiles merge \
  --components OUTPUT_1500/components OUTPUT_2520/components --out NEW_UNION.json
```

A selection's `optimization_component` names either one `components` directory
or the immutable union JSON. Loading replays sample verification, training fit,
and independent holdout audit. Overlapping domains or different base profiles,
models, tokenizers, and TP identities are rejected. All new consumption remains
development-only. Mixed queries require exact measured batch/frequency/chunk
and `prefill_rate_rps`, plus covered actual context:
`mixed_power_supported` / `mixed_power_w`. The independent holdout error bound
is available through `mixed_power_residual_bound_w` and `power_validation`.

`calibration.optimization_profiles.missing_domain_ledger(model, queries)` ranks
actual missed queries using `query_count * abs(decision_sensitivity_j)`. It does
not invent demand or promote domains. Each job separates measured, unqualified,
prefill/settle/cleanup, synchronization, and load GPU seconds; useful GPU time
counts only measurement windows accepted by the independent audit.
