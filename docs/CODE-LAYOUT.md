# Code responsibilities

Canonical modules are grouped by responsibility. Legacy import paths resolve to the same module objects so old CLI invocations, monkeypatches, and pickle references remain usable. Existing frozen experiment sources are untouched.

| Directory | Responsibility |
|---|---|
| `pdblend/profile/query` | CPU-only scalar query, coverage, profile versions |
| `pdblend/profile/collection` | GPU sampling and resumable collection |
| `pdblend/profile/calibration` | Fitting, holdout audit, profile publication |
| `pdblend/planner` | Forecasts and offline pool/topology planning |
| `pdblend/online` | Runtime control, proxy routing and resharding |
| `pdblend/legacy` | Historical shared baseline policy implementations |
| `pdblend_baselines` | Independent baseline implementations and own profiles |

Dataset pickle classes remain at their existing definitions. Shared engine, metering and workload modules keep their established directories. New source manifests include canonical implementations and compatibility modules; old manifests are immutable and do not implicitly qualify changed numerical implementations.

## Module mapping

- `pdblend.profile.model` → `pdblend.profile.query.model`
- `pdblend.profile.power_table` → `pdblend.profile.query.power_table`
- `pdblend.profile.versions` → `pdblend.profile.query.versions`
- `pdblend.profile.long_domain` → `pdblend.profile.query.long_domain`
- `pdblend.profile.profiler` → `pdblend.profile.collection.profiler`
- `pdblend.profile.wave` → `pdblend.profile.collection.wave`
- `pdblend.profile.window_sampling` → `pdblend.profile.collection.window_sampling`
- `pdblend.profile.parallel` → `pdblend.profile.collection.parallel`
- `pdblend.profile.sampling_guard` → `pdblend.profile.collection.sampling_guard`
- `pdblend.profile.sampling_epochs` → `pdblend.profile.collection.sampling_epochs`
- `pdblend.profile.long_context_collect` → `pdblend.profile.collection.long_context_collect`
- `pdblend.profile.short_domain_collect` → `pdblend.profile.collection.short_domain_collect`
- `pdblend.profile.resident_domain_job` → `pdblend.profile.collection.resident_domain_job`
- `pdblend.profile.local_power_job` → `pdblend.profile.collection.local_power_job`
- `pdblend.profile.power_job` → `pdblend.profile.collection.power_job`
- `pdblend.profile.long_holdout_only` → `pdblend.profile.collection.long_holdout_only`
- `pdblend.profile.resident_long_holdout` → `pdblend.profile.collection.resident_long_holdout`
- `pdblend.profile.long_context_followup` → `pdblend.profile.collection.long_context_followup`
- `pdblend.profile.decode_fit` → `pdblend.profile.calibration.decode_fit`
- `pdblend.profile.timing_calibration` → `pdblend.profile.calibration.timing_calibration`
- `pdblend.profile.power_calibration` → `pdblend.profile.calibration.power_calibration`
- `pdblend.profile.local_power` → `pdblend.profile.calibration.local_power`
- `pdblend.profile.local_mixed_repair` → `pdblend.profile.calibration.local_mixed_repair`
- `pdblend.profile.acceptance` → `pdblend.profile.calibration.acceptance`
- `pdblend.profile.validation` → `pdblend.profile.calibration.validation`
- `pdblend.profile.merge` → `pdblend.profile.calibration.merge`
- `pdblend.profile.calibration_repair` → `pdblend.profile.calibration.calibration_repair`
- `pdblend.profile.holdout_inheritance` → `pdblend.profile.calibration.holdout_inheritance`
- `pdblend.profile.short_domain` → `pdblend.profile.calibration.short_domain`
- `pdblend.profile.short_version` → `pdblend.profile.calibration.short_version`
- `pdblend.profile.calibration` → `pdblend.profile.calibration.core`
- `pdblend.control.forecast` → `pdblend.planner.forecast`
- `pdblend.control.topology` → `pdblend.planner.topology`
- `pdblend.control.planner` → `pdblend.planner.pool`
- `pdblend.control.controller` → `pdblend.online.controller`
- `pdblend.control.shield` → `pdblend.online.shield`
- `pdblend.control.reshard` → `pdblend.online.reshard`
- `pdblend.control.tp_modes` → `pdblend.online.tp_modes`
- `pdblend.proxy.router` → `pdblend.online.router`
- `pdblend.proxy.server` → `pdblend.online.server`
- `pdblend.proxy.sse` → `pdblend.online.sse`
- `pdblend.bench.tp_runtime` → `pdblend.online.tp_runtime`
- `pdblend.control.policies` → `pdblend.online.policies`
- `pdblend.control.policies.baselines` → `pdblend.legacy.baselines`

- `pdblend.bench.native_mixed` → `pdblend_baselines.mixed.run_native`
- `pdblend_baselines.mixed_policy` → `pdblend_baselines.mixed.policy`

## Source identity and compatibility

`pdblend.source_inventory` resolves the source tree independently of module depth. Newly prepared collection artifacts bind the complete selected package, including both canonical implementations and old aliases. Existing frozen source bytes and manifests remain immutable. Consumers of published calibration versions verify the original frozen source separately from the current consumer implementation; relocation alone must not revoke their measured evidence.

Legacy module aliases share the same Python module object with the canonical implementation, including the calibration package/core facade. Query modules do not import GPU collectors or calibration runners. `scripts/profile` and `scripts/campaign` provide maintained entrances; active dated dependencies remain available until their final reference retires.
