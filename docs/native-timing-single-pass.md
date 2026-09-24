# Single-pass development timing

`scripts/2026-09-24_prepare_pdblend_single_pass_timing.py` prepares immutable
7B/14B timing jobs without enqueuing or running GPUs. It binds each original
model-owned v2 point plan and keeps every unique shape, frequency, training seed,
holdout seed and holdout threshold. Each point has one measurement window.
Repeated isolation probes, runtime collection and power collection are absent.

Each job owns eight GPUs exclusively and runs eight model replicas concurrently.
The package freezes the complete current `src` tree. The existing collector's
per-window CUDA, native capacity, request completion, frequency and final
physical cleanup checks still apply. Capacity exclusions remain receipts and
never become measured timing observations.

The plan, inputs, completion, partition and replay identify the data as
`single_pass_development`. `component_qualified`, `parallel_qualified`,
`formal_eligible` and `full_profile_qualified` remain false. A successful holdout
diagnostic does not establish repeatability or isolated/concurrent equivalence.
Legacy model curves and existing power/runtime observations retain their
original provenance and qualification; token timestamps are not CUDA samples.

Prepare with explicit parent plans and the one predecessor that must terminate:

```sh
PYTHONPATH=src python3 -B scripts/2026-09-24_prepare_pdblend_single_pass_timing.py \
  --out results/2026-09-24/pdblend-single-pass-timing-v1 \
  --parent-7b results/2026-09-24/pdblend-profile-recovery-v4/7b/collection/point-plan.json \
  --parent-14b results/2026-09-24/pdblend-profile-recovery-v4/14b/collection/point-plan.json \
  --after-terminal pdblend-native-timing-32b-77286f41f7bf8c04
```

The output directory must not exist. `jobs.json` has priority 1000, one attempt,
eight-GPU exclusive ownership and no successful-job dependency. Queue submission
and actual-image CPU preflight are separate operator actions. Native timing's
existing `capture_evidence` / `replay_evidence` entry points recognize the new
schemas and recheck the complete raw inventory after the queue attempt finishes.

After a succeeded or failed single-pass attempt, inspect its point inventory:

```sh
PYTHONPATH=src python3 -B -m pdblend.profile.collection.native_timing_partial \
  --attempt /absolute/path/to/terminal/attempt \
  --queue results/2026-09-22/three-model/queue.json \
  --out /absolute/path/to/new-terminal-inventory
```

This prints a new read-only inventory with `measured`, `unsupported_capacity`,
`failed`, `invalid` and `missing` states. It verifies frozen source bytes, the
terminal task and each raw point's expected owner. Complete audited observations
are marked to skip remeasurement, even when the overall job failed. It does not
retry anything: missing points require a new selective immutable plan, and failed
or invalid points require their actual cause to be reviewed first. Attempts that
failed before recording the full native launch/capability inventory cannot be
audited by this point-level tool and fail explicitly.

The worker binds required receipts only after exit zero. For a collector saved
failure (exit 2), the tool instead requires its original terminal lease and
ordered physical cleanup evidence, then records a **new** post-terminal completion
SHA. `original_worker_completion_binding.present` remains false. The new directory
contains the exact queue snapshot plus the inventory; no original receipt or hash
is backfilled. An existing but mismatched worker hash remains a hard failure.
