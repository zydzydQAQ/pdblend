# Incremental 7B TP4 decode-power validation

This path changes only PDBlend decode power. `power_table.py` supplies a bounded table over batch and actual effective context. `PerfModel.decode_power_w(batch, frequency, *, ctx=...)` uses it only when `decode_power_overrides` is present. Historical affine models retain their original behavior. A new override rejects missing context, unmeasured frequency, B2/B3 (including fractional batches between 1 and 4), and unsupported contexts; it cannot fall back to affine coefficients or nearest frequencies.

`decode_power_supported` and timing's `decode_supported` are separate predicates. Planner candidates require both domains. If no covered fallback exists, planning raises `PowerCoverageError` with `missing_profile`; it cannot silently select a fixed layout using an invalid power estimate. Idle duty uses measured B1 power, while the active fractional batches >=4 use the bounded interpolation rule. Timing parameters and original timing coverage remain unchanged.

The six-frequency table is derived exclusively from the checksum-bound training raw. The prepare command checks every training sample checksum and reconstructs the recorded GPU-group power and effective context. It verifies the proposal nodes against those training observations; it does not refit any holdout. The independent validation panel has 24 points: B1, B64, B192 and B256 at each of the six frequencies.

Prepare a new immutable package after the source changes are final:

```bash
PYTHONPATH=src /home/pdblend/.venv/bin/python -m pdblend.profile.power_calibration prepare \
  --proposal /home/pdblend4/results/2026-09-23/7b-tp4-power-training-review-final/power-candidate-proposal.json \
  --original-holdout /home/pdblend4/results/2026-09-22/three-model/queue-attempts/holdout-7b-tp4-22436e6b272c028dd006-shared-bc3b8b870464/attempt-0001-373e3f3f1322460ba85e5ea5818b9cbd \
  --out /path/to/immutable-power-package
```

Preparation is CPU-only. It produces `candidate.json`, `power-plan.json`, `original-timing-component-audit.json` and `manifest.json`. The manifest binds the proposal, original training raw, unchanged base candidate, old holdout raw/completion/manifest, implementation files and environment identities. The candidate remains `training_only`. Never modify a prepared package in place; a changed implementation or input requires a new package and source freeze.

Run only through the coordinated GPU lease queue, in the pinned vLLM/Torch/CUDA image, with one TP4 group and an existing `ProfileWave` cohort:

```bash
python -m pdblend.profile.power_calibration run \
  --package /path/to/immutable-power-package \
  --model /models/Qwen2.5-7B-Instruct \
  --gpus 0 1 2 3 --base-port 14000 --out /path/to/new-attempt
```

The example GPU IDs/port are placeholders for the queue's allocation. Mount the package and all `manifest.inputs` at their declared absolute paths, read-only. Mount the full original holdout directory including its samples for CPU re-audit, and the old candidate directory for its original manifest. Training raw and the proposal must also remain readable and checksum-identical. Outputs use a separate writable attempt directory. The live image/vLLM/Torch/CUDA/hardware identity must match the original timing evidence; source hashes are explicitly recorded separately because the host-side fitting/audit code changes.

The runner keeps one engine resident across all 24 points. The plan reserves all generated tokens inside both timing and power coverage, using real training token counts for prefill/barrier lead plus 15% faster generation and terminal guard tokens. B256 starts earlier in context to leave space for all three windows. A point uses one prefill followed by three distinct >=2 s settle / >=5 s power windows. Their actual contexts, timestamps, >=8 steps, power/frequency samples and raw SHA-256 values are audited separately. The existing 2100 MHz representative paired-layout qualification remains unchanged; six-frequency coverage refers to measurement points.

Each completed window is checkpointed. Resuming a partial point preserves its existing samples and starts a new decode run for the remaining windows, explicitly counting that additional prefill. Duplicate windows, overlapping intervals, wrong repeat indices, changed candidates/plans and corrupted raw samples are rejected. Prediction errors do not trigger automatic retries or become training samples. Raw evidence is retained if sampling escapes its reservation or coverage. No prefill grid, mixed grid, model fitting or full original holdout rerun occurs.

At completion, `power-only-audit.json` evaluates every fresh power window using its actual context, requiring each independent-window error <=10%, aggregate MAPE <=10% and maximum <=15%. `reused-timing-audit.json` recomputes the unchanged original timing and mixed-timing component while preserving typed original power failures as excluded diagnostic records. Unknown, evidence, timing and matrix failures remain fatal. `composite-audit.json` binds both components and both source/environment identities. The original failed `completion.json` and all original raw files remain untouched. Component success is not formal system eligibility; provenance, transfer-protocol correctness, native mechanisms and campaign acceptance remain separate gates.

The plan includes a CPU scheduling proxy from the original single-request prefill measurements, alongside the 8.4-minute decode-window floor. The proxy is neither a measured batch runtime nor a guaranteed bound; load, token barriers, clocks, qualification and cleanup are extra. Full actual-context bounds are available per point in `reservation`; unsupported B1 short contexts and long contexts >=5120 remain `missing_profile`.
