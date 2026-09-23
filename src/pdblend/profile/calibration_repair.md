# Four-point 14B TP1 holdout repair

This entry point repairs a sampling-domain error without changing the frozen
candidate or fitting any holdout observation. The original B128/context256
measurements at 1500, 1800, 2100 and 2520 MHz generated beyond the candidate's
measured KV domain. Their raw files, completion and failed audit stay unchanged.

The replacement points are B96/context256 at those four frequencies, with output
reservations of 180, 183, 185 and 186 tokens respectively. Each point has three
separate prefills, a two-second decode settling period and a five-second
measurement window. Full prompt-plus-output reservations must fit 90% of live
measured KV capacity and the unchanged candidate's domain. Exhausting an output
reservation is a failure, not permission to shorten a window or extrapolate.

Run through a root-coordinated one-GPU lease with the existing ProfileWave
qualification environment. Mount the candidate, repair plan, original holdout,
model and environment receipt read-only, and give the output and coordination
directories separate writable mounts:

```sh
python -B -m pdblend.profile.calibration_repair \
  --candidate-dir /candidate \
  --repair-plan /repair/plan.json \
  --original-holdout /original \
  --model /models/Qwen2.5-14B-Instruct \
  --gpus 0 --base-port 19000 --out /output
```

The host plan is
`results/2026-09-23/14b-tp1-holdout-domain-repair-plan.json`.
Use the port and GPU assignment from the lease, not the example values. The model
loads once; the collector does not rerun the prefill or mixed grids. Existing
2100 MHz representative-layout qualification is still required. It does not
claim that all six frequencies were independently interference-qualified.

Each completed window is checkpointed with token timestamps, power/frequency
samples and checksums. Restart with the same output directory and immutable
inputs to validate and reuse completed windows. The audit resolves original and
repair sample files through explicit, separate evidence roots. It evaluates the
unchanged predictor at each measured effective context, with the existing sample,
timing and power gates.

Outputs include `raw.json` containing only the four new points,
`superseded-domain-evidence.json` retaining the four original domain failures,
`composite-audit-input.json` describing the explicit revised B96 holdout matrix,
and `holdout-audit.json`. `completion.json` separates measurement completion from
calibration success. A successful repair remains non-formal until the other
profile-provenance and system-mechanism gates pass. No original result is
rewritten, and the original B128 matrix is not reported as having passed.
