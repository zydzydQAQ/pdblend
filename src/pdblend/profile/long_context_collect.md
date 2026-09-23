# Incremental long-context training collector

```bash
python -B -m pdblend.profile.long_context_collect \
  --plan /plan/Qwen2.5-7B-Instruct-tp4.json \
  --training-raw /training/raw.json \
  --model /models/Qwen2.5-7B-Instruct \
  --gpus 0 1 2 3 --base-port 18000 --out /output
```

Mount the immutable plan, original training raw, model/tokenizer receipt and
frozen source read-only. Mount `/output`, the load-lock directory `/coord` and a
new cohort directory `/wave` writable. Set `PDBLEND_PROFILE_WAVE=/wave`,
`PDBLEND_PROFILE_MEMBER` to the declared cohort member, and the same engine/image,
source hash and GPU UUID variables as other profile jobs. The raw checksum must
match `plan.training_source_sha256`; a mounted path override never changes that
identity. Root owns the lease, ports, wave membership and queue scheduling.

The collector validates the PDBlend model/TP/PP1 and tokenizer identity, launches
one resident TP group, qualifies its parallel layout with `ProfileWave`, then
visits only `plan.training` in frequency order. It never reads a fitted model or
uses the short-context predictor to extrapolate into the extension. It does not
sample `plan.holdout` or fit anything. New fitting and an independently collected
holdout are later stages.

Each declared point keeps its full output reservation: for a 7168-token prompt,
1024 outputs exactly meet the 8192 model limit. The entire reservation must fit
90% of the live engine's measured KV capacity. A smaller capacity results in
`missing_profile`, rather than silently substituting another batch or topology.
Three separate batch-prefill runs each include the all-stream 16-token barrier,
at least two seconds of continuous decode settling, and at least five seconds of
measurement. Every stream must contribute eight steps; each window needs two
power and one frequency sample. Output exhaustion also records `missing_profile`
and leaves its existing evidence intact.

Every complete window immediately saves raw token timestamps, power, clocks and
their checksum before checkpointing to `decode_pending`. Resume validates plan,
source, model/tokenizer, engine/source environment, GPUs and every raw checksum,
then runs only missing repetitions. A point moves into `decode` after all three
windows are complete. Engine capacity is retained per window and per launch.
`effective_context_tokens` derives from actual generated-token indices; declared
`max_tokens` does not extend the measured domain.

`completion.json` describes collection as `training_evidence`, with
`formal_eligible=false`, `fit_performed=false` and `holdout_points_consumed=0`.
The raw archive records repeat noise for the subsequent calibration audit. This
artifact alone cannot establish a valid planner or benchmark ranking.
