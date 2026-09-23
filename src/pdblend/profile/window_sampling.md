# Optional consecutive PDBlend decode windows

This module is opt-in. Existing frozen source snapshots, queue entries and
`Profiler._decode_batch` remain unchanged.

```python
from pdblend.profile.window_sampling import measure_shared_prefill_windows

result = await measure_shared_prefill_windows(
    profiler, client, instance_gpus,
    model=frozen_candidate, freq_mhz=f, batch=b, context=c, tag=tag,
    training_raw=checksum_verified_training_raw,
)
if result['status'] == 'fallback_required':
    # Preserve result/attempt metadata alongside the independently completed row.
    row = await profiler._decode_batch(client, instance_gpus, b, c, 64, tag + '-legacy')
elif result['status'] == 'sampled':
    row = result['row']
else:
    # Preserve failed prediction evidence. Do not select a new measurement to
    # hide this candidate's error or pretend a partial row is complete.
    raise RuntimeError(result)
```

The caller keeps ownership of clock locking, model identity/provenance, candidate
file checksum, independent holdout status and profile checkpointing. The runner
checks PDBlend/system/model/TP/PP identity and serializes the candidate before and
after sampling. Another baseline's profile cannot be passed to this path.
The calibration CLI selects it with `--shared-prefill-windows`. Containers also
need the original immutable training raw mounted read-only and passed through
`--training-raw /training/raw.json`; the recorded original checksum is mandatory.

The planner uses an existing same-shape latency measurement to reserve enough
output tokens for three 2-second settle plus 5-second measurement windows,
including prefill skew, the existing 16-token all-stream barrier and a terminal
guard. The **entire** reserved prompt plus output length must fit 8192 and 90% of
the measured KV capacity. The entire reserved context interval must also remain
inside the frozen candidate's measured domain. Missing timing, memory or domain
coverage selects the old method before starting requests. Actual progress is
checked again; early exhaustion or newly unsupported context produces an explicit
incomplete attempt and no acceptable profile row.

Every window retains separate power/frequency samples, raw token timestamps,
actual start/end counts, an immutable sample checksum and its own prediction.
Each stream must contribute at least eight decode steps; every window has at
least two power and one frequency sample. Output token j (zero-based) was
generated with context `prompt_length + j`. The measured effective context uses
those observed events. It is never replaced by the nominal prompt length.

Timing error must be ≤10% for **each** window. Power MAPE must be ≤10%, maximum
error ≤15%, and repeated power CV ≤10%. A timing/power failure is returned as
`prediction_failed`, even if a later window runs out of tokens. Such a failure
does not trigger automatic fallback or candidate refitting. The legacy aggregate
median fields are retained for compatibility, but acceptance must inspect the
per-window predictions; `requires_per_window_evaluation=True` is explicit.

These are three nonoverlapping measurement windows from one continuously
decoding batch. They are not three independently loaded instances, cold starts,
prompt-prefill repeats or workload seeds. Removing two repeated batch prefills
saves GPU work where the larger context range is covered. The CPU feasibility
report for the current candidates is
`results/2026-09-23/window-sampling-feasibility.json`; it is not GPU qualification
or an elapsed-time promise. In particular, the 32B TP4 high-batch 4096-context
point cannot sustain all three windows inside its current frozen context domain
and retains the old measurement path.
