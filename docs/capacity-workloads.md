# Independent capacity workload preparation

`pdblend.bench.capacity_workloads` prepares input files only. It never runs GPU
work, changes the queue, emits measurement receipts, or claims a measured capacity.
The current comparison executor/acceptance protocol still uses evaluation,
seed 701 and 150 seconds. Capacity execution integration is pending; prepared
assignments have `execution_ready: false` and `execution_wiring_pending: true`.

A family binds one model/dataset, prepared corpus bytes, calibration or tuning
split, completed independent calibration/tuning rate anchor, standard dataset
SLO, three explicit seeds, systems and minimum scale. Anchor confirmation requests
are replayed from tuning before acceptance. A completed LongBench recovery may
retain Alpaca/ShareGPT request files at their original location: the reader follows
only `prior_inputs.inherited` path/hash bindings, checks the original clean partial
receipt and identical confirmation bytes, then performs the same full replay.
It never searches for similarly named files. Seed 701, the anchor's two seeds,
and evaluation splits cannot be used as capacity repetition seeds.

Before any capacity measurement, duration starts at the declared base window and
doubles until **each** seed at the declared minimum scale offers at least 100
requests. Only arrival counts determine this window. Every rate and every system
in that family then uses the same duration; all systems at the same rate/seed use
identical prompt/output/arrival bytes. Lowering the minimum scale requires a new
family, rather than silently extending one system's window. Families are separate
for each model/dataset, so their windows may differ.

```sh
PYTHONPATH=src python3 -m pdblend.bench.capacity_workloads prepare-family \
  --out /absolute/new-family --corpus /absolute/prepared-corpus \
  --dataset longbench --split tuning --anchor /absolute/anchor/completion.json \
  --seed 8801 --seed 8802 --seed 8803 --minimum-scale .25 --base-duration 150
PYTHONPATH=src python3 -m pdblend.bench.capacity_workloads prepare-rate \
  --family /absolute/new-family/family.json --out /absolute/new-rate-x1 --rate-scale 1
PYTHONPATH=src python3 -m pdblend.bench.capacity_workloads validate-trace \
  --trace /absolute/new-rate-x1/seed-8801.json
```

Existing family/rate output directories are never overwritten. Each family freezes
source copies with SHA256 bindings. The v1 reader compares the frozen pure source
with its known v1 bytes, then calls trusted versioned builtins; it never executes
artifact Python. Later unrelated workspace edits do not invalidate old families.
The recorded Python version must match, because v1 uses its standard-library RNG.
A new RNG algorithm/interpreter needs explicit replay-version support.

The Python interfaces are:

- `prepare_family(out, *, corpus, dataset, split, anchor, seeds, minimum_scale,
  base_duration_s=150., minimum_requests=100, systems=SYSTEMS)` returns
  `{manifest: {path, sha256}, family}`. `anchor` is also an absolute path/hash binding.
- `prepare_rate(family_ref, out, *, rate_scale)` returns `{manifest, workloads}`.
  `workloads.traces[i].trace` is a trace binding; assignments share these same files.
- `validate_trace(trace_ref)` performs read-only full replay and returns
  `{family, trace, family_ref}`. It verifies corpus/anchor/generator bindings,
  common-duration selection, and exact canonical request/cohort metadata.

Trace and assignment records carry `capacity_workload_family: {path, sha256}`.
Assignments also bind it under `inputs`, alongside `inputs.trace`. Consumers must
keep family ID, repeat ID, rate/scale, duration, split, SLO, model/tokenizer identity,
protocol and output workload consistent. The capacity receipt reader applies
those bindings before the capacity state machine consumes actual measurements.
No evaluation performance is read to choose the family, window or rate grid.
