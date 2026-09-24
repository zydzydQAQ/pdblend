# Compiled PDBlend profile queries

Profiles remain JSON. `PerfModel.load` validates and compiles them into memory;
loaded scalar queries perform no file access, table scan, sorting or binary
search. EcoServe's independent CSV profile is unaffected.

`index.py` builds direct frequency tables and integer context directories.
Each context bucket contains at most four exact breakpoints. `floor(context)`
selects a directory entry; interpolation still uses the original floating point
batch/context and exact knots. Compilation rejects excessive density or a
complete profile index larger than 64 MiB. Integer-valued batches such as `4.0`
work for long tables; `4.5` has no long-profile coverage. Short/long gaps remain
uncovered. The access bound is four knot comparisons regardless of table size.

Construction, JSON/pickle loading and explicit calibration mutations compile
indices. Observable table updates invalidate the old compiled entry before
validation, so a rejected edit cannot silently serve the old curve. `to_json`
stores only profile data, not compiled directories. Legacy module aliases and
pickle paths remain readable.

The O(1) contract covers `PerfModel`, `BoundedVersionModel`, `CompiledPowerTable`
and `CompiledLongTable` scalar methods. The old raw-spec `power_table.predict`
and `context_bounds` helpers remain diagnostic compatibility functions and do
not carry runtime qualification. Use a compiled table for repeated queries.

`versions.load_version` still checks registry identity, candidate identity,
raw sample checksums and the numerical source files in the frozen source tree.
It also compares the consumer's numerics to selected pure functions from that
verified frozen tree. The old source identity is preserved separately from the
current consumer implementation digest. No historical manifest is rewritten.

`RuntimeProfile` defines the planner-facing contract. A component version has
qualified timing/power only: `PoolPlanner` refuses it before planning if its
capacity, static/wake, KV transfer or frequency-transition components are not
qualified. Historical `PerfModel` behavior is retained; loading a new index does
not make historical data or a component profile formally comparable.

Run the CPU benchmark with:

```sh
PYTHONPATH=src /home/pdblend/.venv/bin/python -B scripts/profile/benchmark_query.py \
  --registry results/2026-09-23/calibration-versions-v1/registry.json \
  --registry results/2026-09-23/calibration-incremental-versions-v1/registry.json \
  --out /tmp/query-benchmark.json
```

The 2026-09-23 v3 run covers five real component versions, 10,000 calls per
scalar query. All scalar p95 values are at most 5.932 us. At 16/64/256/1024 power
nodes the compiled p95 is 2.729/2.792/2.762/2.857 us; the old scan reaches
466.528 us at 1024 nodes. Checksummed loading takes 64–285 ms, including raw
verification and numerical compatibility checks. Candidate loading/compilation
under allocation instrumentation takes 2–47 ms. Full numeric index storage is
0.37–0.84 MiB per version; the three incremental versions total 1.62 MiB.

A complete eight-slot planner call on one historical M-only workload has an
identical plan with both query implementations: p95 4.317 ms compiled versus
5.490 ms with the old scan. This is a CPU numerical benchmark, not formal GPU
qualification, a full workload sweep, or an O(1) guarantee for planner search.
Operating-system scheduling can still introduce latency outliers; tests assert
constant access bounds and correctness rather than a noisy CI wall-clock cap.
