# Native runtime collection on the existing PD fleet

`await collect_runtime(specs, fleet, meter, sampler, out, gpu_uuids=uuids)` is
provided by `pdblend.profile.collection.native_runtime_collect`. Pass the
existing running eight-device `PowerSampler`, and its owning `Gpus` object.
The collector neither starts nor stops the sampler. It rejects averaged power,
different UUID order/backend, inactive sampling, non-native32 specs, and a fleet
that does not cover the eight-device lease exactly once.

The independent output directory must not exist. It contains the frozen
`runtime-plan.json`, append-only `journal.jsonl`, `power.json`, `audit.json` and
`completion.json`, with SHA bindings. Keep it under the actual host attempt
path. No evaluation requests are used. Native launch arguments retain the
actual `PDNativeTimingWorker` extension identity.

Capacity covers every instance. Static, L1, off/wake, and clock actions use the
first homogeneous instance; remaining instances stay resident. Transfer uses
the first two symmetric instances through PD's EngineClient/P2P path. There
are three training repetitions and a fourth predeclared holdout, each with
two-second settling and at least five seconds of measurement. Transfer uses
seed 9701 for training and 9702 for holdout at 512, 2048, 7168 input tokens.
Four measured off/wake cycles therefore restart only the representative four
times. Actual Fleet start receipts and counts are retained separately from
service energy.

Off requires an owned stop, dead process, and real NVML compute-empty evidence.
Wake restores generation and compares ordinary exact token IDs with two
pre-off references. Transfer stores prompt/token/usage/time journals, compares
two ordinary references with the carried-token PD output, and measures the
same second logical output position. This is HTTP/scheduling-inclusive handoff
overhead, not a claim of physical-copy time. Native clock ACK power fields use
the endpoint's legacy API and are never integrated; energy comes exclusively
from the supplied instant sampler.

`safe_restore_passed` covers final inventory, native generation, 2520 MHz,
all-rank measurement-stop ACKs, native drain and a live bracketing sampler.
`ready_for_timing` additionally requires execution of all requested collection
phases. A runtime exception blocks later timing even after successful cleanup.
A numerical holdout failure or raw power coverage gap leaves the completed
observations intact and does not by itself block unrelated timing.

The replay computes absolute eight-board and per-device energy with the
unchanged comparison reducer and its one-second gap ceiling. It never calls
absolute board energy incremental transition energy. Training-only medians
predict the held-out observation; frozen mean/p95/max relative error limits
are 0.10/0.20/0.25, matching the native timing supplement. Zero held-out values
leave relative error undefined and fail this gate. `runtime_holdout_passed`
is distinct from `component_qualified`, `formal_eligible` and
`energy_comparable`, which stay false here.

The next required steps are CPU replay/composition and explicit compatibility
of the recorded profiling worker with the chosen execution worker. The
extension overrides only `_native_shape`, adds descriptive arrays, and runs
before the inherited CUDA timing start; it does not alter runner execution or
the CUDA timing boundaries. This supports a narrow source-continuity review
for CUDA timing, capacity and idle/clock mechanisms. End-to-end transfer and
startup costs retain the profiling worker identity; they must not be relabeled
as ordinary NativeWorker observations without that review. A true hardware
error, holdout miss, unsupported query domain or incompatible execution scope
remains a concrete blocker, not a flag to overwrite.
