# New-stack measurement adapter boundary

`measurement_v1.py` is the independent baseline measurement boundary for
vLLM 0.10.1.1. It does not import `pdblend`, `PerfModel`, the shared router, or
the old vLLM 0.9.2 runtime.

`detect_vllm_v1_capabilities()` returns a structured receipt. A missing vLLM
installation, missing V1 worker API, version mismatch, missing TP/PP group API,
or unavailable CUDA is `supported=false`; callers must record the receipt as
`unsupported_engine` and may not create a profile point.

`NativeStageCollector` wraps the actual V1
`worker.model_runner.execute_model` and uses CUDA events around that call. It
requires native `is_prompt`, sequence lengths, input-token count, TP rank and PP
rank. Its `gpu_elapsed_ms` includes the local model runner and TP collectives;
it excludes PP wire/queue time. Endpoint TTFT is never used as a stage sample.
`reduce_native_stage_samples()` rejects missing physical ranks, unequal TP sample
counts, and unknown P/D role.

`EcoServeForwardMeter` measures actual full forward CUDA events and writes the
author CSV shape (`Length,Prefill Time`). Each length requires at least five
measurements and the CSV requires 16 and 4096 anchors. It writes the exact
minimum over observed samples, as the independent EcoServe profile core expects.

`DynamoShapeProfileBuilder` requires a declared rectangular grid over model,
TP, frequency, input length, context length and batch. Every cell must be an
actual measured row with a source hash; missing cells prevent a complete profile.
`DynamoLoadProfileBuilder` writes timestamped arrivals separately and never
fills shape cells from load history. `DynamoEndToEndMeasurement` (or
`MixedEndToEndMeasurement`) records native first/last-token timestamps for
Mixed or Dynamo replay, but remains
`hardware_qualified=false` until the campaign's GPU receipt gate passes.

`mixed_policy.py` provides the independent fixed-TP least-load policy. It has no
dynamic TP, P/D, or re-sharding behavior.
