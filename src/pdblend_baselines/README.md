# Independent baseline CPU cores

These modules are copied byte-for-byte from the local independent implementations
in `/home/pdblend/src/pdblend_baselines`. `migration-manifest.json` records every
copied core's SHA256. Package initializers expose only the migrated cores and do
not import the old execution platform or any `pdblend` strategy/profile module.

- DistServe: its own TP/PP enumeration, placement/goodput search, measured stage
  latency surface, original SimPy worker/scheduler/request loop, and stage timers.
- DynamoLLM: its own measured profile interpolation, nine shape pools, MILP,
  1800/300/5-second hierarchy, model-bound BERT predictor, and acknowledged
  reconfiguration/rollback/quarantine state machine.
- EcoServe: its own CSV prefill profile, saved-TPOT admission, macro rotation,
  block rounding, and output hold/flush behavior.

Install the CPU dependencies (`numpy`, `simpy==4.1.1`, `scipy==1.15.3`,
`PuLP==3.2.2`) and run `python -m pdblend_baselines.cpu --model all`.
The command validates source hashes, searches all three model geometries, and
executes synthetic algorithm contracts. It returns `hardware_qualified=false`
and `energy_comparable=false`; its synthetic inputs must never become calibration.

Predictor training/inference additionally requires Torch/Transformers, local
verified BERT assets and a separately trained model/tokenizer-bound checkpoint.
The predictor validates its target model, tokenizer, calibration split and file
hashes. No old 7B checkpoint or measured profile is bundled.

The stage profiling hooks are retained as adaptation source, not claimed to be
compatible with the vLLM 0.10.1.1 V1 worker. The current new-stack hardware adapter
must provide actual stage execution, KV and cancellation receipts for DistServe;
weight movement, lifecycle and rank/generation receipts for DynamoLLM; and native
scheduler state, split/merge and live KV continuity for EcoServe. No old GPU
runtime was imported. CPU tests cannot qualify these hardware mechanisms.

DistServe and EcoServe Apache-2.0 licenses and immutable upstream reference
manifests are under their respective directories. Their source headers retain
the original attribution. DynamoLLM is a local paper reimplementation, not an
author artifact; no claim of an author-provided checkpoint is made.
