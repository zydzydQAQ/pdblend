# Online incremental-energy routing artifacts

`pdblend.online.energy_routing.load_energy_estimator(path, router=resident_router)`
loads one explicit artifact. Enable it through
`router.configure_energy_routing(slo=slo, estimator=estimator)`. Loading an
ordinary affine power profile, setting `qualified: true`, or choosing a
directory named `latest` cannot qualify incremental-energy routing.

The manifest is JSON:

```json
{
  "schema": "pdblend.incremental-route-energy/v1",
  "system": "pdblend",
  "training": {"path": "training.json", "sha256": "<SHA256 of exact file bytes>"},
  "holdout": {"path": "holdout.json", "sha256": "<SHA256 of exact file bytes>"}
}
```

Both components are JSON arrays of raw paired observations. Paths are relative
to the manifest, and both file contents are verified before use. Each row has
this form (numbers below illustrate the schema, not measured hardware data):

```json
{
  "sample_id": "train-tp1-0",
  "request_id": "request-train-tp1-0",
  "measurement": "paired_common_window",
  "power_source": "nvml_total_energy_counter",
  "gpu_uuids": ["GPU-example"],
  "key": {
    "model_id": "Qwen2.5-7B-Instruct",
    "tp": 1,
    "pp": 1,
    "path": "M",
    "profile_keys": ["explicit-approved-profile-key"],
    "frequencies_mhz": {"M": 1500},
    "input_tokens": 512,
    "max_tokens": 32,
    "batch": 1,
    "context_tokens": 544,
    "queued_prefill_tokens": 0,
    "reservation_tokens": {"M": 0}
  },
  "baseline": {"start_s": 1000, "end_s": 1001, "energy_j": 100},
  "with_request": {"start_s": 1002, "end_s": 1003, "energy_j": 110},
  "completion_tokens": 32,
  "error": null,
  "ttft_s": 0.1,
  "tpot_s": 0.02
}
```

For PD, `path` is `PD`; `profile_keys` has prefill then decode keys, frequency
and reservation maps contain `P` and `D`, and GPU UUIDs cover both TP groups.
Instantaneous power integration is accepted using the actual sampler identity
object: `{"mode":"instant","source_id":"nvml:field:186:scope:0:mW","field_id":186,"scope_id":0,"unit":"W"}`.
Legacy averaged or unidentified power sources are rejected. The collector must
run the same background workload in the baseline and additional-request
windows. Each pair uses equal-duration, disjoint windows and meters every GPU
in the route. Its incremental energy is recomputed as the difference of those
integrals. No decode-power estimate is substituted for a Mixed measurement.

Each exact key requires at least three distinct training requests and three
independent holdout requests, with distinct sample IDs, disjoint windows, the
same physical GPU identities, and complete requested output. Every holdout
window must be within 10% of the training mean incremental joules. Each workload
must have at least two alternative routes, and their ordering must agree in the
independent holdout. TTFT and TPOT maxima from those holdouts also constrain
online admission. The file hashes bind the evidence; operators must still
ensure the collector's native power source and sample provenance are genuine.

The estimator admits exact measured model/TP/PP/profile/frequency/path/batch/
context/input/output/queue/reservation domains. It does not interpolate.
Missing energy coverage for any otherwise feasible route causes the entire
choice set to use latency scores, keeping joules and seconds out of the same
ranking. KV capacity, model coverage and SLO screening run first. Actual
frequency and queue state are recorded in each route estimate.

The artifact establishes a bounded functional routing model. It does not
establish formal GPU qualification, a whole-system energy improvement, or safe
online TP resharding. Stage comparisons still require the controlled acceptance
checks described in `optimization-acceptance.md`.

Tests: `tests/pdblend/test_energy_routing_artifact.py` demonstrates a complete
synthetic publisher and rejection of tampering, affine power, reused windows,
wrong devices, incomplete output, ranking reversals and measured SLO failures.
`tests/pdblend/test_resident_energy.py` covers fallback, actual clocks, queues,
capacity and allocation feedback. These fixtures are explicitly CPU evidence.
