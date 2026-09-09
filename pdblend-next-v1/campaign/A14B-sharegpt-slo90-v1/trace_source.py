"""Frozen real ShareGPT development work; CPU-only independent rate traces."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random

import protocol as p

CAMPAIGN = Path(__file__).resolve().parent.parent
SOURCE_SPEC = dict(path=str(CAMPAIGN / "full-matrix-user-slo-v1/spec.json"),
                   sha256="0a46bd8ed64b6575981b7b51d03242eddc1de597a2ca499ca6a8f6984e6b7ef1")
MAX_REQUESTS = 2_000_000


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def frozen(reference):
    path = Path(reference["path"]).resolve()
    p.require(file_sha(path) == reference["sha256"], "frozen source changed: " + str(path))
    return path


def _seal(source):
    return digest({key: value for key, value in source.items() if key != "seal_sha256"})


def verify_source(source):
    p.require(source.get("model") == p.MODEL and source.get("dataset") == p.DATASET,
              "wrong development source model/dataset")
    p.require(source.get("seal_sha256") == _seal(source), "loaded source content/order changed")
    p.require(source["references"][0] == SOURCE_SPEC, "source is not the frozen existing development declaration")
    for reference in source["references"]:
        frozen(reference)


def load_source():
    """Load only the existing 14B ShareGPT development split and frozen order.

    The old module is used exclusively as a strict pool reader. No old matrix,
    rate, minimum-request, phase or execution declaration is invoked.
    """
    spec = json.loads(frozen(SOURCE_SPEC).read_text())
    origin = spec["source_development_spec"]
    original = json.loads(frozen(origin).read_text())
    pool_ref = spec["models"][p.MODEL]["datasets"][p.DATASET]["pool"]
    p.require(pool_ref == original["models"][p.MODEL]["datasets"][p.DATASET]["pool"],
              "development pool differs from its original declaration")
    p.require(spec["allow_repeat_from_dev_pool"] is True and type(spec["sampling_seed"]) is int,
              "original deterministic development resampling declaration required")
    reader_ref = spec["development_pool_reader"]
    reader_path = frozen(reader_ref)
    module_spec = importlib.util.spec_from_file_location("_slo90_frozen_development_pool_reader", reader_path)
    reader = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(reader)
    records, pool = reader.load_pool(pool_ref, model=p.MODEL, dataset=p.DATASET)
    order = list(range(len(records)))
    random.Random(spec["sampling_seed"]).shuffle(order)
    references = [copy.deepcopy(SOURCE_SPEC), copy.deepcopy(origin), copy.deepcopy(reader_ref),
                  dict(path=pool_ref["path"], sha256=pool_ref["sha256"]),
                  dict(path=pool_ref["manifest_path"], sha256=pool_ref["manifest_sha256"])]
    source = dict(model=p.MODEL, dataset=p.DATASET, records=records, pool=pool, order=order,
                  sampling_seed=spec["sampling_seed"], references=references)
    source["seal_sha256"] = _seal(source)
    verify_source(source)
    return source


def window_arrivals(rate):
    value = float(p.number(rate))
    p.require(math.isfinite(value) and value > 0, "rate exceeds finite float representation")
    rng = random.Random(p.SEED)
    unit, arrivals = 0.0, [0.0]
    while True:
        unit += rng.expovariate(1.0)
        arrival = unit / value
        if arrival >= p.WINDOW_S:
            return arrivals
        p.require(len(arrivals) < MAX_REQUESTS, "materialization limit exceeded; no workload truncation permitted")
        arrivals.append(arrival)


def build_trace(rate, source=None):
    source = load_source() if source is None else source
    verify_source(source)
    rate_text = p.number(rate)
    arrivals = window_arrivals(rate_text)
    order = source["order"]
    indices = [order[index % len(order)] for index in range(len(arrivals))]
    records = [source["records"][index] for index in indices]
    requests = [dict(idx=index, arrival_s=arrival, prompt_len=record["input_tokens"],
                     output_len=record["output_tokens"])
                for index, (arrival, record) in enumerate(zip(arrivals, records))]
    content = digest([digest(dict(prompt=r["prompt"], input_tokens=r["input_tokens"],
                                   output_tokens=r["output_tokens"])) for r in records])
    result = dict(schema=2, model=p.MODEL, dataset=p.DATASET, split="development",
                  load="declared_absolute_rate", protocol_id=p.PROTOCOL, measurement_schema=3,
                  seed=p.SEED, arrival_seed=p.SEED, sampling_seed=source["sampling_seed"],
                  rate=float(rate_text), rate_rps=float(rate_text), rate_rps_decimal=rate_text,
                  duration_s=p.WINDOW_S, arrival_window_s=p.WINDOW_S,
                  fixed_observation_window_required_s=p.WINDOW_S,
                  requests=requests, prompts=[copy.deepcopy(r["prompt"]) for r in records],
                  n_requests=len(requests), source_shapes=[r["request_shape_sha256"] for r in records],
                  source_pool_indices=indices, pool=copy.deepcopy(source["pool"]),
                  slo=dict(ttft_s=5.0, tpot_s=0.15, attainment_target=0.9),
                  slo_protocol="per-dataset-slo-v1", allowed_slo_scales=list(p.SCALES),
                  comparison_systems=list(p.SYSTEMS), execute_baselines=True,
                  request_hard_timeout_s=p.REQUEST_TIMEOUT_S,
                  post_window_drain_allowance_s=p.DRAIN_S,
                  planned_arrival_span_s=arrivals[-1], content_pairing_sha256=content,
                  source_indices_sha256=digest(indices), source_references=copy.deepcopy(source["references"]),
                  source_seal_sha256=source["seal_sha256"],
                  arrival_process="first arrival zero; cumulative random.Random(701).expovariate(1)/rate strictly before 100s",
                  content_pairing="one exact trace shared by all systems and both scales at the same rate",
                  unique_selected_pool_records=len(set(indices)),
                  unique_source_shapes=len({r["request_shape_sha256"] for r in records}),
                  within_trace_resampling=len(indices) > len(set(indices)),
                  repeat_from_existing_dev_pool=True, output_lengths_modified=False,
                  prompts_truncated=False, minimum_requests=None, minimum_planned_span_s=None,
                  sparse_screen=len(requests) < 30, formal_eligible=False,
                  purpose="independent paired PDB SLO90 endpoint sweep; historical execution results not reused")
    verify_source(source)
    return result


def execution_row(trace_ref, system, scale):
    """Describe one potential execution. Selection is the runner's responsibility."""
    p.require(system in p.SYSTEMS, "unknown comparison system")
    key = p.scale_key(scale)
    path = frozen(trace_ref)
    trace = json.loads(path.read_text())
    p.require(trace["protocol_id"] == p.PROTOCOL and trace["model"] == p.MODEL
              and trace["dataset"] == p.DATASET and trace["seed"] == p.SEED
              and trace["arrival_window_s"] == p.WINDOW_S, "wrong trace protocol/workload")
    rate = p.number(trace["rate_rps_decimal"])
    workload = f"14b-sharegpt-r{rate}-s701-w100"
    slo = p.effective_slo(scale)
    return dict(cell_id=f"{workload}-{system}-slo{key}", workload_id=workload,
                protocol_id=p.PROTOCOL, measurement_schema=3, model=p.MODEL, dataset=p.DATASET,
                system=system, strategy=None, controller_config=None, policy_binding_required=True,
                phase="scale", part="scale", split="development", load="declared_absolute_rate",
                seed=p.SEED, arrival_seed=p.SEED, rate_rps=float(rate), rate_rps_decimal=rate,
                trace=str(path), trace_path=str(path), trace_sha256=trace_ref["sha256"],
                trace_bytes=path.stat().st_size, n_requests=trace["n_requests"],
                arrival_window_s=p.WINDOW_S, trace_duration_s=p.WINDOW_S,
                planned_arrival_span_s=trace["planned_arrival_span_s"],
                content_pairing_sha256=trace["content_pairing_sha256"],
                source_indices_sha256=trace["source_indices_sha256"],
                slo_scale=float(key), slo_ttft_s=slo["ttft_s"], slo_tpot_s=slo["tpot_s"],
                slo_attainment_target=0.9, slo_protocol="per-dataset-slo-v1",
                reuse_main_cell_id=None, sparse_screen=trace["sparse_screen"],
                formal_eligible=False, materialized=True)


def materialize(rate, out, source=None):
    """Write a new immutable directory with one trace and ten potential rows.

    These rows do not authorize measurements beyond either scale's PDB endpoint.
    The caller chooses exact rows using protocol.required_baseline_rates().
    """
    source = load_source() if source is None else source
    trace = build_trace(rate, source)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    path = out / "trace.json"
    with path.open("xb") as handle:
        handle.write(encode(trace))
    trace_ref = dict(path=str(path), sha256=file_sha(path))
    cells = [execution_row(trace_ref, system, scale) for system in p.SYSTEMS for scale in p.SCALES]
    for sequence, row in enumerate(cells, 1):
        row["sequence"] = sequence
    generators = [dict(path=str(Path(module_path).resolve()), sha256=file_sha(module_path))
                  for module_path in (__file__, p.__file__)]
    inputs = [*source["references"], *generators, trace_ref]
    result = dict(schema=1, kind="potential_paired_rate_cells", protocol_id=p.PROTOCOL,
                  model=p.MODEL, dataset=p.DATASET, trace=trace_ref,
                  rate_rps_decimal=p.number(rate), source_references=copy.deepcopy(source["references"]),
                  source_seal_sha256=source["seal_sha256"],
                  generator_references=generators,
                  files={reference["path"]: reference["sha256"] for reference in inputs},
                  cells=cells, all_rows_execution_required=False,
                  selection_rule="only PDB pending scale or baseline rates through its scale-specific PDB endpoint")
    verify_source(source)
    with (out / "manifest.json").open("xb") as handle:
        handle.write(encode(result))
    return result
