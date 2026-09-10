"""Materialize one immutable 14B/ShareGPT workload at a time; CPU only.

Existing uniform-v2 rates retain exact trace bytes, including historical
embedded references. New rates use that campaign's same pinned generators and
source specification, then share one trace across both nodes and all systems.
"""
from __future__ import annotations

import argparse
import copy
from decimal import Decimal
import fcntl
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_slo_rate_generation_contract", HERE / "contract.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
CAMPAIGNS = HERE.parent
UNIFORM_DECLARATION = dict(
    path=str(CAMPAIGNS / "parallel-rate-20260908-v1/common/uniform-rate-20260909-v2/release-002/declaration.json"),
    sha256="1c23e9e5b97887219b4879e095ed9944d8b0d9c0be1a94fe10e63079ca5d8101")
GENERATOR = dict(path=str(CAMPAIGNS / "five-system-fixed-window-v1/generate.py"),
                 sha256="d554c26f9d9608d8ce443720eb21cb4129d6d209819282df3c5c154274fa5b4d")


def write_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        c.need(path.read_bytes() == payload, "immutable workload differs: " + str(path))
    else:
        with path.open("xb") as stream:
            stream.write(payload)


def freeze_inputs():
    """Freeze the verified 14B/ShareGPT input group for standalone node staging.

    Historical reference paths embedded in the group/spec/trace stay untouched;
    they are provenance, not runtime dependencies of the frozen build_trace.
    """
    inputs = HERE / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    with (inputs / ".freeze.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = inputs / "manifest.json"
        if target.exists():
            manifest = c.read(target)
            for reference in manifest["dependencies"].values():
                c.need(c.sha(reference["path"]) == reference["sha256"], "frozen generation dependency changed")
            for old in manifest["known_workloads"].values():
                c.need(c.sha(old["local_trace"]["path"]) == old["local_trace"]["sha256"], "known trace changed")
            return manifest
        generator, parent, source_ref, source_spec, groups, sampling_seed, old = _original_sources()
        write_once(inputs / "generator100.py", Path(GENERATOR["path"]).read_bytes())
        write_once(inputs / "generator300.py", Path(generator.PARENT_GENERATOR).read_bytes())
        write_once(inputs / "source-spec.json", Path(source_ref["path"]).read_bytes())
        # Only fields consumed by the frozen build_trace are needed. The old
        # loader also carries an obsolete Decimal rate grid; do not inherit it.
        group = groups[("14b", "sharegpt")]
        group = {key: group[key] for key in ("order", "records", "record_digests", "strategy", "pool", "policy")}
        write_once(inputs / "source-group.json", c.encode(group))
        known = {}
        for rate, workload in old.items():
            reference = dict(path=workload["trace_path"], sha256=workload["trace_sha256"])
            c.checked(reference)
            local = inputs / "known-traces" / ("r" + rate + ".json")
            write_once(local, Path(reference["path"]).read_bytes())
            known[rate] = dict(original_trace=reference, local_trace=c.ref(local),
                               content_pairing_sha256=workload["content_pairing_sha256"])
        manifest = dict(schema="frozen-sharegpt-generation-inputs-v1", campaign_id=c.CAMPAIGN,
            uniform_declaration=UNIFORM_DECLARATION, source_spec=source_ref,
            original_generator=GENERATOR, original_parent_generator=c.ref(generator.PARENT_GENERATOR),
            sampling_seed=sampling_seed, model="14b", dataset="sharegpt",
            dependencies={name: c.ref(inputs / filename) for name, filename in
                (("generator100", "generator100.py"), ("generator300", "generator300.py"),
                 ("spec", "source-spec.json"), ("group", "source-group.json"))},
            known_workloads=known,
            source_group_provenance="output of the frozen parent.load_sources after full pool identity verification",
            standalone=True, old_results_required_on_execution_nodes=False)
        write_once(target, c.encode(manifest))
        return manifest


def _original_sources():
    declaration = c.checked(UNIFORM_DECLARATION)
    c.need(c.sha(GENERATOR["path"]) == GENERATOR["sha256"], "frozen generator changed")
    generator = c.load_module(GENERATOR["path"], "_slo_rate_frozen_window_generator")
    parent = generator.parent_generator()
    source_ref = declaration["source_specs"]["14b"]
    source_spec = c.checked(source_ref)
    groups, sampling_seed = parent.load_sources(source_spec)
    c.need(sampling_seed == c.SAMPLING_SEED, "frozen sampling seed changed")
    old = {c.number(w["rate_rps"]): w for w in declaration["workloads"]
           if (w["model"], w["dataset"]) == ("14b", "sharegpt")}
    return generator, parent, source_ref, source_spec, groups, sampling_seed, old


def _sources():
    manifest = freeze_inputs()
    dependencies = manifest["dependencies"]
    generator = c.load_module(dependencies["generator100"]["path"], "_slo_rate_packaged_window_generator")
    # The copied source is byte-identical; only its module-local resource path
    # changes. build_trace/prefix_trace and embedded original references do not.
    generator.PARENT_GENERATOR = Path(dependencies["generator300"]["path"])
    parent = generator.parent_generator()
    spec = c.checked(dependencies["spec"])
    group = c.checked(dependencies["group"])
    old = {rate: dict(trace_path=value["local_trace"]["path"],
                     trace_sha256=value["local_trace"]["sha256"],
                     original_trace=value["original_trace"])
           for rate, value in manifest["known_workloads"].items()}
    return generator, parent, manifest["source_spec"], spec, {("14b", "sharegpt"): group}, manifest["sampling_seed"], old


def _validate_cached(workload, directory, rate):
    c.need(workload["campaign_id"] == c.CAMPAIGN and c.number(workload["rate_rps"]) == rate,
           "cached workload identity differs")
    c.need(Path(workload["trace_reference"]["path"]) == directory / "trace.json", "cached trace escaped campaign")
    trace = c.checked(workload["trace_reference"])
    c.need(c.sha(directory / "trace.json") == workload["trace_sha256"], "cached trace hash differs")
    c.need(trace["content_pairing_sha256"] == workload["content_pairing_sha256"], "cached content hash differs")
    c.need(trace["n_requests"] == workload["n_expected"], "cached request count differs")
    manifest = c.checked(workload["materialization_manifest"])
    payload = dict(workload)
    payload.pop("materialization_manifest")
    c.need(hashlib.sha256(c.encode(payload)).hexdigest() == manifest["workload_payload_sha256"],
           "cached workload payload changed")
    c.checked(workload["source_300s_local"])
    return workload


def materialize_rate(rate, *, out_root=None):
    """Idempotently materialize an exact grid rate under this campaign only.

    out_root exists for isolated CPU tests. Production callers omit it.
    The advisory lock prevents two node dispatchers from racing on shared traces.
    """
    rate = c.number(rate)
    output = Path(out_root).resolve() if out_root is not None else HERE / "workloads"
    output.mkdir(parents=True, exist_ok=True)
    directory = output / ("r" + rate)
    directory.mkdir(exist_ok=True)
    with (directory / ".materialize.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index = directory / "workload.json"
        if index.exists():
            return _validate_cached(c.read(index), directory, rate)
        generator, parent, source_ref, source_spec, groups, sampling_seed, old = _sources()
        _, original = parent.build_trace(source_spec, "14b", "sharegpt", Decimal(rate), c.SEED,
                                         groups[("14b", "sharegpt")], sampling_seed)
        original_bytes = parent.encode(original)
        source_path = directory / "source300.json"
        write_once(source_path, original_bytes)
        original_local = c.ref(source_path)
        if rate in old:
            old_workload = old[rate]
            trace_reference = dict(path=old_workload["trace_path"], sha256=old_workload["trace_sha256"])
            trace = c.checked(trace_reference)
            original_reference = trace["source_300s_trace"]
            c.need(original_local["sha256"] == original_reference["sha256"], "frozen 300s trace differs")
            c.need(generator.prefix_trace(original, original_reference) == trace, "old prefix differs from frozen sampler")
            trace_bytes = Path(trace_reference["path"]).read_bytes()
            preserved = dict(original_trace=old_workload["original_trace"], original_source_300s=original_reference,
                             exact_local_source_300s=original_local,
                             baseline_scale1_trace_sha_equal=True)
        else:
            original_reference = original_local
            trace = generator.prefix_trace(original, original_reference)
            trace_bytes = generator.encode(trace)
            preserved = dict(original_trace=None, original_source_300s=original_reference,
                             exact_local_source_300s=original_local,
                             baseline_scale1_trace_sha_equal=None)
        trace_path = directory / "trace.json"
        write_once(trace_path, trace_bytes)
        c.need((trace["arrival_window_s"], trace["request_hard_timeout_s"], trace["seed"])
               == (100, 120, 701), "trace time or seed contract changed")
        c.need((trace["slo"]["ttft_s"], trace["slo"]["tpot_s"]) == (5, .15), "trace base SLO changed")
        workload = dict(campaign_id=c.CAMPAIGN, protocol_id=c.PROTOCOL,
            model="14b", dataset="sharegpt", workload_id=f"14b-sharegpt-r{rate}-s701-w100",
            rate_rps=float(rate), rate_rps_decimal=rate, seed=c.SEED, arrival_seed=c.SEED,
            sampling_seed=sampling_seed, arrival_window_s=100., trace_duration_s=100.,
            request_hard_timeout_s=120., drain_after_arrival_window_s=120.,
            trace=str(trace_path), trace_path=str(trace_path), trace_reference=c.ref(trace_path),
            trace_sha256=c.sha(trace_path), trace_bytes=len(trace_bytes),
            content_pairing_sha256=trace["content_pairing_sha256"],
            n_requests=trace["n_requests"], n_expected=trace["n_requests"],
            expected_generated_tokens=sum(q["output_len"] for q in trace["requests"]),
            planned_arrival_span_s=trace["planned_arrival_span_s"],
            source_indices_sha256=hashlib.sha256(c.encode(trace["source_pool_indices"])).hexdigest(),
            source_300s_trace=original_reference, source_300s_local=original_local,
            source_spec=source_ref, generator=GENERATOR,
            parent_generator=freeze_inputs()["original_parent_generator"],
            standalone_generation_inputs=c.ref(HERE / "inputs/manifest.json"),
            base_slo=copy.deepcopy(trace["slo"]), slo=copy.deepcopy(trace["slo"]),
            measurement_schema=3, slo_protocol="per-dataset-slo-v1", load="declared_absolute_rate",
            split="development", materialized=True, sparse_screen=trace["n_requests"] < 30,
            within_trace_resampling=trace["within_trace_resampling"],
            output_lengths_modified=False, prompts_truncated=False, formal_eligible=False,
            request_count_warning="one arrival seed; no independent-seed confidence interval",
            trace_slo_policy="immutable trace keeps base thresholds; executed row/config supply node scaled thresholds")
        manifest = dict(schema="slo-rate-workload-materialization-v1", campaign_id=c.CAMPAIGN,
            rate_rps_decimal=rate, uniform_declaration=UNIFORM_DECLARATION,
            frozen_generator=GENERATOR, source_spec=source_ref, preserved=preserved,
            trace=c.ref(trace_path), source300=original_local,
            workload_payload_sha256=hashlib.sha256(c.encode(workload)).hexdigest(),
            old_release_modified=False, generated_requests_reordered=False)
        write_once(directory / "manifest.json", c.encode(manifest))
        workload["materialization_manifest"] = c.ref(directory / "manifest.json")
        write_once(index, c.encode(workload))
        return _validate_cached(workload, directory, rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rates", nargs="+", help="positive exact multiples of 0.25 rps")
    args = parser.parse_args()
    for value in args.rates:
        w = materialize_rate(value)
        print(c.encode({k: w[k] for k in ("rate_rps", "n_requests", "trace_reference", "content_pairing_sha256")}).decode(), end="")
