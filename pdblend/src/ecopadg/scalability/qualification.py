"""Real resident-engine mechanism checks; this is not an operator reprofiler.

Run only on explicitly prepared TP1 engines. Missing original profile evidence
remains a formal blocker even when these small numerical probes succeed.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
import socket
import time
import uuid

import aiohttp

from ecopadg.measure.backends import PynvmlBackend
from ecopadg.measure.power import PowerSampler, trapezoid_energy
from ecopadg.serving.backend import ClockOwner
from ecopadg.serving.campaign import node_lease
from ecopadg.serving.cli_async import cleanup_timeout, run_async_cli
from ecopadg.serving.interconnect import InterconnectTopology

from .artifacts import object_hash, read_json, sha256, write_json
from .preflight import (REQUIRED_CHECKS, check_profile_inputs, engine_identity,
    inspect_engines, quiescent, measured_source_errors, read_live_topology, topology_errors)

FREQUENCIES = (900, 1500, 2100, 2520)
CONTROL_FIELDS = ("role", "mode", "admit_prefill", "admit_decode")
PROMPT = ([9707, 1879, 13] * 22)[:64]
BODY = dict(prompt=PROMPT, max_tokens=32, temperature=0, top_p=1., seed=0,
            ignore_eos=True, stream=False)


def representative_pairs(instances, topology):
    """One actual ordered pair per physical link class in each direction."""
    groups = {}
    ordered = sorted(instances, key=lambda i: i["gpus"][0])
    for source in ordered:
        for target in ordered:
            if source is target:
                continue
            cls = topology.link_class(source["gpus"], target["gpus"])
            direction = "ascending" if source["gpus"][0] < target["gpus"][0] else "descending"
            groups.setdefault((cls, direction), (source, target))
    return [dict(link_class=cls, direction=direction, source=source, target=target)
            for (cls, direction), (source, target) in sorted(groups.items())]


def active_frequency_pass(samples, requested, *, settle_s=.3):
    active = [s for s in samples if s.get("engine_active") and s.get("clock_idle") is False]
    settled = [s for s in active if s["elapsed_s"] >= settle_s]
    return (len(settled) >= 2 and
            all(abs(s["observed_mhz"] - requested) <= 15 for s in settled))


class RestoreBackend:
    """Let ClockOwner restore the explicitly declared baseline before unlocking."""
    def __init__(self, hardware, targets):
        self.hardware, self.targets = hardware, targets

    def __getattr__(self, name):
        return getattr(self.hardware, name)

    def reset_clock(self, gpu):
        target = self.targets[str(gpu)]
        if target is None:
            self.hardware.reset_clock(gpu)
        else:
            self.hardware.set_clock(gpu, target)


class QualificationClient:
    def __init__(self, session, raw):
        self.session, self.raw = session, raw
        self.owned = {}

    async def call(self, instance, path, body=None, rid=None):
        if path == "/v1/completions":
            rid = rid or "qual-" + uuid.uuid4().hex
            self.owned[rid] = instance
        event = dict(instance_id=instance["id"], path=path, started_s=time.time(),
                     body=copy.deepcopy(body), request_id=rid)
        self.raw["rpc"].append(event)
        kwargs = dict(json=body) if body is not None else {}
        if rid:
            kwargs["headers"] = {"X-Request-Id": rid}
        method = self.session.post if body is not None else self.session.get
        try:
            async with method(instance["url"] + path, **kwargs) as response:
                event["status"] = response.status
                text = await response.text()
                if response.status != 200:
                    raise RuntimeError(f"{path}: HTTP {response.status}: {text[:1000]}")
                result = json.loads(text)
                event["response"] = result
                return result
        except BaseException as exc:
            event["error"] = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            event["finished_s"] = time.time()
            if path == "/v1/completions" and "response" in event:
                self.owned.pop(rid, None)

    async def control(self, instance, **changes):
        state = await self.call(instance, "/runtime")
        if state.get("diagnostic_recompute") or state.get("diagnostic_transport"):
            raise ValueError("diagnostic transport/recomputation cannot qualify production execution")
        payload = {k: state.get(k, True if k == "admit_decode" else None) for k in CONTROL_FIELDS}
        payload.update(changes, generation=state["generation"] + 1)
        await self.call(instance, "/control", payload)
        deadline = time.monotonic() + 5
        while True:
            actual = await self.call(instance, "/runtime")
            if (all(actual.get(k) == payload[k] for k in CONTROL_FIELDS)
                    and actual.get("generation") == payload["generation"]
                    and actual.get("acknowledged_generation") == payload["generation"]):
                return actual
            if time.monotonic() >= deadline:
                raise RuntimeError("native control was not acknowledged")
            await asyncio.sleep(.02)

    async def drained(self, instance, *, native=False):
        deadline = time.monotonic() + cleanup_timeout(10)
        while True:
            state = await self.call(instance, "/runtime")
            if quiescent(state):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("native queues/KV/transport did not drain")
            await asyncio.sleep(.02)
        if native:
            response = await self.call(instance, "/drain", dict(expected_generation=state["generation"]))
            if response.get("drained") is not True:
                raise RuntimeError("native drain did not acknowledge completion")
            state = await self.call(instance, "/runtime")
            if not quiescent(state) or state.get("generation") != state.get("acknowledged_generation"):
                raise RuntimeError("native state is not drained after drain acknowledgement")
        return state

    async def cleanup_owned(self):
        errors = []
        for rid, instance in list(self.owned.items()):
            try:
                await asyncio.wait_for(self.call(instance, "/cancel", dict(request_id=rid)), cleanup_timeout(3))
                self.owned.pop(rid, None)
            except BaseException as exc:
                errors.append(instance["id"] + "/" + rid + ": " + repr(exc))
        if errors:
            raise RuntimeError("owned request cleanup: " + "; ".join(errors))


def output_matches(output, reference=None, count=32):
    tokens = output.get("token_ids")
    return (isinstance(tokens, list) and len(tokens) == count
            and all(type(t) is int for t in tokens)
            and output.get("usage", {}).get("completion_tokens") == count
            and (reference is None or tokens == reference))


async def frequency_probe(client, instance, hardware, clocks, frequency):
    await clocks.set(instance["gpus"], frequency, verify_rise=False)
    row = dict(instance_id=instance["id"], gpu=instance["gpus"][0], requested_mhz=frequency,
               started_s=time.time(), samples=[], passed=False)
    task = asyncio.create_task(client.call(instance, "/v1/completions", BODY))
    try:
        while not task.done():
            state = await client.call(instance, "/runtime")
            observed, idle = await asyncio.to_thread(lambda: (
                hardware.current_freq(row["gpu"]), hardware.clock_idle(row["gpu"])))
            row["samples"].append(dict(elapsed_s=time.time() - row["started_s"],
                                       observed_mhz=observed, clock_idle=bool(idle),
                                       engine_active=bool(state.get("active") or state.get("running"))))
            await asyncio.sleep(.02)
        row["output"] = await task
        row["passed"] = active_frequency_pass(row["samples"], frequency)
    except BaseException as exc:
        row["error"] = type(exc).__name__ + ": " + str(exc)
        if isinstance(exc, asyncio.CancelledError):
            raise
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        row["finished_s"] = time.time()
    return row


async def pd_probe(client, pair, reference):
    a, b = pair["source"], pair["target"]
    row = {k: pair[k] for k in ("link_class", "direction")}
    row.update(source_id=a["id"], target_id=b["id"], pd_output=False, cancellation=False)
    try:
        for instance in (a, b):
            await client.drained(instance)
        await client.control(a, role="prefill", mode="continuous", admit_prefill=True, admit_decode=True)
        await client.control(b, role="decode", mode="continuous", admit_prefill=True, admit_decode=True)
        await client.call(a, "/prepare-peers", dict(peers=[b["id"]]))
        for cancel in (False, True):
            nonce = uuid.uuid4().hex
            pid = f"pdb:{nonce}:p:{a['id']}:{b['id']}"
            did = f"pdb:{nonce}:d:{a['id']}:{b['id']}"
            client.owned[did] = b  # Register consumer before producer sends KV.
            producer = await client.call(a, "/v1/completions", dict(BODY, max_tokens=1), pid)
            if not output_matches(producer, count=1):
                raise RuntimeError("PD producer did not execute exactly one output token")
            if not cancel:
                output = await client.call(b, "/v1/completions", BODY, did)
                row.update(producer=producer, output=output, pd_output=output_matches(output, reference))
                await client.drained(a)
                await client.drained(b)
            else:
                deadline = time.monotonic() + 5
                while True:
                    held = await client.call(b, "/runtime")
                    if held.get("transfer_buffered_tensors") or held.get("transfer_allocations"):
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("cancellation probe never observed real pending imported KV")
                    await asyncio.sleep(.02)
                cancelled = await client.call(b, "/cancel", dict(request_id=did))
                transfers = cancelled.get("transfers", [])
                if len(transfers) != 1 or any(t.get("buffered_tensors") or t.get("inflight_receives")
                                              or t.get("inflight_sends") or t.get("listener_alive") is not True
                                              for t in transfers):
                    raise RuntimeError("cancellation did not clear every TP1 transport rank")
                client.owned.pop(did, None)
                row.update(held_import=held, cancel_response=cancelled,
                           cancellation=True, cancel_terminal=await client.drained(b))
    except BaseException as exc:
        row["error"] = type(exc).__name__ + ": " + str(exc)
        if isinstance(exc, asyncio.CancelledError):
            raise
    finally:
        await client.cleanup_owned()
        for instance in (a, b):
            await client.drained(instance)
            await client.control(instance, role="mixed", mode="continuous", admit_prefill=True, admit_decode=True)
    return row


def clock_restore_targets(config, gpus):
    targets = config.get("qualification_clock_restore")
    if not isinstance(targets, dict) or any(str(g) not in targets for g in gpus):
        raise ValueError("qualification_clock_restore must explicitly cover every target GPU")
    if any(targets[str(g)] is not None and
           (type(targets[str(g)]) is not int or targets[str(g)] not in FREQUENCIES) for g in gpus):
        raise ValueError("clock restoration requires an explicit unlocked or supported locked baseline")
    reset = config.get("qualification_clock_baseline") == "reset_before_qualification"
    if reset and any(targets[str(g)] is not None for g in gpus):
        raise ValueError("an explicitly reset baseline must restore unlocked clocks")
    if not reset and not config.get("qualification_clock_restore_basis"):
        raise ValueError("clock restoration declaration needs its evidence basis")
    return targets, reset


async def qualify(config, out, smoke=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    all_instances = copy.deepcopy(config.get("instances", []))
    instances = all_instances[:3] if smoke else all_instances
    raw = dict(scope="diagnostic_raw", measurement="hardware", host_id=socket.gethostname(),
               purpose="native mechanism qualification; no serving capacity or profile completion claim",
               config_sha256=object_hash(config), smoke=bool(smoke), started_s=time.time(),
               limited_shapes=dict(input_tokens=64, output_tokens=32, prefill_output_tokens=1),
               checks={k: False for k in REQUIRED_CHECKS}, rpc=[], mixed=[], frequency=[], pd=[],
               errors=[], cleanup_errors=[], cleanup_complete=False, passed=False)
    sampler = clocks = hardware = client = None
    original = {}
    try:
        if len(instances) < 3 or any(i.get("tp") != 1 or len(i.get("gpus", [])) != 1 for i in instances):
            raise ValueError("qualification requires at least three explicit TP1 instances")
        gpus = [i["gpus"][0] for i in instances]
        if len(set(gpus)) != len(gpus) or any(type(g) is not int or not 0 <= g < 8 for g in gpus):
            raise ValueError("target instances must have unique physical GPUs 0..7")
        targets, reset_baseline = clock_restore_targets(config, gpus)
        raw.update(clock_restore_targets=targets, clock_baseline="experiment_preparation_reset" if reset_baseline
                   else config["qualification_clock_restore_basis"], allocated_gpu_ids=gpus)
        try:
            raw["profile_errors"] = check_profile_inputs(config)
        except Exception as exc:
            raw["profile_errors"] = [repr(exc)]
        raw["checks"]["profile_coverage"] = not raw["profile_errors"]
        topology = InterconnectTopology.parse(Path(config["interconnect"]).read_text())
        raw["topology_sha256"] = topology.source_sha256
        with node_lease():
            inspected = await inspect_engines(dict(config, instances=instances))
            raw["engines_before"] = inspected
            if len(inspected) != len(instances) or any(r.get("errors") for r in inspected):
                raise RuntimeError("live engines did not pass identity/health/drain inspection")
            original = {r["instance_id"]: r["runtime"] for r in inspected}
            raw["engine_identities"] = {r["instance_id"]: engine_identity(r) for r in inspected}
            raw['profile_errors'].extend(await asyncio.to_thread(measured_source_errors,config,inspected))
            raw['live_topology'] = await read_live_topology()
            raw['profile_errors'].extend(topology_errors(config,raw['live_topology']))
            raw['checks']['profile_coverage'] = not raw['profile_errors']
            profile = read_json(config["profiles"])
            if any(r.get("image_id") != profile.get("engine_image") for r in inspected):
                raw["profile_errors"].append("live engine image differs from measured profile image")
                raw["checks"]["profile_coverage"] = False
            hardware = await asyncio.to_thread(PynvmlBackend, power_mode="instant")
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120), trust_env=False) as session:
                client = QualificationClient(session, raw)
                try:
                    # Construct both owners inside the cleanup region so even
                    # sampler initialization failure restores under the lease.
                    clocks = ClockOwner(RestoreBackend(hardware, targets), gpus)
                    sampler = PowerSampler(range(8), interval=.02, backend=hardware, sample_clocks=True)
                    sampler.start()
                    deadline = time.monotonic() + 5
                    while len(sampler.samples) < 2:
                        if sampler.error or time.monotonic() >= deadline:
                            raise RuntimeError("instant all-eight-GPU power preflight failed")
                        await asyncio.sleep(.02)
                    if sampler.power_source.get("mode") != "instant" or sampler.power_source.get("field_id") != 186:
                        raise RuntimeError("instant-power field 186 is required")
                    if reset_baseline:
                        raw["setup_clock_resets"] = []
                        async with clocks.lock:
                            for gpu in gpus:
                                await asyncio.get_running_loop().run_in_executor(clocks.pool, hardware.reset_clock, gpu)
                                raw["setup_clock_resets"].append(dict(gpu=gpu, finished_s=time.time()))
                    for instance in instances:
                        await client.control(instance, role="mixed", mode="continuous", admit_prefill=True, admit_decode=True)
                    reference = None
                    for instance in instances:
                        await clocks.set(instance["gpus"], 2520, verify_rise=False)
                        output = await client.call(instance, "/v1/completions", BODY)
                        passed = output_matches(output, reference)
                        raw["mixed"].append(dict(instance_id=instance["id"], output=output, passed=passed))
                        if reference is None and passed:
                            reference = output["token_ids"]
                        await client.drained(instance)
                    raw["checks"]["mixed_output"] = all(r["passed"] for r in raw["mixed"])
                    for instance in instances:
                        for frequency in ((2520,) if smoke else FREQUENCIES):
                            row = await frequency_probe(client, instance, hardware, clocks, frequency)
                            row["output_matches"] = output_matches(row.get("output", {}), reference)
                            row["passed"] = row["passed"] and row["output_matches"]
                            raw["frequency"].append(row)
                            await client.cleanup_owned()
                            await client.drained(instance)
                    raw["checks"]["frequency_commands"] = all(r["passed"] for r in raw["frequency"])
                    await clocks.set(gpus, 2520, verify_rise=False)
                    for pair in representative_pairs(instances, topology):
                        raw["pd"].append(await pd_probe(client, pair, reference))
                    raw["checks"]["pd_output"] = bool(raw["pd"]) and all(r["pd_output"] for r in raw["pd"])
                    raw["checks"]["cancellation"] = bool(raw["pd"]) and all(r["cancellation"] for r in raw["pd"])
                except BaseException as exc:
                    raw["errors"].append(type(exc).__name__ + ": " + str(exc))
                finally:
                    # Keep the lease and session throughout independent bounded
                    # request, native control and clock restoration attempts.
                    try:
                        await client.cleanup_owned()
                    except BaseException as exc:
                        raw["cleanup_errors"].append("requests: " + repr(exc))
                    native_results = []
                    for instance in instances:
                        try:
                            await asyncio.wait_for(client.drained(instance, native=True), cleanup_timeout(12))
                            state = original[instance["id"]]
                            restored = await asyncio.wait_for(client.control(instance, **{
                                k: state.get(k, True if k == "admit_decode" else None) for k in CONTROL_FIELDS}), cleanup_timeout(7))
                            if state.get("accepting") is False:
                                restored = await asyncio.wait_for(client.drained(instance, native=True), cleanup_timeout(12))
                            if any(restored.get(k) != state.get(k, True if k == "admit_decode" else None)
                                   for k in CONTROL_FIELDS) or not quiescent(restored):
                                raise RuntimeError("original controls did not restore")
                            native_results.append(dict(instance_id=instance["id"], restored=restored))
                        except BaseException as exc:
                            raw["cleanup_errors"].append(instance["id"] + ": " + repr(exc))
                    raw["native_restoration"] = native_results
                    raw["checks"]["native_drain"] = len(native_results) == len(instances)
                    if clocks is not None:
                        try:
                            await asyncio.wait_for(clocks.close(), cleanup_timeout(5))
                            raw["clock_restore_complete"] = True
                        except BaseException as exc:
                            raw["cleanup_errors"].append("clocks: " + repr(exc))
                        clocks = None
                    raw["engines_after"] = await inspect_engines(dict(config, instances=instances))
                    after = {r["instance_id"]: engine_identity(r) for r in raw["engines_after"]}
                    if after != raw["engine_identities"] or any(r.get("errors") for r in raw["engines_after"]):
                        raw["cleanup_errors"].append("final engine identity/health differs")
                    if sampler is not None:
                        await asyncio.to_thread(sampler.stop)
    except BaseException as exc:
        raw["errors"].append(type(exc).__name__ + ": " + str(exc))
    finally:
        # Covers initialization failures before an HTTP session is available.
        if clocks is not None:
            try:
                await asyncio.wait_for(clocks.close(), cleanup_timeout(5))
                raw["clock_restore_complete"] = True
            except BaseException as exc:
                raw["cleanup_errors"].append("early clocks: " + repr(exc))
        if sampler is not None:
            await asyncio.to_thread(sampler.stop)
            raw.update(power_samples=sampler.samples, power_source=sampler.power_source,
                       power_metadata=sampler.power_metadata, frequency_samples=sampler.frequency_samples,
                       utilization_samples=sampler.utilization_samples, sampling_error=sampler.error)
            if sampler.error:
                raw["cleanup_errors"].append("power: " + str(sampler.error))
            try:
                raw["all_eight_gpu_energy_j"] = trapezoid_energy(sampler.samples)
            except Exception as exc:
                raw["errors"].append("energy integration: " + repr(exc))
        raw["cleanup_complete"] = bool(original) and not raw["cleanup_errors"] and raw.get("clock_restore_complete") is True
        raw["finished_s"] = time.time()
        raw["passed"] = (not smoke and not raw["errors"] and raw["cleanup_complete"]
                         and all(raw["checks"].get(k) is True for k in REQUIRED_CHECKS))
        raw["scope_limit"] = ("three-engine/one-frequency smoke; not formal qualification" if smoke else
                              "64-input/32-output mechanism probes; original profile/heldout evidence is independently required")
        write_json(out / "raw.json", raw)
        profile_path = Path(config.get("profiles", ""))
        proof = dict(scope="hardware_qualification", passed=raw["passed"], checks=raw["checks"],
                     smoke=bool(smoke), cleanup_complete=raw["cleanup_complete"],
                     profile_sha256=sha256(profile_path) if profile_path.is_file() else None,
                     engine_identities=raw.get("engine_identities", {}),
                     errors=raw["errors"], profile_errors=raw.get("profile_errors", []),
                     artifacts={str((out / "raw.json").resolve()): sha256(out / "raw.json")},
                     scope_limit=raw["scope_limit"])
        write_json(out / "qualification.json", proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    proof = run_async_cli(qualify(read_json(args.config), args.out, smoke=args.smoke),
                          failure_path=args.out / "termination.json")
    print(json.dumps(dict(passed=proof["passed"], out=str(args.out.resolve()),
                          profile_errors=len(proof["profile_errors"]))))
    raise SystemExit(0 if proof["passed"] else 2)


if __name__ == "__main__":
    main()
