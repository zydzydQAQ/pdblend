"""New-campaign baseline handoff: terminal PDB -> fresh 8TP1 -> full raw gates.

--run is the only hardware entry. All new declarations, raw data and readiness
files stay below this campaign. Frozen historical sources are read only.
"""
import argparse
import asyncio
import copy
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import sources as s
p = s.p


def observation_with_metadata(audit_ref, value=None):
    """Only scheduler attempt metadata may be added to immutable audit facts."""
    import contract as c
    observation = p.checked(audit_ref)
    if observation.get("audit_reference"):
        p.need(observation["audit_reference"] == audit_ref, "foreign observation audit reference")
    observation["audit_reference"] = audit_ref
    if value is not None:
        p.need(isinstance(value, dict), "boundary observation value must be an object")
        for key in ("engineering_attempt", "repair_reference"):
            if key in value and key not in observation:
                observation[key] = value[key]
    observation.setdefault("engineering_attempt", 1)
    if value is not None:
        normalized_value = dict(value)
        normalized_value.setdefault("engineering_attempt", 1)
        p.need(normalized_value == observation, "boundary observation values differ from audited files")
    attempt = observation["engineering_attempt"]
    p.need(type(attempt) is int and 1 <= attempt <= c.MAX_ENGINEERING_ATTEMPTS,
           "invalid boundary engineering attempt")
    repair = observation.get("repair_reference")
    if attempt == 1:
        p.need("repair_reference" not in observation, "first attempt cannot carry repair metadata")
    else:
        p.need(c.reference_shape(repair), "engineering replacement requires repair reference")
        p.checked(repair)
    return observation


def terminal(reference, node):
    """The supervisor may live; its preceding cell must have exited cleanly."""
    import contract as c
    saved = p.checked(reference)
    p.need(saved.get("campaign_id") == c.CAMPAIGN and saved.get("node") == node
        and saved.get("model") == "14b" and saved.get("dataset") == "sharegpt"
        and saved.get("pdb_boundary_complete") is True, "foreign or incomplete PDB boundary")
    last = p.checked(saved["last_cell_status"])
    p.need(last.get("complete") is True and last.get("finished_s")
        and last.get("node_lease_held") is False and not p.active_owner(last)
        and not last.get("error") and not last.get("failed") and last.get("cleanup_complete") is True,
        "preceding cell is live, unclean or failed")
    values = saved.get("observation_values")
    p.need(values is None or isinstance(values, list) and len(values) == len(saved["observations"]),
           "boundary observation metadata count differs")
    observations = [observation_with_metadata(audit_ref, None if values is None else values[index])
                    for index, audit_ref in enumerate(saved["observations"])]
    decision = c.evaluate_group(node, observations)
    p.need(decision["status"] == "cap_confirmed" and decision["cap_observed"], "PDB cap/confirmation not independently established")
    binding = p.checked(saved["binding"])
    p.need(binding["model"] == "14b" and binding["system"] == "pdblend"
        and binding["hostname"] == s.HOSTS[node], "foreign predecessor native binding")
    if last.get("checkpoint"):
        checkpoint = p.checked(last["checkpoint"])
        p.need(checkpoint["row"]["system"] == "pdblend"
            and checkpoint["row"]["node"] == node, "last cell is not this node's PDB")
    return saved, decision


def make_spec(template, boundary_ref, binding_ref, node, out):
    """Mechanical original producer specification with actual node identity."""
    destination = Path(out) / "deployment"
    p.need(not destination.exists(), "new baseline deployment required")
    destination.mkdir(parents=True)
    configs, records = {}, []
    for index, item in enumerate(template["instances"]):
        p.need(item["tp"] == 1 and item["gpus"] == [index] and item["role"] == "mixed",
               "fixed original 8TP1 native layout required")
        rid = "sloscale" + node.lower() + "r" + str(index)
        config = copy.deepcopy(item["engine_template"])
        p.need(config["model"] == "/models/Qwen2.5-14B-Instruct" and config["tp"] == 1
            and config["max_model_len"] == config["max_num_batched_tokens"] == 8192
            and config["max_num_seqs"] == 32, "original legacy work limits differ")
        config.pop("retained_weights", None)
        config.update(id=rid, tp=1, role="mixed", port=30100 + index, kv_port=31000 + 32 * index,
            runtime_dir=str(destination / "native"), weight_cache_root=str(destination / "weights"),
            initial_generation=0)
        configs[rid] = config
        records.append(dict(id=rid, tp=1, gpus=[index], role="mixed", port=config["port"],
            kv_port=config["kv_port"], url="http://127.0.0.1:" + str(config["port"]),
            container_name="pdb-v2-" + rid, config=str(destination / "engines" / (rid + ".json")),
            engine_entry=template["source_entry"], image=template["image"],
            environment=item["environment"] + ["CUDA_VISIBLE_DEVICES=" + str(index), "PYTHONDONTWRITEBYTECODE=1"],
            mounts=item["mounts"], native_kind="legacy_sync_put", scheduler_cache_observed=False))
    peers = {i["id"]: dict(host="127.0.0.1", tp=1, kv_port=i["kv_port"]) for i in records}
    frozen = dict(template["files"])
    for instance in records:
        configs[instance["id"]]["peers"] = peers
        p.save(instance["config"], configs[instance["id"]])
        frozen[instance["config"]] = p.sha(instance["config"])
    for reference in (boundary_ref, binding_ref, p.ref(__file__), p.ref(s.HERE / "sources.py"),
                      p.ref(s.HERE / "dependencies.json")):
        frozen[reference["path"]] = reference["sha256"]
    spec = dict(schema="slo14-new-node-baseline-deployment-v1",
        protocol_id="per-dataset-slo-five-system-fixed-window-v1", model="14b",
        node=s.NATIVE_NODES[node], layout="resident", hostname=s.HOSTS[node], deadline_s=None,
        campaign_lifecycle="until_declared_complete_v1", out=str(destination),
        host_release=template["host_release"], executor_release=str(s.EXECUTOR.parent),
        instances=records, required_predecessors=[], predecessor_terminal=boundary_ref,
        previous_binding=binding_ref["path"], pdb_binding=binding_ref["path"],
        image=template["image"], files=frozen, supported_datasets=["sharegpt"],
        source_entry=template["source_entry"], deployment_budget_s=720, cleanup_budget_s=120,
        independent_boundary_checked=True, original_creation_cleanup_primitives_preserved=True)
    p.save(destination / "deployment.json", spec)
    return p.ref(destination / "deployment.json")


def qualification_child(argv, out, state):
    logfile = out / "qualification.log"
    with logfile.open("xb") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        state["child"] = dict(p.process_identity(process.pid), argv=argv)
        p.save(out / "status.json", state)
        interrupted = False
        def stop(signum, frame):
            nonlocal interrupted
            interrupted = True
            if process.poll() is None:
                process.send_signal(signum)
        prior = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            code = process.wait()
        finally:
            for sig, handler in prior.items():
                signal.signal(sig, handler)
        state["child"].update(exitcode=code, finished_s=time.time())
        p.save(out / "status.json", state)
        p.need(code == 0 and not interrupted, "fresh qualification child failed; no automatic retry")


def run(node, predecessor, out):
    p.need(node in s.HOSTS and socket.gethostname() == s.HOSTS[node], "wrong physical node")
    p.need("PDBLEND_NODE_LOCK_FD" not in os.environ, "inherited lease forbidden")
    destination = Path(out).resolve()
    p.need(destination.is_relative_to(s.NEW / node) and not destination.exists(), "new node experiment output required")
    dependencies = s.checked_dependencies()
    boundary_ref = p.ref(predecessor)
    boundary, decision = terminal(boundary_ref, node)
    destination.mkdir(parents=True)
    state = dict(schema="slo14-baseline-stage-status-v1", campaign_id=s.NEW.name, node=node,
        model="14b", dataset="sharegpt", pid=os.getpid(),
        startticks=p.process_identity(os.getpid())["startticks"], started_s=time.time(),
        complete=False, node_lease_held=False, phase="deploying", predecessor_terminal=boundary_ref,
        boundary_decision=decision)
    p.save(destination / "status.json", state)
    try:
        template = p.checked(dependencies["original_template"])
        template.update(node=s.NATIVE_NODES[node], expected_hostname=s.HOSTS[node], profile=dependencies["profile"])
        spec = make_spec(template, boundary_ref, boundary["binding"], node, destination)
        state["deployment"] = spec
        producer = p.load(dependencies["producer"], "slo14_frozen_baseline_creation")
        async def controlled():
            task = asyncio.current_task()
            for sig in (signal.SIGTERM, signal.SIGINT):
                asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
            await producer.perform(template, spec, destination, state)
        asyncio.run(controlled())
        p.need(state.get("binding"), "fresh baseline bootstrap missing")
        state.update(phase="qualifying", node_lease_held=False)
        p.save(destination / "status.json", state)
        qualification = destination / "qualification"
        qualification_child([sys.executable, "-B", dependencies["qualifier"]["path"],
            "--bootstrap", state["binding"]["path"], "--out", str(qualification),
            "--node", s.NATIVE_NODES[node], "--hostname", s.HOSTS[node],
            "--profile", dependencies["profile"]["path"], "--datasets", "sharegpt",
            "--policy-adapter", dependencies["policy_adapter"]["path"], "--run"], destination, state)
        state["phase"] = "independent_raw_verification"
        p.save(destination / "status.json", state)
        validator_ref = p.ref(s.HERE / "verify.py")
        verifier = p.load(validator_ref, "slo14_new_baseline_verifier")
        bindings = p.read(qualification / "bindings.json")
        p.need(set(bindings) == {"mixed", "distserve", "dynamollm", "ecoserve"}, "four baselines required")
        handoffs = {}
        for system, reference in bindings.items():
            raw_proof = verifier.verify(reference)
            p.save(destination / (system + "-raw-verification.json"), raw_proof)
            binding = p.checked(reference)
            host = Path(binding["host_release"])
            handoffs[system] = dict(qualification=reference, qualification_validator=validator_ref,
                binding=reference, runtime_pythonpath=[str(host / "src"), str(host), "/root/workspace/pdblend/.runtime-deps"],
                measurement_executor=p.ref(s.EXECUTOR))
        p.save(destination / "ready.json", handoffs)
        state.update(complete=True, phase="ready", handoffs=p.ref(destination / "ready.json"))
    except BaseException as exc:
        state.update(error=repr(exc), phase="blocked_failure")
        raise
    finally:
        state.update(finished_s=time.time(), node_lease_held=False)
        p.save(destination / "status.json", state)
    return state


def command_descriptor(node):
    p.need(node in s.HOSTS, "unknown node")
    destination = s.NEW / node / "baseline-001"
    own = [Path(__file__).resolve(), s.HERE / "sources.py", s.HERE / "verify.py",
           s.HERE / "dependencies.json", s.HERE / "original-B-template.json"]
    host = Path(p.read(s.HERE / "original-B-template.json")["host_release"])
    return dict(schema="slo-rate-baseline-command-v1", node=node,
        files={str(path): p.sha(path) for path in own},
        argv=[sys.executable, "-B", str(Path(__file__).resolve()), "--node", node,
              "--predecessor-terminal", "{pdb_boundary}", "--out", str(destination), "--run"],
        handoffs_path=str(destination / "ready.json"),
        runtime_pythonpath=[str(host / "src"), str(host), "/root/workspace/pdblend/.runtime-deps"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", choices=tuple(s.HOSTS), required=True)
    parser.add_argument("--predecessor-terminal", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--write-command", type=Path)
    args = parser.parse_args()
    if args.write_command:
        p.need(not args.run and not args.write_command.exists(), "fresh CPU command declaration only")
        p.save(args.write_command, command_descriptor(args.node))
        print(p.ref(args.write_command))
    elif args.run:
        p.need(args.predecessor_terminal and args.out, "explicit terminal and output required")
        print(run(args.node, args.predecessor_terminal, args.out)["handoffs"])
    else:
        s.checked_dependencies()
        print("CPU baseline source validation passed; no GPU operation")


if __name__ == "__main__":
    main()
