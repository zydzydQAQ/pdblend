"""Fresh baseline qualification with the actual B pipeline-009 control package.

A resumes its existing native deployment after the recorded setup-only failure.
C deploys once using B's actual template. Historical and v1 sources stay intact.
"""
import argparse
import asyncio
import copy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import prepare as old
import sources as s
p = s.p
REGISTRATION = s.HERE / "control-registration-v2.json"


def registered():
    value = p.read(REGISTRATION)
    template = p.checked(value["template"])
    p.need(value["template"]["sha256"] == value["historical_template"]["sha256"]
        == "a055607172c3ae40bf92f597a28823a694d18e7402757da8b932ba4f7db5279b", "wrong B actual template")
    p.need(template["host_release"] == value["host_release"]
        and template["profile"] == value["profile"]
        and value["profile"]["sha256"] == "1f840a8c058e3bd650602d6b2be08dacdb5e95f7db4b9e3447df9bd7d89dca77",
        "B actual control/profile registration changed")
    for path, digest in value["files"].items():
        p.need(p.sha(path) == digest, "registered B template dependency changed: " + path)
    p.checked(value["profile"])
    return value, template


def runtime_paths(registration):
    host = Path(registration["host_release"])
    return [str(host / "src"), str(host), "/root/workspace/pdblend/.runtime-deps"]


def cold_compatibility(registration):
    # The child mimics the gate's binding-selected import order. The isolated
    # interpreter cannot inherit an already imported legacy ClockOwner.
    program = """import inspect,json,sys
sys.path[:0]=json.loads(sys.argv[1])
import aiohttp
from ecopadg.serving.backend import ClockOwner
signature=inspect.signature(ClockOwner)
signature.bind(object(),tuple(range(8)),max_frequency=2100)
print(json.dumps(dict(passed=True,clock_owner_file=inspect.getfile(ClockOwner),signature=str(signature),aiohttp_file=aiohttp.__file__,gpu_constructed=False)))
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", program, json.dumps(runtime_paths(registration))],
        check=True, text=True, capture_output=True)
    proof = json.loads(result.stdout)
    p.need(proof["clock_owner_file"] == str(Path(registration["host_release"]) / "src/ecopadg/serving/backend.py"),
           "cold gate resolved foreign ClockOwner")
    return proof


def native_template(binding, template):
    """Prove existing native work/startup config is B's mechanically named layout."""
    p.need(binding["model"] == "14b" and len(binding["instances"]) == 8, "eight 14B engines required")
    naming = ("id", "port", "kv_port", "peers", "runtime_dir", "weight_cache_root")
    for index, (actual, source) in enumerate(zip(binding["instances"], template["instances"])):
        p.need(actual["tp"] == source["tp"] == 1 and actual["gpus"] == source["gpus"] == [index]
            and actual["native_kind"] == "legacy_sync_put" and not actual.get("scheduler_cache_observed")
            and actual["container"]["image"] == template["image"], "native layout/image changed")
        config = p.read(actual["engine_config"])
        p.need(binding["files"].get(actual["engine_config"]) == p.sha(actual["engine_config"]), "native config not frozen")
        expected = copy.deepcopy(source["engine_template"])
        expected.pop("retained_weights", None)
        expected.update({key: config[key] for key in naming})
        expected.update(tp=1, role="mixed", initial_generation=0)
        p.need(config == expected, "existing native startup config differs from actual B template")


def diagnosed_prior(reference, bootstrap_ref, node):
    prior = p.checked(reference)
    p.need(prior["node"] == node and prior["binding"] == bootstrap_ref and not prior["complete"]
        and prior.get("finished_s") and not prior.get("node_lease_held") and not p.active_owner(prior),
        "prior preparation must be terminal and own this bootstrap")
    child = prior.get("child", {})
    p.need(child.get("finished_s") and child.get("exitcode") == 1 and not p.active_owner(child),
           "prior qualifier has not exited")
    directory = Path(reference["path"]).parent / "qualification"
    qualification_ref = p.ref(directory / "status.json")
    qualification = p.checked(qualification_ref)
    p.need(not qualification["complete"] and qualification.get("finished_s") and not qualification.get("node_lease_held")
        and not p.active_owner(qualification) and qualification["child"]["exitcode"] == 1
        and not p.active_owner(qualification["child"]), "prior native gate still active")
    native_ref = p.ref(directory / "native/status.json")
    native = p.checked(native_ref)
    p.need(native["complete"] and native["passed"] is False and native.get("finished_s")
        and native.get("native_cleanup_complete") is True and native.get("clock_restore_complete") is True
        and not native.get("cleanup_errors")
        and native["error"] == 'TypeError("ClockOwner.__init__() got an unexpected keyword argument \'max_frequency\'")',
        "prior failure is not the diagnosed clean setup-only API mismatch")
    return dict(failed_setup_status=reference, failed_qualification_status=qualification_ref, failed_native_status=native_ref)


def adapt_bootstrap(reference, node, out, registration, template, repair_ref):
    original = p.checked(reference)
    p.need(original["hostname"] == s.HOSTS[node] and original["fresh_node_native_identity"] is True
        and original["old_node_qualification_inherited"] is False
        and original["output_correctness_verified"] is False
        and original["correctness_gate_required_before_performance"] is True, "foreign or already qualified bootstrap")
    native_template(original, template)
    updated = copy.deepcopy(original)
    updated.update(host_release=registration["host_release"], qualified_profile_required=registration["profile"],
        baseline_control_registration=dict(source_bootstrap=reference, registration=p.ref(REGISTRATION), repair=repair_ref))
    updated["files"].update(registration["files"])
    for ref in (reference, p.ref(REGISTRATION), registration["template"], repair_ref, p.ref(__file__)):
        updated["files"][ref["path"]] = ref["sha256"]
    target = Path(out) / "registered-bootstrap.json"
    p.need(not target.exists(), "fresh registered bootstrap required")
    p.save(target, updated)
    return p.ref(target)


def verify_control_bootstrap(reference):
    registration, template = registered()
    actual = p.checked(reference)
    declaration = actual["baseline_control_registration"]
    p.need(declaration["registration"] == p.ref(REGISTRATION), "foreign control registration")
    original = p.checked(declaration["source_bootstrap"])
    allowed = set(registration["allowed_bootstrap_changes"])
    p.need({k:v for k,v in actual.items() if k not in allowed}
        == {k:v for k,v in original.items() if k not in allowed}, "native bootstrap fields changed")
    p.need(actual["host_release"] == registration["host_release"]
        and actual["qualified_profile_required"] == registration["profile"], "wrong registered gate control source/profile")
    p.need(all(actual["files"].get(path) == digest for path, digest in original["files"].items()), "original native source closure changed")
    native_template(actual, template)
    repair = p.checked(declaration["repair"])
    p.need(repair["implementation_unchanged"] is True and repair["reference_implementation"] == "B actual pipeline-009 frozen baselines",
           "unregistered implementation change")
    if repair.get("failed_setup_status"):
        p.need(diagnosed_prior(repair["failed_setup_status"], declaration["source_bootstrap"], repair["node"])
            == {key:repair[key] for key in ("failed_setup_status", "failed_qualification_status", "failed_native_status")},
            "repair failure references differ")
    return declaration


def run(node, predecessor, out, bootstrap=None, prior_attempt=None):
    p.need(socket.gethostname() == s.HOSTS[node] and "PDBLEND_NODE_LOCK_FD" not in os.environ, "wrong host or inherited lease")
    destination = Path(out).resolve()
    p.need(destination.is_relative_to(s.NEW / node) and not destination.exists(), "fresh node output required")
    dependencies = s.checked_dependencies()
    registration, template = registered()
    compatibility = cold_compatibility(registration)
    boundary_ref = p.ref(predecessor)
    boundary, decision = old.terminal(boundary_ref, node)
    p.need((bootstrap is None) == (prior_attempt is None), "resume requires both existing bootstrap and diagnosed prior status")
    prior_proof = diagnosed_prior(p.ref(prior_attempt), p.ref(bootstrap), node) if bootstrap else {}
    destination.mkdir(parents=True)
    state = dict(schema="slo14-baseline-stage-status-v2", node=node, campaign_id=s.NEW.name, model="14b", dataset="sharegpt",
        pid=os.getpid(), startticks=p.process_identity(os.getpid())["startticks"], started_s=time.time(), complete=False,
        node_lease_held=False, phase="control_registration", predecessor_terminal=boundary_ref, boundary_decision=decision)
    repair = dict(schema="slo14-baseline-setup-repair-v2", node=node, implementation_unchanged=True,
        reference_implementation="B actual pipeline-009 frozen baselines", diagnosis="Default template selected pre-platform ClockOwner and PDB profile; use actual B pipeline-009 template host/profile before native qualification.",
        registration=p.ref(REGISTRATION), native_deployment_reused=bool(bootstrap), failed_setup_status=None,
        preventive_setup_correction=bootstrap is None,
        cold_compatibility=compatibility)
    repair.update(prior_proof)
    p.save(destination / "repair.json", repair)
    p.save(destination / "status.json", state)
    try:
        if bootstrap:
            original_bootstrap = p.ref(bootstrap)
            state["deployment_reused"] = original_bootstrap
        else:
            template = copy.deepcopy(template)
            template.update(node=s.NATIVE_NODES[node], expected_hostname=s.HOSTS[node])
            spec = old.make_spec(template, boundary_ref, boundary["binding"], node, destination)
            producer = p.load(dependencies["producer"], "slo14_actual_B_legacy_creation")
            async def controlled():
                task = asyncio.current_task()
                for sig in (signal.SIGTERM, signal.SIGINT):
                    asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
                await producer.perform(template, spec, destination, state)
            asyncio.run(controlled())
            original_bootstrap = state["binding"]
        registered_bootstrap = adapt_bootstrap(original_bootstrap, node, destination, registration, template, p.ref(destination / "repair.json"))
        verify_control_bootstrap(registered_bootstrap)
        state.update(phase="qualifying", binding=registered_bootstrap, node_lease_held=False)
        p.save(destination / "status.json", state)
        os.environ["PYTHONPATH"] = ":".join(runtime_paths(registration))
        qualification = destination / "qualification"
        old.qualification_child([sys.executable, "-B", dependencies["qualifier"]["path"],
            "--bootstrap", registered_bootstrap["path"], "--out", str(qualification),
            "--node", s.NATIVE_NODES[node], "--hostname", s.HOSTS[node],
            "--profile", registration["profile"]["path"], "--datasets", "sharegpt",
            "--policy-adapter", dependencies["policy_adapter"]["path"], "--run"], destination, state)
        validator_ref = p.ref(s.HERE / "verify_v2.py")
        validator = p.load(validator_ref, "slo14_registered_baseline_validator")
        bindings = p.read(qualification / "bindings.json")
        p.need(set(bindings) == {"mixed", "distserve", "dynamollm", "ecoserve"}, "four qualified baselines required")
        handoffs = {}
        for system, reference in bindings.items():
            p.save(destination / (system + "-raw-verification.json"), validator.verify(reference))
            binding = p.checked(reference)
            host = Path(binding["host_release"])
            handoffs[system] = dict(qualification=reference, qualification_validator=validator_ref, binding=reference,
                runtime_pythonpath=[str(host / "src"), str(host), "/root/workspace/pdblend/.runtime-deps"], measurement_executor=p.ref(s.EXECUTOR))
        p.save(destination / "ready.json", handoffs)
        state.update(complete=True, phase="ready", handoffs=p.ref(destination / "ready.json"))
    except BaseException as exc:
        state.update(error=repr(exc), phase="blocked_failure")
        raise
    finally:
        state.update(finished_s=time.time(), node_lease_held=False)
        p.save(destination / "status.json", state)
    return state


def descriptor(node):
    registration, _ = registered()
    destination = s.NEW / node / ("baseline-qualification-002" if node == "A" else "baseline-002")
    own = [Path(__file__).resolve(), s.HERE / "verify_v2.py", REGISTRATION,
           s.HERE / "template-B-pipeline009-v2.json", s.HERE / "prepare.py", s.HERE / "verify.py",
           s.HERE / "sources.py", s.HERE / "dependencies.json", s.HERE / "original-B-template.json"]
    argv = [sys.executable, "-B", str(Path(__file__).resolve()), "--node", node,
        "--predecessor-terminal", "{pdb_boundary}", "--out", str(destination)]
    if node == "A":
        argv += ["--bootstrap", str(s.NEW / node / "baseline-001/bootstrap/binding.json"),
                 "--prior-attempt", str(s.NEW / node / "baseline-001/status.json")]
    return dict(schema="slo-rate-baseline-command-v1", node=node, files={str(f):p.sha(f) for f in own},
        argv=argv + ["--run"], handoffs_path=str(destination / "ready.json"), runtime_pythonpath=runtime_paths(registration))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", choices=("A", "C"), required=True)
    parser.add_argument("--predecessor-terminal", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--bootstrap", type=Path)
    parser.add_argument("--prior-attempt", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        p.need(args.predecessor_terminal and args.out, "explicit boundary/output required")
        print(run(args.node, args.predecessor_terminal, args.out, args.bootstrap, args.prior_attempt)["handoffs"])
    else:
        registration, _ = registered()
        print(json.dumps(cold_compatibility(registration)))


if __name__ == "__main__":
    main()
