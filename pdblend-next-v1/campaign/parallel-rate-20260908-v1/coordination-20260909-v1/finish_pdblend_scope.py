"""Finish the user's PDB-only scope at an existing baseline stage boundary.

This stage producer performs file validation only. It never starts a service or
measurement. The original supervisor consumes its STOP after the producer exits.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(reference):
    assert ref(reference["path"]) == reference, reference["path"]
    return read(reference["path"])


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "pipeline-status", "priority", "out", "stop"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    priority = read(args.priority)
    assert priority["effective_systems"] == ["pdblend"]
    assert priority["future_pdb_progress_must_not_wait_for_baseline_completion"] is True
    plan, state = read(args.plan), read(args.pipeline_status)
    assert socket.gethostname() == plan["expected_hostname"]
    assert state["plan"] == ref(args.plan)
    assert state["node"] == plan["node"] and state["model"] == plan["model"]
    assert not state["node_lease_held"] and not state.get("error")
    assert str(args.stop) in plan["stop_paths"] and not args.stop.exists()
    contract_ref = plan["contract"]
    assert ref(contract_ref["path"]) == contract_ref
    spec = importlib.util.spec_from_file_location("pdb_scope_contract", contract_ref["path"])
    contract = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = contract
    spec.loader.exec_module(contract)
    observations = [checked(item) for item in state["observations"]]
    assert all(item["system"] == "pdblend" for item in observations)
    boundaries = {}
    for dataset in plan["dataset_order"]:
        group = contract.resolve_group(state["declaration"], plan["model"], dataset,
                                       actual_host=plan["node"])
        reuse = [checked(item) for item in plan.get("dynamic_reuse_observations", [])]
        reuse = [item for item in reuse if item["dataset"] == dataset]
        if reuse:
            group = contract.apply_audited_reuse(group, reuse)
        selected = contract.select_group(group, [item for item in observations
                                                if item["dataset"] == dataset])
        assert selected["phase"] in ("baselines", "complete"), dataset
        assert selected.get("cap_rate_rps") is not None, dataset
        boundaries[dataset] = selected
    assert set(boundaries) == {"alpaca", "sharegpt", "longbench"}
    # Snapshot the still-live supervisor status before its own graceful STOP.
    snapshot = args.out.with_name(args.out.stem + ".supervisor-snapshot.json")
    write_new(snapshot, state)
    result = dict(schema="user-pdblend-only-scope-complete-v1", completed_s=time.time(),
                  node=plan["node"], model=plan["model"], pdb_complete=True,
                  five_system_complete=False, baseline_dispatched=False,
                  user_priority=ref(args.priority), plan=ref(args.plan),
                  supervisor_snapshot=ref(snapshot), observations=state["observations"],
                  group_decisions=boundaries, execution_performed="CPU_only")
    write_new(args.out, result)
    with args.stop.open("x") as stream:
        stream.write("PDBlend scope completed; baselines disabled by the user's request. "
                     "See " + str(args.out) + "\n")
    print(json.dumps({"pdb_complete": True, "baseline_dispatched": False,
                      "completion": ref(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
