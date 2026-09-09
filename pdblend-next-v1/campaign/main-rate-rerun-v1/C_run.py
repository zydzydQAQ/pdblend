#!/usr/bin/env python3
"""User-authorized C12 original-policy repeats; default is CPU verification only.

Consumes measured peer-readiness evidence. It never repairs peers, restarts an
engine or edits existing experiment/configuration files. Original run_one owns
every measurement, timeout, request cancellation, native drain and clock cleanup.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent
HOST = "iZwz9gfq11hx1sbob59yrgZ"
DECLARATION_SHA = "c51feb0af47f623dc28200abebeee6d439e7544f4cbac7e42a162549c064e529"
COMMON = CAMPAIGN / "five-system-execution-v3/run.py"
COMMON_SHA = "7c7dbe217243b42a8f93b57476ed457a6111e8f90c71ac4269130c9b46420f92"
CHILD_SHA = "e636789c1af2b2077840d14ee8ec293a6318270c70f365f98da84ad123f1ed35"
REPAIR = CAMPAIGN / "C7B-retained-peer-repair-v1"
REPAIR_MANIFEST_SHA = "29cf57f1575bcfd12ddfb7a887427ffa01008c5b0d43f4e7126c4c483ee5a840"
START_CUTOFF = 1788861600.0  # Existing work cutoff, 2026-09-08 18:00 CST.
DEADLINE = 1788868800.0      # Existing delivery cutoff, 20:00 CST.
EXPECTED = {("alpaca", 9.0), ("alpaca", 12.0), ("sharegpt", 2.0),
            ("sharegpt", 3.0), ("longbench", 1.5), ("longbench", 3.0)}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        temp.replace(path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_declaration(path):
    require(sha(path) == DECLARATION_SHA, "frozen user declaration changed")
    require(sha(COMMON) == COMMON_SHA and sha(COMMON.with_name("child.py")) == CHILD_SHA,
            "frozen measurement implementation changed")
    declaration = read(path)
    require(declaration["schema"] == "main-rate-repeat-v1" and declaration["authorized"] is True
            and declaration["automatic_retries"] is False
            and declaration["repeats_per_workload"] == 2, "wrong authorized repeat contract")
    cells = [c for c in declaration["cells"] if c["model"] == "7b"]
    validate_cells(cells)
    for cell in cells:
        for key in ("trace", "config", "source_binding", "source_manifest", "host_manifest", "original_receipt"):
            ref = cell[key]
            require(sha(ref["path"]) == ref["sha256"], "frozen reference changed: " + key)
        declared_rows = read(cell["source_manifest"]["path"])["cells"]
        source = next(r for r in declared_rows if r["cell_id"] == cell["original_cell_id"])
        require(source == cell["source_row"], "original declared row changed")
        original_binding = read(cell["source_binding"]["path"])
        require(original_binding["configs"][cell["dataset"]] == cell["config"]["path"],
                "original configuration path changed")
        require(original_binding["files"].get(cell["config"]["path"]) == cell["config"]["sha256"],
                "original configuration hash changed")
    return declaration, cells


def validate_cells(cells):
    require(len(cells) == 12 and len({c["cell_id"] for c in cells}) == 12,
            "exactly twelve unique C repeats required")
    require({(c["dataset"], c["rate_rps"], c["repeat"]) for c in cells}
            == {(d, r, repeat) for d, r in EXPECTED for repeat in (1, 2)},
            "C six workload/two repeat domain changed")
    for cell in cells:
        row = cell["source_row"]
        require(cell["system"] == row["system"] == "pdblend" and cell["policy_diff"] == {}
                and cell["fresh_execution_required"] is True, "original policy required")
        require(cell["seed"] == row["seed"] == 701 and cell["arrival_window_s"] == 100
                and row["trace_duration_s"] == 100 and cell["request_hard_timeout_s"] == 120
                and cell["drain_after_arrival_window_s"] == 120, "work protocol changed")
        require(cell["slo_scale"] == row["slo_scale"] == 1
                and cell["slo_ttft_s"] == row["slo_ttft_s"]
                and cell["slo_tpot_s"] == row["slo_tpot_s"]
                and cell["n_requests"] == row["n_requests"]
                and cell["rate_rps"] == row["rate_rps"]
                and cell["trace"]["path"] == row["trace"]
                and cell["trace"]["sha256"] == row["trace_sha256"], "work, trace or SLO changed")
        require(cell["original_cell_id"] == row["cell_id"]
                and cell["cell_id"] != row["cell_id"], "new result ID and original reference required")


def execution_rows(cells):
    rows = []
    for cell in cells:
        row = copy.deepcopy(cell["source_row"])
        row.update(cell_id=cell["cell_id"], original_cell_id=cell["original_cell_id"],
                   rerun_repeat=cell["repeat"], rerun_scope="fixed original policy and seed701 trace",
                   point_kind=cell["point_kind"])
        rows.append(row)
    return rows


def repair_evidence(args):
    require(args.repair_spec and args.repair_receipt, "actual measured peer repair evidence is required")
    require(sha(REPAIR / "manifest.json") == REPAIR_MANIFEST_SHA, "reviewed peer repair package changed")
    audit = load(REPAIR / "audit.py", "c_repeat_peer_audit")
    evidence = audit.audit(args.repair_spec, args.repair_receipt)
    require(evidence["passed"] is True, "peer repair raw audit failed")
    repaired = read(evidence["ready_binding"]["path"])
    require(repaired["performance_authorized"] is False, "peer repair must not authorize performance")
    return evidence, repaired


def original_policy_binding(cells, repaired):
    original = read(cells[0]["source_binding"]["path"])
    require(original["model"] == repaired["model"] == "7b"
            and original["host_release"] == repaired["host_release"]
            and original["configs"] == repaired["configs"], "restored source/config differs")
    require([i["id"] for i in original["instances"]] == [i["id"] for i in repaired["instances"]],
            "restored owner set or order changed")
    for old, new in zip(original["instances"], repaired["instances"]):
        require(old["tp"] == new["tp"] == 1 and old["role"] == new["role"] == "mixed"
                and old["gpus"] == new["gpus"], "original physical layout changed")
        for key, value in old.items():
            if key in ("container", "provenance", "host_pid"):
                continue
            require(new.get(key) == value, "original instance configuration changed: " + key)
        for key, value in old["container"].items():
            if key != "StartedAt":
                require(new["container"].get(key) == value, "original container changed: " + key)
        for key, value in old["provenance"].items():
            if key != "pid":
                require(new["provenance"].get(key) == value, "original source/model provenance changed: " + key)
        require(type(new.get("host_pid")) is int and new["host_pid"] > 0, "current host PID missing")
    binding = copy.deepcopy(original)
    binding["instances"] = copy.deepcopy(repaired["instances"])
    binding["identity_file"] = repaired["identity_file"]
    binding["files"][binding["identity_file"]] = sha(binding["identity_file"])
    binding["peer_readiness_reference"] = copy.deepcopy(repaired["peer_repair"])
    binding["deadline_s"] = DEADLINE
    return binding


def other_work():
    names = {"run.py", "run_b32.py", "run_selective.py", "child.py", "restore.py", "C_run.py"}
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().decode().split("\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError, UnicodeDecodeError):
            continue
        if argv and "python" in Path(argv[0]).name:
            scripts = [a for a in argv[1:] if a.endswith(".py") and a.startswith(str(CAMPAIGN))]
            if scripts and Path(scripts[0]).name in names:
                found.append({"pid": int(proc.name), "argv": argv[:-1]})
    return found


async def execute(args, cells, evidence, binding, common):
    import aiohttp
    from ecopadg.measure.backends import PynvmlBackend

    require(not args.out.exists(), "new output directory required; no automatic resume/retry")
    require(not other_work(), "another experiment or recovery driver is live")
    require(not (ROOT / "STOP").exists(), "rerun STOP present")
    require(time.time() + 400 < START_CUTOFF, "insufficient time before original start cutoff")
    common.validate_binding(binding)
    async with aiohttp.ClientSession(trust_env=False) as session:
        before = await common.identity(session, binding)
        for record in before:
            owner = next(i for i in binding["instances"] if i["id"] == record["provenance"]["instance_id"])
            require(record["container"]["State"]["Pid"] == owner["host_pid"], "actual host PID changed")
        gpu_pids = await common.command("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits")
        require({int(line.strip()) for line in gpu_pids.splitlines() if line.strip()}
                == {i["host_pid"] for i in binding["instances"]}, "unexpected GPU process owner")
        require(not other_work(), "another experiment appeared before claim")
        claim = dict(pid=os.getpid(), started_s=time.time(), declaration_sha256=DECLARATION_SHA,
                     cell_ids=[c["cell_id"] for c in cells], output=str(args.out),
                     scope="new user-authorized C12 main-rate repeats; no old ablation attempt consumed")
        write(ROOT / "C-execution-claim.json", claim, exclusive=True)
        args.out.mkdir(parents=True, exist_ok=False)
        rows = execution_rows(cells)
        manifest_path = args.out / "execution-manifest.json"
        write(manifest_path, dict(model="7b", protocol_id=binding["protocol_id"], cells=rows,
                                 declaration=str(args.declaration), declaration_sha256=DECLARATION_SHA), exclusive=True)
        write(args.out / "identity.before.json", before, exclusive=True)
        write(args.out / "peer-readiness-audit.json", evidence, exclusive=True)
        binding.update(output=str(args.out / "results"),
                       experiment_scope="user-authorized fixed original main policy, trace701 twice",
                       rerun_declaration=dict(path=str(args.declaration), sha256=DECLARATION_SHA),
                       fresh_main_rate_rerun_binding=True)
        for path in (args.declaration, Path(__file__).resolve(), manifest_path,
                     args.out / "identity.before.json", args.out / "peer-readiness-audit.json",
                     args.repair_spec, args.repair_receipt):
            binding["files"][str(path)] = sha(path)
        binding_path = args.out / "binding.json"
        write(binding_path, binding, exclusive=True)
        common.validate_binding(binding)
        output = Path(binding["output"])
        state = dict(schema="main-rate-rerun-status-v1", model="7b", started_s=time.time(),
                     pid=os.getpid(), fresh_node_lease_held=True, status="running", complete=False,
                     declared=[r["cell_id"] for r in rows], attempted=[], completed=[], failed=[],
                     remaining=[r["cell_id"] for r in rows], binding_path=str(binding_path),
                     binding_sha256=sha(binding_path), declaration_sha256=DECLARATION_SHA,
                     no_automatic_retry=True, start_cutoff_s=START_CUTOFF, deadline_s=DEADLINE)
        write(args.out / "status.json", state, exclusive=True)
        task = asyncio.current_task()
        interrupted = False

        def stop():
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                task.cancel()

        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, stop)
        try:
            hardware = await asyncio.to_thread(PynvmlBackend, power_mode="instant")
            for row in rows:
                if (ROOT / "STOP").exists() or (args.out / "STOP").exists() or time.time() + 400 >= START_CUTOFF:
                    state.update(status="stopped_at_boundary", stop_reason="STOP or start cutoff")
                    break
                check_declaration(args.declaration)
                common.validate_binding(binding)
                state["current_cell"] = row["cell_id"]
                state["attempted"].append(row["cell_id"])
                state["remaining"].remove(row["cell_id"])
                write(args.out / "status.json", state)
                receipt = await common.run_one(session, binding, row, output, hardware)
                require(receipt["measurement_valid"] is True, "invalid measurement retained")
                receipt_path = output / "operations" / row["cell_id"] / "receipt.json"
                artifacts = {str(p): sha(p) for base in (receipt_path.parent, output / "cells" / row["cell_id"])
                             for p in base.rglob("*") if p.is_file()}
                write(output / "checkpoints" / (row["cell_id"] + ".json"),
                      dict(row=row, receipt=str(receipt_path), receipt_sha256=sha(receipt_path),
                           artifacts=artifacts, completed_s=time.time(), measurement_valid=True,
                           work_complete=receipt["summary"].get("work_complete")), exclusive=True)
                state["completed"].append(row["cell_id"])
                state.pop("current_cell", None)
                write(args.out / "status.json", state)
            state["complete"] = len(state["completed"]) == len(rows)
            if state["complete"]:
                state["status"] = "complete"
        except BaseException as exc:
            state.update(status="technical_failure", error=repr(exc))
            if state.get("current_cell"):
                state["failed"].append(dict(cell_id=state["current_cell"], error=repr(exc)))
            raise
        finally:
            state["finished_s"] = time.time()
            write(args.out / "status.json", state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, default=ROOT / "declarations.json")
    parser.add_argument("--repair-spec", type=Path)
    parser.add_argument("--repair-receipt", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    args.declaration = args.declaration.resolve()
    _, cells = check_declaration(args.declaration)
    if not args.run and not (args.repair_spec or args.repair_receipt):
        print(json.dumps(dict(cpu_only=True, declared_cells=len(cells), hardware_actions=False)))
        return
    evidence, repaired = repair_evidence(args)
    binding = original_policy_binding(cells, repaired)
    if not args.run:
        print(json.dumps(dict(cpu_only=True, peer_repair_verified=True, original_policy_verified=True,
                              declared_cells=len(cells), hardware_actions=False)))
        return
    require(args.out is not None and socket.gethostname() == HOST, "new output path on actual C host required")
    args.out = args.out.resolve()
    require("PDBLEND_NODE_LOCK_FD" not in os.environ, "fresh non-inherited original node lease required")
    host = Path(binding["host_release"])
    sys.path[:0] = [str(host / "src"), str(host), "/root/workspace/pdblend/.runtime-deps"]
    os.environ["PYTHONPATH"] = ":".join(sys.path[:3])
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    common = load(COMMON, "c_repeat_frozen_measurement")
    common.GLOBAL_DEADLINE = DEADLINE  # Private wrapper retains the earlier existing deadline.
    from ecopadg.serving.campaign import node_lease
    watcher_lock = CAMPAIGN / "pdblend-ablation-20260908-v1/watcher-7b.lock"
    with watcher_lock.open("a") as watcher:
        fcntl.flock(watcher, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with node_lease():
            _, cells = check_declaration(args.declaration)
            evidence, repaired = repair_evidence(args)
            binding = original_policy_binding(cells, repaired)
            print(json.dumps(asyncio.run(execute(args, cells, evidence, binding, common))))


if __name__ == "__main__":
    main()
