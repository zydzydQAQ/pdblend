"""Read-only idle-host observations. A recommendation does not acquire a lease.

The startup owner calls poll_once repeatedly. This module never starts a monitor,
serving process, container, or clock change, and never writes a status/lock file.
It reads /proc/locks instead of acquiring or creating the experiment lock.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import time

HOSTS = {
    "A": dict(address=None, alias=None, hostname="iZwz92bdfqihqp38tekqjyZ"),
    "B": dict(address="172.16.50.102", alias="39.108.209.97", hostname="iZwz9i5bte3xkpmcoes3t2Z"),
    "C": dict(address="172.16.50.105", alias="47.106.163.29", hostname="iZwz9gfq11hx1sbob59yrgZ"),
}
NODE_LOCK = Path("/root/workspace/pdblend/new-results/campaigns/node-experiment.lock")
SCHEMA = "slo90-idle-observation-v1"
READ_ONLY_PREFIXES = ("mirror", "collect", "report", "audit", "plot", "render", "summarize")
WORK_WORDS = ("qualification", "qualify", "correctness", "continue", "continuation", "waiting",
              "supervise", "supervisor", "runner", "execute", "benchmark", "calibrate", "capacity_load")
ENGINE_MODULES = ("ecopadg.serving.engine", "vllm.entrypoints.openai.api_server",
                  "vllm.entrypoints.api_server", "vllm.engine.multiprocessing.engine")


def _program(argv):
    """Get the executable script/module, never classify an output path as code."""
    if not argv:
        return "", []
    executable = Path(argv[0]).name
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable):
        for index, value in enumerate(argv[1:], 1):
            if value in ("-m", "-c"):
                return (argv[index + 1] if index + 1 < len(argv) else value), argv[index + 2:]
            if value == "-" or not value.startswith("-"):
                return value, argv[index + 1:]
        return "<python-interactive>", []
    return argv[0], argv[1:]


def classify_process(argv):
    """Conservative entrypoint classification, independent of utilization.

    Resident engine leaves are separate from work queues. An observer's output
    directory cannot turn a queue into a read-only process. Waiting continuations
    and qualification owners remain busy even while no GPU is currently active.
    """
    if not argv:
        return "system"
    program, arguments = _program(argv)
    name = Path(program).name.lower()
    stem = name.rsplit(".", 1)[0]
    if (program in ENGINE_MODULES or name in ("ray::raylet", "raylet", "gcs_server")
            or (name == "engine.py" and "--config" in arguments)):
        return "resident_engine"
    if argv[0].startswith("ray::") or any(program.startswith(prefix) for prefix in
                                           ("from multiprocessing.resource_tracker", "from multiprocessing.spawn")):
        return "resident_helper"
    if name in ("bash", "sh", "dash", "zsh") and "-c" in arguments:
        index = arguments.index("-c")
        if index + 1 >= len(arguments):
            return "unknown_controller"
        command = arguments[index + 1]
        try:
            tokens = list(shlex.shlex(command, posix=True, punctuation_chars=True))
        except ValueError:
            return "unknown_controller"
        # A shell waiting to invoke a controller is still its owner. Inspect
        # executable tokens, not text merely containing "reports" or "audit".
        kinds = []
        for position, token in enumerate(tokens):
            if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(token).name):
                kinds.append(classify_process(tokens[position:]))
            elif token.endswith((".py", ".sh")) and ("campaign/" in token or "ecopadg" in token):
                kinds.append(classify_process([token]))
        return "gpu_work" if any(kind in ("gpu_work", "unknown_controller") for kind in kinds) else "system"
    if name == "dispatcher.py" and "--probe-local" in arguments:
        return "read_only"
    if (program == "/root/workspace/pdblend-next-v1/campaign/A14B-sharegpt-slo90-v1/supervisor.py"
            and "--wait-only" in arguments
            and not any(value in arguments for value in ("--run", "--execute", "--launch"))):
        return "read_only"
    explicit_execution = any(value in ("--run", "--execute", "--launch") for value in arguments)
    starts_read_only = any(stem == prefix or stem.startswith(prefix + "_") or stem.startswith(prefix + "-")
                          for prefix in READ_ONLY_PREFIXES)
    if starts_read_only and not explicit_execution:
        # mirror_progress_*until_complete.py observes another queue, while a
        # file explicitly named audit_and_launch.py cannot use this exemption.
        if not any(word in stem for word in ("and_launch", "and_run", "and_execute", "qualification")):
            return "read_only"
    if explicit_execution or any(word in stem for word in WORK_WORDS):
        return "gpu_work"
    if ("campaign/" in program or "campaigns/" in program or program.startswith(("ecopadg.", "benchmarks."))):
        return "gpu_work"
    executable = Path(argv[0]).name
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable) and (
            program in ("-", "<python-interactive>") or (not program.startswith("/") and not program.endswith(".py"))):
        return "unknown_controller"
    return "system"


def process_inventory(proc_root=Path("/proc"), own_pid=None):
    proc_root = Path(proc_root)
    own_pid = os.getpid() if own_pid is None else own_pid
    blockers, engines, observers, errors = [], [], [], []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit() or int(directory.name) == own_pid:
            continue
        try:
            before = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
            if before[0] == "Z":
                continue
            argv = [value.decode(errors="replace") for value in (directory / "cmdline").read_bytes().split(b"\0") if value]
            after = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
            if before[19] != after[19] or after[0] == "Z" or not argv:
                continue
            kind = classify_process(argv)
            row = dict(pid=int(directory.name), ppid=int(after[1]), start_ticks=int(after[19]),
                       entry=_program(argv)[0][:500], kind=kind)
            if kind in ("gpu_work", "unknown_controller"):
                blockers.append(row)
            elif kind in ("resident_engine", "resident_helper"):
                engines.append(row)
            elif kind == "read_only":
                observers.append(row)
        except (FileNotFoundError, ProcessLookupError):
            continue  # A process which actually exited cannot own future work.
        except (OSError, IndexError, ValueError) as exc:
            errors.append(dict(pid=int(directory.name), error=type(exc).__name__))
    return dict(blockers=blockers, resident_engine_count=len(engines), read_only_count=len(observers), errors=errors)


def parse_locks(text, device, inode):
    matches = []
    expected = (os.major(device), os.minor(device), inode)
    for line in text.splitlines():
        fields = line.split()
        for position, field in enumerate(fields):
            if re.fullmatch(r"[0-9a-fA-F]+:[0-9a-fA-F]+:\d+", field):
                major, minor, number = field.split(":")
                if (int(major, 16), int(minor, 16), int(number)) == expected:
                    matches.append(dict(pid=int(fields[position - 1]), type=fields[2] if fields[1] == "->" else fields[1],
                                        waiting="->" in fields, inode=inode))
                break
    return matches


def node_lease_probe(lock_path=NODE_LOCK, proc_root=Path("/proc")):
    """Read kernel lock evidence; never open/create/acquire the lock file."""
    try:
        before = Path(lock_path).stat()
    except FileNotFoundError:
        return dict(free=True, exists=False, identity=None, holders=[])
    text = (Path(proc_root) / "locks").read_text()
    after = Path(lock_path).stat()
    identity = dict(device=before.st_dev, inode=before.st_ino)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise ValueError("node lease inode changed during observation")
    holders = parse_locks(text, before.st_dev, before.st_ino)
    return dict(free=not holders, exists=True, identity=identity, holders=holders)


def parse_gpu_csv(text):
    values = []
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if len(row) != 3:
            raise ValueError("incomplete GPU sensor row")
        index, uuid, utilization = [value.strip() for value in row]
        value = float(utilization)
        if not 0 <= value <= 100 or not uuid:
            raise ValueError("invalid GPU sensor value")
        values.append(dict(index=int(index), uuid=uuid, utilization_percent=value))
    values.sort(key=lambda value: value["index"])
    if [value["index"] for value in values] != list(range(8)) or len({value["uuid"] for value in values}) != 8:
        raise ValueError("all eight unique GPU boards must be observed")
    return values


def current_ancestry(proc_root=Path("/proc"), pid=None):
    """Exact live identity chain used only for an already owned startup claim."""
    current = os.getpid() if pid is None else pid
    result, visited = [], set()
    while current > 0 and current not in visited:
        visited.add(current)
        text = (Path(proc_root) / str(current) / "stat").read_text()
        fields = text.rsplit(") ", 1)[1].split()
        result.append(dict(pid=current, start_ticks=int(fields[19])))
        current = int(fields[1])
    return result


def verify_owned_fd(fd, lock_path=NODE_LOCK, proc_root=Path("/proc")):
    """Verify a real inherited FLOCK on this exact node lock, without acquiring it."""
    if type(fd) is not int or fd < 0:
        raise ValueError("a real owned node-lock file descriptor is required")
    descriptor, target = os.fstat(fd), Path(lock_path).stat()
    if (descriptor.st_dev, descriptor.st_ino) != (target.st_dev, target.st_ino):
        raise ValueError("owned descriptor points to another file")
    text = (Path(proc_root) / "self" / "fdinfo" / str(fd)).read_text()
    locks = parse_locks("\n".join(line for line in text.splitlines() if line.startswith("lock:")),
                        target.st_dev, target.st_ino)
    if not locks or "FLOCK" not in text or "WRITE" not in text or any(row["waiting"] for row in locks):
        raise ValueError("descriptor has no actual exclusive node lease")
    return dict(verified=True, device=target.st_dev, inode=target.st_ino, fd=fd)


def owned_exclusions(identities, ancestry):
    allowed = {(row["pid"], row["start_ticks"]) for row in ancestry}
    supplied = {(row["pid"], row["start_ticks"]) for row in identities}
    if not supplied <= allowed:
        raise ValueError("only exact current process and ancestor identities may be excluded")
    return supplied


def probe_local(*, owned_lock_fd=None, ignore_identities=()):
    """One bounded read-only observation. No background polling is started."""
    result = dict(schema=SCHEMA, hostname=socket.gethostname(), started_s=time.time(),
                  observer_pid=os.getpid(), eligible_snapshot=False, errors=[])
    try:
        before = node_lease_probe()
        processes = process_inventory()
        owned = None
        if owned_lock_fd is not None:
            owned = verify_owned_fd(owned_lock_fd)
            excluded = owned_exclusions(ignore_identities, current_ancestry())
            processes["blockers"] = [row for row in processes["blockers"]
                                      if (row["pid"], row["start_ticks"]) not in excluded]
        elif ignore_identities:
            raise ValueError("process exclusions require an actually owned node lease")
        command = ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu", "--format=csv,noheader,nounits"]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
        if completed.returncode:
            raise RuntimeError("GPU read-only query failed: " + completed.stderr[-500:])
        gpu = parse_gpu_csv(completed.stdout)
        after = node_lease_probe()
        lease = dict(before=before, after=after,
                     free=before["free"] and after["free"] and before["identity"] == after["identity"])
        result.update(node_lease=lease, gpu=gpu, blockers=processes["blockers"][:30],
                      blocker_count=len(processes["blockers"]),
                      resident_engine_count=processes["resident_engine_count"],
                      read_only_count=processes["read_only_count"])
        result["errors"].extend(processes["errors"][:30])
        result["eligible_snapshot"] = bool(lease["free"] and not processes["blockers"]
                                            and not processes["errors"]
                                            and all(value["utilization_percent"] == 0 for value in gpu))
        if owned is not None:
            verify_owned_fd(owned_lock_fd)  # The lock must still be ours after all probes.
            result["owned_node_lease"] = owned
            result["eligible_for_owned_start"] = bool(not processes["blockers"] and not processes["errors"]
                                                        and all(value["utilization_percent"] == 0 for value in gpu))
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        result["errors"].append(dict(error=type(exc).__name__, message=str(exc)[:700]))
    result.update(observed_s=time.time(), recommendation_only=True, acquired_node_lease=False)
    return result


def probe_host(label):
    host = HOSTS[label]
    if socket.gethostname() == host["hostname"]:
        result = probe_local()
    elif host["address"] is None:
        result = dict(schema=SCHEMA, eligible_snapshot=False, hostname=None,
                      observed_s=time.time(), errors=["A must be probed from its local coordinator"])
    else:
        ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o",
               "StrictHostKeyChecking=yes", "-o", "HostKeyAlias=" + host["alias"],
               host["address"], "python3 -B - --probe-local"]
        try:
            completed = subprocess.run(ssh, input=Path(__file__).read_text(), capture_output=True,
                                       text=True, timeout=20)
            if completed.returncode:
                raise RuntimeError("read-only remote probe failed: " + completed.stderr[-500:])
            result = json.loads(completed.stdout)
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            result = dict(schema=SCHEMA, eligible_snapshot=False, hostname=None,
                          observed_s=time.time(), errors=[str(exc)[:700]])
    if result.get("hostname") != host["hostname"]:
        result["eligible_snapshot"] = False
        result.setdefault("errors", []).append("actual host identity differs")
    result["host"] = label
    return result


def new_state():
    return dict(schema=SCHEMA, hosts={label: dict(count=0, samples=[], last_received_s=None,
                                                last_observed_s=None, signature=None)
                                    for label in HOSTS}, candidate=None)


def observe(state, label, evidence, *, received_s=None, min_interval_s=10.0, required=3, max_gap_s=120.0):
    """Return new in-memory state after a fresh observation, without writes."""
    if (state.get("schema") != SCHEMA or label not in HOSTS or type(required) is not int
            or required < 3 or min_interval_s < 1 or max_gap_s < min_interval_s):
        raise ValueError("invalid observation state or stability requirement")
    result = copy.deepcopy(state)
    track = result["hosts"][label]
    now = time.time() if received_s is None else received_s
    if type(now) not in (float, int) or not math.isfinite(now):
        raise ValueError("finite local observation time required")
    observed = evidence.get("observed_s")
    eligible = (evidence.get("schema") == SCHEMA and evidence.get("hostname") == HOSTS[label]["hostname"]
                and evidence.get("eligible_snapshot") is True and not evidence.get("errors")
                and evidence.get("node_lease", {}).get("free") is True and evidence.get("blocker_count") == 0
                and len(evidence.get("gpu", [])) == 8
                and [value.get("index") for value in evidence["gpu"]] == list(range(8))
                and all(value.get("utilization_percent") == 0 for value in evidence["gpu"])
                and len({value.get("uuid") for value in evidence["gpu"]}) == 8
                and all(isinstance(value.get("uuid"), str) and value["uuid"] for value in evidence["gpu"])
                and type(observed) in (float, int) and math.isfinite(observed))
    signature = ([value.get("uuid") for value in evidence.get("gpu", [])],
                 evidence.get("node_lease", {}).get("after", {}).get("identity"))
    fresh = (type(observed) in (float, int) and math.isfinite(observed)
             and (track["last_observed_s"] is None or observed > track["last_observed_s"]))
    spaced = track["last_received_s"] is None or now - track["last_received_s"] >= min_interval_s
    if not eligible or not fresh:
        track.update(count=0, samples=[], signature=None)
        if result["candidate"] == label:
            result["candidate"] = None
    elif spaced:
        if track["signature"] != signature or (track["last_received_s"] is not None and now - track["last_received_s"] > max_gap_s):
            track.update(count=0, samples=[], signature=signature)
        track["count"] += 1
        track["samples"] = (track["samples"] + [copy.deepcopy(evidence)])[-required:]
        track["last_received_s"] = now
        track["last_observed_s"] = observed
        if track["count"] >= required and result["candidate"] is None:
            result["candidate"] = label
    result["recommendation_only"] = True
    result["startup_rule"] = "caller must atomically acquire the node lease and recheck other work before any GPU mutation"
    return result


def poll_once(state, *, min_interval_s=10.0):
    """One A/B/C pass for the startup owner's loop; does not sleep or launch a monitor."""
    result = copy.deepcopy(state)
    evidence = []
    for label in HOSTS:
        observation = probe_host(label)
        evidence.append(observation)
        result = observe(result, label, observation, min_interval_s=min_interval_s)
    return result, dict(schema=SCHEMA, candidate=result["candidate"],
                        consecutive_counts={label: value["count"] for label, value in result["hosts"].items()},
                        observations=evidence, recommendation_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-local", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(probe_local(), sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
