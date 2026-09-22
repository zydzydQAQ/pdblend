"""Crash-safe leases for exclusive, resumable GPU campaigns.

The queue is a small JSON database guarded by a side-car ``flock``.  Commits
use fsync plus ``os.replace``; lease tokens prevent old workers from finishing
reclaimed attempts. Every claim gets a new immutable attempt directory.
"""
from __future__ import annotations

import contextlib
import dataclasses
import errno
import fcntl
import json
import os
import pathlib
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence

DEFAULT_HEARTBEAT_S = 30.0
DEFAULT_TTL_S = 180.0


class LeaseError(RuntimeError):
    pass


class LeaseConflict(LeaseError):
    pass


class LeaseExpired(LeaseError):
    pass


class QueueCorrupt(LeaseError):
    pass


@dataclasses.dataclass(frozen=True)
class GPUInfo:
    uuid: str
    index: str | None = None
    pids: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class Job:
    job_id: str
    payload: dict
    priority: int
    max_attempts: int
    attempts: int
    status: str
    created_at: float
    updated_at: float
    lease_id: str | None = None
    last_error: str | None = None


@dataclasses.dataclass(frozen=True)
class Lease:
    lease_id: str
    token: str
    job_id: str
    owner: str
    owner_pid: int
    gpu_uuids: tuple[str, ...]
    attempt: int
    attempt_dir: str
    claimed_at: float
    expires_at: float
    heartbeat_interval_s: float
    gpu_indices: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        return exc.errno != errno.ESRCH
    return True


def gpu_snapshot() -> dict[int, dict]:
    """Return UUID/memory/compute-process information from ``nvidia-smi``."""
    try:
        rows = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return {}
    result: dict[int, dict] = {}
    for line in rows:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        result[idx] = {"uuid": parts[1], "memory_used_mib": int(parts[2]) if len(parts) > 2 else None,
                       "memory_total_mib": int(parts[3]) if len(parts) > 3 else None,
                       "utilization_gpu": int(parts[4]) if len(parts) > 4 else None, "pids": []}
    try:
        proc_rows = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.splitlines()
        by_uuid = {x["uuid"]: x for x in result.values()}
        for line in proc_rows:
            parts = [x.strip() for x in line.split(",", 1)]
            if len(parts) == 2 and parts[0] in by_uuid:
                with contextlib.suppress(ValueError):
                    by_uuid[parts[0]].setdefault("pids", []).append(int(parts[1]))
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _default_probe() -> list[GPUInfo]:
    return [GPUInfo(str(v["uuid"]), str(k), tuple(v.get("pids", ()))) for k, v in gpu_snapshot().items()]


def _safe(value: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_ ." else "_" for c in str(value)).replace(" ", "_")
    return out[:160] or "job"


class GPULeaseQueue:
    """Atomic queue with dependency-aware jobs and GPU UUID leases."""

    _locks: dict[str, threading.RLock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, path: str | os.PathLike[str], *, attempts_dir: str | os.PathLike[str] | None = None,
                 heartbeat_interval_s: float = DEFAULT_HEARTBEAT_S, ttl_s: float = DEFAULT_TTL_S,
                 heartbeat_s: float | None = None,
                 gpu_probe: Callable[[], Iterable[GPUInfo | Mapping]] | None = None,
                 clock: Callable[[], float] = time.time,
                 pid_probe: Callable[[int], bool] = _pid_alive):
        hb = heartbeat_interval_s if heartbeat_s is None else heartbeat_s
        if not (0 < hb < ttl_s):
            raise ValueError("heartbeat interval must be positive and below TTL")
        self.path = pathlib.Path(path).expanduser()
        self.lock_path = pathlib.Path(str(self.path) + ".lock")
        self.attempts_dir = pathlib.Path(attempts_dir) if attempts_dir else self.path.parent / f"{self.path.stem}-attempts"
        self.heartbeat_s, self.ttl_s = float(hb), float(ttl_s)
        self.gpu_probe, self.clock, self.pid_probe = gpu_probe or _default_probe, clock, pid_probe
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.attempts_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @classmethod
    def _thread_lock(cls, path: pathlib.Path) -> threading.RLock:
        key = str(path.resolve())
        with cls._locks_guard:
            return cls._locks.setdefault(key, threading.RLock())

    @contextlib.contextmanager
    def _lock(self):
        with self._thread_lock(self.lock_path):
            with self.lock_path.open("a+") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def _initialize(self):
        with self._lock():
            if not self.path.exists():
                self._write({"schema": 2, "created_at": self.clock(), "jobs": {}, "leases": {}})

    def _read(self) -> dict:
        try:
            state = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise QueueCorrupt(f"cannot read {self.path}: {exc}") from exc
        if not isinstance(state, dict) or not isinstance(state.get("leases", {}), dict):
            raise QueueCorrupt(f"invalid queue state in {self.path}")
        state.setdefault("jobs", {})
        return state

    def _write(self, state: dict):
        fd, tmp = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent, text=True)
        try:
            with os.fdopen(fd, "w") as out:
                json.dump(state, out, sort_keys=True, indent=2, ensure_ascii=False)
                out.write("\n"); out.flush(); os.fsync(out.fileno())
            os.replace(tmp, self.path)
            with contextlib.suppress(OSError):
                dfd = os.open(self.path.parent, os.O_DIRECTORY); os.fsync(dfd); os.close(dfd)
        finally:
            with contextlib.suppress(FileNotFoundError): os.unlink(tmp)

    def _probe(self) -> list[GPUInfo]:
        values = self.gpu_probe() or []
        out = []
        for value in values:
            if isinstance(value, GPUInfo):
                out.append(value)
            else:
                out.append(GPUInfo(str(value["uuid"]), str(value.get("index")) if value.get("index") is not None else None, tuple(int(p) for p in value.get("pids", ()))))
        return out

    def _event(self, lease: Mapping, name: str, fields: Mapping | None = None):
        path = pathlib.Path(lease["attempt_dir"]); path.mkdir(parents=True, exist_ok=True)
        with (path / "events.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps({"event": name, "at": self.clock(), **dict(fields or {})}, sort_keys=True) + "\n")
            out.flush(); os.fsync(out.fileno())

    def _reclaim(self, state: dict, now: float) -> list[str]:
        reclaimed = []
        visible = {g.uuid: g for g in self._probe()}
        for lid, lease in list(state["leases"].items()):
            if lease.get("status") != "active" or now < float(lease.get("expires_at", 0)):
                continue
            owner_live = self.pid_probe(int(lease.get("owner_pid", lease.get("pid", -1))))
            compute_live = any(p for u in lease.get("gpu_uuids", lease.get("gpus", ())) for p in visible.get(u, GPUInfo(u)).pids)
            # Expiry is a recovery request, not permission to steal a live
            # worker's GPUs. A deterministic test can inject pid_probe;
            # production uses the OS process table.
            if owner_live or compute_live or any(u not in visible for u in lease.get("gpu_uuids", ())):
                lease["stale_owner"], lease["stale_compute_process"] = owner_live, compute_live
                continue
            lease["status"], lease["expired_at"] = "expired", now
            reclaimed.append(lid); self._event(lease, "expired", {"owner_live": owner_live, "compute_live": compute_live})
            job = state["jobs"].get(lease.get("job_id"))
            if job and job.get("status") == "running":
                job["lease_id"], job["updated_at"] = None, now
                if int(job.get("attempts", 0)) >= int(job.get("max_attempts", 1)):
                    job["status"], job["last_error"] = "blocked", "lease expired; retry budget exhausted"
                else:
                    job["status"], job["last_error"] = "queued", "lease expired; reclaimed"
        return reclaimed

    @staticmethod
    def _job_obj(job: Mapping) -> Job:
        return Job(str(job["job_id"]), dict(job.get("payload", {})), int(job.get("priority", 0)), int(job.get("max_attempts", 1)), int(job.get("attempts", 0)), str(job["status"]), float(job["created_at"]), float(job["updated_at"]), job.get("lease_id"), job.get("last_error"))

    @staticmethod
    def _lease_obj(lease: Mapping) -> Lease:
        return Lease(str(lease["lease_id"]), str(lease["token"]), str(lease["job_id"]), str(lease["owner"]), int(lease["owner_pid"]), tuple(lease.get("gpu_uuids", lease.get("gpus", ()))), int(lease["attempt"]), str(lease["attempt_dir"]), float(lease["claimed_at"]), float(lease["expires_at"]), float(lease.get("heartbeat_interval_s", DEFAULT_HEARTBEAT_S)), tuple(str(x) for x in lease.get("gpu_indices", ())))

    def enqueue(self, job_id: str, payload: Mapping | None = None, *, priority: int = 0, max_attempts: int = 3, depends_on: Sequence[str] = ()) -> Job:
        if not job_id or max_attempts < 1: raise ValueError("job_id and max_attempts >= 1 required")
        now, spec = self.clock(), dict(payload or {}); spec.setdefault("depends_on", list(depends_on))
        with self._lock():
            state = self._read(); self._reclaim(state, now); old = state["jobs"].get(job_id)
            if old:
                if old.get("payload", {}) != spec or int(old.get("max_attempts", max_attempts)) != max_attempts: raise LeaseConflict(f"job {job_id} immutable spec differs")
                self._write(state); return self._job_obj(old)
            state["jobs"][job_id] = {"job_id": job_id, "payload": spec, "priority": int(priority), "max_attempts": int(max_attempts), "attempts": 0, "status": "queued", "created_at": now, "updated_at": now, "lease_id": None, "last_error": None}
            self._write(state); return self._job_obj(state["jobs"][job_id])

    def _deps_ready(self, state: dict, job: Mapping) -> bool:
        return all(state["jobs"].get(dep, {}).get("status") == "succeeded" for dep in job.get("payload", {}).get("depends_on", ()))

    def claim(self, *, owner: str | None = None, owner_pid: int | None = None,
              gpu_uuids: Sequence[str] | None = None, gpu_count: int | None = None,
              now: float | None = None, lock_mode: bool | None = None) -> Lease | None:
        """Claim a ready job and a disjoint GPU set.

        When no explicit GPUs/count are supplied, select the first ready job
        whose payload ``gpu_count`` fits the currently free devices.  This is
        what permits several profile workers to use disjoint GPU groups.
        ``lock_mode`` filters jobs for the worker's shared (False) or formal
        exclusive (True) host lock.
        """
        owner, owner_pid = owner or f"{os.uname().nodename}:{os.getpid()}", int(owner_pid or os.getpid()); now = self.clock() if now is None else float(now)
        requested = tuple(str(g) for g in (gpu_uuids or ()))
        if len(requested) != len(set(requested)):
            raise LeaseConflict("duplicate GPU UUID")
        selected = requested
        if not selected and gpu_count is not None and gpu_count < 1:
            raise ValueError("gpu_count must be positive")
        with self._lock():
            state = self._read(); self._reclaim(state, now); visible = {g.uuid: g for g in self._probe()}
            used = {u for x in state["leases"].values() if x.get("status") == "active" for u in x.get("gpu_uuids", x.get("gpus", ()))}
            if not visible:
                raise LeaseConflict("GPU inventory is unavailable; refusing an unverifiable lease")
            jobs = [j for j in state["jobs"].values() if j.get("status") == "queued" and self._deps_ready(state, j)]
            jobs.sort(key=lambda j: (-int(j.get("priority", 0)), float(j.get("created_at", 0)), j["job_id"]))
            if lock_mode is not None:
                jobs = [j for j in jobs if bool(j.get("payload", {}).get("global_lock", j.get("payload", {}).get("exclusive", False))) == lock_mode]
            available = tuple(g.uuid for g in visible.values() if g.uuid not in used)
            if not selected:
                if gpu_count is not None:
                    candidates = [j for j in jobs if int(j.get("payload", {}).get("gpu_count", gpu_count)) == gpu_count]
                else:
                    candidates = jobs
                chosen = None
                requested_count = None
                for candidate in candidates:
                    # An unspecified placement requirement means one GPU.
                    # Falling back to all free devices serialized ordinary
                    # jobs whenever workers omitted ``gpu_count``.
                    count = int(candidate.get("payload", {}).get("gpu_count", gpu_count or 1))
                    if count < 1 or len(available) < count:
                        continue
                    if (bool(candidate.get("payload", {}).get("exclusive", False))
                            or bool(candidate.get("payload", {}).get("global_lock", False))) and count != len(visible):
                        continue
                    chosen, requested_count = candidate, count
                    break
                if chosen is None:
                    self._write(state)
                    return None
                selected = available[:requested_count]
                gpu_count = requested_count
            if gpu_count is not None and len(selected) != gpu_count:
                raise LeaseConflict(f"requested {gpu_count} GPUs but only {len(selected)} are available")
            if any(g not in visible for g in selected): raise LeaseConflict("requested GPU UUID is not visible")
            if any(g in used for g in selected): raise LeaseConflict("requested GPU is already leased")
            foreign = {g: visible[g].pids for g in selected if g in visible and any(p != owner_pid for p in visible[g].pids)}
            if foreign: raise LeaseConflict(f"GPU process conflict: {foreign}")
            if not jobs: self._write(state); return None
            jobs = [j for j in jobs if int(j["payload"].get("gpu_count", len(selected))) == len(selected)
                    and (not (j["payload"].get("exclusive", False)
                              or j["payload"].get("global_lock", False))
                         or set(selected) == set(visible))]
            if not jobs:
                self._write(state)
                return None
            job, attempt = jobs[0], int(jobs[0].get("attempts", 0)) + 1
            if attempt > int(job.get("max_attempts", 1)): job["status"] = "blocked"; self._write(state); return None
            lid, token = uuid.uuid4().hex, uuid.uuid4().hex; adir = self.attempts_dir / _safe(job["job_id"]) / f"attempt-{attempt:04d}-{lid}"; adir.mkdir(parents=True, exist_ok=False)
            gpu_indices = tuple(str(visible[g].index) for g in selected)
            manifest = {"schema": 1, "immutable": True, "job_id": job["job_id"], "attempt": attempt, "lease_id": lid, "owner": owner, "owner_pid": owner_pid, "gpu_uuids": list(selected), "gpu_indices": list(gpu_indices), "claimed_at": now, "payload": job.get("payload", {})}
            with (adir / "manifest.json").open("x") as out: json.dump(manifest, out, sort_keys=True, indent=2); out.write("\n"); out.flush(); os.fsync(out.fileno())
            lease = {"lease_id": lid, "token": token, "job_id": job["job_id"], "owner": owner, "owner_pid": owner_pid, "gpu_uuids": list(selected), "gpu_indices": list(gpu_indices), "gpus": list(selected), "attempt": attempt, "attempt_dir": str(adir), "claimed_at": now, "heartbeat_s": now, "expires_at": now + self.ttl_s, "heartbeat_interval_s": self.heartbeat_s, "status": "active"}
            state["leases"][lid] = lease; job.update({"status": "running", "lease_id": lid, "attempts": attempt, "updated_at": now, "last_error": None}); self._event(lease, "claimed", {"gpu_uuids": list(selected)}); self._write(state); return self._lease_obj(lease)

    def _resolve(self, state: dict, lease_id: str, token: str | None) -> dict:
        lease = state["leases"].get(lease_id)
        if not token:
            raise LeaseExpired("lease token is required")
        if not lease or lease.get("status") != "active" or token != lease.get("token"): raise LeaseExpired(f"invalid or inactive lease: {lease_id}")
        if self.clock() >= float(lease.get("expires_at", 0)): raise LeaseExpired(f"lease expired: {lease_id}")
        return lease

    def heartbeat(self, lease_id: str, token: str | None = None, *, owner_pid: int | None = None, process_pids: Sequence[int] = ()) -> Lease:
        now = self.clock()
        with self._lock():
            state = self._read(); lease = self._resolve(state, lease_id, token); pid = int(owner_pid or lease["owner_pid"])
            if not self.pid_probe(pid): raise LeaseExpired(f"owner process is gone: {pid}")
            previous = {int(x) for x in lease.get("process_pids", ()) if self.pid_probe(int(x))}
            allowed = {pid, int(lease["owner_pid"])} | previous | {int(x) for x in process_pids}; visible = {g.uuid: g for g in self._probe()}
            foreign = {g: visible[g].pids for g in lease.get("gpu_uuids", ()) if g in visible and any(p not in allowed for p in visible[g].pids)}
            if foreign: raise LeaseConflict(f"foreign GPU process detected: {foreign}")
            lease.update({"heartbeat_s": now, "expires_at": now + self.ttl_s, "process_pids": sorted(allowed)}); self._event(lease, "heartbeat", {"process_pids": sorted(allowed)}); self._write(state); return self._lease_obj(lease)

    def _require_released(self, lease: Mapping) -> None:
        visible = {g.uuid: g for g in self._probe()}
        if any(g not in visible for g in lease["gpu_uuids"]):
            raise LeaseConflict("GPU inventory unavailable during release")
        if any(visible[g].pids for g in lease["gpu_uuids"]):
            raise LeaseConflict("GPU compute processes remain during release")

    def complete(self, lease_id: str, token: str | None = None, *, status: str = "succeeded", metadata: Mapping | None = None) -> Job:
        if status not in {"succeeded", "failed", "cancelled"}: raise ValueError(status)
        with self._lock():
            state = self._read(); lease = self._resolve(state, lease_id, token); self._require_released(lease); job = state["jobs"][lease["job_id"]]; self._event(lease, status, metadata); lease["status"] = status; job.update({"status": status, "lease_id": None, "updated_at": self.clock()}); self._write(state); return self._job_obj(job)

    def retry(self, lease_id: str, token: str | None = None, *, error: str | None = None) -> Job:
        with self._lock():
            state = self._read(); lease = self._resolve(state, lease_id, token); self._require_released(lease); job = state["jobs"][lease["job_id"]]; blocked = int(job.get("attempts", 0)) >= int(job.get("max_attempts", 1)); self._event(lease, "blocked" if blocked else "retry", {"error": error}); lease["status"] = "blocked" if blocked else "failed"; job.update({"status": "blocked" if blocked else "queued", "lease_id": None, "last_error": error, "updated_at": self.clock()}); self._write(state); return self._job_obj(job)

    def reclaim(self) -> list[str]:
        with self._lock():
            state = self._read(); ids = self._reclaim(state, self.clock()); self._write(state); return ids

    def block(self, job_id: str, *, reason: str = "blocked") -> Job:
        with self._lock():
            state = self._read(); job = state["jobs"].get(job_id)
            if not job: raise KeyError(job_id)
            if job.get("status") == "running": raise LeaseConflict("stop the worker and release its GPUs before blocking a job")
            job.update({"status": "blocked", "lease_id": None, "last_error": reason, "updated_at": self.clock()}); self._write(state); return self._job_obj(job)

    def list_jobs(self, status: str | None = None) -> list[Job]:
        with self._lock():
            state = self._read(); changed = bool(self._reclaim(state, self.clock()));
            if changed: self._write(state)
            result = [self._job_obj(x) for x in state["jobs"].values() if status is None or x.get("status") == status]; return sorted(result, key=lambda j: (-j.priority, j.created_at, j.job_id))

    def active_leases(self) -> list[Lease]:
        with self._lock():
            state = self._read(); changed = bool(self._reclaim(state, self.clock()));
            if changed: self._write(state)
            return [self._lease_obj(x) for x in state["leases"].values() if x.get("status") == "active"]

    def snapshot(self) -> dict:
        with self._lock():
            state = self._read(); changed = bool(self._reclaim(state, self.clock()));
            if changed: self._write(state)
            return json.loads(json.dumps(state))


def attempt_dir(root: str | os.PathLike[str], *, run_id: str, point: str, attempt: int) -> pathlib.Path:
    path = pathlib.Path(root) / _safe(run_id) / _safe(point) / f"attempt-{attempt:02d}"; path.mkdir(parents=True, exist_ok=False)
    payload = {"run_id": run_id, "point": point, "attempt": attempt, "immutable": True}
    (path / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    (path / "status.json").write_text(json.dumps({"status": "running", "attempt": attempt}, indent=2) + "\n")
    return path


GpuLeaseQueue = GPULeaseQueue
