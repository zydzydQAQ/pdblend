"""Host-side engine lifecycle service.

The controller is intentionally not allowed to invoke Docker directly.  This
module keeps the privileged lifecycle boundary on the host and validates every
image, model, GPU and port against an explicit allow-list before delegating to
an injectable backend.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple

from ecopadg.runner import build_vllm_cmd
from ecopadg.types import (
    ROLE_DECODE,
    ROLE_MIXED,
    ROLE_PREFILL,
    InstanceSpec,
)


ENGINE_STARTING = "starting"
ENGINE_READY = "ready"
ENGINE_UNHEALTHY = "unhealthy"
ENGINE_STOPPING = "stopping"
ENGINE_STOPPED = "stopped"

_ACTIVE_STATES = {
    ENGINE_STARTING,
    ENGINE_READY,
    ENGINE_UNHEALTHY,
    ENGINE_STOPPING,
}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class SupervisorError(RuntimeError):
    """Base class for errors returned by the supervisor."""


class ValidationError(SupervisorError):
    """A launch request is outside the configured policy."""


class ConflictError(SupervisorError):
    """A requested name, GPU, or port is already owned."""


class EngineNotFoundError(SupervisorError):
    """No engine with the requested stable name exists."""


@dataclass(frozen=True)
class EngineLaunch:
    """Fully specified, policy-validated engine launch request."""

    name: str
    image: str
    model: str
    role: str
    gpus: Tuple[int, ...]
    port: int
    tp: int
    max_model_len: int = 8192
    gpu_mem_util: float = 0.85
    kv_pair: Optional[Tuple[int, ...]] = None
    engine_v1: bool = True
    strict_padg: bool = False

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EngineLaunch":
        if not isinstance(payload, Mapping):
            raise ValidationError("request body must be a JSON object")
        allowed = {
            "name",
            "image",
            "model",
            "role",
            "gpus",
            "port",
            "tp",
            "max_model_len",
            "gpu_mem_util",
            "kv_pair",
            "engine_v1",
            "strict_padg",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValidationError(
                "unknown launch fields: %s" % ", ".join(unknown)
            )
        try:
            raw_gpus = payload.get("gpus", ())
            if isinstance(raw_gpus, str):
                raw_gpus = [
                    item.strip() for item in raw_gpus.split(",")
                    if item.strip()
                ]
            gpus = tuple(int(item) for item in raw_gpus)
            raw_pair = payload.get("kv_pair")
            kv_pair = (
                None if raw_pair is None
                else tuple(int(item) for item in raw_pair)
            )
            return cls(
                name=str(payload["name"]),
                image=str(payload["image"]),
                model=str(payload["model"]),
                role=str(payload["role"]),
                gpus=gpus,
                port=int(payload["port"]),
                tp=int(payload.get("tp", len(gpus))),
                max_model_len=int(payload.get("max_model_len", 8192)),
                gpu_mem_util=float(payload.get("gpu_mem_util", 0.85)),
                kv_pair=kv_pair,
                engine_v1=bool(payload.get("engine_v1", True)),
                strict_padg=bool(payload.get("strict_padg", False)),
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValidationError("invalid launch request: %s" % exc) from exc

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "image": self.image,
            "model": self.model,
            "role": self.role,
            "gpus": list(self.gpus),
            "port": int(self.port),
            "tp": int(self.tp),
            "max_model_len": int(self.max_model_len),
            "gpu_mem_util": float(self.gpu_mem_util),
            "kv_pair": list(self.kv_pair) if self.kv_pair is not None else None,
            "engine_v1": bool(self.engine_v1),
            "strict_padg": bool(self.strict_padg),
        }


@dataclass(frozen=True)
class SupervisorPolicy:
    """Exact resources the supervisor is permitted to materialize."""

    allowed_images: frozenset[str]
    allowed_models: frozenset[str]
    allowed_gpus: frozenset[int]
    allowed_ports: frozenset[int]

    def __init__(
        self,
        allowed_images: Iterable[str],
        allowed_models: Iterable[str],
        allowed_gpus: Iterable[int],
        allowed_ports: Iterable[int],
    ):
        object.__setattr__(
            self, "allowed_images",
            frozenset(str(item) for item in allowed_images if str(item)),
        )
        object.__setattr__(
            self, "allowed_models",
            frozenset(str(item) for item in allowed_models if str(item)),
        )
        object.__setattr__(
            self, "allowed_gpus",
            frozenset(int(item) for item in allowed_gpus),
        )
        object.__setattr__(
            self, "allowed_ports",
            frozenset(int(item) for item in allowed_ports),
        )
        if not self.allowed_images:
            raise ValueError("allowed_images must not be empty")
        if not self.allowed_models:
            raise ValueError("allowed_models must not be empty")
        if not self.allowed_gpus:
            raise ValueError("allowed_gpus must not be empty")
        if not self.allowed_ports:
            raise ValueError("allowed_ports must not be empty")

    def validate(self, launch: EngineLaunch) -> None:
        if not _SAFE_NAME.fullmatch(launch.name):
            raise ValidationError("unsafe engine name")
        if launch.image not in self.allowed_images:
            raise ValidationError("image is not allowed: %s" % launch.image)
        if launch.model not in self.allowed_models:
            raise ValidationError("model is not allowed: %s" % launch.model)
        if not launch.gpus or len(set(launch.gpus)) != len(launch.gpus):
            raise ValidationError("gpus must be unique and non-empty")
        if any(gpu < 0 for gpu in launch.gpus):
            raise ValidationError("GPU ids must be non-negative")
        if not set(launch.gpus).issubset(self.allowed_gpus):
            raise ValidationError("one or more GPUs are not allowed")
        if not 1 <= launch.port <= 65535:
            raise ValidationError("port must be in 1..65535")
        if launch.port not in self.allowed_ports:
            raise ValidationError("port is not allowed: %s" % launch.port)
        if launch.tp <= 0 or launch.tp != len(launch.gpus):
            raise ValidationError("tp must equal the number of assigned GPUs")
        if launch.max_model_len <= 0:
            raise ValidationError("max_model_len must be positive")
        if not 0.0 < launch.gpu_mem_util <= 1.0:
            raise ValidationError("gpu_mem_util must be in (0, 1]")
        if launch.role not in (ROLE_MIXED, ROLE_PREFILL, ROLE_DECODE):
            raise ValidationError("unknown role: %s" % launch.role)
        if launch.strict_padg and (
            launch.role != ROLE_MIXED or launch.engine_v1
        ):
            raise ValidationError("strict PaDG requires a V0 mixed engine")
        if launch.role in (ROLE_PREFILL, ROLE_DECODE):
            if launch.engine_v1:
                raise ValidationError("P/D engines require the V0 engine")
            if launch.kv_pair is None:
                raise ValidationError("P/D engine requires kv_pair")
            if len(launch.kv_pair) != 3:
                raise ValidationError(
                    "supervised P/D engine requires (rank, size, kv_port)"
                )
            rank, parallel = launch.kv_pair[:2]
            expected_rank = 0 if launch.role == ROLE_PREFILL else 1
            if rank != expected_rank or parallel != 2:
                raise ValidationError("P/D kv_pair must be (role-rank, 2)")
            kv_port = int(launch.kv_pair[2])
            if not 1 <= kv_port <= 65535:
                raise ValidationError("KV port must be in 1..65535")
            if kv_port not in self.allowed_ports:
                raise ValidationError(
                    "KV port is not allowed: %s" % kv_port
                )
        elif launch.kv_pair is not None:
            raise ValidationError("mixed engine must not set kv_pair")


class EngineBackend(Protocol):
    """Unprivileged interface used by :class:`EngineSupervisor`."""

    def start(self, launch: EngineLaunch) -> object:
        ...

    def stop(self, handle: object, launch: EngineLaunch) -> None:
        ...

    def is_healthy(self, handle: object, launch: EngineLaunch) -> bool:
        ...

    def list(self) -> List[object]:
        ...


class DockerBackend:
    """Production backend using argument-vector subprocess calls only."""

    def __init__(
        self,
        *,
        docker_binary: str = "docker",
        models_host_dir: str = "/root/workspace/models",
        models_container_dir: str = "/models",
        cache_volume: str = "vllm-pd-hf-cache",
        stop_timeout_s: int = 10,
        health_timeout_s: float = 1.0,
    ):
        self.docker_binary = str(docker_binary)
        self.models_host_dir = str(models_host_dir)
        self.models_container_dir = str(models_container_dir)
        self.cache_volume = str(cache_volume)
        self.stop_timeout_s = max(int(stop_timeout_s), 0)
        self.health_timeout_s = max(float(health_timeout_s), 0.1)

    def _run(
        self, argv: List[str], *, check: bool = True
    ) -> subprocess.CompletedProcess:
        # Never use shell=True: every user-controlled value has already passed
        # an exact allow-list and remains one argv element.
        return subprocess.run(
            argv,
            check=check,
            capture_output=True,
            text=True,
            shell=False,
        )

    @staticmethod
    def _container_name(name: str) -> str:
        return "pdblend-%s" % name

    def start(self, launch: EngineLaunch) -> object:
        spec = InstanceSpec(
            role=launch.role,
            model=launch.model,
            tp=launch.tp,
            gpus=launch.gpus,
            max_model_len=launch.max_model_len,
            gpu_mem_util=launch.gpu_mem_util,
        )
        engine_cmd = build_vllm_cmd(
            spec,
            launch.port,
            engine_v1=launch.engine_v1,
            kv_pair=launch.kv_pair,
            strict_padg=launch.strict_padg,
        )
        argv = [
            self.docker_binary,
            "run",
            "-d",
            "--rm",
            "--name",
            self._container_name(launch.name),
            "--label",
            "pdblend.engine-supervisor=1",
            "--gpus",
            "all",
            "--ipc=host",
            "--network",
            "host",
            "-e",
            "CUDA_VISIBLE_DEVICES=%s"
            % ",".join(str(gpu) for gpu in launch.gpus),
            "-v",
            "%s:%s:ro"
            % (self.models_host_dir, self.models_container_dir),
        ]
        if self.cache_volume:
            argv.extend([
                "-v",
                "%s:/root/.cache/huggingface" % self.cache_volume,
                "-e",
                "HF_HOME=/root/.cache/huggingface",
            ])
        if not launch.engine_v1:
            argv.extend(["-e", "VLLM_USE_V1=0"])
        argv.append(launch.image)
        argv.extend(engine_cmd)
        result = self._run(argv)
        container_id = result.stdout.strip()
        if not container_id:
            raise SupervisorError("docker returned an empty container id")
        return container_id

    def stop(self, handle: object, launch: EngineLaunch) -> None:
        container = str(handle or self._container_name(launch.name))
        self._run([
            self.docker_binary,
            "stop",
            "-t",
            str(self.stop_timeout_s),
            container,
        ])
        # --rm normally removes the container.  A best-effort rm covers a
        # daemon race without turning successful stop into a failed RPC.
        self._run(
            [self.docker_binary, "rm", "-f", container],
            check=False,
        )

    def is_healthy(self, handle: object, launch: EngineLaunch) -> bool:
        container = str(handle or self._container_name(launch.name))
        try:
            result = self._run([
                self.docker_binary,
                "inspect",
                "-f",
                "{{.State.Running}}",
                container,
            ])
        except (OSError, subprocess.SubprocessError):
            return False
        if result.stdout.strip().lower() != "true":
            return False
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(
                "http://127.0.0.1:%d/v1/models" % launch.port,
                timeout=self.health_timeout_s,
            ) as response:
                return 200 <= int(response.status) < 300
        except (OSError, urllib.error.URLError, ValueError):
            return False

    def list(self) -> List[object]:
        try:
            result = self._run([
                self.docker_binary,
                "ps",
                "-q",
                "--filter",
                "label=pdblend.engine-supervisor=1",
            ])
        except (OSError, subprocess.SubprocessError):
            return []
        return [line for line in result.stdout.splitlines() if line.strip()]


class FakeBackend:
    """Deterministic backend for tests and ``--fake`` supervisor mode."""

    def __init__(self, *, healthy: bool = True):
        self.healthy_default = bool(healthy)
        self.started: List[EngineLaunch] = []
        self.stopped: List[str] = []
        self.handles: Dict[str, str] = {}
        self.unhealthy: set[str] = set()
        self.fail_start: set[str] = set()
        self.fail_stop: set[str] = set()
        self._counter = 0

    def start(self, launch: EngineLaunch) -> object:
        if launch.name in self.fail_start:
            raise SupervisorError("injected start failure: %s" % launch.name)
        self._counter += 1
        handle = "fake-%d" % self._counter
        self.started.append(launch)
        self.handles[launch.name] = handle
        return handle

    def stop(self, handle: object, launch: EngineLaunch) -> None:
        if launch.name in self.fail_stop:
            raise SupervisorError("injected stop failure: %s" % launch.name)
        self.stopped.append(launch.name)
        self.handles.pop(launch.name, None)

    def is_healthy(self, handle: object, launch: EngineLaunch) -> bool:
        return (
            self.healthy_default
            and launch.name in self.handles
            and launch.name not in self.unhealthy
        )

    def list(self) -> List[object]:
        return list(self.handles.values())


@dataclass
class EngineRecord:
    launch: EngineLaunch
    handle: Optional[object]
    state: str = ENGINE_STARTING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: str = ""

    def to_dict(self) -> dict:
        data = self.launch.to_dict()
        data.update({
            "handle": None if self.handle is None else str(self.handle),
            "state": self.state,
            "healthy": self.state == ENGINE_READY,
            "url": "http://127.0.0.1:%d" % self.launch.port,
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
            "error": self.error,
        })
        return data


class EngineSupervisor:
    """Thread-safe owner of engine names and physical resources."""

    def __init__(
        self,
        policy: SupervisorPolicy,
        backend: EngineBackend,
        *,
        clock=time.time,
    ):
        self.policy = policy
        self.backend = backend
        self._clock = clock
        self._lock = threading.RLock()
        self._records: Dict[str, EngineRecord] = {}

    def _refresh_locked(self, record: EngineRecord) -> None:
        if record.state not in (
            ENGINE_STARTING,
            ENGINE_READY,
            ENGINE_UNHEALTHY,
        ):
            return
        try:
            healthy = bool(
                self.backend.is_healthy(record.handle, record.launch)
            )
        except Exception as exc:  # backend boundary, fail closed
            healthy = False
            record.error = "health:%s" % type(exc).__name__
        record.state = ENGINE_READY if healthy else ENGINE_UNHEALTHY
        record.updated_at = float(self._clock())

    def start(self, launch: EngineLaunch | Mapping[str, Any]) -> EngineRecord:
        if not isinstance(launch, EngineLaunch):
            launch = EngineLaunch.from_dict(launch)
        self.policy.validate(launch)
        with self._lock:
            old = self._records.get(launch.name)
            if old is not None and old.state in _ACTIVE_STATES:
                if old.launch == launch:
                    self._refresh_locked(old)
                    return old
                raise ConflictError("engine name already exists: %s" % launch.name)
            for record in self._records.values():
                if record.state not in _ACTIVE_STATES:
                    continue
                if record.launch.port == launch.port:
                    raise ConflictError("port is already allocated: %s" % launch.port)
                old_kv_port = (
                    int(record.launch.kv_pair[2])
                    if record.launch.kv_pair is not None
                    and len(record.launch.kv_pair) == 3 else None
                )
                new_kv_port = (
                    int(launch.kv_pair[2])
                    if launch.kv_pair is not None
                    and len(launch.kv_pair) == 3 else None
                )
                if (
                    launch.port == old_kv_port
                    or record.launch.port == new_kv_port
                ):
                    raise ConflictError("HTTP/KV port collision")
                if old_kv_port is not None and old_kv_port == new_kv_port:
                    old_rank = int(record.launch.kv_pair[0])
                    new_rank = int(launch.kv_pair[0])
                    if old_rank == new_rank:
                        raise ConflictError(
                            "KV port/rank is already allocated"
                        )
                overlap = set(record.launch.gpus).intersection(launch.gpus)
                if overlap:
                    raise ConflictError(
                        "GPU is already allocated: %s"
                        % ",".join(str(item) for item in sorted(overlap))
                    )
            record = EngineRecord(
                launch=launch,
                handle=None,
                state=ENGINE_STARTING,
                created_at=float(self._clock()),
                updated_at=float(self._clock()),
            )
            # Store the reservation before backend start.  The supervisor lock
            # makes check+reserve+start atomic with respect to concurrent RPCs.
            self._records[launch.name] = record
            try:
                record.handle = self.backend.start(launch)
            except Exception as exc:
                record.state = ENGINE_STOPPED
                record.error = "start:%s" % type(exc).__name__
                record.updated_at = float(self._clock())
                raise SupervisorError(str(exc)) from exc
            self._refresh_locked(record)
            return record

    def stop(self, name: str) -> EngineRecord:
        with self._lock:
            record = self._records.get(str(name))
            if record is None:
                raise EngineNotFoundError("unknown engine: %s" % name)
            if record.state == ENGINE_STOPPED:
                return record
            record.state = ENGINE_STOPPING
            record.updated_at = float(self._clock())
            try:
                self.backend.stop(record.handle, record.launch)
            except Exception as exc:
                record.state = ENGINE_UNHEALTHY
                record.error = "stop:%s" % type(exc).__name__
                record.updated_at = float(self._clock())
                raise SupervisorError(str(exc)) from exc
            record.handle = None
            record.state = ENGINE_STOPPED
            record.error = ""
            record.updated_at = float(self._clock())
            return record

    def health(self, name: str) -> EngineRecord:
        with self._lock:
            record = self._records.get(str(name))
            if record is None:
                raise EngineNotFoundError("unknown engine: %s" % name)
            self._refresh_locked(record)
            return record

    def list(self, *, include_stopped: bool = False) -> List[EngineRecord]:
        with self._lock:
            records = sorted(
                self._records.values(), key=lambda item: item.launch.name
            )
            for record in records:
                self._refresh_locked(record)
            if include_stopped:
                return list(records)
            return [
                record for record in records
                if record.state != ENGINE_STOPPED
            ]

    def healthy(self) -> bool:
        with self._lock:
            records = self.list()
            return all(record.state == ENGINE_READY for record in records)


def create_app(supervisor: EngineSupervisor):
    """Create the small aiohttp JSON RPC application."""
    try:
        from aiohttp import web
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime packaging
        raise RuntimeError("engine supervisor requires aiohttp") from exc

    def _error(exc: Exception):
        if isinstance(exc, ValidationError):
            status = 400
        elif isinstance(exc, ConflictError):
            status = 409
        elif isinstance(exc, EngineNotFoundError):
            status = 404
        else:
            status = 502
        return web.json_response(
            {"ok": False, "error": str(exc), "type": type(exc).__name__},
            status=status,
        )

    async def service_health(_request):
        records = supervisor.list()
        return web.json_response({
            "ok": supervisor.healthy(),
            "engines": len(records),
            "ready": sum(r.state == ENGINE_READY for r in records),
        })

    async def list_engines(request):
        include_stopped = request.query.get("include_stopped", "") in (
            "1", "true", "yes"
        )
        return web.json_response({
            "ok": True,
            "engines": [
                record.to_dict()
                for record in supervisor.list(
                    include_stopped=include_stopped
                )
            ],
        })

    async def start_engine(request):
        try:
            payload = await request.json()
            record = supervisor.start(payload)
        except (
            json.JSONDecodeError,
            SupervisorError,
            TypeError,
            ValueError,
        ) as exc:
            return _error(
                exc if isinstance(exc, SupervisorError)
                else ValidationError(str(exc))
            )
        return web.json_response(
            {"ok": True, "engine": record.to_dict()},
            status=201,
        )

    async def stop_engine(request):
        try:
            payload = await request.json()
            if not isinstance(payload, Mapping) or "name" not in payload:
                raise ValidationError("stop request requires name")
            record = supervisor.stop(str(payload["name"]))
        except (
            json.JSONDecodeError,
            SupervisorError,
            TypeError,
            ValueError,
        ) as exc:
            return _error(
                exc if isinstance(exc, SupervisorError)
                else ValidationError(str(exc))
            )
        return web.json_response({"ok": True, "engine": record.to_dict()})

    async def stop_engine_named(request):
        try:
            record = supervisor.stop(request.match_info["name"])
        except SupervisorError as exc:
            return _error(exc)
        return web.json_response({"ok": True, "engine": record.to_dict()})

    async def engine_health(request):
        try:
            record = supervisor.health(request.match_info["name"])
        except SupervisorError as exc:
            return _error(exc)
        status = 200 if record.state == ENGINE_READY else 503
        return web.json_response(
            {"ok": record.state == ENGINE_READY,
             "engine": record.to_dict()},
            status=status,
        )

    app = web.Application(client_max_size=64 * 1024)
    app.router.add_get("/health", service_health)
    app.router.add_get("/v1/engines", list_engines)
    app.router.add_post("/v1/engines/start", start_engine)
    app.router.add_post("/v1/engines/stop", stop_engine)
    app.router.add_delete("/v1/engines/{name}", stop_engine_named)
    app.router.add_get("/v1/engines/{name}/health", engine_health)
    return app

