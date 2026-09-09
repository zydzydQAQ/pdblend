"""Controller-side JSON RPC client for the host engine supervisor."""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence

from ecopadg.types import ROLE_MIXED, InstanceSpec


class SupervisorRPCError(RuntimeError):
    """The host supervisor rejected or could not complete an RPC."""

    def __init__(self, message: str, *, status: int = 0):
        super().__init__(message)
        self.status = int(status)


@dataclass(frozen=True)
class SupervisorHandle:
    """Stable logical name plus the backend's current opaque handle."""

    name: str
    backend_id: str = ""


class SupervisorClient:
    """Small synchronous client suitable for PoolManager's executor API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 5.0,
        opener=None,
    ):
        url = str(base_url or "").strip().rstrip("/")
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("supervisor URL must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError("credentials must not be embedded in supervisor URL")
        self.base_url = url
        self.timeout_s = max(float(timeout_s), 0.1)
        self._opener = opener or urllib.request.build_opener(
            urllib.request.ProxyHandler({})
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(dict(payload)).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=str(method).upper(),
        )
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                raw = response.read()
                status = int(getattr(response, "status", 200))
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                detail = json.loads(raw.decode("utf-8")).get("error", "")
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                detail = ""
            raise SupervisorRPCError(
                detail or "supervisor HTTP %d" % exc.code,
                status=int(exc.code),
            ) from exc
        except (OSError, urllib.error.URLError) as exc:
            raise SupervisorRPCError(
                "supervisor unavailable: %s" % type(exc).__name__
            ) from exc
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SupervisorRPCError(
                "supervisor returned invalid JSON", status=status
            ) from exc
        if not isinstance(body, dict):
            raise SupervisorRPCError(
                "supervisor response must be an object", status=status
            )
        if not (200 <= status < 300) or body.get("ok") is False:
            raise SupervisorRPCError(
                str(body.get("error") or "supervisor request failed"),
                status=status,
            )
        return body

    def service_health(self) -> dict:
        return self._request("GET", "/health")

    def list_engines(self) -> List[dict]:
        body = self._request("GET", "/v1/engines")
        engines = body.get("engines", [])
        if not isinstance(engines, list) or not all(
            isinstance(item, dict) for item in engines
        ):
            raise SupervisorRPCError("invalid engines response")
        return list(engines)

    # Compatibility alias for executor-style callers.
    list = list_engines

    def start_engine(self, payload: Mapping[str, Any]) -> dict:
        body = self._request("POST", "/v1/engines/start", payload)
        engine = body.get("engine")
        if not isinstance(engine, dict):
            raise SupervisorRPCError("start response omitted engine")
        return engine

    def stop_engine(self, name: str) -> dict:
        body = self._request(
            "POST", "/v1/engines/stop", {"name": str(name)}
        )
        engine = body.get("engine")
        if not isinstance(engine, dict):
            raise SupervisorRPCError("stop response omitted engine")
        return engine

    def engine_health(self, name: str) -> dict:
        body = self._request(
            "GET",
            "/v1/engines/%s/health"
            % urllib.parse.quote(str(name), safe=""),
        )
        engine = body.get("engine")
        if not isinstance(engine, dict):
            raise SupervisorRPCError("health response omitted engine")
        return engine


class InProcessSupervisorClient:
    """No-socket adapter for deterministic unit and integration tests."""

    def __init__(self, supervisor):
        self.supervisor = supervisor

    def service_health(self) -> dict:
        records = self.supervisor.list()
        return {
            "ok": self.supervisor.healthy(),
            "engines": len(records),
        }

    def list_engines(self) -> List[dict]:
        return [record.to_dict() for record in self.supervisor.list()]

    list = list_engines

    def start_engine(self, payload: Mapping[str, Any]) -> dict:
        return self.supervisor.start(payload).to_dict()

    def stop_engine(self, name: str) -> dict:
        return self.supervisor.stop(name).to_dict()

    def engine_health(self, name: str) -> dict:
        record = self.supervisor.health(name)
        if not record.to_dict().get("healthy"):
            raise SupervisorRPCError("engine is unhealthy", status=503)
        return record.to_dict()


class SupervisorExecutor:
    """PoolManager executor implemented by supervisor RPCs."""

    def __init__(
        self,
        client,
        *,
        image: str,
        mixed_engine_v1: bool = True,
        strict_padg: bool = False,
    ):
        if not str(image):
            raise ValueError("supervisor executor requires an image")
        self.client = client
        self.image = str(image)
        self.mixed_engine_v1 = bool(mixed_engine_v1)
        self.strict_padg = bool(strict_padg)

    @staticmethod
    def _name(handle: object, fallback: str) -> str:
        if isinstance(handle, SupervisorHandle):
            return handle.name
        if isinstance(handle, Mapping) and handle.get("name"):
            return str(handle["name"])
        return str(fallback)

    def start(
        self,
        name: str,
        spec: InstanceSpec,
        port: int,
        kv_pair: Optional[Sequence[int]] = None,
    ) -> SupervisorHandle:
        engine_v1 = (
            self.mixed_engine_v1 if spec.role == ROLE_MIXED else False
        )
        engine = self.client.start_engine({
            "name": str(name),
            "image": self.image,
            "model": spec.model,
            "role": spec.role,
            "gpus": list(spec.gpus),
            "port": int(port),
            "tp": int(spec.tp),
            "max_model_len": int(spec.max_model_len),
            "gpu_mem_util": float(spec.gpu_mem_util),
            "kv_pair": list(kv_pair) if kv_pair is not None else None,
            "engine_v1": bool(engine_v1),
            "strict_padg": bool(
                self.strict_padg and spec.role == ROLE_MIXED
            ),
        })
        return SupervisorHandle(
            name=str(engine.get("name") or name),
            backend_id=str(engine.get("handle") or ""),
        )

    def stop(self, name: str, handle: object) -> None:
        self.client.stop_engine(self._name(handle, name))

    def is_healthy(self, name: str, handle: object, port: int) -> bool:
        del port
        try:
            engine = self.client.engine_health(
                self._name(handle, name)
            )
        except SupervisorRPCError:
            return False
        return bool(
            engine.get("healthy") or engine.get("state") == "ready"
        )

    def list(self) -> List[dict]:
        return self.client.list_engines()

    def canary(self, name: str, handle: object, port: int) -> bool:
        """A second fail-closed readiness probe used during validation."""
        return self.is_healthy(name, handle, port)

