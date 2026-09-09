"""Host supervisor safety and fake lifecycle tests (never invoke Docker)."""
from __future__ import annotations

import inspect
import threading

import pytest

from ecopadg.engine_supervisor import (
    ConflictError,
    DockerBackend,
    EngineLaunch,
    EngineSupervisor,
    FakeBackend,
    SupervisorPolicy,
    ValidationError,
    create_app,
)
from ecopadg.supervisor_client import (
    InProcessSupervisorClient,
    SupervisorClient,
    SupervisorExecutor,
)
from script.bench.engine_supervisor import create_stdlib_server
from ecopadg.types import InstanceSpec


IMAGE = "vllm-pd:allowed"
MODEL = "/models/Qwen"


def _policy():
    return SupervisorPolicy(
        allowed_images=[IMAGE],
        allowed_models=[MODEL],
        allowed_gpus=range(8),
        allowed_ports=list(range(8100, 8400)) + [14579, 14679],
    )


def _launch(name="mixed-0", gpus=(0, 1), port=8300, **overrides):
    values = dict(
        name=name,
        image=IMAGE,
        model=MODEL,
        role="mixed",
        gpus=tuple(gpus),
        port=port,
        tp=len(gpus),
    )
    values.update(overrides)
    return EngineLaunch(**values)


def test_supervisor_fake_lifecycle_and_health():
    backend = FakeBackend()
    supervisor = EngineSupervisor(_policy(), backend)
    record = supervisor.start(_launch())
    assert record.to_dict()["healthy"]
    assert [item.launch.name for item in supervisor.list()] == ["mixed-0"]
    assert supervisor.health("mixed-0").to_dict()["healthy"]
    stopped = supervisor.stop("mixed-0")
    assert stopped.state == "stopped"
    assert supervisor.list() == []
    assert backend.stopped == ["mixed-0"]


@pytest.mark.parametrize(
    "launch",
    [
        _launch(name="../escape"),
        _launch(image="other"),
        _launch(model="/models/not-allowed"),
        _launch(gpus=(0, 9), tp=2),
        _launch(port=9000),
        _launch(gpus=(0, 0), tp=2),
    ],
)
def test_supervisor_rejects_resources_outside_allowlist(launch):
    supervisor = EngineSupervisor(_policy(), FakeBackend())
    with pytest.raises(ValidationError):
        supervisor.start(launch)


def test_supervisor_reserves_gpu_port_and_name():
    supervisor = EngineSupervisor(_policy(), FakeBackend())
    supervisor.start(_launch())
    with pytest.raises(ConflictError):
        supervisor.start(_launch(name="other", gpus=(2, 3), port=8300))
    with pytest.raises(ConflictError):
        supervisor.start(_launch(name="other", gpus=(1, 2), port=8301))


def test_inprocess_controller_executor_roundtrip():
    backend = FakeBackend()
    supervisor = EngineSupervisor(_policy(), backend)
    executor = SupervisorExecutor(
        InProcessSupervisorClient(supervisor), image=IMAGE
    )
    spec = InstanceSpec(
        role="mixed", model=MODEL, tp=2, gpus=(0, 1)
    )
    handle = executor.start("mixed-0", spec, 8300)
    assert executor.is_healthy("mixed-0", handle, 8300)
    assert executor.list()[0]["name"] == "mixed-0"
    executor.stop("mixed-0", handle)
    assert executor.list() == []


def test_supervised_pd_pair_requires_and_shares_explicit_kv_port():
    supervisor = EngineSupervisor(_policy(), FakeBackend())
    prefill = _launch(
        name="prefill-0",
        role="prefill",
        gpus=(0, 1),
        port=8100,
        tp=2,
        kv_pair=(0, 2, 14579),
        engine_v1=False,
    )
    decode = _launch(
        name="decode-0",
        role="decode",
        gpus=(2, 3),
        port=8200,
        tp=2,
        kv_pair=(1, 2, 14579),
        engine_v1=False,
    )
    supervisor.start(prefill)
    supervisor.start(decode)
    assert len(supervisor.list()) == 2
    with pytest.raises(ConflictError):
        supervisor.start(_launch(
            name="prefill-1",
            role="prefill",
            gpus=(4, 5),
            port=8101,
            tp=2,
            kv_pair=(0, 2, 14579),
            engine_v1=False,
        ))
    with pytest.raises(ValidationError):
        EngineSupervisor(_policy(), FakeBackend()).start(_launch(
            name="prefill-no-port",
            role="prefill",
            gpus=(0, 1),
            port=8100,
            tp=2,
            kv_pair=(0, 2),
            engine_v1=False,
        ))


def test_docker_backend_uses_argv_and_shell_false():
    source = inspect.getsource(DockerBackend._run)
    assert "shell=False" in source
    assert "subprocess.run(" in source


def test_http_app_exposes_health_start_stop_and_list():
    aiohttp = pytest.importorskip("aiohttp")
    if not hasattr(getattr(aiohttp, "web", None), "Application"):
        pytest.skip("aiohttp runtime is not installed")
    app = create_app(EngineSupervisor(_policy(), FakeBackend()))
    routes = {
        (route.method, route.resource.canonical)
        for route in app.router.routes()
    }
    assert ("GET", "/health") in routes
    assert ("GET", "/v1/engines") in routes
    assert ("POST", "/v1/engines/start") in routes
    assert ("POST", "/v1/engines/stop") in routes
    assert ("GET", "/v1/engines/{name}/health") in routes


def test_stdlib_http_backend_roundtrip_without_aiohttp():
    supervisor = EngineSupervisor(_policy(), FakeBackend())
    server = create_stdlib_server(supervisor, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        client = SupervisorClient("http://%s:%d" % (host, port))
        assert client.service_health()["ok"] is True
        engine = client.start_engine(_launch().to_dict())
        assert engine["name"] == "mixed-0"
        assert client.engine_health("mixed-0")["healthy"] is True
        assert len(client.list_engines()) == 1
        client.stop_engine("mixed-0")
        assert client.list_engines() == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

