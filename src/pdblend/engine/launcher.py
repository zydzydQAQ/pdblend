"""Launch and manage vLLM V1 instances as subprocesses (one per GPU group)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

MODELS_DIR = Path(os.environ.get("PDBLEND_MODELS_DIR", "/models"))
SERVED_NAME = "m"


@dataclass(frozen=True)
class InstanceSpec:
    instance_id: str
    gpus: tuple[int, ...]
    port: int
    model: str
    tp: int = 1
    max_model_len: int = 8192
    # 0.90 leaves no room for the P2pNccl receive pool + sampler temporaries on the decode side
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    kv_connector: Optional[str] = "P2pNcclConnector"
    kv_role: str = "kv_both"
    kv_port: Optional[int] = None
    side_channel_port: Optional[int] = None
    extra_args: tuple[str, ...] = ()
    # New identity fields are after the historical positional arguments so
    # callers that constructed InstanceSpec positionally keep their meaning.
    pp: int = 1
    generation: int = 0
    pool_id: str = ""
    profile_key: str = ""
    native_control: bool = False

    def __post_init__(self):
        if self.tp < 1 or self.pp < 1 or len(self.gpus) != self.tp * self.pp:
            raise ValueError(f"{self.instance_id}: gpus must contain exactly TP*PP devices")
        if self.kv_connector == "P2pNcclConnector" and self.pp != 1:
            raise ValueError("P2pNcclConnector currently supports symmetric PP1 handoff only")

    @property
    def model_path(self) -> Path:
        path = Path(self.model)
        return path if path.is_absolute() else MODELS_DIR / self.model

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def zmq_address(self) -> str:
        return f"127.0.0.1:{self.kv_port or (self.port + 20000)}"

    @property
    def stage_map(self) -> dict[int, tuple[int, ...]]:
        """Deterministic pipeline stage to local-rank mapping for manifests."""
        if len(self.gpus) != self.tp * self.pp:
            raise ValueError(f"{self.instance_id}: expected TP*PP={self.tp * self.pp} GPUs, got {len(self.gpus)}")
        return {stage: tuple(self.gpus[stage * self.tp:(stage + 1) * self.tp]) for stage in range(self.pp)}

    def kv_transfer_config(self) -> Optional[dict]:
        if not self.kv_connector:
            return None
        if self.kv_connector == "P2pNcclConnector":
            # received KV stays on-GPU up to kv_buffer_size, then spills to a pinned host pool; the GPU
            # budget beside the KV cache is ~4 GiB (CUDA context, NCCL, activations), so keep this small
            return {"kv_connector": self.kv_connector, "kv_role": self.kv_role,
                    "kv_buffer_size": "1e9",
                    "kv_port": self.zmq_address.split(":")[1],
                    "kv_connector_extra_config": {"http_port": str(self.port), "send_type": "PUT_ASYNC",
                                                  "nccl_num_channels": "8", "mem_pool_size_gb": "4"}}
        return {"kv_connector": self.kv_connector, "kv_role": self.kv_role}

    def command(self) -> list[str]:
        cmd = [
            "vllm", "serve", str(self.model_path),
            "--host", "127.0.0.1", "--port", str(self.port),
            "--served-model-name", SERVED_NAME,
            "--tensor-parallel-size", str(self.tp),
            "--pipeline-parallel-size", str(self.pp),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-num-seqs", str(self.max_num_seqs),
            "--max-num-batched-tokens", str(self.max_num_batched_tokens),
            "--dtype", "bfloat16",
            "--no-enable-prefix-caching",
            "--enable-chunked-prefill",
            "--enable-sleep-mode",
            "--disable-log-requests",
        ]
        kv_config = self.kv_transfer_config()
        if kv_config:
            cmd += ["--kv-transfer-config", json.dumps(kv_config)]
        cmd += list(self.extra_args)
        if self.native_control:
            cmd = [sys.executable, '-m', 'pdblend_runtime.serve', *cmd[2:]]
        return cmd

    def environment(self) -> dict[str, str]:
        env = dict(os.environ)
        # Container ordinals and host NVML indices need not agree. Bind both
        # CUDA and the meter to the UUIDs returned by the physical lease.
        # A queue worker restricts the container to lease UUIDs and passes a
        # UUID map for metering; inside that container the launcher receives
        # local ordinals. On a host process (without the map), retain physical
        # ordinals for backwards compatibility.
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpus)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["VLLM_SERVER_DEV_MODE"] = "1"      # exposes /sleep and /wake_up
        env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
        if self.kv_connector == "NixlConnector":
            env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"
            env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(self.side_channel_port or (self.port + 10000))
        elif self.kv_connector == "P2pNcclConnector":
            env["VLLM_HOST_IP"] = "127.0.0.1"
        return env


@dataclass
class Instance:
    spec: InstanceSpec
    log_dir: Path
    process: Optional[subprocess.Popen] = None
    state: str = "off"  # off | starting | ready | sleeping
    sleep_level: int = 0
    events: list = field(default_factory=list)

    def _event(self, kind: str, **extra) -> None:
        self.events.append(dict(t_s=time.time(), instance=self.spec.instance_id, kind=kind, **extra))

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.log_dir / f"{self.spec.instance_id}.log", "ab")
        self.process = subprocess.Popen(self.spec.command(), env=self.spec.environment(),
                                        stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        self.state = "starting"
        self._event("start", pid=self.process.pid)

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _http(self, method: str, path: str, timeout: float = 5.0) -> tuple[int, bytes]:
        req = urllib.request.Request(self.spec.base_url + path, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def healthy(self) -> bool:
        try:
            return self._http("GET", "/health", timeout=2.0)[0] == 200
        except (urllib.error.URLError, OSError):
            return False

    def health_report(self) -> dict:
        report = {"instance_id": self.spec.instance_id, "url": self.spec.base_url,
                  "tp": self.spec.tp, "pp": self.spec.pp, "stage_map": self.spec.stage_map,
                  "alive": self.alive(), "healthy": self.healthy()}
        try:
            rows = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,power.draw",
                 "--format=csv,noheader,nounits"], text=True, timeout=5).splitlines()
            report["gpu_report"] = [x.strip() for x in rows if x.strip()]
        except (OSError, subprocess.SubprocessError):
            report["gpu_report"] = []
        return report

    def wait_ready(self, timeout_s: float = 600.0, poll_s: float = 1.0) -> float:
        started = time.time()
        while time.time() - started < timeout_s:
            if not self.alive():
                raise RuntimeError(f"{self.spec.instance_id} exited with {self.process.returncode}; "
                                   f"see {self.log_dir / (self.spec.instance_id + '.log')}")
            if self.healthy():
                self.state = "ready"
                elapsed = time.time() - started
                self._event("ready", elapsed_s=elapsed, health=self.health_report())
                return elapsed
            time.sleep(poll_s)
        raise TimeoutError(f"{self.spec.instance_id} not ready after {timeout_s}s")

    def stop(self, timeout_s: float = 30.0) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        self._event("stop", returncode=self.process.returncode)
        self.process = None
        self.state = "off"
        self.sleep_level = 0

    def sleep(self, level: int = 1) -> float:
        started = time.time()
        status, body = self._http("POST", f"/sleep?level={level}", timeout=120.0)
        if status != 200:
            raise RuntimeError(f"sleep failed: {status} {body[:200]!r}")
        self.state, self.sleep_level = "sleeping", level
        elapsed = time.time() - started
        self._event("sleep", level=level, elapsed_s=elapsed)
        return elapsed

    def wake_up(self) -> float:
        started = time.time()
        status, body = self._http("POST", "/wake_up", timeout=300.0)
        if status != 200:
            raise RuntimeError(f"wake_up failed: {status} {body[:200]!r}")
        self.state, self.sleep_level = "ready", 0
        elapsed = time.time() - started
        self._event("wake_up", elapsed_s=elapsed)
        return elapsed


def tp_groups(gpus: Sequence[int], tp: int) -> list[tuple[int, ...]]:
    gpus = list(gpus)
    if len(gpus) % tp:
        raise ValueError(f"{len(gpus)} GPUs not divisible by tp={tp}")
    return [tuple(gpus[i:i + tp]) for i in range(0, len(gpus), tp)]


def make_specs(model: str, gpus: Sequence[int], tp: int = 1, base_port: int = 8100,
               *, pp: int = 1, **overrides) -> list[InstanceSpec]:
    if tp < 1 or pp < 1:
        raise ValueError("tp and pp must be positive")
    # One vLLM process owns TP*PP devices.  The stage/rank mapping is kept in
    # the spec; the V1 engine receives the same physical group through
    # CUDA_VISIBLE_DEVICES and creates the pipeline ranks itself.
    return [InstanceSpec(instance_id=f"i{index}", gpus=group, port=base_port + group[0],
                         model=model, tp=tp, pp=pp, **overrides)
            for index, group in enumerate(tp_groups(gpus, tp * pp))]


class Fleet:
    """A set of instances sharing one log directory; starts them concurrently."""

    def __init__(self, specs: Sequence[InstanceSpec], log_dir: Path):
        self.instances = {s.instance_id: Instance(s, Path(log_dir)) for s in specs}

    def __getitem__(self, instance_id: str) -> Instance:
        return self.instances[instance_id]

    def start_all(self, timeout_s: float = 900.0) -> dict[str, float]:
        for inst in self.instances.values():
            inst.start()
        return {k: inst.wait_ready(timeout_s) for k, inst in self.instances.items()}

    def stop_all(self) -> None:
        for inst in self.instances.values():
            inst.stop()

    def events(self) -> list:
        return sorted((e for i in self.instances.values() for e in i.events), key=lambda e: e["t_s"])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop_all()
