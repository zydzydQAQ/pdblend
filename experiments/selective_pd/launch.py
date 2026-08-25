#!/usr/bin/env python3
"""Lifecycle management for vLLM experiment topologies."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO


@dataclass(frozen=True)
class ServerSpec:
    name: str
    gpu: int
    port: int
    kv_config: dict[str, object] | None = None


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen[bytes]
    log_stream: IO[bytes]
    log_path: Path


def _host_ip() -> str:
    try:
        output = subprocess.check_output(
            ["hostname", "-I"], text=True, timeout=5
        ).strip()
        if output:
            return output.split()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return "127.0.0.1"


def _wait_for_health(
    url: str, processes: list[ManagedProcess], timeout_s: float
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "not attempted"
    while time.monotonic() < deadline:
        for managed in processes:
            exit_code = managed.process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"{managed.name} exited with {exit_code}; "
                    f"see {managed.log_path}"
                )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for {url}: {last_error}")


class Topology:
    def __init__(
        self,
        *,
        mode: str,
        model: str,
        log_dir: Path,
        profile_gpu: int = 0,
        prefill_gpu: int = 0,
        decode_gpu: int = 1,
        weights: tuple[int, int] = (1, 1),
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.75,
        kv_buffer_size: float = 3e9,
        kv_connector: str = "nixl",
        startup_timeout_s: float = 900,
    ):
        if mode not in {"profile", "colocated", "disagg"}:
            raise ValueError(f"unsupported topology mode: {mode}")
        self.mode = mode
        self.model = model
        self.log_dir = log_dir
        self.profile_gpu = profile_gpu
        self.prefill_gpu = prefill_gpu
        self.decode_gpu = decode_gpu
        self.weights = weights
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.kv_buffer_size = kv_buffer_size
        self.kv_connector = kv_connector
        self.startup_timeout_s = startup_timeout_s
        self.processes: list[ManagedProcess] = []

    @property
    def base_url(self) -> str:
        return (
            "http://127.0.0.1:8100"
            if self.mode == "profile"
            else "http://127.0.0.1:8000"
        )

    @property
    def measured_gpus(self) -> list[int]:
        if self.mode == "profile":
            return [self.profile_gpu]
        return [0, 1]

    def _server_specs(self) -> list[ServerSpec]:
        if self.mode == "profile":
            return [ServerSpec("profile", self.profile_gpu, 8100)]
        if self.mode == "colocated":
            return [
                ServerSpec("mixed-a100", 0, 8100),
                ServerSpec("mixed-v100", 1, 8200),
            ]
        if self.kv_connector == "nixl":
            config = {
                "kv_connector": "NixlConnector",
                "kv_role": "kv_both",
                "kv_buffer_device": "cuda",
            }
            return [
                ServerSpec("prefill", self.prefill_gpu, 8100, config),
                ServerSpec("decode", self.decode_gpu, 8200, config),
            ]
        if self.kv_connector != "p2p":
            raise ValueError(
                f"unsupported KV connector: {self.kv_connector}"
            )
        producer = {
            "kv_connector": "P2pNcclConnector",
            "kv_role": "kv_producer",
            "kv_port": 21001,
            "kv_buffer_size": self.kv_buffer_size,
            "kv_connector_extra_config": {
                "send_type": "PUT_ASYNC",
                "nccl_num_channels": "8",
            },
        }
        consumer = {
            "kv_connector": "P2pNcclConnector",
            "kv_role": "kv_consumer",
            "kv_port": 22001,
            "kv_buffer_size": self.kv_buffer_size,
            "kv_connector_extra_config": {
                "send_type": "PUT_ASYNC",
                "nccl_num_channels": "8",
            },
        }
        return [
            ServerSpec("prefill", self.prefill_gpu, 8100, producer),
            ServerSpec("decode", self.decode_gpu, 8200, consumer),
        ]

    def _spawn(
        self, name: str, command: list[str], log_path: Path, env: dict[str, str]
    ) -> ManagedProcess:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("wb")
        process = subprocess.Popen(
            command,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        managed = ManagedProcess(name, process, log_stream, log_path)
        self.processes.append(managed)
        return managed

    def start(self) -> None:
        if self.processes:
            raise RuntimeError("topology is already started")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        transfer_host = _host_ip()
        common_env = os.environ.copy()
        common_env.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "VLLM_HOST_IP": transfer_host,
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        try:
            server_processes: list[tuple[ServerSpec, ManagedProcess]] = []
            for spec in self._server_specs():
                env = dict(common_env)
                env["CUDA_VISIBLE_DEVICES"] = str(spec.gpu)
                if self.mode == "disagg" and self.kv_connector == "nixl":
                    env.update(
                        {
                            "UCX_NET_DEVICES": "all",
                            "UCX_TLS": "self,sm,tcp,cuda_copy",
                            "VLLM_KV_CACHE_LAYOUT": "HND",
                            "VLLM_NIXL_SIDE_CHANNEL_PORT": (
                                "5559"
                                if spec.name == "prefill"
                                else "5659"
                            ),
                        }
                    )
                command = [
                    "vllm",
                    "serve",
                    self.model,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(spec.port),
                    "--dtype",
                    "half",
                    "--max-model-len",
                    str(self.max_model_len),
                    "--max-num-batched-tokens",
                    str(self.max_model_len),
                    "--max-num-seqs",
                    "64",
                    "--gpu-memory-utilization",
                    str(self.gpu_memory_utilization),
                    "--enforce-eager",
                    "--no-enable-prefix-caching",
                    "--disable-log-stats",
                ]
                if spec.kv_config is not None:
                    command.extend(
                        [
                            "--kv-transfer-config",
                            json.dumps(spec.kv_config, separators=(",", ":")),
                        ]
                    )
                server_processes.append(
                    (
                        spec,
                        self._spawn(
                            spec.name,
                            command,
                            self.log_dir / f"{spec.name}.log",
                            env,
                        ),
                    )
                )

            for spec, managed in server_processes:
                _wait_for_health(
                    f"http://127.0.0.1:{spec.port}/health",
                    [managed],
                    self.startup_timeout_s,
                )

            if self.mode != "profile":
                proxy_command = [
                    sys.executable,
                    "-m",
                    "experiments.selective_pd.proxy",
                    "--mode",
                    self.mode,
                ]
                if self.mode == "colocated":
                    proxy_command.extend(
                        [
                            "--weights",
                            str(self.weights[0]),
                            str(self.weights[1]),
                        ]
                    )
                else:
                    proxy_command.extend(
                        ["--connector", self.kv_connector]
                    )
                    if self.kv_connector == "p2p":
                        proxy_command.extend(
                            [
                                "--prefill-zmq",
                                f"{transfer_host}:21001",
                                "--decode-zmq",
                                f"{transfer_host}:22001",
                            ]
                        )
                self._spawn(
                    "proxy",
                    proxy_command,
                    self.log_dir / "proxy.log",
                    common_env,
                )
                _wait_for_health(
                    "http://127.0.0.1:8000/health",
                    self.processes,
                    30,
                )
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        for managed in reversed(self.processes):
            if managed.process.poll() is None:
                try:
                    os.killpg(managed.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 20
        for managed in reversed(self.processes):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                managed.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(managed.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                managed.process.wait(timeout=10)
            managed.log_stream.close()
        self.processes.clear()
        time.sleep(2)

    def __enter__(self) -> "Topology":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
