"""Explicit bridge between EcoSpdController runtime objects and PoolManager."""
from __future__ import annotations

import time
import urllib.parse
from typing import Dict, List, Tuple

from ecopadg.pool_manager import (
    STATE_DRAINING,
    STATE_PARKED,
    STATE_READY,
    PoolManager,
)
from ecopadg.supervisor_client import (
    SupervisorClient,
    SupervisorExecutor,
    SupervisorHandle,
)
from ecopadg.topology import TopologyCoordinator
from ecopadg.types import (
    ROLE_DECODE,
    ROLE_MIXED,
    ROLE_PREFILL,
    InstanceSpec,
)


class TopologyAdapterError(RuntimeError):
    """Controller endpoints do not match supervisor-owned engines."""


def _port(url: str) -> int:
    parsed = urllib.parse.urlsplit(str(url))
    if parsed.scheme not in ("http", "https") or parsed.port is None:
        raise TopologyAdapterError("engine endpoint has no valid port: %s" % url)
    return int(parsed.port)


class ControllerTopologyAdapter:
    """Adopt supervisor engines and rebuild request-plane objects on commit."""

    def __init__(self, controller, *, client=None):
        self.controller = controller
        args = controller.args
        self.client = client or SupervisorClient(
            str(args.supervisor_url), timeout_s=5.0
        )
        records = self.client.list_engines()
        expected = self._expected_engines()
        by_port: Dict[int, dict] = {}
        for record in records:
            try:
                port = int(record["port"])
            except (KeyError, TypeError, ValueError) as exc:
                raise TopologyAdapterError(
                    "supervisor record has invalid port"
                ) from exc
            if port in by_port:
                raise TopologyAdapterError(
                    "supervisor returned duplicate port %d" % port
                )
            by_port[port] = record
        expected_ports = {item[1] for item in expected}
        if expected_ports != set(by_port):
            raise TopologyAdapterError(
                "controller endpoints are not exactly supervisor-owned"
            )
        images = {str(record.get("image") or "") for record in records}
        models = {str(record.get("model") or "") for record in records}
        if len(images) != 1 or "" in images:
            raise TopologyAdapterError("engines must use one known image")
        if len(models) != 1 or "" in models:
            raise TopologyAdapterError("engines must use one known model")
        image = next(iter(images))
        model = next(iter(models))
        self.executor = SupervisorExecutor(
            self.client,
            image=image,
            mixed_engine_v1=not bool(controller.strict_padg),
            strict_padg=bool(controller.strict_padg),
        )
        self.pool_manager = PoolManager(
            model=model,
            tp=int(controller.mcfg["tp"]),
            gpus=range(int(args.gpu_count)),
            executor=self.executor,
            port_base=min(expected_ports or {8100}),
            kv_port_base=14579,
            kv_port_stride=100,
        )
        for role, port, gpus, segment, parked in expected:
            record = by_port[port]
            record_role = str(record.get("role") or "")
            record_gpus = tuple(int(item) for item in record.get("gpus", ()))
            if record_role != role or record_gpus != tuple(gpus):
                raise TopologyAdapterError(
                    "controller/supervisor engine specification mismatch"
                )
            if not (
                bool(record.get("healthy"))
                or str(record.get("state")) == "ready"
            ):
                raise TopologyAdapterError(
                    "cannot adopt unhealthy engine: %s"
                    % record.get("name", port)
                )
            raw_kv_pair = record.get("kv_pair")
            kv_pair = (
                tuple(int(item) for item in raw_kv_pair)
                if raw_kv_pair is not None else None
            )
            if role in (ROLE_PREFILL, ROLE_DECODE):
                if kv_pair is None or len(kv_pair) != 3:
                    raise TopologyAdapterError(
                        "supervised P/D record requires a unique KV port"
                    )
                expected_rank = 0 if role == ROLE_PREFILL else 1
                if (
                    int(kv_pair[0]) != expected_rank
                    or int(kv_pair[1]) != 2
                ):
                    raise TopologyAdapterError(
                        "supervised P/D record has invalid KV rank"
                    )
            spec = InstanceSpec(
                role=role,
                model=model,
                tp=len(gpus),
                gpus=tuple(gpus),
                max_model_len=int(record.get("max_model_len", 8192)),
                gpu_mem_util=float(record.get("gpu_mem_util", 0.85)),
            )
            self.pool_manager.adopt_instance(
                str(record["name"]),
                spec,
                port,
                SupervisorHandle(
                    name=str(record["name"]),
                    backend_id=str(record.get("handle") or ""),
                ),
                state=STATE_PARKED if parked else STATE_READY,
                segment=segment,
                kv_pair=kv_pair,
            )
        self.topology = TopologyCoordinator(
            self.pool_manager,
            controller.partition,
            min_dwell_s=float(args.topology_min_dwell),
            role_switch_cost_j=float(args.role_switch_cost_j),
            drain_timeout_s=float(controller.cfg.drain_timeout_s),
            health_timeout_s=max(
                float(getattr(args, "topology_health_timeout", 600.0)),
                1.0,
            ),
            queue_empty=controller._topology_queues_empty,
            slo_safe=controller._topology_slo_safe,
            trusted=controller._topology_trusted,
            canary_validator=self.validate_runtime,
            on_commit=self.apply_runtime,
            on_rollback=self.apply_runtime,
            # Spatial dwell starts when the supervisor attachment becomes
            # active; L3 park/unpark has an independent dwell/cost domain.
            last_switch=time.time(),
        )
        self._bind_runtime_handles()

    def _expected_engines(
        self,
    ) -> List[Tuple[str, int, Tuple[int, ...], int | None, bool]]:
        expected = []
        for mixed in self.controller.mixed:
            expected.append((
                ROLE_MIXED,
                _port(mixed.url),
                tuple(int(gpu) for gpu in mixed.gpus),
                None,
                bool(mixed.parked),
            ))
        for index, segment in enumerate(self.controller.segments):
            expected.append((
                ROLE_PREFILL,
                _port(segment.purl),
                tuple(int(gpu) for gpu in segment.pgpus),
                index,
                bool(segment.parked),
            ))
            expected.append((
                ROLE_DECODE,
                _port(segment.durl),
                tuple(int(gpu) for gpu in segment.dgpus),
                index,
                bool(segment.parked),
            ))
        if not expected:
            raise TopologyAdapterError("controller has no engines to adopt")
        return expected

    def validate_runtime(self, target) -> bool:
        live = self.pool_manager._live()
        if any(
            instance.state not in (STATE_READY, STATE_PARKED)
            for instance in live
        ):
            return False
        mixed = [
            item for item in live
            if item.spec.role == ROLE_MIXED and item.state == STATE_READY
        ]
        prefills = [
            item for item in live
            if item.spec.role == ROLE_PREFILL and item.state == STATE_READY
        ]
        decodes = [
            item for item in live
            if item.spec.role == ROLE_DECODE and item.state == STATE_READY
        ]
        if (
            len(mixed) != int(target.n_mixed)
            or len(prefills) != int(target.n_prefill)
            or len(decodes) != int(target.n_decode)
        ):
            return False
        pre_segments = {item.segment for item in prefills}
        dec_segments = {item.segment for item in decodes}
        return (
            pre_segments == dec_segments
            and (not pre_segments or None not in pre_segments)
        )

    def _by_port(self) -> Dict[int, object]:
        return {
            int(instance.port): instance
            for instance in self.pool_manager._live()
        }

    def _bind_runtime_handles(self) -> None:
        by_port = self._by_port()
        for mixed in self.controller.mixed:
            record = by_port[_port(mixed.url)]
            mixed.name = record.name
            mixed.handle = record.handle
        for segment in self.controller.segments:
            prefill = by_port[_port(segment.purl)]
            decode = by_port[_port(segment.durl)]
            segment.prefill_name = prefill.name
            segment.decode_name = decode.name
            segment.prefill_handle = prefill.handle
            segment.decode_handle = decode.handle

    def reconcile_controller(self) -> None:
        """Mirror existing inplace park/drain state before a restart request."""
        if not self.topology.is_steady:
            return
        by_port = self._by_port()
        for mixed in self.controller.mixed:
            record = by_port[_port(mixed.url)]
            record.inflight = int(mixed.inflight)
            if mixed.draining:
                record.state = STATE_DRAINING
            elif mixed.parked:
                record.state = STATE_PARKED
            else:
                record.state = STATE_READY
        for segment in self.controller.segments:
            for url in (segment.purl, segment.durl):
                record = by_port[_port(url)]
                record.inflight = int(segment.inflight)
                if segment.draining:
                    record.state = STATE_DRAINING
                elif segment.parked:
                    record.state = STATE_PARKED
                else:
                    record.state = STATE_READY
        self.topology.reconcile_steady(
            self.controller.partition,
        )

    def apply_runtime(self, partition) -> None:
        """Publish validated PoolManager handles/URLs to the request plane."""
        self.controller._rebuild_runtime_from_pool(
            self.pool_manager, partition
        )


class _UnavailableExecutor:
    def start(self, *args, **kwargs):
        raise RuntimeError("supervisor unavailable")

    def stop(self, *args, **kwargs):
        raise RuntimeError("supervisor unavailable")

    def is_healthy(self, *args, **kwargs):
        return False


def degraded_topology(controller, error: str) -> TopologyCoordinator:
    """Build a status-bearing fail-closed coordinator after attach failure."""
    pool_manager = PoolManager(
        model="unavailable",
        tp=int(controller.mcfg["tp"]),
        gpus=range(int(controller.args.gpu_count)),
        executor=_UnavailableExecutor(),
    )
    topology = TopologyCoordinator(
        pool_manager,
        controller.partition,
        min_dwell_s=float(controller.args.topology_min_dwell),
        role_switch_cost_j=float(controller.args.role_switch_cost_j),
        clock=time.time,
        last_switch=float(controller.last_switch),
    )
    topology.fail_closed("adapter:%s" % error)
    return topology

