from types import SimpleNamespace
from ecopadg.serving.engine import EngineService


def test_control_query_stops_persistent_tp_execution_loop(tmp_path):
    calls = []
    class Executor:
        running = True
        def stop_remote_worker_execution_loop(self):
            calls.append("stop")
            self.running = False
        def collective_rpc(self, method):
            assert not self.running, "TP workers cannot process RPC inside model loop"
            calls.append("rpc")
            return [method(None)]
    service = EngineService(dict(runtime_dir=str(tmp_path), id="i0"))
    service.engine = SimpleNamespace(model_executor=Executor())
    try:
        assert service.control_rpc(lambda worker: "state") == ["state"]
        assert calls == ["stop", "rpc"]
    finally:
        service.worker.shutdown()
