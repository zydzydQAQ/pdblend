import json

from pdblend.engine.client import PDTransfer, remote_decode_params, remote_prefill_params
from pdblend.engine.launcher import InstanceSpec, make_specs, tp_groups


def test_spec_command_and_env():
    spec = InstanceSpec("i0", (3,), 8103, "/m/Qwen", tp=1, kv_connector="NixlConnector")
    cmd = spec.command()
    assert cmd[:3] == ["vllm", "serve", "/m/Qwen"]
    assert "--enable-sleep-mode" in cmd and "--enable-chunked-prefill" in cmd
    kv = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
    assert kv == {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
    env = spec.environment()
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["VLLM_NIXL_SIDE_CHANNEL_PORT"] == "18103"


def test_spec_without_connector():
    spec = InstanceSpec("i0", (0,), 8100, "/m", kv_connector=None)
    assert "--kv-transfer-config" not in spec.command()
    assert "VLLM_NIXL_SIDE_CHANNEL_PORT" not in spec.environment()


def test_tp_groups_and_make_specs():
    assert tp_groups(range(8), 2) == [(0, 1), (2, 3), (4, 5), (6, 7)]
    specs = make_specs("Qwen", range(4), tp=2, base_port=9000)
    assert [s.port for s in specs] == [9000, 9002]
    assert specs[1].gpus == (2, 3)


def test_kv_transfer_param_roundtrip():
    prefill = remote_decode_params()
    assert prefill["do_remote_decode"] and not prefill["do_remote_prefill"]
    handoff = dict(remote_engine_id="e", remote_block_ids=[1, 2], remote_host="h", remote_port=5)
    decode = remote_prefill_params(handoff)
    assert decode["do_remote_prefill"] and not decode["do_remote_decode"]
    assert decode["remote_block_ids"] == [1, 2]
    assert handoff.get("do_remote_prefill") is None


def test_spec_p2p_nccl_kv_both():
    spec = InstanceSpec("i1", (1,), 8101, "/m", kv_connector="P2pNcclConnector")
    cmd = spec.command()
    kv = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
    assert kv["kv_role"] == "kv_both" and kv["kv_port"] == "28101"
    assert kv["kv_connector_extra_config"]["http_port"] == "8101"
    assert spec.zmq_address == "127.0.0.1:28101"
    assert spec.environment()["VLLM_HOST_IP"] == "127.0.0.1"


def test_pd_transfer_per_connector():
    nixl = PDTransfer("NixlConnector")
    assert nixl.request_id("i0", "i1", "r7") == "r7"
    assert nixl.prefill_params()["do_remote_decode"]
    assert nixl.decode_params(dict(remote_block_ids=[3]))["do_remote_prefill"]
    p2p = PDTransfer("P2pNcclConnector", {"i0": "127.0.0.1:28100", "i1": "127.0.0.1:28101"})
    assert p2p.request_id("i0", "i1", "r7") == "___prefill_addr_127.0.0.1:28100___decode_addr_127.0.0.1:28101_r7"
    assert p2p.prefill_params() is None and p2p.decode_params(None) is None
