"""CPU tensors through installed hooks and the pinned connector's real methods.

Run in the pinned image with --network none and without --gpus. We read/compile
only the connector's three relevant methods; no vLLM module or CUDA backend is
imported. The fake wire replaces NCCL, not connector gathering or KV injection.
"""
import ast
from collections import defaultdict
import hashlib
import importlib.util
import logging
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from pdblend_runtime.kv import KVCapabilityError, install, operation
from pdblend_runtime.kv_digest import compare_digests


CONNECTOR_SHA256 = 'c8698b99a58fec5e35559fbcf0b94bf22cb174e093a86a48afe188cb7129f9dc'
LAYERS = ('model.layers.0.self_attn.attn', 'model.layers.1.self_attn.attn')
HOLD = 'distserve-hold-cpu'
DECODE = 'decode-cpu'
TRANSACTION = 'cpu-transaction'
GENERATION = 7


class Metadata:
    def __init__(self, requests=(), loads=()):
        self.requests, self.loads = list(requests), list(loads)


@pytest.fixture(scope='module')
def pinned_connector_class():
    # find_spec locates installed bytes without executing vLLM/__init__.py.
    spec = importlib.util.find_spec('vllm')
    if spec is None:
        pytest.skip('pinned vLLM source is available in the CPU test image')
    root = Path(next(iter(spec.submodule_search_locations)))
    source = root/'distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py'
    content = source.read_bytes()
    assert hashlib.sha256(content).hexdigest() == CONNECTOR_SHA256, 'test requires reviewed pinned connector bytes'
    parsed = ast.parse(content)
    klass = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == 'P2pNcclConnector')
    names = {'start_load_kv', 'save_kv_layer', 'check_tensors_except_dim'}
    methods = [node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == names
    narrowed = ast.Module(body=[ast.ClassDef(name='PinnedConnector', bases=[], keywords=[],
                                            body=methods, decorator_list=[])], type_ignores=[])
    namespace = dict(torch=torch, KVConnectorMetadata=Metadata, P2pNcclConnectorMetadata=Metadata,
                     MLACommonMetadata=type('MLACommonMetadata', (), {}), logger=logging.getLogger(__name__))
    exec(compile(ast.fix_missing_locations(narrowed), str(source), 'exec'), namespace)
    assert 'vllm' not in sys.modules
    assert not torch.cuda.is_initialized()
    return namespace['PinnedConnector']


class Wire:
    """CPU stand-in for successful sender completion and actual received tensors."""
    def __init__(self):
        self.tensors = {}
        self.sent = []
        self.send_queue_cv = threading.Condition()
        self.send_queue = []
        self.send_request_id_to_tensor_ids = defaultdict(set)
        self._send_thread = SimpleNamespace(is_alive=lambda: True)
        self.omit_completion = set()
        self.recv_failure = None

    def send_tensor(self, tensor_id, tensor, remote_address, slot_mapping, is_mla=False):
        assert tensor.device.type == 'cpu' and not is_mla
        self.sent.append((tensor_id, remote_address))
        # Same logical gather used by P2pNcclEngine.extract_kv_from_layer.
        self.tensors[tensor_id] = tensor.reshape(2, tensor.shape[1]*tensor.shape[2], -1)[:, slot_mapping].clone()
        if tensor_id not in self.omit_completion:
            self.send_request_id_to_tensor_ids[tensor_id.rsplit('#', 1)[0]].add(tensor_id)
        return True

    def recv_tensor(self, tensor_id, *args, **kwargs):
        if self.recv_failure:
            raise self.recv_failure
        return self.tensors.pop(tensor_id, None)

    def wait_for_sent(self):
        raise AssertionError('installed worker wait must inspect actual completed tensor IDs')


@pytest.fixture
def rig(monkeypatch, pinned_connector_class):
    wire = Wire()
    workers, connectors = {}, {}
    for name in ('source', 'target'):
        connector = pinned_connector_class()
        connector.p2p_nccl_engine = wire
        connector.is_producer = name == 'source'
        connector.is_consumer = name == 'target'
        connector._rank = 1
        connector.metadata = Metadata()
        connector._get_connector_metadata = lambda c=connector: c.metadata
        connector.parse_request_id = lambda *args: ('127.0.0.1', 19000)
        worker = SimpleNamespace(rank=1, _native_generation=GENERATION,
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
            model_config=SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=len(LAYERS))))
        for module in ('vllm', 'vllm.distributed', 'vllm.distributed.kv_transfer'):
            value = ModuleType(module)
            value.__path__ = []
            monkeypatch.setitem(sys.modules, module, value)
        sys.modules['vllm.distributed.kv_transfer'].get_kv_transfer_group = lambda c=connector: c
        assert install(worker) is connector
        workers[name], connectors[name] = worker, connector

    source_slots = torch.tensor([1, 2, 9, 11], dtype=torch.int64)
    target_slots = torch.tensor([18, 19, 20, 21, 22], dtype=torch.int64)
    layers = {name: (torch.arange(2*4*8*2*4).reshape(2, 4, 8, 2, 4) + index*1024).to(torch.bfloat16)
              for index, name in enumerate(LAYERS)}
    target_layers = {name: SimpleNamespace(kv_cache=[torch.zeros_like(tensor)]) for name, tensor in layers.items()}
    context = SimpleNamespace(no_compile_layers=target_layers, virtual_engine=0, attn_metadata=object())
    context.no_compile_layers['non_attention'] = SimpleNamespace()
    request = SimpleNamespace(request_id=HOLD, slot_mapping=source_slots)
    connectors['source'].metadata = Metadata(requests=[request])
    connectors['target'].metadata = Metadata(loads=[SimpleNamespace(request_id=DECODE, slot_mapping=target_slots)])
    return SimpleNamespace(wire=wire, source=workers['source'], target=workers['target'],
        producer=connectors['source'], consumer=connectors['target'], context=context,
        layers=layers, source_slots=source_slots, target_slots=target_slots)


def hold(rig, *, layers=LAYERS):
    for layer in layers:
        rig.producer.save_kv_layer(layer, rig.layers[layer], object())


def expect(rig, **updates):
    return operation(rig.target, 'expect_load', dict(target_request_id=DECODE, transaction_id=TRANSACTION,
        generation=GENERATION, source_tokens=4, expected_layers=len(LAYERS), kv_digest=True, **updates))


def transfer(rig):
    return operation(rig.source, 'transfer', dict(held_request_id=HOLD, target_request_id=DECODE,
        target_address='127.0.0.1:20000', target_tp=2, generation=GENERATION,
        transaction_id=TRANSACTION, kv_digest=True))


def query(rig):
    return operation(rig.target, 'query_load', dict(target_request_id=DECODE, transaction_id=TRANSACTION))


def test_installed_hold_send_inject_and_digest_cover_all_layers_and_survive_empty_decode(rig):
    hold(rig)
    held = rig.source._pdblend_native_kv.held[HOLD]
    assert set(held.layers) == set(LAYERS) == set(held.slots)
    assert not rig.wire.sent, 'holding must suppress automatic connector transmission'
    for layer in LAYERS:
        assert held.layers[layer] is rig.layers[layer]
        assert held.slots[layer] is rig.source_slots
    assert expect(rig)['acknowledged'] and not query(rig)['acknowledged']
    sent = transfer(rig)
    assert sent['acknowledged'] and sent['layers'] == 2
    assert {address for _, address in rig.wire.sent} == {'127.0.0.1:20001'}
    original_recv = rig.wire.recv_tensor
    rig.consumer.start_load_kv(rig.context)
    assert rig.wire.recv_tensor == original_recv
    received = query(rig)
    assert received['acknowledged'] and received['loaded_layers'] == 2
    assert received['generation'] == GENERATION and received['source_tokens'] == 4
    for row in sent['layer_receipts']:
        layer = row['layer']
        digests = received['digests'][layer]
        compared = compare_digests(row['source_digest'], digests['received'], digests['injected'])
        assert compared['exact_payload_match'] and compared['injection_checked']
        assert row['source_digest']['physical_slot_sha256'] != digests['injected']['physical_slot_sha256']
        expected_payload = rig.layers[layer].reshape(2, 32, -1)[:, rig.source_slots]
        actual = rig.context.no_compile_layers[layer].kv_cache[0].reshape(2, 32, -1)
        assert torch.equal(actual[:, rig.target_slots[:4]], expected_payload)
        assert not actual[:, rig.target_slots[4]].count_nonzero(), 'final prompt slot remains for local recompute'
    rig.consumer.metadata = Metadata()  # Subsequent decode has no load metadata.
    rig.consumer.start_load_kv(rig.context)
    assert query(rig) == received, 'one-shot actual load evidence must persist across decode steps'
    rig.target._native_generation += 1
    assert not query(rig)['acknowledged'], 'old success cannot acknowledge a new generation'


def test_ordinary_requests_still_reach_original_save_and_metadata_restores_on_error(rig):
    ordinary = SimpleNamespace(request_id='ordinary', slot_mapping=rig.source_slots)
    rig.producer.metadata.requests.append(ordinary)
    original_requests = rig.producer.metadata.requests
    rig.producer.save_kv_layer(LAYERS[0], rig.layers[LAYERS[0]], object())
    assert rig.wire.sent == [('ordinary#'+LAYERS[0], '127.0.0.1:19001')]
    assert rig.producer.metadata.requests is original_requests
    def failed_send(*args, **kwargs):
        raise RuntimeError('normal connector failed')
    rig.wire.send_tensor = failed_send
    with pytest.raises(RuntimeError, match='normal connector'):
        rig.producer.save_kv_layer(LAYERS[1], rig.layers[LAYERS[1]], object())
    assert rig.producer.metadata.requests is original_requests


def test_incomplete_held_source_cannot_start_transfer(rig):
    hold(rig, layers=LAYERS[:1])
    with pytest.raises(KVCapabilityError, match='all attention layers'):
        transfer(rig)
    assert not rig.wire.sent


@pytest.mark.parametrize('generation', [GENERATION-1, str(GENERATION), True])
def test_expect_load_requires_actual_integer_generation(rig, generation):
    payload = dict(target_request_id=DECODE, transaction_id=TRANSACTION, generation=generation, source_tokens=4)
    with pytest.raises(KVCapabilityError, match='generation'):
        operation(rig.target, 'expect_load', payload)
    assert not query(rig)['acknowledged']


@pytest.mark.parametrize('fault', ['missing', 'token_count', 'descriptor', 'exception', 'missing_layer'])
def test_received_value_and_complete_injection_are_required_before_ack(rig, fault):
    hold(rig)
    expect(rig)
    transfer(rig)
    tensor_id = DECODE+'#'+LAYERS[1]
    if fault == 'missing':
        rig.wire.tensors.pop(tensor_id)
    elif fault == 'token_count':
        rig.wire.tensors[tensor_id] = rig.wire.tensors[tensor_id][:, :3]
    elif fault == 'descriptor':
        rig.wire.tensors[tensor_id] = (1234, torch.bfloat16, (2, 4, 8))
    elif fault == 'exception':
        rig.wire.recv_failure = RuntimeError('receive failed after staging')
    else:
        del rig.context.no_compile_layers[LAYERS[1]]
    original_recv = rig.wire.recv_tensor
    with pytest.raises((KVCapabilityError, RuntimeError, AttributeError)):
        rig.consumer.start_load_kv(rig.context)
    assert rig.wire.recv_tensor == original_recv
    receipt = query(rig)
    assert not receipt['acknowledged'] and receipt['failed'] and receipt['loaded_layers'] == 0


def test_empty_sender_queue_without_every_completed_layer_never_acknowledges(rig):
    hold(rig)
    rig.wire.omit_completion.add(DECODE+'#'+LAYERS[1])
    assert rig.wire.send_queue == []
    with pytest.raises(KVCapabilityError, match='complete successful sends'):
        transfer(rig)
    assert TRANSACTION not in rig.source._pdblend_native_kv.transactions
    with pytest.raises(KVCapabilityError, match='uncertain completion'):
        transfer(rig)


def test_generation_change_between_registration_and_receive_fails_closed(rig):
    hold(rig)
    expect(rig)
    transfer(rig)
    rig.target._native_generation += 1
    with pytest.raises(KVCapabilityError, match='stale'):
        rig.consumer.start_load_kv(rig.context)
    assert query(rig)['failed'] and not query(rig)['acknowledged']


def test_injected_corruption_remains_visible_to_three_stage_digest_comparison(rig):
    hold(rig)
    expect(rig)
    sent = transfer(rig)
    # Corrupt the actual receive tensor. Injection itself remains successful,
    # but exact source/received/injected comparison must flag the mismatch.
    rig.wire.tensors[DECODE+'#'+LAYERS[0]][0, 0, 0] = -123
    rig.consumer.start_load_kv(rig.context)
    receipts = query(rig)
    first = next(row for row in sent['layer_receipts'] if row['layer'] == LAYERS[0])
    digests = receipts['digests'][LAYERS[0]]
    comparison = compare_digests(first['source_digest'], digests['received'], digests['injected'])
    assert comparison['status'] == 'mismatch' and not comparison['exact_payload_match']
    assert digests['received']['sha256'] == digests['injected']['sha256']
    assert not torch.cuda.is_initialized()
