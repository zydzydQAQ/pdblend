"""Read actual pinned vLLM class bodies without loading vLLM or CUDA.

CPU-only image invocation mounts /home/models at /models to read the three
verified config.json files. Qwen constructors run on torch's meta device, so
this checks full parameter names/shapes without allocating model weights.
"""
import ast
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from pdblend_baselines.dynamollm.deployment import SubprocessLifecycle
from pdblend_baselines.dynamollm.gpu_weights import DynamoWorkerExtension, transfer_pieces
from pdblend_baselines.dynamollm.relay import _PortView


PINNED = {
    'platforms/interface.py': 'bc1f48c2ebc221dd42ad822c8b78237f7959055a527d6e26a42372fb5b476e5f',
    'worker/worker_base.py': 'a218ea7d5bd9f498fb4b582b0b8d7d0d6cd32c31359a007688596c0aba71fa54',
    'v1/worker/gpu_worker.py': '8b286ed930e2def68e994e752c8207ee818583692cbcefb85d53103a01d6c295',
    'model_executor/models/qwen2.py': '0f5c0c303f77e14c0f7e09bdfef86d8fb54e1e904e89f83bfb7113ea05c4fa94',
}


@pytest.fixture(scope='module')
def pinned():
    spec = importlib.util.find_spec('vllm')
    if spec is None:
        pytest.skip('run this CPU contract test in the pinned vLLM image')
    root = Path(next(iter(spec.submodule_search_locations)))
    result = {}
    for relative, digest in PINNED.items():
        source = root/relative
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, 'pinned vLLM source changed'
        result[relative] = ast.parse(source.read_text())
    assert 'vllm' not in sys.modules
    return result


def klass(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def method(tree, class_name, method_name):
    return next(node for node in klass(tree, class_name).body
                if isinstance(node, ast.FunctionDef) and node.name == method_name)


def execute_ast(nodes, namespace):
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                             *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<reviewed-pinned-contract>', 'exec'), namespace)


def test_real_wrapper_mixes_extension_into_native_worker_and_runs_load_hook(pinned, monkeypatch):
    native_path = Path(__file__).resolve().parents[2]/'src/pdblend_runtime/native_v1.py'
    native_tree = ast.parse(native_path.read_text())
    native_node = klass(native_tree, 'NativeWorker')
    real_load = method(pinned['v1/worker/gpu_worker.py'], 'Worker', 'load_model')
    assert [arg.arg for arg in real_load.args.args] == ['self']
    calls = []
    class Worker:
        def __init__(self, **kwargs):
            self.parallel_config = kwargs['vllm_config'].parallel_config
            self.model_runner = SimpleNamespace(execute_model=lambda *args, **kw: None)
        def load_model(self):
            calls.append('real-base-load-placeholder')
    namespace = dict(Worker=Worker, os=__import__('os'))
    execute_ast([native_node], namespace)
    native = namespace['NativeWorker']
    modules = {}
    for name in ('vllm', 'vllm.plugins', 'vllm.distributed', 'vllm.distributed.kv_transfer'):
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    modules['vllm.plugins'].load_general_plugins = lambda: None
    modules['vllm.distributed.kv_transfer'].has_kv_transfer_group = lambda: False
    wrapper_namespace = dict(enable_trace_function_call_for_thread=lambda _: None,
        resolve_obj_by_qualname=lambda name: native if name == 'native' else DynamoWorkerExtension,
        set_current_vllm_config=lambda _: nullcontext(), logger=logging.getLogger(__name__))
    execute_ast([method(pinned['worker/worker_base.py'], 'WorkerWrapperBase', 'init_worker')], wrapper_namespace)
    wrapper = SimpleNamespace(rpc_rank=0)
    config = SimpleNamespace(parallel_config=SimpleNamespace(worker_cls='native', worker_extension_cls='dynamo',
                                                            pipeline_parallel_size=1, tensor_parallel_size=1))
    wrapper_namespace['init_worker'](wrapper, [dict(vllm_config=config)])
    assert DynamoWorkerExtension in native.__bases__
    assert isinstance(wrapper.worker, native) and callable(wrapper.worker.dynamo_operation)
    monkeypatch.setenv('DYNAMO_GENERATION', '3')
    wrapper.worker.load_model()
    assert calls == ['real-base-load-placeholder']
    assert wrapper.worker._native_generation == 3
    assert wrapper.worker._native_scope is None
    assert not torch.cuda.is_initialized()


def qwen_constructors(tree, *, tp):
    nn = torch.nn
    def weight(rows, columns=None, *, bias=False):
        result = nn.Module()
        result.weight = nn.Parameter(torch.empty((rows,) if columns is None else (rows, columns),
                                                 dtype=torch.bfloat16, device='meta'))
        if bias:
            result.bias = nn.Parameter(torch.empty(rows, dtype=torch.bfloat16, device='meta'))
        return result
    def make_layers(count, factory, prefix):
        return 0, count, nn.ModuleList(factory(prefix+'.'+str(index)) for index in range(count))
    namespace = dict(nn=nn, get_tensor_model_parallel_world_size=lambda: tp,
        QKVParallelLinear=lambda hidden, dim, heads, kv_heads, **kw:
            weight((heads+2*kv_heads)*dim//tp, hidden, bias=kw['bias']),
        MergedColumnParallelLinear=lambda hidden, sizes, **kw: weight(sum(sizes)//tp, hidden, bias=kw['bias']),
        RowParallelLinear=lambda inputs, outputs, **kw: weight(outputs, inputs//tp, bias=kw['bias']),
        RMSNorm=lambda hidden, **kw: weight(hidden),
        VocabParallelEmbedding=lambda vocab, hidden, **kw: weight(((vocab+63)//64*64)//tp, hidden),
        ParallelLMHead=lambda vocab, hidden, **kw: weight(((vocab+63)//64*64)//tp, hidden),
        SiluAndMul=nn.Identity, Attention=lambda *args, **kw: nn.Identity(),
        get_rope=lambda *args, **kw: nn.Identity(), LogitsProcessor=lambda *args: nn.Identity(),
        AttentionType=SimpleNamespace(DECODER='decoder', ENCODER_ONLY='encoder'),
        get_pp_group=lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
        is_interleaved=lambda _: False, make_layers=make_layers,
        make_empty_intermediate_tensors_factory=lambda *args: None,
        maybe_prefix=lambda prefix, name: prefix+'.'+name if prefix else name)
    classes = []
    for name in ('Qwen2MLP', 'Qwen2Attention', 'Qwen2DecoderLayer', 'Qwen2Model', 'Qwen2ForCausalLM'):
        # Every attribute/constructor/bias choice comes from actual pinned Qwen
        # source. Only the low-level parameter allocator is a CPU meta adapter.
        classes.append(ast.ClassDef(name=name, bases=[ast.Attribute(value=ast.Name(id='nn', ctx=ast.Load()),
            attr='Module', ctx=ast.Load())], keywords=[],
            body=[method(tree, name, '__init__')], decorator_list=[]))
    execute_ast(classes, namespace)
    return namespace['Qwen2ForCausalLM']


@pytest.mark.parametrize('size', ['7B', '14B', '32B'])
def test_all_real_qwen_parameter_names_have_complete_tp_shards_including_qkv_bias(pinned, size):
    directory = Path('/models')/('Qwen2.5-'+size+'-Instruct')
    if not directory.is_dir():
        pytest.skip('mount verified three-model config directories read-only at /models')
    config_path = directory/'config.json'
    manifest = json.loads((directory/'pdblend-model-manifest.json').read_text())
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == next(
        row['sha256'] for row in manifest['files'] if row['path'] == 'config.json')
    config = SimpleNamespace(**json.loads(config_path.read_text()))
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=config, hf_text_config=config),
                                  cache_config=None, quant_config=None, lora_config=None)
    geometry = {name: getattr(config, name) for name in
                ('hidden_size', 'intermediate_size', 'num_attention_heads', 'num_key_value_heads')}
    shapes = {}
    for tp in (1, 2, 4):
        cls = qwen_constructors(pinned['model_executor/models/qwen2.py'], tp=tp)
        model = cls(vllm_config=vllm_config)
        shapes[tp] = {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}
        assert len(shapes[tp]) == 7*config.num_hidden_layers+3
        assert sum(name.endswith('qkv_proj.bias') for name in shapes[tp]) == config.num_hidden_layers
        assert 'model.embed_tokens.weight' in shapes[tp] and 'lm_head.weight' in shapes[tp]
    for source_tp, target_tp in ((1, 2), (2, 4), (4, 2)):
        for name, target_shape in shapes[target_tp].items():
            for target_rank in range(target_tp):
                pieces = [piece for source_rank in range(source_tp) for piece in transfer_pieces(name,
                    shapes[source_tp][name], target_shape, source_tp, source_rank, target_tp, target_rank, geometry)]
                assert pieces, name
                axis = pieces[0]['axis']
                if axis is None:
                    assert len(pieces) == 1 and pieces[0]['length'] == target_shape[0]
                else:
                    cursor = 0
                    for piece in sorted(pieces, key=lambda row: row['target_offset']):
                        assert piece['axis'] == axis and piece['target_offset'] == cursor, name
                        cursor += piece['length']
                    assert cursor == target_shape[axis], name
    assert not torch.cuda.is_initialized()


def test_pinned_torch_exposes_exact_nccl_backend_protocol_without_initializing_cuda():
    import torch.distributed as distributed
    cls = distributed.ProcessGroupNCCL
    assert 'timeout: datetime.timedelta' in cls.__init__.__doc__
    for method_name, argument in (('send', 'dstRank'), ('recv', 'srcRank'), ('allreduce', 'tensors')):
        assert argument in getattr(cls, method_name).__doc__
    assert callable(cls.shutdown)
    assert not torch.cuda.is_initialized()


def test_dynamo_launch_environment_obeys_pinned_numeric_cuda_parser(pinned, monkeypatch, tmp_path):
    import os
    from pdblend_baselines.dynamollm.deployment import gpu_devices
    namespace = {'os': os}
    execute_ast([method(pinned['platforms/interface.py'], 'Platform', 'device_id_to_physical_device_id')], namespace)
    actual_mapping = namespace['device_id_to_physical_device_id'].__func__
    platform = SimpleNamespace(device_control_env_var='CUDA_VISIBLE_DEVICES')
    monkeypatch.setenv('PDBLEND_GPU_UUIDS', 'GPU-seven,GPU-three,GPU-five')
    # Reproduce the real old launch failure before any model/CUDA allocation.
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-five,GPU-seven')
    with pytest.raises(ValueError, match='invalid literal'):
        actual_mapping(platform, 0)
    lifecycle = SubprocessLifecycle(dict(base_port=19000, model_id='Qwen2.5-7B-Instruct'),
                                    None, lambda *a, **k: None, tmp_path)
    env = lifecycle.environment(dict(id='target', gpus=[2, 0], generation=4), dummy=True,
                                golden={'tp': 2, 'token_ids': [1]})
    assert env['CUDA_VISIBLE_DEVICES'] == '2,0' and env['DYNAMO_GENERATION'] == '4'
    assert env['DYNAMO_DUMMY'] == '1' and env['DYNAMO_INSTANCE_ID'] == 'target'
    assert env['PDBLEND_GPU_UUIDS'] == 'GPU-seven,GPU-three,GPU-five'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', env['CUDA_VISIBLE_DEVICES'])
    assert [actual_mapping(platform, rank) for rank in range(2)] == [2, 0]
    assert gpu_devices([2, 0]) == ['GPU-five', 'GPU-seven']
    assert not torch.cuda.is_initialized()


def test_all_dynamo_port_offsets_stay_inside_one_hundred_port_lease(tmp_path):
    config = dict(base_port=19000, node_gpus=list(range(8)))
    lifecycle = SubprocessLifecycle(config, None, lambda *a, **k: None, tmp_path)
    assert lifecycle.target_port == 19032
    lifecycle.instances = {str(i): dict(port=19032+i*2) for i in range(8)}
    assert [_PortView(lifecycle, i*2).target_port for i in range(8)] == [19048]*8
    lifecycle.instances = {str(i): dict(port=19000+i*2) for i in range(40)}
    with pytest.raises(RuntimeError, match='exhausted'):
        _PortView(lifecycle, 0).target_port
    for invalid in (dict(target_port=19128), dict(store_port=19512), dict(base_port=65500)):
        with pytest.raises(ValueError, match='100-port'):
            SubprocessLifecycle(dict(config, **invalid), None, lambda *a, **k: None, tmp_path)
