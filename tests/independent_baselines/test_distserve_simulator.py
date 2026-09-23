import ast
from pathlib import Path
import pytest

from pdblend_baselines.distserve.simulator import OfficialSimulator, MeasuredLatency
from pdblend_baselines.distserve.planning import enumerate_configs


def test_enum_is_differentially_identical_to_official_search():
    path=Path(__file__).resolve().parent/'fixtures'/'baselines/distserve/references/upstream/simdistserve/benchmarks/search_configs.py'
    tree=ast.parse(path.read_text())
    func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_distserve_configs')
    class Models:
        @staticmethod
        def formalize_model_name(name):return name
    from itertools import product
    namespace=dict(ModelTypes=Models,product=product,get_model_possible_tp=lambda name:[1,2,4],
                   get_model_possible_pp=lambda name:[1,2,4,7,14,28])
    exec(compile(ast.Module(body=[func],type_ignores=[]),str(path),'exec'),namespace)
    for high in (True,False):
        original=namespace['get_distserve_configs']('qwen',1,8,high)
        ours=enumerate_configs(layers=28,attention_heads=28,num_nodes=1,gpus_per_node=8,
                               allowed_tps=(1,2,4),high_affinity=high)
        assert tuple(original)==ours


def test_official_core_ast_is_unchanged_except_import_namespace():
    root=Path(__file__).resolve().parent/'fixtures'
    for name in ['base/request.py','base/worker.py','base/scheduler.py','clusters/disagg.py','utils.py']:
        original=(root/'baselines/distserve/references/upstream/simdistserve'/name).read_text()
        current=(Path(__file__).resolve().parents[2]/'src/pdblend_baselines/distserve/_sim'/name).read_text()
        expected=original.replace('from simdistserve.', 'from pdblend_baselines.distserve._sim.')
        assert ast.dump(ast.parse(current))==ast.dump(ast.parse(expected))


def test_actual_official_simpy_loop_returns_request_and_worker_events():
    def latency(role,tp,pp,batch,inputs,contexts):return 10. if role=='prefill' else 2.
    simulator=OfficialSimulator([(16,3),(32,4),(48,2)],latency=latency,
        capacities={(1,1):8192},seed=701)
    result=simulator((1,1,1,1,1),1.)
    assert len(result['ttft_s'])==3 and len(result['tpot_s'])==3
    assert all(x>0 for x in result['ttft_s'])
    assert result['upstream_revision']=='82831f1604cc6b10bebd360f6c437a07790dde9f'
    assert len(result['request_events'])==3 and result['worker_events']
    assert result['gpu_qualified'] is False
    assert result['output_convention']=='API total outputs mapped to upstream decode steps = max_tokens - 1'


def test_simulator_refuses_missing_pp_profile_instead_of_dividing_tp_latency():
    simulator=OfficialSimulator([(32,4)],latency=lambda *args:1.,capacities={(1,1):8192})
    with pytest.raises(ValueError,match='capacity'):
        simulator((1,1,2,1,1),1.)


def test_measured_latency_rejects_unmeasured_pp_and_accepts_valid_shapes():
    points=[dict(role='prefill',tp=1,pp=1,batch=2,max_input_tokens=64,max_context_tokens=64,
                 stage_latency_ms=3.,source_sha256='a'*64)]
    table=MeasuredLatency(points)
    assert table('prefill',1,1,2,[16,32],[16,32])==3.
    with pytest.raises(ValueError,match='coverage'):table('prefill',1,2,2,[16,32],[16,32])
    with pytest.raises(ValueError,match='coverage'):table('prefill',1,1,2,[128,32],[128,32])


def test_official_simulator_binds_each_real_pipeline_stage_without_changing_worker_core():
    seen=[]
    class Latency:
        def __call__(self,*args):raise AssertionError('stage identity was lost')
        def stage_latency(self,role,tp,pp,stage,batch,inputs,contexts):
            seen.append((role,stage));return [1.,9.][stage]
    sim=OfficialSimulator([(32,4),(16,3)],latency=Latency(),capacities={(1,2):8192})
    result=sim((1,1,2,1,2),1.)
    assert set(seen)=={('prefill',0),('prefill',1),('decode',0),('decode',1)}
    assert all(value>.010 for value in result['ttft_s'])
    from pdblend_baselines.distserve._sim.estimators.time_estimator import PIPE_STAGE
    assert PIPE_STAGE.get() is None


def test_pp_provider_rejects_rank_cuda_only_and_requires_actual_stage_identity():
    point=dict(role='prefill',tp=1,pp=2,batch=2,max_input_tokens=64,max_context_tokens=64,
        stage_latency_ms=3.,source_sha256='a'*64)
    with pytest.raises(ValueError,match='stage'):
        MeasuredLatency([point])
    with pytest.raises(ValueError,match='PP wire'):
        MeasuredLatency([dict(point,stage_index=0,timing_scope='local_model_runner_cuda')])
    table=MeasuredLatency([dict(point,stage_index=i,stage_latency_ms=3+i,
        timing_scope='stage_service_including_pp_and_host') for i in range(2)])
    assert table.stage_latency('prefill',1,2,0,2,[32,32],[32,32])==3
    assert table.stage_latency('prefill',1,2,1,2,[32,32],[32,32])==4
