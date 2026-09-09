"""Focused real-class CPU checks. Never initializes Controller or GPU backends."""
import argparse
import asyncio
import copy
import hashlib
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path('/root/workspace/pdblend-next-v1')
HOST=ROOT/'releases/five-system100-B32B-v1-runtime'
sys.path.insert(0,str(HOST/'src'))
from ecopadg.serving.runtime import Controller
from ecopadg.serving.types import InstanceState,RequestBudget,RuntimeSnapshot
from ecopadg.serving.topology import DockerLifecycle,InstanceSpec,TopologyManager,retained_cache_available
from ecopadg.serving.pd_topology import PDBlendTopologyPlanner,MeasuredCapacity
from ecopadg.serving.dynamo_topology import TopologyCost


async def check(prepared):
    checks=[]
    coverage=[]
    with tempfile.TemporaryDirectory(prefix='b-capacity-cpu-') as temporary:
        for ds in ('alpaca','sharegpt','longbench'):
            baseline=json.loads((prepared/'static2'/f'{ds}.json').read_text())
            candidate=json.loads((prepared/'static3'/f'{ds}.json').read_text())
            assert {k for k in candidate if candidate[k]!=baseline[k]}=={'instances'}
            original=copy.deepcopy(candidate)
            candidate['journal']=str(Path(temporary)/(ds+'.jsonl'))
            c=Controller(candidate)
            try:
                assert c.topology_manager is None and c.pd_topology is None
                assert c.retained_weights is None and c.strategy=='pdblend-joint'
                now=100.
                instances=tuple(InstanceState(i['id'],'mixed',2,tuple(i['gpus']),now,1,2520,
                                               47104,0,0) for i in candidate['instances'])
                request=RequestBudget('cpu-only',now,128,211,candidate['slo_ttft_s'],
                    candidate['slo_tpot_s'],output_limit=256,hard_deadline_s=now+120)
                routes={r.decode_id for p in c.planner.candidates(RuntimeSnapshot(1,now,instances),request,now)
                        for r in p.routes}
                assert routes=={'nextv3b0','nextv3b1','cap3b2'}
                checks.append(dict(case=ds+'_real_controller_static3',passed=True,
                                   cfg_except_instances_equal=True,available_route_ids=sorted(routes),
                                   dynamic_manager=None))
                if ds=='alpaca':
                    blocked=RuntimeSnapshot(1,now,tuple(replace(i,free_kv_tokens=0) for i in instances[:2]))
                    recovery=c.planner.plan(blocked,(request,),now=now)
                    assert not recovery.feasible and not recovery.routes and not recovery.roles
                    assert {a.instance_id for a in recovery.frequencies}=={'nextv3b0','nextv3b1'}
                    assert all(a.frequency_mhz==2520 for a in recovery.frequencies)
                    checks.append(dict(case='current_infeasible_recovery_only_existing_frequency',
                                       passed=True,reason=recovery.reason,additions=0))
                    # Fake transition cost is used only to show the actual planner rejects
                    # an infeasible original layout before considering any transition.
                    cost=TopologyCost((),(2,),1.,1.,'synthetic-not-measured',True)
                    capacity=MeasuredCapacity(2,47104,2147483648,1024,'synthetic-not-measured')
                    dynamic=PDBlendTopologyPlanner(c.planner,[cost],[capacity])
                    forecast=SimpleNamespace(requests=(request,),horizon_s=100.,rate_lower_rps=1.,rate_upper_rps=1.)
                    assert dynamic.choose(blocked,forecast,now,cached_weights=True) is None
                    checks.append(dict(case='dynamic_original_infeasible_returns_none',passed=True,
                                       synthetic_transition_fixture_not_profile=True))
                    for inp,ctx in ((512,768),(2048,2560),(4096,4352),(4096,4353),(7168,7680)):
                        for frequency in (1500,2520):
                            row=dict(input_tokens=inp,context_tokens=ctx,frequency_mhz=frequency,batches={})
                            for batch in (1,4,8,16):
                                p=c.profiles.lookup_execution_phase('mixed',2,frequency,inp,ctx,batch)
                                row['batches'][str(batch)]=None if p is None else dict(role=p.role,
                                    measured_bucket=[p.input_tokens,p.context_tokens,p.batch],source_sha256=p.source_sha256)
                            coverage.append(row)
                    static2=RuntimeSnapshot(1,now,instances[:2]);static3=RuntimeSnapshot(1,now,instances)
                    predicted=dict(two_w=c.planner.node_residency(static2),three_w=c.planner.node_residency(static3))
                assert {k for k in original if original[k]!=baseline[k]}=={'instances'}
            finally:
                c.planning_executor._executor.shutdown(wait=True,cancel_futures=True)
        events=[]
        async def freeze(*args):events.append(('freeze',args))
        async def commit(*args):events.append(('commit',args))
        class Journal:
            async def emit(self,item):events.append(('journal',item))
        specs=[InstanceSpec('old',2,(0,1),33500,33700)]
        manager=TopologyManager(SimpleNamespace(clocks=None),None,specs,tuple(range(8)),Journal(),freeze=freeze,commit=commit)
        async def direct(function,*args,**kwargs):
            return function(*args,**kwargs)
        try:
            # The CPU sandbox cannot wake asyncio's cross-thread socket. The
            # exact cache predicate executes synchronously; no HTTP is invoked.
            with patch('asyncio.to_thread',direct):
                await manager.reconfigure((),(InstanceSpec('third',2,(4,5),34702,55764),),
                                          savings_lower_j=2.,cost_upper_j=1.,retained_weights=None)
        except ValueError as exc:
            assert 'retained-weight cache' in str(exc)
        else:
            raise AssertionError('pure add without actual cache should fail')
        assert not events and not retained_cache_available(None)
        checks.append(dict(case='pure_add_without_cache_refused_before_control',passed=True,side_effects=events,
                           synchronous_cpu_thread_handoff=True))
        lifecycle=DockerLifecycle(Path(temporary)/'lifecycle','synthetic-image',{'model':'synthetic-model'})
        commands=[]
        async def command(*argv,**kw):commands.append(list(argv));return 'cpu-only'
        lifecycle.command=command
        with patch('asyncio.to_thread',direct):
            await lifecycle.start(InstanceSpec('cpu-only',2,(4,5),34702,55764),{})
        assert 'PYTHONPATH=/root/workspace/pdblend/src' in commands[0]
        assert 'ecopadg.serving.engine' in commands[0]
        checks.append(dict(case='original_lifecycle_uses_old_source',passed=True,
                           mock_only_argv=commands[0],candidate_uses_explicit_v3_file_instead=True,
                           synchronous_cpu_thread_handoff=True))
    return dict(cpu_only=True,gpu_executed=False,controller_initialized=False,
        http_dependency_import_stub=bool(getattr(sys.modules.get('aiohttp'),'CPU_IMPORT_STUB',False)),
        actual_host=str(HOST),synthetic_state=True,historical_free_kv_not_future_measurement=47104,
        checks=checks,coverage=coverage,profile_sha256=hashlib.sha256(Path(baseline['profiles']).read_bytes()).hexdigest(),
        predicted_residency_at_2520=predicted,measured_static3_throughput=None)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepared',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();result=asyncio.run(check(args.prepared.resolve()))
    with args.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=len(result['checks']),gpu=False,out=str(args.out))))
