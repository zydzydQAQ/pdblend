"""Both real independent runtimes over one pair of native HTTP fixtures."""
import asyncio
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace as NS

import pytest
pytest.importorskip('fastapi')

from pdblend_baselines import resident_campaign as rc


def load(name, filename):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(filename))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

_eco=load('resident_eco_http','test_ecoserve_run_native_http.py')
_dist=load('resident_dist_http','test_distserve_request_runtime_http.py')


def specs(config, endpoints):
    return [NS(instance_id=iid,model=config['model_id'],tp=1,pp=1,gpus=(index,),
               base_url=url,zmq_address='127.0.0.1:'+str(31000+16*index))
            for index,(iid,url) in enumerate(endpoints.items())]


@pytest.mark.asyncio
async def test_actual_distserve_then_ecoserve_reuses_pair_with_acknowledged_reopen(tmp_path,monkeypatch):
    config,identity,trace=_eco.inputs(tmp_path,monkeypatch,count=2)
    async with _eco.services(monkeypatch,identity,count=2,engine_class=_dist.Engine) as (engines,endpoints,calls):
        selected=specs(config,endpoints)
        monkeypatch.setattr(rc,'build_specs',lambda *a:selected)
        monkeypatch.setattr(rc,'Fleet',lambda *a:pytest.fail('external resident pair must not be deployed again'))
        result=await rc.execute(model=config['model_id'],tp=1,gpus=[0,1],base_port=19000,
            trace=trace,eco_profile=Path(config['eco_prefill_csv']),out=tmp_path/'run',duration=.08)
        assert result['complete'], result
        assert result['distserve']['complete'] and result['ecoserve']['complete']
        assert result['ecoserve']['automatic_policy_status']=='inconclusive'
        assert not result['complete_reproduction'] and not result['energy_comparable']
        assert result['distserve']['observed_duration_s']>=.08
        assert result['ecoserve']['service_finished_s']-result['ecoserve']['service_started_s']>=.08
        assert result['engine_loads']==result['engine_load_cycles']==0
        assert len(result['drain_states'])==len(result['final_drain_states'])==2
        rng=random.Random(9701);expected=[rng.randint(1000,60000) for _ in range(128)]
        for label in ('distserve','ecoserve'):
            assert len(result['warmup_'+label])==2
            assert all(row['prompt']==expected and row['seed']==9701 for row in result['warmup_'+label])
        for engine in engines.values():
            warmups=[row for op,row in engine.calls if op=='warmup_seed']
            assert len(warmups)==2 and all(row['prompt']['prompt_token_ids']==expected for row in warmups)
            assert not engine.scheduler.requests and not engine.kv.held
        assert set(result['system_receipt_sha256'])=={'distserve','ecoserve'}
        saved=json.loads((tmp_path/'run/completion.json').read_text())
        assert saved['trace_sha256']==saved['distserve']['trace_sha256']==saved['ecoserve']['trace_sha256']


@pytest.mark.asyncio
@pytest.mark.parametrize('failure',['lost_block','missing_inventory','stale_scheduler'])
async def test_empty_queue_cannot_hide_incomplete_drain_inventory(tmp_path,monkeypatch,failure):
    config,identity,trace=_eco.inputs(tmp_path,monkeypatch)
    async with _eco.services(monkeypatch,identity) as (engines,endpoints,_):
        engine=engines['eco0']
        if failure=='lost_block':
            engine.scheduler.free_override=1023
        else:
            original=engine.scheduler.native_state
            def changed():
                state=original()
                if failure=='missing_inventory':state.pop('kv_allocations')
                else:state['native_at_s']-=10
                return state
            engine.scheduler.native_state=changed
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(rc.drain_endpoints(specs(config,endpoints)),2)
