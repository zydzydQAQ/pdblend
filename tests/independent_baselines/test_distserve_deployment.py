import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdblend_baselines.distserve.deployment import deployment_specs, execute_deployment, search_deployment


def plan():
    return dict(system='distserve',model_id='Qwen2.5-7B-Instruct',status='ready_for_native_execution',
        selected=dict(config=[1,1,1,1,1],tp=1,pp=1,replicas=2,total_gpu_count=4),
        profiles=[dict(tp=1,sha256='a'*64)],scope='symmetric PP1 paired replicas')


def test_selected_tp_replica_allocation_controls_real_native_specs_and_rank_ports():
    value=plan();specs=deployment_specs(value,[0,1,2,3],19000)
    assert [s.instance_id for s in specs]==['dist-0-P','dist-0-D','dist-1-P','dist-1-D']
    assert [s.gpus for s in specs]==[(0,),(1,),(2,),(3,)]
    assert len({s.zmq_address for s in specs})==4
    assert all('pdblend_runtime.serve' in s.command() for s in specs)
    value['selected']['config']=[1,1,1,2,1]
    with pytest.raises(ValueError,match='symmetric'):deployment_specs(value,[0,1,2,3],19000)


def test_failed_profile_and_evaluation_never_mint_deployment(tmp_path):
    corpus=tmp_path/'corpus.json';corpus.write_text(json.dumps(dict(calibration=[dict(input_tokens=128,output_tokens=16)])))
    profile=tmp_path/'profile.json';profile.write_text(json.dumps(dict(identity=dict(model_id='Qwen2.5-7B-Instruct'),qualified=False)))
    result=search_deployment([profile],corpus,model='Qwen2.5-7B-Instruct',rate_rps=1,ttft_s=5,tpot_s=.15)
    assert result['status']=='missing_profile' and result['selected'] is None
    assert result['selection_split']=='calibration' and not result['evaluation_used_for_selection']
    corpus.write_text(json.dumps(dict(evaluation=[dict(input_tokens=128,output_tokens=16)])))
    with pytest.raises(ValueError,match='calibration'):search_deployment([],corpus,model='Qwen2.5-7B-Instruct',rate_rps=1,ttft_s=5,tpot_s=.15)
