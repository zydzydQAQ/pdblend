import asyncio
import copy
from pathlib import Path

import pytest

from ecopadg.scalability.protocol import build_config
from ecopadg.scalability.run import replay, validate_inputs


def inputs():
    base=dict(model_name='Qwen2.5-14B-Instruct',max_service_frequency_mhz=2520,
              instances=[dict(id=f'i{g}',gpus=[g],tp=1,role='mixed',url=f'http://127.0.0.1:{30000+g}')
                         for g in range(8)])
    config=build_config(base,system='pdblend',dataset='sharegpt',allocated_gpu_ids=[0,1,2])
    manifest=dict(system='pdblend',dataset='sharegpt',n_gpus=3,seed=701,stage='diagnostic',
                  rate_rps=1.,allocated_gpu_ids=[0,1,2],arrival_window_s=2.,slo_ttft_s=5.,slo_tpot_s=.15)
    trace=dict(requests=[dict(arrival_s=0.,prompt_len=2,output_len=2,timeout_s=120.)],prompts=[[1,2]])
    return config,trace,manifest


def test_gpu_subset_and_selective_policy_are_both_required():
    config,trace,manifest=inputs()
    validate_inputs(config,trace,manifest,diagnostic=True)
    config['allow_pd']=False
    with pytest.raises(ValueError,match='selective PD'):
        validate_inputs(config,trace,manifest,diagnostic=True)


@pytest.mark.parametrize('mutation',[
    lambda c:c['instances'][0].update(gpus=[7]),
    lambda c:c.update(strategy='mixed'),
    lambda c:c['instances'][0].update(role='decode'),
    lambda c:c.update(dynamic_pools=True),
    lambda c:c.update(manage_clocks=False),
])
def test_invalid_actual_configuration_cannot_be_relabelled(mutation):
    config,trace,manifest=inputs()
    mutation(config)
    with pytest.raises(ValueError):
        validate_inputs(config,trace,manifest,diagnostic=True)


def test_long_output_timeout_is_not_capped_to_120():
    config,trace,manifest=inputs()
    trace['requests'][0].update(output_len=2000,timeout_s=120.)
    with pytest.raises(ValueError,match='timeout'):
        validate_inputs(config,trace,manifest,diagnostic=True)
    trace['requests'][0]['timeout_s']=5+1999*.15+30
    validate_inputs(config,trace,manifest,diagnostic=True)


def test_diagnostic_cannot_inherit_a_formal_stage():
    config,trace,manifest=inputs()
    manifest['stage']='capacity'
    with pytest.raises(ValueError,match='labelled diagnostic'):
        validate_inputs(config,trace,manifest,diagnostic=True)


def test_oversized_context_is_rejected_before_gpu_execution():
    config,trace,manifest=inputs()
    config['max_model_len']=8192
    trace['requests'][0].update(prompt_len=8191,output_len=2)
    with pytest.raises(ValueError,match='context limit'):
        validate_inputs(config,trace,manifest,diagnostic=True)


def test_replay_keeps_individual_deadlines_and_absolute_arrivals(monkeypatch):
    observed=[]
    async def fake_send(session,api,model,prompt,length,**kw):
        observed.append((session.timeout_s,kw))
        return dict(request_id=kw['request_id'],success=True)
    monkeypatch.setattr('ecopadg.scalability.run.send_request',fake_send)
    trace=dict(requests=[dict(arrival_s=0.,output_len=2,timeout_s=120.),
                         dict(arrival_s=.001,output_len=2000,timeout_s=430.)],prompts=[[1],[2]])
    progress={'terminal':0}
    result=asyncio.run(replay(trace,'http://127.0.0.1:1','test',start_s=100.,progress=progress))
    assert progress['terminal']==2
    assert [r['declared_timeout_s'] for r in result]==[120.,430.]
    assert observed[0][1]['arrival_s']==100.
    assert observed[1][1]['arrival_s']==100.001
    assert 119 < observed[0][0] <= 120
    assert 429 < observed[1][0] <= 430
