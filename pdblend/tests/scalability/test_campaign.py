import copy
import json

import pytest

from ecopadg.scalability.campaign import (new_campaign,next_task,apply_observation,
    settle_selections,bind_freeze,verify_freeze_binding)
from ecopadg.scalability.artifacts import sha256
from ecopadg.scalability.prepare_inputs import export_pool


def prepared(tmp_path):
    config=tmp_path/'base.json'
    config.write_text(json.dumps(dict(model_name='Qwen2.5-14B-Instruct',max_service_frequency_mhz=2520,
        distserve_prefill_batch=1,distserve_decode_batch=32,
        instances=[dict(id=f'i{g}',gpus=[g],tp=1,url=f'http://127.0.0.1:{30000+g}') for g in range(8)])))
    pools={}
    for dataset in ('sharegpt','longbench'):
        pools[dataset]=tmp_path/f'{dataset}.json'
        pools[dataset].write_text(json.dumps(dict(dataset=dataset,records=[dict(prompt=[1,2],prompt_len=2,output_len=4)])))
    return new_campaign(config,pools,tmp_path/'campaign')


def test_preparation_does_not_claim_qualification(tmp_path):
    state=prepared(tmp_path)
    assert state['status']=='prepared' and state['freeze_path'] is None
    task=next_task(state)
    assert task['stage']=='pilot' and task['purpose']=='pd_split_selection'


def test_failed_engineering_attempt_is_kept_and_stops_queue(tmp_path):
    state=prepared(tmp_path);task=next_task(state)
    changed=apply_observation(state,task,dict(measurement_valid=False,capacity_pass=False,error='missing raw power'))
    assert changed['observations'][0]['error']=='missing raw power'
    assert next_task(changed)['action']=='blocked'
    assert not state['observations']


def test_valid_overload_is_a_capacity_boundary_not_engineering_block(tmp_path):
    state=prepared(tmp_path);task=next_task(state)
    changed=apply_observation(state,task,dict(measurement_valid=True,capacity_pass=False))
    assert changed['status']=='running'
    assert next_task(changed)['rate_rps']==task['rate_rps']/2


def test_stale_dispatch_cannot_consume_an_observation(tmp_path):
    state=prepared(tmp_path);task=next_task(state);task['rate_rps']*=2
    with pytest.raises(ValueError,match='stale task'):
        apply_observation(state,task,dict(measurement_valid=True,capacity_pass=True))


def test_one_token_exclusions_are_declared_before_measurement(tmp_path):
    source=tmp_path/'source.json'
    source.write_text(json.dumps(dict(dataset='sharegpt',formal={'101':[
        dict(prompt=[1],input_tokens=1,output_tokens=1),dict(prompt=[2],input_tokens=1,output_tokens=2)]})))
    result=export_pool(source,'sharegpt',tmp_path/'pool.json')
    assert len(result['records'])==1 and len(result['exclusions'])==1
    assert result['source_sha256']


def bound_inputs(state):
    return dict(schema='pdblend-scalability-freeze-v1',host_id=state['host_id'],
        config_path=state['config_path'],pools=state['pools'],
        files={state['config_path']:state['config_sha256'],
               **{p:state['pool_hashes'][d] for d,p in state['pools'].items()}})


def test_binding_freeze_is_exact_and_one_time(tmp_path,monkeypatch):
    # Full qualification contents are exercised by test_preflight; this test
    # isolates the campaign's exact-input and one-time binding contract.
    monkeypatch.setattr('ecopadg.scalability.campaign.verify_freeze',lambda freeze: [])
    state=prepared(tmp_path)
    path=tmp_path/'freeze.json'
    path.write_text(json.dumps(bound_inputs(state)))
    bind_freeze(tmp_path/'campaign',path)
    saved=json.loads((tmp_path/'campaign'/'state.json').read_text())
    assert saved['freeze_sha256']==sha256(path)
    assert (tmp_path/'campaign'/'freeze-binding.json').is_file()
    with pytest.raises(ValueError,match='only be bound once'):
        bind_freeze(tmp_path/'campaign',path)


def test_another_valid_pool_freeze_cannot_be_attached(tmp_path):
    state=prepared(tmp_path)
    freeze=bound_inputs(state)
    freeze['pools']=dict(freeze['pools'],sharegpt='/other-pool.json')
    with pytest.raises(ValueError,match='different dataset pools'):
        verify_freeze_binding(state,freeze)
    freeze=bound_inputs(state)
    freeze['files'][state['config_path']]='0'*64
    with pytest.raises(ValueError,match='does not hash'):
        verify_freeze_binding(state,freeze)


def test_changed_input_freeze_leaves_prepared_state_untouched(tmp_path):
    state=prepared(tmp_path)
    path=tmp_path/'freeze.json'
    path.write_text(json.dumps(bound_inputs(state)))
    (tmp_path/'sharegpt.json').write_text('{}')
    with pytest.raises(ValueError,match='campaign input changed'):
        bind_freeze(tmp_path/'campaign',path)
    assert json.loads((tmp_path/'campaign'/'state.json').read_text())==state
    assert not (tmp_path/'campaign'/'freeze-binding.json').exists()
