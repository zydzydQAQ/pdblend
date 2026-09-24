"""Replay the first real nonstream P response and preserve strict D/epoch gates."""
import asyncio
from copy import deepcopy
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path

import pytest

from pdblend.profile.collection.native_runtime_audit import validate_transfer_payload,validate_transfer_request
from pdblend.profile.collection.native_runtime_collect import NativeRuntimeCollector
from tests.test_native_runtime_collect import inventory,state,UUIDS

FIXTURE=Path(__file__).parent/'fixtures/native_runtime/2026-09-24_first_512_request.json'


def raw_request():return json.loads(FIXTURE.read_text())['request']


def full_request():
    row=raw_request();specs,*_=inventory()
    specs=[replace(s,generation=1) for s in specs[:2]]
    pre=row['result']['prefill'];combined=row['result']['combined']
    oldp,oldd=pre['instance_id'],combined['instance_id']
    pre['instance_id']=specs[0].instance_id
    for value in [*row['result']['ordinary'],combined]:value['instance_id']=specs[1].instance_id
    timestamp=row['result']['ordinary'][0]['submitted_s']
    before={s.instance_id:dict(state(1),native_at_s=timestamp-.2) for s in specs}
    admission={s.instance_id:dict(control=dict(generation=1,acknowledged=True),
        state=dict(state(1),native_at_s=timestamp-.1,accepting=True)) for s in specs}
    row['native_epoch']=dict(prefill_instance=specs[0].instance_id,decode_instance=specs[1].instance_id,
                             before=before,admission=admission)
    return row,{s.instance_id:asdict(s) for s in specs}


def validate_payload(row):
    return validate_transfer_payload(row['result'],input_tokens=row['input_tokens'],
        prefill_instance=row['result']['prefill']['instance_id'],decode_instance=row['result']['combined']['instance_id'])


def test_real_512_nonstream_first_token_replays_without_changing_frozen_raw():
    fixture=json.loads(FIXTURE.read_text());source=Path(fixture['provenance']['path'])
    before=source.read_bytes() if source.exists() else None
    if before is not None:
        assert hashlib.sha256(before).hexdigest()==fixture['provenance']['sha256']
        assert json.loads(before.splitlines()[fixture['provenance']['sequence']])==fixture['request']
    row=fixture['request'];assert row['result']['prefill']['stream_done'] is False
    receipt=validate_payload(row)
    assert receipt['protocol']=='carry_first_token_second_output_v1'
    assert receipt['physical_copy_time'] is False
    assert row['result']['combined']['token_ids']==row['result']['ordinary'][0]['token_ids']
    if before is not None:assert source.read_bytes()==before
    # Historical first request remains insufficient for a new full epoch/window receipt.
    with pytest.raises(KeyError,match='native_epoch'):validate_transfer_request(row,specs={})


@pytest.mark.parametrize('fault',['ordinary_partial','combined_partial','prefill_error','prefill_usage',
    'prefill_no_error_field','prefill_missing_token','prefill_wrong_first','prefill_wrong_time',
    'prefill_wrong_usage','prefill_wrong_request','prefill_wrong_endpoint','prefill_missing_finish',
    'combined_missing_decode_count','combined_wrong_second_time','combined_wrong_protocol','ordinary_disagrees'])
def test_missing_or_partial_transfer_payloads_still_fail(fault):
    row=raw_request();result=row['result'];pre=result['prefill'];combined=result['combined']
    if fault=='ordinary_partial':result['ordinary'][0]['stream_done']=False
    elif fault=='combined_partial':combined['stream_done']=False
    elif fault=='prefill_error':pre['error']='HTTP 500'
    elif fault=='prefill_usage':pre['usage_received']=False
    elif fault=='prefill_no_error_field':del pre['error']
    elif fault=='prefill_missing_token':pre['token_ids']=[]
    elif fault=='prefill_wrong_first':pre['token_ids'][0]+=1
    elif fault=='prefill_wrong_time':pre['token_times_s'][0]-=1
    elif fault=='prefill_wrong_usage':pre['completion_tokens']=2
    elif fault=='prefill_wrong_request':pre['request_id']='wrong'
    elif fault=='prefill_wrong_endpoint':pre['instance_id']=combined['instance_id']
    elif fault=='prefill_missing_finish':del pre['finished_s']
    elif fault=='combined_missing_decode_count':del combined['decode_completion_tokens']
    elif fault=='combined_wrong_second_time':combined['decode_first_token_s']+=1
    elif fault=='combined_wrong_protocol':combined['pd_protocol']='recompute'
    elif fault=='ordinary_disagrees':result['ordinary'][1]['token_ids'][2]+=1
    with pytest.raises(ValueError):validate_payload(row)


def test_new_collector_and_replayer_share_the_same_native_epoch_contract():
    row,specs=full_request()
    assert validate_transfer_request(row,specs=specs)==validate_payload(row)


@pytest.mark.parametrize('fault',['old_epoch','rank_epoch','missing_admission','admission_closed','bad_ack','admission_after_request','peer_epoch','wrong_holdout_seed'])
def test_prefill_generation_comes_from_actual_native_evidence_not_invented_completion_field(fault):
    row,specs=full_request();epochs=row['native_epoch'];prefill=epochs['prefill_instance']
    if fault=='old_epoch':epochs['before'][prefill]['generation']=0
    elif fault=='rank_epoch':epochs['admission'][prefill]['state']['ranks'][0]['generation']=0
    elif fault=='missing_admission':del epochs['admission'][prefill]
    elif fault=='admission_closed':epochs['admission'][prefill]['state']['accepting']=False
    elif fault=='bad_ack':epochs['admission'][prefill]['control']['acknowledged']=False
    elif fault=='admission_after_request':epochs['admission'][prefill]['state']['native_at_s']+=10
    elif fault=='peer_epoch':specs[prefill]['generation']=2
    elif fault=='wrong_holdout_seed':row['repeat']=3
    with pytest.raises((ValueError,KeyError,RuntimeError)):validate_transfer_request(row,specs=specs)


def test_memory_clock_boundary_is_after_settle_without_shortening_five_second_measurement(tmp_path,monkeypatch):
    import pdblend.profile.collection.native_runtime_collect as module
    specs,fleet,meter,sampler=inventory();runner=NativeRuntimeCollector(specs,fleet,meter,sampler,tmp_path,gpu_uuids=UUIDS)
    now=[100.];memory_reads=[]
    def memory(gpu):
        memory_reads.append(now[0]);return 9001 if now[0]<102 else 405
    async def sleep(duration):now[0]+=duration
    async def native_state(spec,**kwargs):return state()
    meter.mem_freq=memory;runner.state=native_state
    monkeypatch.setattr(module.time,'time',lambda:now[0]);monkeypatch.setattr(module.asyncio,'sleep',sleep)
    asyncio.run(runner.static_window(specs[0],'L1',0))
    row=runner.rows[0]
    assert memory_reads==[100.,102.,107.]
    assert row['memory_settle_before_mhz']==[9001] and row['memory_before_mhz']==row['memory_after_mhz']==[405]
    assert row['started_s']-row['settle_started_s']==2 and row['finished_s']-row['started_s']==5
