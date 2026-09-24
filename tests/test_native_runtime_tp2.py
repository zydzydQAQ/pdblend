"""Whole TP2 group power/off/clock/transfer evidence; no GPU operations."""
import asyncio
from copy import deepcopy
from dataclasses import asdict,replace
from types import SimpleNamespace

import pytest

from pdblend.profile.collection.native_runtime_collect import NativeRuntimeCollector,validate_inventory
from pdblend.profile.collection.native_runtime_audit import replay_runtime,validate_transfer_request
from pdblend.profile.collection.native_runtime_topology import validate_phase,with_off_start_proof
from tests.test_native_runtime_collect import inventory,artifact,replace_bound,state,UUIDS
from tests.test_native_runtime_transfer_contract import full_request


def tp2_inventory():
    old,fleet,meter,sampler=inventory()
    specs=[replace(old[i],model='/models/Qwen2.5-32B-Instruct',tp=2,gpus=(2*i,2*i+1)) for i in range(4)]
    fleet.instances.clear()
    fleet.instances.update({s.instance_id:SimpleNamespace(spec=s,alive=lambda:True,events=[dict(kind='start',t_s=1)]) for s in specs})
    return specs,fleet,meter,sampler


def state2(value):
    value=deepcopy(value);value['tp']=2
    value['ranks']=[dict(deepcopy(value['ranks'][0]),rank=r) for r in range(2)]
    return value


def tp2_artifact(tmp_path):
    report,power,rows=artifact(tmp_path);specs,*_=tp2_inventory()
    report['actual_launch']=[dict(spec=asdict(s),argv=s.command()) for s in specs]
    caps={s.instance_id:dict(supported=True,model_id='Qwen2.5-32B-Instruct',tp=2,pp=1,
        gpu_uuids=[UUIDS[g] for g in s.gpus]) for s in specs}
    report['initial_capabilities']=caps
    def convert(value):
        if isinstance(value,dict):
            if 'tp' in value and 'ranks' in value:
                value.update(state2(value));return
            if 'requested_frequency_mhz' in value and 'gpus' in value:
                value['gpus']=[dict(gpu_uuid=UUIDS[g]) for g in (0,1)]
            if 'frequencies_mhz' in value:value['frequencies_mhz']*=2
            if 'devices' in value:
                value['devices']=[dict(gpu=g,uuid=UUIDS[g],compute_pids=[]) for g in (0,1)]
            for child in value.values():convert(child)
        elif isinstance(value,list):
            for child in value:convert(child)
    converted=[]
    for row in rows:
        if row['kind']=='capacity':
            if row['instance_id'] not in caps:continue
            row['capability']=caps[row['instance_id']];row['state']=state2(row['state'])
        else:
            convert(row)
            if row['kind'] in ('static','operation'):row['gpus']=[0,1]
            if row['kind']=='static':row['memory_before_mhz']*=2;row['memory_after_mhz']*=2
            if row.get('operation')=='park':
                row['receipt']['memory_mhz']*=2;row['receipt']['observed_mhz']*=2
            if row.get('operation')=='wake':row['receipt']['capability']=caps['i0']
        row['sequence']=len(converted);converted.append(row)
    report['restoration']['instances']=report['restoration']['instances'][:4]
    for row in report['restoration']['instances']:
        row['drain']=state2(row['drain']);row['resume']['state']=state2(row['resume']['state'])
        row['measurement_stop']['ranks'].append(dict(rank=1,acknowledged=True))
    for _,freq in power['frequency_samples']:freq[1]=freq[0]
    replace_bound(tmp_path,report,'power',power);replace_bound(tmp_path,report,'journal',converted)
    return report,power,converted


def test_tp2_runtime_targets_two_board_watts_but_meter_keeps_eight_board_energy(tmp_path):
    report,_,_=tp2_artifact(tmp_path);audit=replay_runtime(report)
    assert audit['raw_components_complete'],audit['errors']
    row=next(r for r in audit['holdout_comparisons'] if r['component']=='active_idle@1500')
    assert row['training_prediction']==row['heldout']==200.
    assert audit['phases'][0]['energy_j']==4000 and len(audit['capacity'])==4
    assert not audit['component_qualified'] and not audit['formal_eligible']


@pytest.mark.parametrize('fault',['target_board','off_board','off_uuid','static_rank','clock_board','frequency_board','power_order'])
def test_runtime_replay_rejects_missing_second_tp_rank_or_substituted_gpu(tmp_path,fault):
    report,power,rows=tp2_artifact(tmp_path)
    if fault=='target_board':next(r for r in rows if r['kind']=='static')['gpus']=[0]
    elif fault=='off_board':next(r for r in rows if r.get('state')=='off')['off_evidence']['observations'][0]['devices'].pop()
    elif fault=='off_uuid':next(r for r in rows if r.get('operation')=='off')['receipt']['off']['observations'][0]['devices'][1]['uuid']=UUIDS[7]
    elif fault=='static_rank':next(r for r in rows if r['kind']=='static')['after']['ranks'].pop()
    elif fault=='clock_board':next(r for r in rows if r.get('operation')=='clock_2520_to_1500')['receipt']['ack']['gpus'].pop()
    elif fault=='frequency_board':
        power['frequency_samples'][14][1][1]=2520
    elif fault=='power_order':power['gpu_uuids'][0],power['gpu_uuids'][1]=power['gpu_uuids'][1],power['gpu_uuids'][0]
    if fault in ('frequency_board','power_order'):
        report['power']=__import__('pdblend.profile.collection.native_runtime_collect',fromlist=['write_new']).write_new(tmp_path/'damaged-power.json',power)
    else:
        path=tmp_path/'damaged-journal.jsonl'
        import json,hashlib
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows));report['journal']=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    audit=replay_runtime(report);assert not audit['raw_components_complete'] and audit['errors']


def test_tp2_off_checks_both_devices_and_cannot_ignore_remaining_rank_process(tmp_path):
    specs,fleet,meter,sampler=tp2_inventory();validate_inventory(specs,fleet,meter,sampler,UUIDS)
    instance=fleet['i0'];instance.alive=lambda:False;instance.state='off';instance.events.append(dict(kind='stop',t_s=2))
    meter.backend._handle=lambda g:g
    seen=[]
    def processes(gpu):
        seen.append(gpu);return [SimpleNamespace(pid=77)] if gpu==1 else []
    meter.backend._nvml=SimpleNamespace(nvmlDeviceGetComputeRunningProcesses=processes)
    runner=NativeRuntimeCollector(specs,fleet,meter,sampler,tmp_path,gpu_uuids=UUIDS)
    with pytest.raises(RuntimeError,match='retains compute'):asyncio.run(runner.compute_empty(specs[0],timeout_s=0))
    assert seen==[0,1]
    meter.backend._nvml.nvmlDeviceGetComputeRunningProcesses=lambda gpu:[]
    assert len(asyncio.run(runner.compute_empty(specs[0],timeout_s=0))['observations'][-1]['devices'])==2


def test_tp2_transfer_validates_two_complete_rank_groups_and_four_target_boards():
    row,oldspecs=full_request();specs={}
    for index,(iid,spec) in enumerate(oldspecs.items()):
        specs[iid]=dict(spec,tp=2,gpus=[2*index,2*index+1],model='/models/Qwen2.5-32B-Instruct')
        row['native_epoch']['before'][iid]=state2(row['native_epoch']['before'][iid])
        row['native_epoch']['admission'][iid]['state']=state2(row['native_epoch']['admission'][iid]['state'])
    assert validate_transfer_request(row,specs=specs)['physical_copy_time'] is False
    ids=list(specs);phase=dict(kind='transfer',prefill_instance=ids[0],decode_instance=ids[1],
        gpus=[0,1,2,3],before=row['native_epoch']['before'],
        after={iid:dict(state2(state(1)),native_at_s=1001.) for iid in ids},finished_s=1000.)
    validate_phase(phase,specs,dict(enumerate(UUIDS)))
    phase['gpus']=[0,2]
    with pytest.raises(ValueError,match='two-peer TP'):validate_phase(phase,specs,dict(enumerate(UUIDS)))
    row['native_epoch']['admission'][ids[1]]['state']['ranks'].pop()
    with pytest.raises(RuntimeError,match='rank'):validate_transfer_request(row,specs=specs)


def test_old_off_format_reuses_only_actual_immediately_preceding_stop_proof():
    row=dict(kind='static',state='off',instance_id='i0',settle_started_s=10.)
    receipt={'actual':'owned stop'}
    operation=dict(kind='operation',operation='off',instance_id='i0',finished_s=9.,receipt={'off':receipt})
    rebuilt=with_off_start_proof(row,[operation,row])
    assert rebuilt['off_before'] is receipt and 'off_before' not in row
    later=dict(operation,operation='wake',finished_s=9.5)
    with pytest.raises(ValueError,match='different operation'):with_off_start_proof(row,[operation,later,row])
