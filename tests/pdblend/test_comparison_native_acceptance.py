"""Shared transport checks must bind the actual full native launch command."""
from copy import deepcopy

import pytest

from pdblend.bench.comparison_native_acceptance import (
    native_topology, audit_native_startup, audit_native_reset,
    audit_native_meter, audit_native_metrics,
)
from pdblend.bench.resident_session import engine_signature
from pdblend_runtime.probe import NativeSpec
from test_comparison_acceptance import fixture


def native_fixture(tmp_path):
    args=fixture(tmp_path)
    point,identity,startup=(args[k] for k in ('point','engine_identity','startup_qualification'))
    point.update(system='distserve',seed=701,duration_s=150)
    for i,row in enumerate(identity['instances']):
        row['launch_options'].update(kv_connector='P2pNcclConnector',extra_args=['--enforce-eager'])
        launch=startup['actual_launch_identity']['instances'][i]
        launch.update(launch_options=deepcopy(row['launch_options']),argv=NativeSpec(
            row['instance_id'],(i,),20000+i*4,'/models/'+point['model_id'],tp=1,pp=1,
            **row['launch_options']).command())
        for rank in startup['capabilities'][row['instance_id']]['state']['ranks']:
            rank['retained_kv_supported']=True
    startup['engine_signature']=engine_signature(identity)
    return args


def test_valid_native_startup_and_fresh_boundary(tmp_path):
    args=native_fixture(tmp_path)
    instances=native_topology(args['point'],args['engine_identity'])
    audit_native_startup(args['point'],args['engine_identity'],args['startup_qualification'],instances,args['raw_refs'])
    audit_native_reset(args['reset'],instances,100.)


@pytest.mark.parametrize('change', ['memory','connector','extra','worker','tp','cache','mapping'])
def test_actual_unbound_execution_switch_is_rejected(tmp_path,change):
    args=native_fixture(tmp_path)
    row=args['startup_qualification']['actual_launch_identity']['instances'][0]
    if change=='mapping':row['environment']['CUDA_VISIBLE_DEVICES']='7'
    elif change=='extra':row['argv']+=['--max-num-seqs','12']
    elif change=='worker':row['argv']+=['--worker-cls','alternate.Worker']
    elif change=='cache':row['argv']+=['--enable-prefix-caching']
    else:
        flag={'memory':'--gpu-memory-utilization','connector':'--kv-transfer-config','tp':'--tensor-parallel-size'}[change]
        row['argv'][row['argv'].index(flag)+1]={'memory':'.50','connector':'{}','tp':'2'}[change]
    with pytest.raises(ValueError):
        audit_native_startup(args['point'],args['engine_identity'],args['startup_qualification'],
            native_topology(args['point'],args['engine_identity']),args['raw_refs'])


def test_cross_generation_peer_cannot_pass_reset(tmp_path):
    args=native_fixture(tmp_path)
    state=args['reset']['reopen_state']['mixed1'];state['generation']=6
    with pytest.raises(ValueError):
        audit_native_reset(args['reset'],native_topology(args['point'],args['engine_identity']),100.)


def test_resident_subset_keeps_all_eight_power_devices(tmp_path):
    args=native_fixture(tmp_path)
    identity=args['engine_identity'];identity['instances']=identity['instances'][:2]
    assert len(native_topology(args['point'],identity))==2
    assert len(identity['fleet_gpu_uuids'])==8
