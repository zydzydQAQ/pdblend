import asyncio,copy,time
import pytest
from checks import Checks,check_ack,check_ranks,difference,tokens
from validate import validate_scope

def instance():return dict(id='b',url='http://127.0.0.1:34100',tp=2,gpus=[0,1],native_kind='legacy_sync_put',container={'image':'sha256:d11407cd827a43a0dec8ad7d4d7037c97c39bbe93c6f4b4fd951c94e67509a8b'})
def raw():return dict(id='b',generation=7,acknowledged_generation=7,error=None,runtime_error=None,transport_healthy=True,timestamp=time.time(),transfer_observed_s=time.time())
def rank():return dict(buffered_tensors=0,inflight_receives=0,buffered_gpu_bytes=0,allocations={},listener_alive=True)
def test_actual_legacy_owner_without_v3_cache_is_valid():check_ack(raw(),instance())
@pytest.mark.parametrize('field',['timestamp','transfer_observed_s'])
def test_stale_actual_observation_rejected(field):
    r=raw();r[field]-=2
    with pytest.raises(RuntimeError):check_ack(r,instance())
@pytest.mark.parametrize('field',['buffered_tensors','inflight_receives','buffered_gpu_bytes','allocations'])
def test_missing_native_rank_observation_rejected(field):
    r=rank();r.pop(field)
    with pytest.raises(RuntimeError):check_ranks([rank(),r],2)
def test_missing_rank_rejected():
    with pytest.raises(RuntimeError):check_ranks([rank()],2)
def test_token_position_and_work_unchanged():
    a=list(range(64));b=list(a);b[31]=999;assert difference(a,b)==dict(position_one_based=32,reference=31,observed=999)
    with pytest.raises(RuntimeError):tokens(dict(token_ids=a,usage={'prompt_tokens':128,'completion_tokens':63}),128)
def test_scope_does_not_invent_dynamic_legacy_budget():
    b={'model':'32b','instances':[dict(instance(),id=str(j),gpus=[2*j,2*j+1]) for j in range(4)]};validate_scope(b)
    b['instances'][0]['service_budget_tokens']=8192
    with pytest.raises(RuntimeError):validate_scope(b)
def test_failed_native_proof_still_resumes_and_stays_failed(tmp_path):
    class Fake(Checks):
        async def control(self,i,**changes):self.controls.append(changes);return {'after':{'accepting':True}}
        async def idle(self,i,timeout=15):return dict(generation=7,accepting=True)
        async def http(self,i,path,payload=None,rid=None,timeout=45):
            assert path=='/drain';return dict(drained=True,accepting=False,generation=8,drain_proof_type='synchronous_put_owner_barrier',transfers=[rank()])
    f=Fake(None,{'instances':[instance()]},tmp_path/'out');f.controls=[]
    assert asyncio.run(f.cleanup()) is False
    assert f.controls[-1]==dict(role='mixed',mode='continuous',admit_prefill=True,admit_decode=True)
    assert f.state['cleanup']['instances']['b']['errors'];f.close()
def test_cleanup_cancels_only_owned_ids(tmp_path):
    class Fake(Checks):
        async def control(self,i,**changes):return {}
        async def idle(self,i,timeout=15):return dict(generation=7,accepting=True)
        async def http(self,i,path,payload=None,rid=None,timeout=45):
            if path=='/cancel':self.cancelled.append(payload['request_id']);return {}
            return dict(drained=True,accepting=False,generation=8,drain_proof_type='synchronous_put_owner_barrier',transfers=[rank(),rank()])
    f=Fake(None,{'instances':[instance()]},tmp_path/'out');f.cancelled=[];f.own(instance(),'owned-request')
    assert asyncio.run(f.cleanup()) is True;assert f.cancelled==['owned-request'];f.close()


def test_temporal_failure_preserves_only_verified_other_mechanisms():
    from validate import mechanism_gates
    checks=dict(ordinary_cross_replica_exact=True,pd_exact_all_declared_pairs=True,cancel_all_tp_ranks=True)
    assert mechanism_gates(checks,True,True)==dict(ordinary=True,pd=True,temporal=False)
    assert mechanism_gates(checks,False,True)==dict(ordinary=False,pd=False,temporal=False)
