import pytest
from types import SimpleNamespace
from pdblend_runtime.kv import KVCapabilityError, operation

class Worker:
    def __init__(self):
        self._native_generation=2
        self.parallel_config=SimpleNamespace(tensor_parallel_size=1)
        self.model_config=SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=1))
    def send_kv_layer(self,*a,**k): return {'submitted':True,'layer':k['layer']}
    def wait_for_sent(self,tx,ranks): return {'acknowledged':True,'ranks':[0]}
    def start_load_kv(self,**k): return True
    def consume_loaded_kv(self,**k): return True

def test_hold_transfer_load_release_state():
    w=Worker(); rid='distserve-hold-a___prefill_addr_x___decode_addr_127.0.0.1:0_tag'
    assert operation(w,'hold',{'held_request_id':rid,'source_address':'x','layers':{0:'tensor'},'slots':{0:'slot'}})['acknowledged']
    result=operation(w,'transfer',{'held_request_id':rid,'target_request_id':'d','target_address':'y','target_tp':1,'generation':2,'transaction_id':'tx'})
    assert result['acknowledged']
    operation(w,'expect_load',{'target_request_id':'d','transaction_id':'tx','generation':2,'source_tokens':16,'expected_layers':1})
    assert operation(w,'load_ack',{'target_request_id':'d','transaction_id':'tx','layer':0,'generation':2})['acknowledged']
    assert operation(w,'release',{'held_request_id':rid})['released']

def test_missing_worker_hook_fails_closed():
    class Bare: pass
    with pytest.raises(KVCapabilityError): operation(Bare(),'transfer',{'held_request_id':'x','target_request_id':'d','target_address':'y','target_tp':1,'generation':1,'transaction_id':'t'})

def test_expect_load_is_required_and_persistent():
    w=Worker(); w._native_generation=2
    operation(w,'expect_load',{'target_request_id':'d','transaction_id':'t','generation':2,'source_tokens':16,'expected_layers':1})
    with pytest.raises(KVCapabilityError): operation(w,'load_ack',{'target_request_id':'d','transaction_id':'other','layer':'x','generation':2})
    assert operation(w,'query_load',{'target_request_id':'d','transaction_id':'t'})['acknowledged'] is False
