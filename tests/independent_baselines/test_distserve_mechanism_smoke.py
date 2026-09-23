import asyncio, json
from types import SimpleNamespace
from pdblend_baselines.distserve import mechanism_smoke as mod

class Fake:
    def __init__(self): self.calls=[]
    async def capability(self, role): return {'supported':True,'tp':1,'pp':1,'model_hash':'m','tokenizer_hash':'t','image_digest':'i','source_revision':'s'}
    async def state(self, role): return {'native_evidence_complete':True,'free_blocks':4096,'block_size':16,'max_num_batched_tokens':8192,'generation':2,'acknowledged_generation':2}
    async def prefill(self, p): self.calls.append(('prefill',p)); return {'acknowledged':True,'retained_handle':'h','source_address':'p:9000','outputs':[{'token_ids':[777]}]}
    async def expect_load(self,p): self.calls.append(('expect',p)); return {'acknowledged':True}
    async def transfer(self,p): self.calls.append(('transfer',p)); return {'acknowledged':True}
    async def load_ack(self,p): self.calls.append(('load',p)); return {'acknowledged':True,'request_id':p['target_request_id'],'transaction_id':p['transaction_id'],'generation':p['generation'],'ranks':[{'rank':0,'acknowledged':True,'generation':p['generation'],'transaction_id':p['transaction_id'],'target_request_id':p['target_request_id'],'expected_layers':1,'loaded_layers':1}]}
    async def release(self,p): self.calls.append(('release',p)); return {'acknowledged':True}
    async def cancel(self,r,p): self.calls.append(('cancel',p)); return {'acknowledged':True}
    async def generate_decode(self,p):
        self.calls.append(('generate',p)); yield {'token_ids':list(range(31)),'finished':True,'finish_reason':'stop'}

def test_distserve_cli_protocol_and_carry(tmp_path, monkeypatch):
    fake=Fake(); monkeypatch.setattr(mod,'MappedDistServeTransport',lambda p,d:fake)
    args=SimpleNamespace(prefill_url='p',decode_url='d',target_address='d:9000',tp=1,generation=2,prompt_tokens=8,max_tokens=32,request_id='r',carry_first_token=True,out=str(tmp_path/'a.json'))
    result=asyncio.run(mod.run(args)); artifact=json.loads((tmp_path/'a.json').read_text())
    assert result['status']=='passed'; assert artifact['mode']=='carry_first_token'
    transfer=next(x[1] for x in fake.calls if x[0]=='transfer'); assert '___decode_addr_d:9000_' in transfer['target_request_id']
    generated=next(x[1] for x in fake.calls if x[0]=='generate'); assert generated['prompt'][-1]==777
    assert fake.calls[-1][0]=='release'
