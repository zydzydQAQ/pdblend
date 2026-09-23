"""Real DistServe mechanism_smoke rejection-path tests (CPU fake HTTP)."""
import asyncio, json
from types import SimpleNamespace
import pytest
from pdblend_baselines.distserve import mechanism_smoke as mod
class FakeTransport:
    def __init__(self, *, transfer=True, expect=True, release=True, tokens=31): self.transfer_ok=transfer; self.expect_ok=expect; self.release_ok=release; self.tokens=tokens; self.calls=[]
    async def capability(self, role): return {'supported':True,'tp':1,'pp':1,'model_hash':'m','tokenizer_hash':'t','image_digest':'i','source_revision':'s'}
    async def state(self, role): return {'native_evidence_complete':True,'free_blocks':4096,'block_size':16,'max_num_batched_tokens':8192,'generation':2,'acknowledged_generation':2}
    async def prefill(self,p): self.calls.append(('prefill',p)); return {'acknowledged':True,'retained_handle':'h','source_address':'p:9000','outputs':[{'token_ids':[777]}]}
    async def expect_load(self,p): self.calls.append(('expect_load',p)); return {'acknowledged':self.expect_ok}
    async def transfer(self,p): self.calls.append(('transfer',p)); return {'acknowledged':self.transfer_ok}
    async def load_ack(self,p): self.calls.append(('load_ack',p)); return {'acknowledged':True,'request_id':p['target_request_id'],'transaction_id':p['transaction_id'],'generation':p['generation'],'ranks':[{'rank':0,'acknowledged':True,'generation':p['generation'],'transaction_id':p['transaction_id'],'target_request_id':p['target_request_id'],'expected_layers':1,'loaded_layers':1}]}
    async def release(self,p): self.calls.append(('release',p)); return {'acknowledged':self.release_ok}
    async def cancel(self,r,p): self.calls.append(('cancel',p)); return {'acknowledged':True}
    async def generate_decode(self,p): self.calls.append(('generate',p)); yield {'token_ids':list(range(self.tokens)),'finished':True,'finish_reason':'stop'}
def args(path): return SimpleNamespace(prefill_url='p',decode_url='d',target_address='d:9000',tp=1,generation=2,prompt_tokens=8,max_tokens=32,request_id='reject',carry_first_token=False,out=str(path))
def invoke(monkeypatch,tmp_path,transport):
    monkeypatch.setattr(mod,'MappedDistServeTransport',lambda p,d:transport)
    try: result=asyncio.run(mod.run(args(tmp_path/'artifact.json')))
    except Exception: return None,transport
    return result,transport
@pytest.mark.parametrize('kwargs',[{'transfer':False},{'expect':False},{'tokens':2},{'release':False}])
def test_distserve_mechanism_rejections_never_pass(monkeypatch,tmp_path,kwargs):
    result,transport=invoke(monkeypatch,tmp_path,FakeTransport(**kwargs))
    assert result is None or result['status']!='passed'
    assert any(name in {'prefill','expect_load','transfer','generate','release','cancel'} for name,_ in transport.calls)
    if result is not None: assert json.loads((tmp_path/'artifact.json').read_text())['formal_eligible'] is False
def test_wrong_generation_path_is_sent_to_native_expect_load(monkeypatch,tmp_path):
    result,transport=invoke(monkeypatch,tmp_path,FakeTransport(expect=False))
    assert result is None or result['status']!='passed'
    expect=next(payload for name,payload in transport.calls if name=='expect_load'); assert expect['generation']==2

@pytest.mark.parametrize('kind',['missing_rank','wrong_generation'])
def test_load_ack_requires_complete_rank_identity(monkeypatch,tmp_path,kind):
    class BadLoad(FakeTransport):
        async def load_ack(self,p):
            value=await super().load_ack(p)
            if kind=='missing_rank': value['ranks']=[]
            else: value['ranks'][0]['generation']=999
            return value
    result,transport=invoke(monkeypatch,tmp_path,BadLoad())
    assert result is None or result['status']!='passed'
