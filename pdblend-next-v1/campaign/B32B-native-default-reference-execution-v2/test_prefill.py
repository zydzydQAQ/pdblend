import copy
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace as N
import pytest
from prefill_evidence import verify

ROOT=Path(__file__).resolve().parent

def load(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

class T:
    reads=0
    def __init__(self,value):self.value=copy.deepcopy(value);self.shape=(len(value),len(value[0])) if value and isinstance(value[0],list) else (len(value),);self.ndim=len(self.shape)
    def numel(self):return self.shape[0]*(self.shape[1] if len(self.shape)>1 else 1)
    def detach(self):return self
    def clone(self):return T(self.value)
    def cpu(self):T.reads+=1;return self
    def tolist(self):return copy.deepcopy(self.value)


def module(monkeypatch,name,spec):
    old=load(name+'_old',ROOT/'image-context/pdblend_diagnostics.py')
    old.state=lambda:N(spec=spec,spec_sha=__import__('hashlib').sha256(json.dumps(spec).encode()).hexdigest())
    package=types.ModuleType('vllm');package.pdblend_diagnostics=old;monkeypatch.setitem(sys.modules,'vllm',package)
    return load(name,ROOT/'image-context/pdblend_prefill_reference.py')


def test_disabled_does_not_read_tensor_or_create_writer(monkeypatch):
    monkeypatch.delenv('PDBLEND_PREFILL_REFERENCE',raising=False)
    m=module(monkeypatch,'prefill_disabled',{})
    assert m.begin(None,None,None,None) is None and m._STATE is None


def fixture(monkeypatch,tmp_path):
    from adapter import sha
    monkeypatch.setenv('PDBLEND_PREFILL_REFERENCE','1')
    spec=json.loads((ROOT/'request-spec.json').read_text());spec['output_dir']=str(tmp_path/'capture-live')
    sp=tmp_path/'observation.json';sp.write_text(json.dumps(spec));runtime=tmp_path/'runtime';runtime.mkdir()
    cfg=runtime/'engine.json';cfg.write_text(json.dumps(dict(runtime_dir=str(runtime))))
    job=dict(observation_spec=str(sp),diagnostic_instance=dict(config=str(cfg)))
    records=[]
    for rank in (0,1):
        m=module(monkeypatch,'prefill_rank'+str(rank),spec)
        c=N(chunked_prefill_enabled=True,max_num_batched_tokens=8192,max_num_seqs=32)
        runner=N(scheduler_config=c,builder=N(scheduler_config=c,chunked_prefill_enabled=True))
        records.append(m.flags(runner,rank))
        for request in spec['requests']:
            tokens=request['body']['prompt'];n=len(tokens);table=list(range(n//16))
            p=N(block_tables=T([table]),slot_mapping=T(list(range(n))),seq_lens_tensor=T([n]),
                query_start_loc=T([0,n]),context_lens_tensor=T([0]))
            a=N(num_prefills=1,num_prefill_tokens=n,num_decode_tokens=0,prefill_metadata=p)
            inp=N(attn_metadata=a,request_ids_to_seq_ids={request['request_uuid']:[rank+1]},input_tokens=T(tokens),input_positions=T(list(range(n))))
            before=T.reads;context=m.begin(inp,runner,rank,[T([1]) for _ in range(64)])
            assert context is not None and T.reads==before
            m.finish(context);assert T.reads==before+7
        m._STATE['writer'].close()
    (runtime/'worker-config.json').write_text(json.dumps(dict(before=records,after=records)))
    return job


def test_actual_capture_writer_to_offline_eight_records_and_true_flags(monkeypatch,tmp_path):
    job=fixture(monkeypatch,tmp_path)
    result=verify(job,tmp_path)
    assert result['complete'] and result['records']==8 and result['branch_is_source_inference']


def test_changed_worker_flag_cannot_pass_metadata_gate(monkeypatch,tmp_path):
    job=fixture(monkeypatch,tmp_path)
    p=tmp_path/'runtime/worker-config.json';s=json.loads(p.read_text());s['after'][1]['builder_chunked']=False;p.write_text(json.dumps(s))
    with pytest.raises(RuntimeError,match='all worker configs'):verify(job,tmp_path)


def test_slot_mismatch_retained_and_rejected(monkeypatch,tmp_path):
    job=fixture(monkeypatch,tmp_path)
    p=next((tmp_path/'prefill-capture-live').glob('rank0-*.jsonl'));rows=[json.loads(x) for x in p.read_text().splitlines()]
    rows[0]['metadata']['slots'][0]+=1;p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(RuntimeError,match='block/slot'):verify(job,tmp_path)
    assert p.exists() and not (tmp_path/'prefill-capture-frozen').exists()


def test_initialization_dummy_prefill_is_not_a_capture_failure(monkeypatch):
    monkeypatch.setenv('PDBLEND_PREFILL_REFERENCE','1')
    spec=json.loads((ROOT/'request-spec.json').read_text())
    m=module(monkeypatch,'prefill_warmup',spec)
    inp=N(attn_metadata=N(num_prefills=1),request_ids_to_seq_ids={'warmup':[0]})
    assert m.begin(inp,None,0,None) is None
    assert m._STATE['writer'] is None and 'error' not in m._STATE
