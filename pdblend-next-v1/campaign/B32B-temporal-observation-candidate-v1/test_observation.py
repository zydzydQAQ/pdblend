import ast
import dataclasses
import hashlib
import json
import pathlib
import sys
import time
from types import SimpleNamespace as N
import numpy as np
import pytest
import pdblend_diagnostics as d
import build_candidate as build

ROOT=pathlib.Path(__file__).resolve().parent
IDS=['a'*32,'b'*32]

def spec():
    return dict(schema=1,enabled=True,parent_image=d.IMAGE,parent_model_runner_sha256=d.PARENT,
        request_ids=IDS.copy(),output_indices=list(d.STEPS),tp=2,pp=1,max_output_tokens=64,
        output_dir='/root/workspace/pdblend-next-v1/campaign/test-only-observation/unused',
        diagnostic_only=True,performance_claims_allowed=False)

class Sink:
    def __init__(self):self.rank=0;self.rows=[];self.failed=False;self.error=None
    def submit(self,v):self.rows.append(json.loads(json.dumps(v)))

class Tensor:
    reads=0
    def __init__(self,a):self.a=np.array(a)
    @property
    def device(self):return 'cuda:%d' % d._STATE.writer.rank
    @property
    def shape(self):return self.a.shape
    @property
    def ndim(self):return self.a.ndim
    @property
    def dtype(self):return self.a.dtype
    def numel(self):return self.a.size
    def __getitem__(self,key):return Tensor(self.a[key])
    def detach(self):return self
    def reshape(self,*shape):return Tensor(self.a.reshape(*shape))
    def to(self,dtype=None):return Tensor(self.a.astype(dtype))
    def clone(self):return Tensor(self.a.copy())
    def cpu(self):Tensor.reads+=1;return self
    def tolist(self):return self.a.tolist()

def torch_fake():
    def topk(t,n):
        ix=np.argsort(t.a)[-n:][::-1]
        return Tensor(t.a[ix]),Tensor(ix)
    return N(int64=np.int64,float64=np.float64,cat=lambda xs:Tensor(np.concatenate([x.a for x in xs])),
             topk=topk,argmax=lambda t:Tensor(np.argmax(t.a)))

@pytest.fixture
def active(monkeypatch):
    s=d.State(spec(),'f'*64);s.writer=Sink();d._STATE=s;d._LOADED=True
    monkeypatch.setitem(sys.modules,'torch',torch_fake());Tensor.reads=0
    yield s
    d._STATE=None;d._LOADED=False

def fixture_input(rid=IDS[1],step=32,other_first=True):
    sid=17;other=5
    data=N(get_output_len=lambda:step-1,get_prompt_len=lambda:192,
        get_num_computed_tokens=lambda:192+step-2,get_len=lambda:192+step-1,
        get_last_token_id=lambda:358)
    params=N(max_tokens=64,temperature=0,top_p=1,ignore_eos=True,n=1,seed=0)
    g=N(request_id=rid,is_prompt=False,do_sample=True,seq_data={sid:data},
        sampling_params=params,block_tables={sid:list(range(100,114))})
    mapping={'other':[other],rid:[sid]} if other_first else {rid:[sid],'other':[other]}
    row=list(mapping).index(rid);off=row
    tok=[20,20];pos=[130,130];slot=[1138,1138];seqlens=[131,131];ctx=[130,130]
    tok[row]=358;pos[row]=222;slot[row]=113*16+14;seqlens[row]=223;ctx[row]=222
    tables=[list(range(63,77)),list(range(63,77))];tables[row]=list(range(100,114))
    mi=N(request_ids_to_seq_ids=mapping,query_lens=[1,1],
        input_tokens=Tensor(tok),input_positions=Tensor(pos),diagnostic_bindings=None,
        attn_metadata=N(use_cuda_graph=False,block_tables=Tensor(tables),
            slot_mapping=Tensor(slot),seq_lens_tensor=Tensor(seqlens),context_lens_tensor=Tensor(ctx),
            query_start_loc=Tensor([0,1,2]),num_prefills=0,num_prefill_tokens=0,num_decode_tokens=2))
    # Deliberately sampler-logits row != input sequence row.
    sm=N(seq_groups=[N(seq_ids=[sid],sample_indices=[0]),N(seq_ids=[other],sample_indices=[1])])
    return g,mi,sm


def test_default_off_no_torch_writer_or_env_side_effect(monkeypatch):
    monkeypatch.delenv('PDBLEND_DIAGNOSTIC_SPEC',raising=False)
    monkeypatch.setattr(d,'Writer',lambda *a:pytest.fail('writer started'))
    d._STATE=None;d._LOADED=False
    assert d.bind([],None,None,None) is None
    assert d.state() is None

@pytest.mark.parametrize('key,value',[('request_ids',IDS[:1]),('request_ids',[IDS[0],IDS[0]]),
    ('output_indices',[32]),('tp',1),('pp',2),('max_output_tokens',32),
    ('parent_image','wrong'),('output_dir','/tmp/x'),('performance_claims_allowed',True)])
def test_strict_scope(key,value):
    s=spec();s[key]=value
    with pytest.raises(ValueError):d.validate_spec(s)


def test_unknown_uuid_does_not_read_tensors(active):
    g,mi,sm=fixture_input('c'*32)
    assert d.bind([g],mi,sm,16) is None
    assert Tensor.reads==0 and not active.seen

@pytest.mark.parametrize('step',[1,27,35,64])
def test_only_seven_positions(active,step):
    g,mi,sm=fixture_input(step=step)
    assert d.bind([g],mi,sm,16) is None
    assert Tensor.reads==0

@pytest.mark.parametrize('first',[True,False])
def test_uuid_and_sampling_rows_are_independently_bound(active,first):
    g,mi,sm=fixture_input(other_first=first)
    b=d.bind([g],mi,sm,16)
    r=b['records'][0]
    assert r['seq_id']==17 and r['input_sequence_row']==int(first) and r['logits_row']==0
    assert r['computed_before']==222 and r['sequence_len']==223
    assert not active.error


def test_small_tensor_snapshot_pre_sampler_logits_and_parent_seq_output(active):
    g,mi,sm=fixture_input();mi.diagnostic_bindings=d.bind([g],mi,sm,16)
    contexts=d.begin(mi,0,2,1,True)
    assert contexts and Tensor.reads==0
    logits=Tensor(np.zeros((2,5000)));logits.a[0,2776]=4.2;logits.a[0,4172]=4.1
    logits.a[1,100]=99 # Other sample cannot be mistaken for the configured UUID.
    d.capture_logits(contexts,logits)
    assert Tensor.reads==0
    logits.a[:]=-999 # Real sampler may mutate logits; observation must own tiny copies.
    mi.input_positions.a[:]=999 # Metadata was snapshotted before forward.
    output=N(outputs=[N(samples=[N(parent_seq_id=5,output_token=100)]),
                      N(samples=[N(parent_seq_id=17,output_token=2776)])])
    d.finish(contexts,output)
    r=active.writer.rows[0]
    assert r['actual']['position']==[222]
    assert r['raw_model_logits_pre_sampler']['top2_token_ids']==[2776,4172]
    assert r['raw_model_logits_pre_sampler']['token_2776']==4.2
    assert r['sampler_output_token']==2776 and r['sampler_parent_seq_id']==17
    assert Tensor.reads==2 and len(active.seen)==1 and not active.error


def test_non_driver_uses_broadcast_uuid_binding_without_sampler(active):
    g,mi,sm=fixture_input();mi.diagnostic_bindings=d.bind([g],mi,sm,16)
    # JSON transport changes tuples into lists; explicit mapping still checked.
    mi.diagnostic_bindings=json.loads(json.dumps(mi.diagnostic_bindings))
    active.writer.rank=1
    c=d.begin(mi,1,2,1,True);assert c and Tensor.reads==0
    d.finish(c,None)
    r=active.writer.rows[0]
    assert r['rank']==1 and r['actual']['position']==[222]
    assert r['raw_model_logits_pre_sampler'] is None and Tensor.reads==1


def test_duplicate_marks_invalid_and_preserves_inference(active):
    g,mi,sm=fixture_input();mi.diagnostic_bindings=d.bind([g],mi,sm,16)
    assert d.begin(mi,0,2,1,True)
    assert d.begin(mi,0,2,1,True) is None
    assert active.error and active.writer.failed

@pytest.mark.parametrize('bad',['mapping','blocks','query','sampling'])
def test_unsafe_or_ambiguous_shapes_fail_observation_only(active,bad):
    g,mi,sm=fixture_input()
    if bad=='mapping':mi.request_ids_to_seq_ids[IDS[1]]=[17,5]
    if bad=='blocks':g.block_tables[17]=list(range(65))
    if bad=='query':mi.query_lens[1]=2
    if bad=='sampling':sm.seq_groups[0].sample_indices=[0,1]
    assert d.bind([g],mi,sm,16) is None
    assert active.error and Tensor.reads==0


def extract_method(cls,method):
    tree=ast.parse((ROOT/'model_runner.py').read_text())
    c=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
    fn=next(x for x in c.body if isinstance(x,ast.FunctionDef) and x.name==method)
    fn.decorator_list=[];fn.returns=None
    for a in fn.args.args:a.annotation=None
    ns={'_add_attn_metadata_broadcastable_dict':lambda *a:None,
        '_add_sampling_metadata_broadcastable_dict':lambda *a:None,
        '_init_sampling_metadata_from_tensor_dict':lambda x:x}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'actual-method','exec'),ns)
    return ns[method]

@pytest.mark.parametrize('cls',['ModelInputForGPU','ModelInputForGPUWithSamplingMetadata'])
def test_actual_broadcast_default_key_absent_and_enabled_roundtrip(cls):
    fn=extract_method(cls,'as_broadcastable_tensor_dict')
    keys=['input_tokens','inputs_embeds','input_positions','lora_requests','lora_mapping',
          'multi_modal_kwargs','prompt_adapter_mapping','prompt_adapter_requests','virtual_engine',
          'request_ids_to_seq_ids','finished_requests_ids','attn_metadata','sampling_metadata']
    obj=N(**{k:None for k in keys},diagnostic_bindings=None)
    off=fn(obj);assert 'diagnostic_bindings' not in off
    obj.diagnostic_bindings={'schema':1,'spec_sha256':'x','records':[{'request_id':IDS[0]}]}
    on=fn(obj);assert {k:v for k,v in on.items() if k!='diagnostic_bindings'}==off
    rebuild=extract_method(cls,'from_broadcasted_tensor_dict')
    restored=rebuild(N,on,None)
    assert restored.diagnostic_bindings==obj.diagnostic_bindings


def test_patch_changes_only_expected_actual_methods():
    old=build.PARENT.read_text();new=(ROOT/'model_runner.py').read_text()
    assert build.patch(old)==new
    def methods(s):
        t=ast.parse(s);return {(c.name,n.name):ast.dump(n,include_attributes=False)
            for c in t.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef)}
    a,b=methods(old),methods(new)
    changed={k for k in a if a[k]!=b[k]}
    assert changed=={('ModelInputForGPU','as_broadcastable_tensor_dict'),
        ('ModelInputForGPUWithSamplingMetadata','as_broadcastable_tensor_dict'),
        ('ModelRunner','prepare_model_input'),('ModelRunner','execute_model')}
    assert 'kv_caches' not in (ROOT/'pdblend_diagnostics.py').read_text()
    with pytest.raises(AssertionError):build.patch(old+'\n')


def test_writer_actual_async_limit_and_disk_status(tmp_path):
    w=d.Writer(tmp_path,0)
    try:
        for i in range(14):w.submit({'i':i})
        until=time.time()+2
        while w.written<14 and time.time()<until:time.sleep(.01)
        assert w.written==14
        w.submit({'overflow':True})
        until=time.time()+2
        p=tmp_path/('rank0-pid%d.status.json'%w.pid)
        while not json.loads(p.read_text())['failed'] and time.time()<until:time.sleep(.01)
        assert json.loads(p.read_text())['failed'] and w.q.maxsize==32
        assert len((tmp_path/('rank0-pid%d.jsonl'%w.pid)).read_text().splitlines())==14
    finally:w.close()


def test_actual_broadcast_helpers_preserve_explicit_binding(monkeypatch):
    # Exact actual-source helper bodies, no CUDA imports or replacement algorithms.
    path=ROOT/'source-evidence/worker/model_runner_base.py'
    tree=ast.parse(path.read_text())
    names={'_add_attn_metadata_broadcastable_dict','_init_attn_metadata_from_tensor_dict',
           '_add_sampling_metadata_broadcastable_dict','_init_sampling_metadata_from_tensor_dict'}
    funcs=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    for fn in funcs:
        fn.returns=None
        for arg in fn.args.args:arg.annotation=None
    ns={'dataclasses':dataclasses}
    monkeypatch.setitem(sys.modules,'vllm.model_executor',N(SamplingMetadata=lambda **kw:N(**kw)))
    exec(compile(ast.fix_missing_locations(ast.Module(body=funcs,type_ignores=[])),str(path),'exec'),ns)
    @dataclasses.dataclass
    class Metadata:
        slot_mapping:object
        def asdict_zerocopy(self):return {'slot_mapping':self.slot_mapping}
    backend=N(get_metadata_cls=lambda:Metadata,make_metadata=lambda **kw:Metadata(**kw))
    td={'diagnostic_bindings':{'request_id':IDS[0]},'request_ids_to_seq_ids':{IDS[0]:[12]}}
    ns['_add_attn_metadata_broadcastable_dict'](td,Metadata([123]))
    ns['_add_sampling_metadata_broadcastable_dict'](td,N(selected_token_indices=[0]))
    td=ns['_init_sampling_metadata_from_tensor_dict'](td)
    td=ns['_init_attn_metadata_from_tensor_dict'](backend,td)
    assert td['diagnostic_bindings']['request_id']==IDS[0]
    assert td['request_ids_to_seq_ids']=={IDS[0]:[12]}
    assert td['attn_metadata'].slot_mapping==[123]


def write_complete_fixture(tmp_path,monkeypatch):
    import verify_capture
    cfg=spec();raw=json.dumps(cfg).encode();sp=tmp_path/'spec.json';sp.write_bytes(raw)
    sha=hashlib.sha256(raw).hexdigest();root=tmp_path/'logs';root.mkdir()
    for rank in (0,1):
        s=d.State(cfg,sha);s.writer=Sink();s.writer.rank=rank;d._STATE=s;d._LOADED=True
        monkeypatch.setitem(sys.modules,'torch',torch_fake())
        for rid in IDS:
            for step in d.STEPS:
                g,mi,sm=fixture_input(rid,step)
                mi.diagnostic_bindings=d.bind([g],mi,sm,16)
                c=d.begin(mi,rank,2,1,True)
                if rank==0:
                    logits=Tensor(np.zeros((2,5000)));logits.a[0,2776]=4.2;logits.a[0,4172]=4.1
                    d.capture_logits(c,logits)
                    out=N(outputs=[N(samples=[N(parent_seq_id=17,output_token=2776)])])
                else:out=None
                d.finish(c,out)
        assert len(s.writer.rows)==14 and not s.error
        for r in s.writer.rows:r['pid']=1000+rank # Explicit synthetic per-rank CPU fixture.
        pid=s.writer.rows[0]['pid'];p=root/('rank%d-pid%d.jsonl'%(rank,pid))
        p.write_text(''.join(json.dumps(r)+'\n' for r in s.writer.rows))
        p.with_suffix('.status.json').write_text(json.dumps(dict(rank=rank,pid=pid,complete=True,
                                                               failed=False,error=None,written=14)))
    outputs=tmp_path/'complete-outputs.json'
    outputs.write_text(json.dumps({'token_ids_by_request_uuid':{rid:[2776]*64 for rid in IDS}}))
    d._STATE=None;d._LOADED=False
    return sp,root,outputs


def test_offline_full_capture_stays_observation_not_kv_proof(tmp_path,monkeypatch):
    import verify_capture
    args=write_complete_fixture(tmp_path,monkeypatch)
    report=verify_capture.verify(*args)
    assert report['capture_complete'] and len(report['records'])==14
    assert report['hardware_correctness_proven'] is False and report['performance_evidence'] is False
    assert all(r['rank_metadata_equal'] for r in report['records'])
    assert str(args[2]) in report['inputs']

@pytest.mark.parametrize('fault',['missing_rank','output_mismatch','missing_step','writer_failure'])
def test_offline_partial_or_false_token_evidence_rejected(tmp_path,monkeypatch,fault):
    import verify_capture
    sp,root,outputs=write_complete_fixture(tmp_path,monkeypatch)
    p=next(root.glob('rank1*.jsonl'))
    if fault=='missing_rank':p.unlink()
    elif fault=='output_mismatch':
        x=json.loads(outputs.read_text());x['token_ids_by_request_uuid'][IDS[1]][31]=4172
        outputs.write_text(json.dumps(x))
    elif fault=='missing_step':p.write_text('\n'.join(p.read_text().splitlines()[:-1])+'\n')
    else:
        s=p.with_suffix('.status.json');x=json.loads(s.read_text());x['failed']=True;s.write_text(json.dumps(x))
    with pytest.raises(ValueError):verify_capture.verify(sp,root,outputs)


def test_large_unselected_identifier_cannot_expand_observation_memory(active):
    g,mi,sm=fixture_input()
    mi.request_ids_to_seq_ids={'x'*10000:[5],IDS[1]:[17]}
    assert d.bind([g],mi,sm,16) is None
    assert active.error and Tensor.reads==0
