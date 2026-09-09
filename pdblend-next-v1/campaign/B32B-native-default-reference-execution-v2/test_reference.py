import asyncio, ast, copy, importlib.util, json, os, sys, time, types
from pathlib import Path
from types import SimpleNamespace
import pytest
import adapter

ROOT=Path(__file__).resolve().parent
child=adapter.load('native_reference_child_test',ROOT/'child.py')
oldtests=adapter.load('native_reference_reused_fake_transport',adapter.PARENT/'test_child.py')
oldtests.child=child

class Session(oldtests.Session):
    def __init__(self,spec,*,delay=0,difference=False):super().__init__(spec);self.delay=delay;self.difference=difference
    async def handle(self,path,payload,rid):
        if path!='/native-reference':return await super().handle(path,payload,rid)
        assert payload['requests']==self.spec['requests'] and payload['work_deadline_s']>time.time()
        await asyncio.sleep(self.delay)
        outputs={r['request_uuid']:list(range(64)) for r in self.spec['requests']}
        if self.difference:outputs[self.spec['requests'][3]['request_uuid']][31]=999
        return dict(complete=True,token_ids_by_request_uuid=outputs)

def test_four_outputs_real_frozen_native_cleanup_and_failed_exact_retained(tmp_path):
    jp,job,spec=oldtests.job(tmp_path);session=Session(spec,difference=True)
    r=asyncio.run(child.execute(jp,session_factory=lambda:session))
    assert r['complete'] and r['completed_requests']==4 and r['cleanup_complete'] and r['child_exit_ok']
    assert not r['exact_passed'] and r['first_differences'][1]['position_one_based']==32
    assert [x['path'] for x in session.sent].count('/native-reference')==1
    assert not any(x['path']=='/v1/completions' for x in session.sent)
    proof=child.read(Path(job['output_dir'])/'checks/checks.json')['cleanup']['instances']['diag-only']['proof']
    assert len(proof['transfers'])==2 and session.accepting and session.mode=='continuous'
    assert len(child.read(Path(job['output_dir'])/'full-outputs.json')['token_ids_by_request_uuid'])==4

def test_reference_work_timeout_runs_native_cleanup_without_retry(tmp_path):
    jp,job,spec=oldtests.job(tmp_path,seconds=.025);session=Session(spec,delay=.2)
    r=asyncio.run(child.execute(jp,session_factory=lambda:session))
    assert not r['complete'] and r['cleanup_complete'] and not r['child_exit_ok']
    assert [x['path'] for x in session.sent].count('/native-reference')==1
    assert len([x for x in session.sent if x['path']=='/cancel'])==4

def test_reference_external_cancel_preserves_owned_cleanup(tmp_path):
    async def go():
        jp,job,spec=oldtests.job(tmp_path);session=Session(spec,delay=.2)
        task=asyncio.create_task(child.execute(jp,session_factory=lambda:session));await asyncio.sleep(.015);task.cancel();r=await task
        assert r['cancelled'] and r['cleanup_complete'] and not r['complete']
    asyncio.run(go())

def test_bad_cleanup_cap_rejected_before_session(tmp_path):
    jp,job,spec=oldtests.job(tmp_path);job['cleanup_deadline_s']=job['work_deadline_s']+91;child.write(jp,job)
    def forbidden():raise AssertionError('no HTTP before all job fields verified')
    r=asyncio.run(child.execute(jp,session_factory=forbidden));assert not r['complete'] and r['completed_requests']==0

def test_capture_requires_four_http_native_terminals_not_just_writer():
    term=dict(http_child_exited=True,diagnostic_container_stopped=True,diagnostic_gpu_workers_gone=True)
    adapter.capture_gate(dict(complete=True,cleanup_complete=True,completed_requests=4,exact_passed=False),term)
    with pytest.raises(RuntimeError):adapter.capture_gate(dict(complete=True,cleanup_complete=False,completed_requests=4),term)
    with pytest.raises(RuntimeError):adapter.capture_gate(dict(complete=True,cleanup_complete=True,completed_requests=4),dict(term,http_child_exited=False))

def test_parent_adapter_paths_and_original_restoration_functions_unmodified():
    old=sys.modules.get('common');c,base,loader,Op=adapter.load_operation()
    assert sys.modules.get('common') is old and c.ROOT==ROOT
    assert Op.restore_original is base.Operation.restore_original
    assert Op.stop_diagnostic is base.Operation.stop_diagnostic
    assert Op.native is base.Operation.native
    assert Op.run_child is base.Operation.run_child
    assert Op.run is base.Operation.run
    assert 'self.spec[\'diagnostic_instance\']' in str(base.Operation.run.__code__.co_consts) or 'kv_port' in str(base.Operation.run.__code__.co_consts)

def method(path,name):
    tree=ast.parse(path.read_text());owner=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='EngineService')
    fn=next(x for x in owner.body if isinstance(x,ast.FunctionDef) and x.name==name)
    ns=dict(os=os);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns);return ns[name]

def test_actual_initialize_kwargs_only_native_chunked_flag_changes(monkeypatch,tmp_path):
    captured=[]
    class Captured(Exception):pass
    def args(**kw):captured.append(kw);return kw
    def from_args(a):raise Captured()
    class KV:
        def __init__(self,**kw):self.__dict__.update(kw)
    for name,attrs in {'vllm.engine.arg_utils':dict(EngineArgs=args),
                       'vllm.engine.llm_engine':dict(LLMEngine=SimpleNamespace(from_engine_args=from_args)),
                       'vllm.config':dict(KVTransferConfig=KV)}.items():
        m=types.ModuleType(name);m.__dict__.update(attrs);monkeypatch.setitem(sys.modules,name,m)
    monkeypatch.setenv('PDBLEND_RUNTIME_PATH','not-used-by-official-scheduler')
    config=dict(id='nativefake',model='/models/Qwen2.5-32B-Instruct',tp=2,max_model_len=8192,max_num_seqs=32,
        max_num_batched_tokens=8192,peers={'nativefake':dict(host='127.0.0.1',tp=2,kv_port=34828)},kv_port=34828,port=34504)
    for path in [adapter.C/'B32B-baseline-fixed-window-preparation-v1/legacy-observation-candidate/engine.py',ROOT/'engine/engine.py']:
        owner=SimpleNamespace(runtime_path=tmp_path/'runtime',state={},config=config,write_state=lambda s:None)
        with pytest.raises(Captured):method(path,'initialize')(owner)
    a,b=captured;a['kv_transfer_config']=a['kv_transfer_config'].__dict__;b['kv_transfer_config']=b['kv_transfer_config'].__dict__
    assert a==b
    assert a['enable_chunked_prefill'] is True and b['enable_chunked_prefill'] is True

def test_official_scheduler_has_no_runtime_control_or_telemetry_symbols():
    text=(ROOT/'image-context/scheduler.py').read_text()
    assert adapter.sha(ROOT/'image-context/scheduler.py')==adapter.SCHEDULER_SHA
    assert not any(x in text for x in ['pdblend','apply_scheduler_control','emit_scheduler_snapshot','PDBLEND'])

def test_actual_service_endpoint_calls_actual_driver_once_full_work(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(ROOT/'engine'))
    monkeypatch.syspath_prepend(str(adapter.C/'B32B-native-default-reference-plan-v1'))
    fake=adapter.load('native_reference_driver_fake',adapter.C/'B32B-native-default-reference-plan-v1/test_native_driver.py')
    vllm=types.ModuleType('vllm');vllm.SamplingParams=lambda **kw:kw
    monkeypatch.setitem(sys.modules,'vllm',vllm)
    service_module=adapter.load('native_reference_actual_service',ROOT/'engine/engine.py')
    spec=adapter.read(ROOT/'request-spec.json');sp=tmp_path/'spec.json';sp.write_text(json.dumps(spec))
    monkeypatch.setenv('PDBLEND_DIAGNOSTIC_SPEC',str(sp));monkeypatch.setenv('PDBLEND_DIAGNOSTIC_SPEC_SHA256',adapter.sha(sp))
    svc=service_module.EngineService(dict(id='cpu-ref',runtime_dir=str(tmp_path),role='mixed'))
    from test_selection import engine
    import schedule_selection
    monkeypatch.setattr(schedule_selection,'verify_default',lambda s:{})
    svc.engine=engine();svc.update_snapshot=lambda:None
    svc.control_rpc=lambda fn:[dict(rank=r,runner_chunked=True,builder_chunked=True,builder_config_chunked=True,runner_builder_config_same=True,tokens=8192,seqs=32) for r in (0,1)]
    payload=dict(requests=spec['requests'],work_deadline_s=time.time()+5)
    async def request_json():return copy.deepcopy(payload)
    async def go():
        reply=await svc.native_reference(SimpleNamespace(json=request_json));out=json.loads(reply.text)
        assert out['complete'] and len(out['token_ids_by_request_uuid'])==4 and svc.engine.calls==197
        with pytest.raises(service_module.web.HTTPConflict):await svc.native_reference(SimpleNamespace(json=request_json))
    try:asyncio.run(go())
    finally:svc.worker.shutdown(wait=True)
    events=[json.loads(x) for x in (tmp_path/'native-reference.events.jsonl').read_text().splitlines()]
    assert len([x for x in events if x['kind']=='executed_step'])==197

def test_actual_driver_wrong_executed_shape_is_not_tolerated(monkeypatch):
    monkeypatch.syspath_prepend(str(adapter.C/'B32B-native-default-reference-plan-v1'))
    fake=adapter.load('native_reference_wrong_shape_fake',adapter.C/'B32B-native-default-reference-plan-v1/test_native_driver.py')
    driver=adapter.load('native_reference_actual_driver_negative',ROOT/'engine/native_driver.py')
    from test_selection import engine
    import schedule_selection
    monkeypatch.setattr(schedule_selection,'verify_default',lambda s:{})
    e=engine();original=e.s.schedule
    def wrong():
        m,d,a=original();d.num_batched_tokens+=1;return m,d,a
    e.s.schedule=wrong
    with pytest.raises(RuntimeError,match='trajectory differs'):
        driver.run_reference(e,adapter.read(ROOT/'request-spec.json')['requests'],lambda **k:k,lambda v:None,lambda:None)
    assert e.s.schedule==wrong and e.calls==1
