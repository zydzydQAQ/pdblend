import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'engine'))
import schedule_selection as selection
import native_driver as driver
PLAN=ROOT.parent/'B32B-native-default-reference-plan-v1'
sys.path.insert(0,str(PLAN))
spec=importlib.util.spec_from_file_location('sameflag_fake',PLAN/'test_native_driver.py')
fake=importlib.util.module_from_spec(spec);spec.loader.exec_module(fake)
REQUESTS=json.loads((ROOT/'request-spec.json').read_text())['requests']


def engine():
    e=fake.FakeEngine();e.scheduler_config=e.s.scheduler_config
    e.scheduler_config.chunked_prefill_enabled=True
    # Actual policy composition is the frozen native _schedule_default AST;
    # this fake model supplies synthetic outputs, never experimental evidence.
    def schedule():
        d=e.s._schedule()
        return [SimpleNamespace(request_id=x.seq_group.request_id) for x in d.scheduled_seq_groups],d,False
    e.s.schedule=schedule
    return e


def test_inspect_method_source_hash_exact_official_bytes():
    src=(ROOT/'image-context/scheduler.py').read_text();tree=ast.parse(src)
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_schedule_default')
    raw=''.join(src.splitlines(True)[fn.lineno-1:fn.end_lineno])
    assert hashlib.sha256(raw.encode()).hexdigest()==selection.DEFAULT_SHA
    assert hashlib.sha256(ast.get_source_segment(src,fn).encode()).hexdigest()=='1e041843b18a7f109299fffce917384bca53a603572f1c4f3670dd9ed4e42813'
    assert hashlib.sha256(src.encode()).hexdigest()==selection.SCHEDULER_SHA


@pytest.mark.parametrize('existing',[False,True])
def test_true_configs_actual_default_197_and_exact_instance_restore(monkeypatch,existing):
    e=engine();cfg=e.scheduler_config;before=dict(vars(cfg));sentinel=lambda:None
    if existing:e.s._schedule=sentinel
    monkeypatch.setattr(selection,'verify_default',lambda s:dict(cpu_fake_source_already_checked=True))
    events=[];old_schedule=e.s.schedule
    r=driver.run_reference(e,REQUESTS,lambda **k:k,events.append,lambda:None)
    assert r['complete'] and e.calls==197 and len([x for x in events if x['kind']=='output'])==256
    assert dict(vars(cfg))==before and e.s.scheduler_config is cfg and e.scheduler_config is cfg
    assert e.s.schedule is old_schedule
    assert ('_schedule' in vars(e.s)) is existing
    if existing:assert e.s._schedule is sentinel
    assert events[-1]['kind']=='default_selection_restored'


def test_partial_install_failure_restores_no_config_or_dispatch_change(monkeypatch):
    e=engine();cfg=e.scheduler_config;old_schedule=e.s.schedule
    monkeypatch.setattr(selection,'verify_default',lambda s:{})
    def emit(x):
        if x['kind']=='default_selection':raise RuntimeError('writer broken')
    with pytest.raises(RuntimeError,match='writer broken'):
        driver.run_reference(e,REQUESTS,lambda **k:k,emit,lambda:None)
    assert '_schedule' not in vars(e.s) and e.s.schedule is old_schedule and cfg.chunked_prefill_enabled is True and e.calls==0


def test_step_failure_restores_both_instance_attributes(monkeypatch):
    e=engine();e.fail_at=3;prior=e.s.schedule
    monkeypatch.setattr(selection,'verify_default',lambda s:{})
    with pytest.raises(RuntimeError,match='injected'):
        driver.run_reference(e,REQUESTS,lambda **k:k,lambda x:None,lambda:None)
    assert e.s.schedule is prior and '_schedule' not in vars(e.s) and e.scheduler_config.chunked_prefill_enabled is True


def test_false_config_rejected_before_step(monkeypatch):
    e=engine();e.scheduler_config.chunked_prefill_enabled=False
    monkeypatch.setattr(selection,'verify_default',lambda s:{})
    with pytest.raises(RuntimeError,match='original chunked'):
        driver.run_reference(e,REQUESTS,lambda **k:k,lambda x:None,lambda:None)
    assert e.calls==0
