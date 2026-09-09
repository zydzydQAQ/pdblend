import asyncio
import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
from unittest.mock import patch
import pytest

ROOT=Path(__file__).resolve().parent
HOST=ROOT/'hosts/14b-fixed-p4'
sys.path[:0]=[str(HOST/'src'),'/root/workspace/pdblend/.runtime-deps']
SPEC=importlib.util.spec_from_file_location('until_complete_executor',ROOT/'common/execution-until-complete-v1/run.py')
mod=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(mod)

def binding():
    return dict(protocol_id=mod.PROTOCOL,hostname=socket.gethostname(),deadline_s=None,
        campaign_lifecycle='until_declared_complete_v1',files={},large_inputs={},
        instances=[],system='pdblend',configs={})

def test_explicit_user_cancelled_deadline_survives_old_cutoff():
    with patch.object(mod.time,'time',return_value=1788872770+86400):
        mod.validate_binding(binding())

@pytest.mark.parametrize('change',[dict(deadline_s=1788872770.0400891),dict(campaign_lifecycle=None)])
def test_old_or_implicit_binding_cannot_enter_new_executor(change):
    b=binding();b.update(change)
    with pytest.raises(RuntimeError,match='explicit until-complete'):
        mod.validate_binding(b)

def test_real_job_keeps_finite_full_cell_budget_after_old_cutoff(tmp_path):
    class StopBeforeHardware(Exception):pass
    async def identity(*args):return dict(cpu_mock=True)
    trace=tmp_path/'trace.json';trace.write_text('{}')
    row=dict(trace=str(trace),trace_sha256=mod.sha(trace),dataset='alpaca',cell_id='cpu-cell',system='pdblend',n_requests=1)
    b=binding();b.update(configs=dict(alpaca=str(tmp_path/'config.json')),instances=[dict(port=9999)])
    def stop(*args,**kwargs):raise StopBeforeHardware()
    import ecopadg.measure.power as power
    now=1788872770+86400
    with patch.object(mod.time,'time',return_value=now),patch.object(mod,'identity',identity),patch.object(power,'PowerSampler',stop):
        with pytest.raises(StopBeforeHardware):asyncio.run(mod.run_one(None,b,row,tmp_path/'out',None))
    job=json.loads((tmp_path/'out/operations/cpu-cell/job.json').read_text())
    assert job['latest_arrival_epoch_s']==now+90
    assert job['execution_deadline_s']==now+90+100+120

def test_child_and_all_non_deadline_controller_interfaces_are_unchanged():
    spec=importlib.util.spec_from_file_location('until_builder',ROOT/'build_execution_until_complete.py')
    build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
    assert (build.OUT/'child.py').read_bytes()==(build.PARENT/'child.py').read_bytes()
    assert (build.OUT/'run.py').read_text()==build.transform((build.PARENT/'run.py').read_text())
    source=(build.OUT/'run.py').read_text()
    assert "cleanup_end = min(time.time() + 90, job['execution_deadline_s'] + 90)" in source
