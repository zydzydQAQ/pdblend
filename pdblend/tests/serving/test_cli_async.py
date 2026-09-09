import asyncio
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


MODULE=Path(__file__).resolve().parents[2]/'src/ecopadg/serving/cli_async.py'
spec=importlib.util.spec_from_file_location('isolated_cli_async',MODULE)
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


def wait_file(path,process):
    deadline=time.monotonic()+5
    while not path.exists():
        if process.poll() is not None or time.monotonic()>deadline:
            raise AssertionError('CPU child did not reach its synchronization point')
        time.sleep(.01)


@pytest.mark.parametrize('signum,repeat,suppress',[
    (signal.SIGTERM,False,False),(signal.SIGTERM,True,False),
    (signal.SIGTERM,False,True),(signal.SIGINT,False,False),
])
def test_real_child_signal_preserves_finally_and_cannot_exit_success(tmp_path,signum,repeat,suppress):
    script=tmp_path/'child.py'
    script.write_text('''import asyncio, importlib.util, json, sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('isolated',sys.argv[1])
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
root=Path(sys.argv[2]);suppress=sys.argv[3]=='true'
assert 'pynvml' not in sys.modules and 'torch' not in sys.modules
class Sampler:
    samples=[];frequency_samples=[];error=None
    def stop(self):(root/'sampler.stopped').write_text('yes')
class Clocks:
    async def close(self):
        (root/'cleanup.started').write_text('yes')
        await asyncio.sleep(.25)
        (root/'clocks.restored').write_text('yes')
async def main():
    raw={'complete':False,'passed':False}
    try:
        (root/'ready').write_text('yes')
        await asyncio.Event().wait()
    except asyncio.CancelledError as exc:
        cli.record_failure(raw,exc)
        if not suppress:raise
        return 'swallowed cancellation'
    finally:
        await cli.finish_measurement(raw,root,Sampler(),Clocks())
cli.run_async_cli(main(),failure_path=root/'interrupted.json')
''')
    process=subprocess.Popen([sys.executable,str(script),str(MODULE),str(tmp_path),str(suppress).lower()],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        wait_file(tmp_path/'ready',process)
        # Only the child created in this test is ever signalled.
        process.send_signal(signum)
        if repeat:
            wait_file(tmp_path/'cleanup.started',process);process.send_signal(signum)
        stdout,stderr=process.communicate(timeout=5)
        assert process.returncode==128+signum,(stdout,stderr)
        assert (tmp_path/'sampler.stopped').exists() and (tmp_path/'clocks.restored').exists()
        interrupted=json.loads((tmp_path/'interrupted.json').read_text())
        assert interrupted['main_task_exited'] and not interrupted['passed']
        assert interrupted['signals_received']==(2 if repeat else 1)
        raw=json.loads((tmp_path/'raw.json').read_text())
        assert not raw['complete'] and not raw['passed'] and raw['cleanup_complete']
        assert any('CancelledError' in e for e in raw['errors'])
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=5)


def test_normal_return_and_original_signal_handler_are_preserved(tmp_path):
    before=signal.getsignal(signal.SIGTERM)
    async def value():return 7
    assert cli.run_async_cli(value(),failure_path=tmp_path/'unused.json')==7
    assert signal.getsignal(signal.SIGTERM)==before
    assert not (tmp_path/'unused.json').exists()


@pytest.mark.parametrize('failing',['sampler','clock','derive'])
def test_cleanup_failure_keeps_raw_and_attempts_both_devices(tmp_path,failing):
    events=[]
    class Sampler:
        samples=[];frequency_samples=[];error=None
        def stop(self):
            events.append('sampler')
            if failing=='sampler':raise RuntimeError('sampling thread failure')
    class Clocks:
        async def close(self):
            partial=json.loads((tmp_path/'raw.json').read_text())
            assert not partial['complete'] and not partial['passed']
            events.append('clock')
            if failing=='clock':raise RuntimeError('GPU reset failure')
    def derive():
        if failing=='derive':raise ValueError('missing power window')
    with pytest.raises(RuntimeError,match='cleanup incomplete'):
        asyncio.run(cli.finish_measurement(dict(complete=True,passed=True),tmp_path,Sampler(),Clocks(),derive=derive))
    assert events==['sampler','clock']
    raw=json.loads((tmp_path/'raw.json').read_text())
    assert not raw['complete'] and not raw['passed'] and raw['cleanup_errors']


def test_cleanup_deadline_is_shared_across_nested_helpers():
    state={'deadline':time.monotonic()+.03};token=cli._cleanup_state.set(state)
    try:
        assert 0<cli.cleanup_timeout(120)<=.03
        state['deadline']=time.monotonic()-1
        assert cli.cleanup_timeout(120)==.001
    finally:cli._cleanup_state.reset(token)


def test_initial_raw_write_error_does_not_skip_sampler_or_clock_cleanup(tmp_path,monkeypatch):
    writes=[];events=[];original=cli.write_json
    def write(path,value):
        writes.append(path)
        if len(writes)==1:raise OSError('temporary evidence write failure')
        original(path,value)
    monkeypatch.setattr(cli,'write_json',write)
    sampler=SimpleNamespace(samples=[],frequency_samples=[],error=None,stop=lambda:events.append('sampler'))
    async def close():events.append('clock')
    with pytest.raises(RuntimeError,match='partial evidence'):
        asyncio.run(cli.finish_measurement(dict(complete=True),tmp_path,sampler,SimpleNamespace(close=close)))
    assert events==['sampler','clock'] and len(writes)==2
    assert not json.loads((tmp_path/'raw.json').read_text())['complete']


@pytest.mark.parametrize('name',['profiling','interference','prefill_batch','clock_profiling','residency','runtime_validation'])
def test_hardware_cli_uses_common_signal_runner_without_importing_gpu_modules(name):
    import ast
    tree=ast.parse(MODULE.with_name(name+'.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
        and n.func.id=='run_async_cli']
    assert len(calls)==1 and any(k.arg=='failure_path' for k in calls[0].keywords)
