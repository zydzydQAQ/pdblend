"""Host CLI cancellation and best-effort measurement finalization; stdlib only.

The campaign retains its original deadline and hard-kill grace. A termination
signal cancels the main task once, allowing its finally blocks to run. Repeated
signals do not interrupt those blocks or turn an interrupted run into success.
"""
import asyncio
from contextvars import ContextVar
import json
from pathlib import Path
import signal
import time


_cleanup_state=ContextVar('cli_cleanup_state',default=None)


def cleanup_timeout(limit):
    state=_cleanup_state.get()
    if not state or state.get('deadline') is None:return limit
    return max(.001,min(limit,state['deadline']-time.monotonic()))


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,allow_nan=False))
    temp.replace(path)


def run_async_cli(awaitable, *, failure_path):
    interrupted={}
    async def run():
        loop=asyncio.get_running_loop();task=asyncio.current_task()
        state={'deadline':None};token=_cleanup_state.set(state)
        previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
        def stop(signum):
            if interrupted:
                interrupted['signals_received']+=1
                return
            interrupted.update(complete=False,passed=False,signal=signum,
                signal_name=signal.Signals(signum).name,signals_received=1,
                received_s=time.time(),main_task_exited=False,
                note='Termination requested; hardware cleanup is only established by the raw cleanup record.')
            # Leave margin inside Campaign's unchanged ten-second hard-kill grace.
            state['deadline']=time.monotonic()+8.
            try:write_json(failure_path,interrupted)
            finally:task.cancel()
        for signum in previous:loop.add_signal_handler(signum,stop,signum)
        try:
            return await awaitable
        finally:
            for signum,handler in previous.items():
                loop.remove_signal_handler(signum);signal.signal(signum,handler)
            if interrupted:
                interrupted.update(main_task_exited=True,finished_s=time.time())
                write_json(failure_path,interrupted)
            _cleanup_state.reset(token)
    try:
        result=asyncio.run(run())
    except BaseException:
        if interrupted:raise SystemExit(128+interrupted['signal']) from None
        raise
    if interrupted:raise SystemExit(128+interrupted['signal'])
    return result


def record_failure(raw, error):
    raw.update(complete=False,passed=False)
    raw.setdefault('errors',[]).append(type(error).__name__+': '+str(error))


async def finish_measurement(raw, out, sampler, clocks, *, derive=None):
    """Save failure evidence before cleanup; one cleanup error cannot skip another."""
    path=Path(out)/'raw.json';raw['cleanup_complete']=False
    errors=[]
    try:write_json(path,dict(raw,complete=False,passed=False))
    except BaseException as exc:errors.append('partial evidence: '+type(exc).__name__+': '+str(exc))
    async def stop_sampling():
        # Preserve the sample after the last observed interval boundary.
        await asyncio.sleep(.1)
        await asyncio.to_thread(sampler.stop)
    for name,action,timeout in (
            ('power sampler',stop_sampling,2.5),
            ('clock restoration',clocks.close,3.)):
        try:await asyncio.wait_for(action(),cleanup_timeout(timeout))
        except BaseException as exc:errors.append(name+': '+type(exc).__name__+': '+str(exc))
    raw.update(power_samples=sampler.samples,frequency_samples=sampler.frequency_samples,
        sampling_error=sampler.error)
    for field in ('power_source','power_metadata','utilization_samples'):
        if hasattr(sampler,field):raw[field]=getattr(sampler,field)
    if sampler.error:errors.append('power sampling: '+str(sampler.error))
    if derive:
        try:derive()
        except BaseException as exc:errors.append('derived evidence: '+type(exc).__name__+': '+str(exc))
    if errors:
        raw.setdefault('cleanup_errors',[]).extend(errors);raw.update(complete=False,passed=False)
    raw['cleanup_complete']=not errors
    write_json(path,raw)
    if errors:raise RuntimeError('measurement cleanup incomplete: '+'; '.join(errors))


async def cancel_tasks(tasks, timeout=.5):
    """Cancel local request tasks without an unbounded gather during shutdown."""
    tasks=tuple(tasks)
    for task in tasks:
        if not task.done():task.cancel()
    if not tasks:return
    done,left=await asyncio.wait(tasks,timeout=cleanup_timeout(timeout))
    await asyncio.gather(*done,return_exceptions=True)
    for task in left:
        task.cancel()
        task.add_done_callback(lambda t:None if t.cancelled() else t.exception())
    if left:raise TimeoutError(f'{len(left)} local request tasks did not cancel')
