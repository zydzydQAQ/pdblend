"""Actual frozen ClockOwner with CPU hardware stand-ins and isolated lock files."""
import asyncio
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import adapter
import ablation_queue
import clock_commands


class Hardware:
    def __init__(self):
        self.actions=[]
    def set_clock(self,gpu,frequency):
        self.actions.append(('set',gpu,frequency))
    def reset_clock(self,gpu):
        self.actions.append(('reset',gpu))
    def current_freq(self,gpu):return 2520
    def clock_idle(self,gpu):return False


def run():
    from ecopadg.serving import backend,runtime
    assert backend.ClockOwner is runtime.ClockOwner
    original=backend.ClockOwner
    count=0
    with tempfile.TemporaryDirectory(prefix='dvfs-command-cpu-') as name:
        root=Path(name)
        for frequency in (2520,1500):
            cid=str(frequency);operation=root/'operations'/cid;operation.mkdir(parents=True)
            before=time.time();hardware=Hardware();markers=[]
            class MarkerOwner(original):
                def __init__(self,*args,**kwargs):
                    super().__init__(*args,**kwargs)
                    markers.append(tuple(self.gpus))
            # Mirrors C's existing ownership subclass without replacing its
            # behavior: the observed subclass must call through it unchanged.
            with patch.object(backend,'ClockOwner',MarkerOwner),patch.object(runtime,'ClockOwner',MarkerOwner):
                with clock_commands.capture_clock_commands(operation):
                    assert backend.ClockOwner is runtime.ClockOwner and issubclass(backend.ClockOwner,MarkerOwner)
                    async def work():
                        owner=runtime.ClockOwner(hardware,range(8),lock_dir=str(root/'locks'))
                        await owner.set((i for i in range(8)),frequency,verify_rise=False)
                        await owner.park((i for i in (6,7)))
                        await owner.verify_deferred()
                        await owner.close()
                    asyncio.run(work())
                assert backend.ClockOwner is MarkerOwner and runtime.ClockOwner is MarkerOwner
            after=time.time()
            assert markers==[tuple(range(8))]
            assert hardware.actions==[('set',i,frequency) for i in range(8)]+[('reset',6),('reset',7)]+[('reset',i) for i in range(8)]
            receipt=dict(child_pid=os.getpid(),operation_start_s=before,operation_end_s=after)
            adapter.immutable(root/'receipts'/(cid+'.json'),receipt)
            power=operation/'power';power.mkdir()
            with (power/'clocks.csv').open('w') as f:
                writer=csv.writer(f);writer.writerow(['t_s']+[f'gpu{i}_sm_mhz' for i in range(8)])
                writer.writerow([before-.1]+[2520]*8);writer.writerow([after+.1]+[2520]*8)
            b=SimpleNamespace(ROOT=root)
            if frequency==2520:
                proof=ablation_queue.clock_proof(b,cid)
                assert proof['service_maximum_commands_verified'] and not proof['actual_physical_fixed_2520_certified']
                assert proof['hardware_set_writes']==8 and proof['hardware_reset_writes']==10
                # Reject a partial journal even if the actual samples are all max.
                status=adapter.read(operation/'clock-commands-status.json');status['records']+=1
                (operation/'clock-commands-status.json').write_text(json.dumps(status))
                try:ablation_queue.clock_proof(b,cid)
                except RuntimeError:pass
                else:raise AssertionError('truncated command journal accepted')
                count+=1
            else:
                try:ablation_queue.clock_proof(b,cid)
                except RuntimeError as exc:assert 'nonmaximum' in str(exc)
                else:raise AssertionError('nonmaximum real command accepted')
            count+=2

        operation=root/'recording-error';operation.mkdir();hardware=Hardware()
        with clock_commands.capture_clock_commands(operation):
            async def recording_failure():
                owner=runtime.ClockOwner(hardware,(0,),lock_dir=str(root/'locks'))
                def fail(*a,**k):raise OSError('CPU injected journal failure')
                with patch.object(clock_commands.json,'dumps',fail):
                    await owner.set((0,),2520,verify_rise=False)
                    await owner.close()
            asyncio.run(recording_failure())
        assert hardware.actions==[('set',0,2520),('reset',0)]
        assert adapter.read(operation/'clock-commands-status.json')['complete'] is False
        assert backend.ClockOwner is original and runtime.ClockOwner is original
        count+=1

        operation=root/'hardware-error';operation.mkdir();hardware=Hardware()
        def hardware_fail(*args):raise RuntimeError('CPU injected driver failure')
        hardware.set_clock=hardware_fail
        with clock_commands.capture_clock_commands(operation):
            async def failing_write():
                owner=runtime.ClockOwner(hardware,(0,),lock_dir=str(root/'locks'))
                try:await owner.set((0,),2520,verify_rise=False)
                finally:await owner.close()
            try:asyncio.run(failing_write())
            except RuntimeError as exc:assert 'driver failure' in str(exc)
            else:raise AssertionError('original hardware error was swallowed')
        assert hardware.actions==[('reset',0)]
        events=[json.loads(line) for line in (operation/'clock-commands.jsonl').read_text().splitlines()]
        assert any(r['kind']=='hardware_end' and r['complete'] is False for r in events)
        assert any(r['kind']=='owner_close_end' and r['complete'] is True for r in events)
        count+=1
    return dict(cpu_command_cases_passed=count,gpu_executed=False,real_clockowner_source=backend.__file__)


if __name__=='__main__':print(json.dumps(run()))
