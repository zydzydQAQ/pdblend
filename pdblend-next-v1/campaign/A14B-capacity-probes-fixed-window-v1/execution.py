"""Bounded host-child execution; original engines and baseline artifacts stay resident."""
import asyncio
import csv
import json
from pathlib import Path
import time


def dispatch_ids(path,ports):
    owned=set()
    if not path.exists():return owned
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b'\n'):continue  # Partial dispatch write did not reach its following network await.
        row=json.loads(line);rid=row['request_id'];port=row['port']
        if port not in ports or not isinstance(rid,str) or len(rid)!=32 or any(c not in '0123456789abcdef' for c in rid):
            raise ValueError('foreign dispatch record; no cancellation authorized')
        owned.add((ports.index(port),rid))
    return owned


async def stop_child(child):
    if child is None or child.returncode is not None:return
    child.terminate()
    try:await asyncio.wait_for(child.wait(),3)
    except asyncio.TimeoutError:child.kill();await asyncio.wait_for(child.wait(),2)


async def wait_idle(b,session,index,*,tokens=None,generation=None,seconds=12,accepting=True):
    until=time.monotonic()+seconds
    while True:
        raw=await b.http(session,index,'/runtime',timeout=min(3,max(.01,until-time.monotonic())))
        b.require(raw.get('id')==b.IDS[index] and not raw.get('error') and not raw.get('runtime_error'),'bad/foreign owner')
        try:
            b.native_idle(raw,b.IDS[index],accepting=accepting,tokens=tokens)
            b.require(generation is None or raw['generation']==generation,'awaiting actual next ACK')
            return raw
        except RuntimeError:
            b.require(time.monotonic()<until,'owner did not become idle/ACKed');await asyncio.sleep(.02)


async def set_one(b,session,index,tokens):
    before=await wait_idle(b,session,index)
    target=before['generation']+1
    payload=dict(generation=target,role='mixed',mode='continuous',admit_prefill=True,admit_decode=True,
        scheduler_budget=dict(schema_version=1,max_num_batched_tokens=tokens,max_num_seqs=32))
    response=await b.http(session,index,'/control',payload)
    after=await wait_idle(b,session,index,tokens=tokens,generation=target)
    b.require(response.get('generation')==target,'control declaration differs')
    return dict(before=before,payload=payload,response=response,after=after)


async def restore_one(b,session,index):
    result=dict(complete=False);errors=[]
    try:
        async def proof():
            before=await wait_idle(b,session,index,accepting=None,seconds=10)
            barrier=await b.http(session,index,'/drain',dict(expected_generation=before['generation']),timeout=8)
            result.update(before=before,barrier=barrier);b.native_barrier(before,barrier)
        await asyncio.wait_for(proof(),20)
    except BaseException as exc:errors.append('native proof: '+repr(exc))
    finally:
        try:
            async def resume():
                current=await b.http(session,index,'/runtime',timeout=3)
                b.require(current.get('id')==b.IDS[index],'refuse control for foreign instance')
                target=current['generation']+1
                payload=dict(generation=target,role='mixed',mode='continuous',admit_prefill=True,admit_decode=True,
                    scheduler_budget=dict(schema_version=1,max_num_batched_tokens=8192,max_num_seqs=32))
                result['response']=await b.http(session,index,'/control',payload,timeout=8)
                after=await wait_idle(b,session,index,tokens=8192,generation=target,seconds=8)
                b.require(after['role']=='mixed' and after['mode']=='continuous' and after['admit_prefill'] and after['admit_decode'],'wrong resumed role')
                result['after']=after
            await asyncio.wait_for(resume(),20)
        except BaseException as exc:errors.append('resume: '+repr(exc))
    result.update(complete=not errors,errors=errors);return result


class Cell:
    def __init__(self,b,session,row,freeze,status,status_name,hardware,*,limits):
        self.b,self.session,self.row,self.freeze=b,session,row,freeze
        self.status,self.status_name,self.hardware=status,status_name,hardware
        self.limits=limits
        self.operation=b.ROOT/'operations'/row['cell_id'];self.operation.mkdir(parents=True,exist_ok=False)
        b.write('operations/'+row['cell_id']+'/deadline-limits.json',limits)
        self.out=b.ROOT/'cells'/row['cell_id'];self.child=None;self.child_log=None;self.verified=False;self.mutated=False
        self.offsets={};self.positions={};self.tails={};self.event_counts={n:0 for n in b.IDS}
        self.receipt=dict(cell_id=row['cell_id'],sequence=row['sequence'],trace_sha256=row['trace_sha256'],
            config_sha256=freeze['policy_config_sha256'],slo_protocol=row['slo_protocol'],
            slo_ttft_s=row['slo_ttft_s'],slo_tpot_s=row['slo_tpot_s'],started_s=time.time(),screen_valid=False)
        status['cells'].append(self.receipt);self.save()

    def save(self):
        self.b.write('receipts/'+self.row['cell_id']+'.json',self.receipt)
        self.b.write(self.status_name,self.status)

    def poll_events(self):
        for name in self.b.IDS:
            path=self.b.EVENT_ROOT/(name+'.control.events.jsonl')
            with path.open('rb') as f:f.seek(self.positions[name]);data=f.read();self.positions[name]+=len(data)
            parts=(self.tails[name]+data).split(b'\n');self.tails[name]=parts.pop()
            for line in parts:
                if not line:continue
                event=json.loads(line)
                if not event.get('tokens',0):continue
                self.event_counts[name]+=1
                self.b.require(event.get('mode')=='continuous' and event.get('role')=='mixed'
                    and type(event.get('tokens')) is int and 0<event['tokens']<=2048,'wrong owner mode/budget execution')

    async def cleanup(self):
        b=self.b;errors=[];native={};closed_clock=False;child_stopped=False
        try:
            try:await stop_child(self.child);child_stopped=True
            except BaseException as exc:errors.append('own child stop: '+repr(exc))
            if self.verified and self.mutated and child_stopped:
                try:
                    owned=dispatch_ids(self.operation/'dispatch.jsonl',b.PORTS)
                    self.receipt['owned_dispatched_requests']=len(owned)
                    async def cancel(index,rid):
                        try:await b.http(self.session,index,'/cancel',dict(request_id=rid),timeout=5)
                        except BaseException as exc:errors.append('owned cancel: '+repr(exc))
                    await asyncio.wait_for(asyncio.gather(*(cancel(i,rid) for i,rid in owned)),8)
                except BaseException as exc:errors.append('dispatch/cancel: '+repr(exc))
                # Neither a child exception nor an ownership-log error skips either native restoration.
                outcomes=await asyncio.gather(*(restore_one(b,self.session,i) for i in range(2)),return_exceptions=True)
                for name,outcome in zip(b.IDS,outcomes):
                    native[name]=dict(complete=False,error=repr(outcome)) if isinstance(outcome,BaseException) else outcome
                if not all(v['complete'] for v in native.values()):errors.append('one or more native restorations failed')
        except BaseException as exc:errors.append('outer cleanup: '+repr(exc))
        finally:
            if self.verified and self.mutated and child_stopped:
                try:
                    from ecopadg.serving.backend import ClockOwner
                    # All eight locks belonged to the stopped cell host. Reacquire/reset
                    # only after that host has exited; never reset a live owner lock.
                    clocks=await asyncio.to_thread(ClockOwner,self.hardware,tuple(range(8)))
                    await asyncio.wait_for(clocks.close(),10);closed_clock=True
                except BaseException as exc:errors.append('clock release: '+repr(exc))
            elif not self.mutated:closed_clock=True
            else:errors.append('clock reset skipped because own child exit was not verified')
        result=dict(complete=not errors,errors=errors,native=native,own_child_stopped=child_stopped,
            clock_release_complete=closed_clock,mutations_started=self.mutated,finished_s=time.time())
        self.b.write('operations/'+self.row['cell_id']+'/outer-cleanup.json',result)
        return result

    async def execute(self):
        from ecopadg.measure.power import PowerSampler,trapezoid_energy
        from ecopadg.serving.measurement import power_evidence,save_raw
        from ecopadg.metrics import clip_power_window
        b=self.b;failure=None;summary=None;start=None;cleanup={}
        sampler=PowerSampler(range(8),interval=.02,backend=self.hardware,sample_clocks=True)
        # Read-only identity completes before any cleanup permission or budget control.
        try:
            before=await b.live(self.session,expected_inventory=self.freeze['inventory'])
            b.require(time.time()<self.limits['latest_arrival_epoch_s'],'preflight consumed reserved startup time')
            b.write('operations/'+self.row['cell_id']+'/identity.before.json',before);self.verified=True
            for name in b.IDS:
                path=b.EVENT_ROOT/(name+'.control.events.jsonl');self.offsets[name]=path.stat().st_size
        except BaseException as exc:
            self.receipt.update(error=repr(exc),phase='preflight_failed',finished_s=time.time());self.save();raise
        self.positions=dict(self.offsets);self.tails={name:b'' for name in b.IDS}
        sampler.start()
        try:
            until=time.monotonic()+5
            while True:
                power=list(sampler.samples);meta=list(sampler.power_metadata)[:len(power)]
                b.require(not sampler.error and time.monotonic()<until,'instant outer power not ready')
                if len(power)>=2 and power_evidence(power,sampler.power_source,meta)['power_source_verified']:break
                await asyncio.sleep(.01)
            start=self.receipt['operation_start_s']=time.time();self.mutated=True
            setup=await asyncio.gather(*(set_one(b,self.session,i,2048) for i in range(2)),return_exceptions=True)
            b.write('operations/'+self.row['cell_id']+'/budget.before.json',[
                dict(error=repr(v)) if isinstance(v,BaseException) else v for v in setup])
            b.require(not any(isinstance(v,BaseException) for v in setup),'one or both budget setup controls failed')
            self.child_log=(self.operation/'child.log').open('xb')
            b.require(time.time()<self.limits['latest_arrival_epoch_s'],'budget setup consumed reserved startup time')
            self.child=await asyncio.create_subprocess_exec(sys_executable(),'-u',str(b.ROOT/'child.py'),self.row['cell_id'],
                cwd=str(b.ROOT),stdout=self.child_log,stderr=asyncio.subprocess.STDOUT,start_new_session=True)
            self.receipt['child_pid']=self.child.pid;self.save()
            until=time.monotonic()+max(0,self.limits['cell_execution_deadline_s']-time.time())
            waiter=asyncio.create_task(self.child.wait())
            try:
                while not waiter.done():
                    await asyncio.wait({waiter},timeout=.25);self.poll_events()
                    b.require(not sampler.error,'outer power sampler failed: '+str(sampler.error))
                    b.require(time.monotonic()<until,'host exceeded trace plus bounded drain window')
                self.receipt['child_exitcode']=waiter.result();self.poll_events()
            finally:
                if not waiter.done():waiter.cancel()
                await asyncio.gather(waiter,return_exceptions=True)
            b.require((self.out/'summary.json').is_file(),'inner summary missing; failed attempt retained')
            summary=b.read(self.out/'summary.json');self.receipt['summary']=summary
            actual_config=b.read(self.out/'runtime_config.json')
            expected_config=b.read(b.CONFIG);expected_config['journal']=str(self.out/'control.jsonl')
            expected_config.update({k:self.row[k] for k in ('slo_ttft_s','slo_tpot_s')})
            expected_config.update(slo_scale=self.row['slo_scale'],slo_protocol='per-dataset-slo-v1')
            b.require(actual_config==expected_config and summary.get('slo_protocol')=='per-dataset-slo-v1'
                and summary.get('declared_dataset_slo')=={k:self.row[k] for k in ('slo_ttft_s','slo_tpot_s')},
                'actual controller/scoring SLO or strategy differs from frozen user contract')
            b.require(summary.get('fixed_window_valid') is True
                and summary.get('fixed_window',{}).get('arrival_epoch_s',float('inf'))<=self.limits['latest_arrival_epoch_s']
                and summary.get('measurement_end_s',float('inf'))<=self.limits['cell_execution_deadline_s'],
                'fixed arrival window or reserved measurement deadline invalid')
            self.receipt['effective_config_sha256']=b.sha(self.out/'runtime_config.json')
            controller_cleanup=b.read(self.out/'cleanup.json') if (self.out/'cleanup.json').exists() else {}
            self.receipt['controller_cleanup']=controller_cleanup
            # A successful schema3 host quiesces admission before it exits.
            # Verify real idle/budget/ACK here; cleanup restores accepting=True.
            after=await asyncio.gather(*(wait_idle(b,self.session,i,tokens=2048,accepting=None) for i in range(2)))
            b.write('operations/'+self.row['cell_id']+'/budget.after.json',after)
            b.require(self.receipt['child_exitcode']==0 and summary.get('measurement_valid') is True
                and controller_cleanup.get('cleanup_complete') is True
                and summary.get('post_measurement_cleanup',{}).get('cleanup_complete') is True,'invalid inner measurement/cleanup')
        except BaseException as exc:failure=exc;self.receipt['error']=repr(exc)
        finally:
            try:
                cleanup=await asyncio.wait_for(self.cleanup(),max(.01,self.limits['restore_deadline_s']-time.time()))
                self.receipt['outer_cleanup']=cleanup
            except BaseException as exc:failure=failure or exc;self.receipt['outer_cleanup_error']=repr(exc)
            end=self.receipt['operation_end_s']=time.time()
            try:
                until=time.monotonic()+3
                while not sampler.samples or sampler.samples[-1][0]<end:
                    b.require(not sampler.error and time.monotonic()<until,'outer power tail not bracketed');await asyncio.sleep(.01)
            except BaseException as exc:self.receipt['sampling_tail_error']=repr(exc)
            await asyncio.to_thread(sampler.stop)
            for name,offset in self.offsets.items():
                try:
                    with (b.EVENT_ROOT/(name+'.control.events.jsonl')).open('rb') as f:f.seek(offset);data=f.read()
                    (self.operation/(name+'.events.jsonl')).write_bytes(data)
                except BaseException as exc:self.receipt.setdefault('event_capture_errors',[]).append(repr(exc))
            pdir=self.operation/'power';pdir.mkdir()
            save_raw(pdir,[],sampler.samples,sampler.utilization_samples,power_source=sampler.power_source,power_metadata=sampler.power_metadata)
            with (pdir/'clocks.csv').open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['t_s']+[f'gpu{i}_sm_mhz' for i in range(8)])
                writer.writerows([t,*values] for t,values in sampler.frequency_samples)
            evidence=power_evidence(sampler.samples,sampler.power_source,sampler.power_metadata)
            self.receipt.update(power_evidence=evidence,sampling_error=sampler.error,owner_model_steps=self.event_counts,
                energy_boundary='separate all-eight-GPU full operation integral: budget setup through child/native cleanup/clock release; original schema3 cell integral retained')
            try:self.receipt['full_operation_energy_j']=trapezoid_energy(clip_power_window(sampler.samples,start,end,pad_s=0)) if start is not None else None
            except BaseException as exc:self.receipt['integration_error']=repr(exc)
            self.receipt['screen_valid']=bool(failure is None and summary and summary.get('measurement_valid') is True
                and cleanup.get('complete') is True and evidence['power_source_verified'] and not sampler.error
                and not self.receipt.get('integration_error') and not self.receipt.get('event_capture_errors')
                and not self.receipt.get('sampling_tail_error'))
            self.receipt['finished_s']=time.time();self.save()
            if self.child_log:self.child_log.close()
        if failure is not None:raise failure
        b.require(self.receipt['screen_valid'],'invalid point retained; stop remaining cells')


def sys_executable():
    import sys
    return sys.executable


async def sweep(part,b):
    import aiohttp
    import inspect
    from ecopadg.serving.cell import run_cell
    from ecopadg.measure.backends import PynvmlBackend
    b.require(Path(inspect.getfile(run_cell)).resolve()==b.HOST/'src/ecopadg/serving/cell.py','wrong frozen host module')
    name='status.'+part+'.json';b.require(not (b.ROOT/name).exists(),'part already attempted; preserve evidence')
    status=dict(part=part,phase='preflight',complete=False,started_s=time.time(),cells=[],baseline_execution=False,
        scope='A14B first three long development anchors only; not verified capacity or formal baseline inference',
        selected_cells=3,declared_matrix_cells=513,full_matrix_complete=False,
        long_workload_acceptance='requires independent terminal raw actual-dispatch-span/full-output audit; summary validity alone is insufficient')
    b.write(name,status);freeze=None
    try:
        freeze=await asyncio.to_thread(b.frozen_check,True)
        hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
        async with aiohttp.ClientSession(trust_env=False) as session:
            for row in [c for c in b.read(b.ROOT/'runspec.json')['cells'] if c['part']==part]:
                await asyncio.to_thread(b.frozen_check,False)
                with b.socket.socket() as sock:sock.bind(('127.0.0.1',b.read(b.CONFIG)['port']))
                status['phase']='cell:'+row['cell_id'];b.write(name,status)
                cell=Cell(b,session,row,freeze,status,name,hardware)
                await cell.execute()
                await asyncio.to_thread(b.frozen_check,False)
                b.write('identities/'+row['cell_id']+'.json',await b.live(session,expected_inventory=freeze['inventory']))
            await asyncio.to_thread(b.frozen_check,True)
            b.write('after.'+part+'.json',await b.live(session,expected_inventory=freeze['inventory']))
        status.update(phase='finished',complete=True)
    except BaseException as exc:status.update(phase='failed',error=repr(exc));raise
    finally:
        # Every cell owns its mutation and native/clock cleanup scope. No late
        # unmeasured control is sent from a failing file/identity preflight.
        if freeze is not None:
            try:
                for path,digest in freeze['dependencies']['protected_files'].items():
                    b.require(b.sha(path)==digest,'protected baseline changed: '+path)
                status['baseline_preservation_verified']=True
            except BaseException as exc:status.update(complete=False,phase='failed',baseline_preservation_error=repr(exc))
        status['finished_s']=time.time();b.write(name,status)
    b.require(status['complete'],'part did not complete; retain failed evidence')
