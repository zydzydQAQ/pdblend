"""Small pure-role power pilot on the caller's live native timing fleet.

No engine load, sampler restart, queue mutation, registry write or profile fit.
The caller must stop the session if ready_for_timing is false.
"""
from __future__ import annotations
import asyncio
from dataclasses import asdict
import random
import time
import uuid
from pathlib import Path

import aiohttp
from pdblend_runtime.probe import call
from .native_timing_collect import _request, WORKER
from .native_timing_audit import need
from .native_power_plan import validate_plan
from .native_power_audit import audit_power_window, PowerFrequencyQualificationError
from .native_runtime_collect import NativeRuntimeCollector, validate_inventory, snapshot_sampler, write_new
from .native_runtime_audit import power_slice


class PowerDomainUnavailable(ValueError):
    """A real capacity/output budget cannot provide this continuous window."""


def require_running(tasks, clients, output_tokens):
    ended=[i for i,t in enumerate(tasks) if t.done()]
    if not ended:return
    if all(clients[i].get('terminal') is True and not clients[i].get('error')
           and clients[i].get('completion_tokens')==output_tokens for i in ended):
        raise PowerDomainUnavailable('normal output budget ended before the continuous power boundary')
    raise RuntimeError('native decode request failed before the power boundary')


async def _wait(predicate, tasks, timeout_s=120.):
    deadline=time.monotonic()+timeout_s
    while not await predicate():
        need(not any(t.done() for t in tasks),'request ended before native admission/measurement barrier')
        if time.monotonic()>=deadline:raise TimeoutError('native pure-power barrier expired')
        await asyncio.sleep(.01)


async def power_window(runner, spec, point, path):
    raw=dict(schema='pdblend-native-power-window/v1',system='pdblend',status='failed',point=point,
        spec=asdict(spec),lease=runner.lease,cleanup_errors=[],client_requests=[],cancel_receipts={},
        formal_eligible=False,power_component_qualified=False,full_profile_qualified=False,
        invocation_id='pd-power-'+uuid.uuid4().hex)
    tasks=[];session=runner.session;url=spec.base_url
    async def control(**kwargs):
        result=await runner.request(spec,'control',kwargs)
        need(result.get('acknowledged') is True,'native admission control lacks ACK')
        return result
    async def submit(index):
        rid=raw['invocation_id']+'-'+str(index)
        row=dict(request_id=rid,seen_tokens=0);raw['client_requests'].append(row)
        rng=random.Random(point['seed']+point['repeat'])
        prompt=[rng.randrange(100,60000) for _ in range(point['prompt_tokens'])]
        payload=dict(request_id=rid,prompt=prompt,max_tokens=point['output_tokens'],temperature=0,
                     seed=point['seed']+point['repeat'],ignore_eos=True)
        task=asyncio.create_task(_request(session,url,payload,row));tasks.append(task)
        return task
    try:
        raw['capability']=await runner.capability(spec)
        capacity=raw['capability']['state']
        if point['batch']*(point['prompt_tokens']+point['output_tokens'])>=.9*capacity['total_kv_tokens']:
            raise PowerDomainUnavailable('pilot actual full output reservation exceeds native KV capacity')
        raw['before']=await runner.drain(spec)
        await runner.resume(spec)
        raw['clock']=await runner.clock(spec,point['frequency_mhz'])
        raw['measurement_start']=await runner.request(spec,'measurement/start',dict(system='pdblend',scope='runner'))
        need(raw['measurement_start'].get('acknowledged') is True,'idle native measurement arm failed')
        if point['role']=='decode':
            raw['queue_gate']=await control(accepting=True,admit_prefill=False,admit_decode=False)
            for i in range(point['batch']):await submit(i)
            async def queued():
                state=await runner.state(spec)
                return {r['request_id'] for r in raw['client_requests']}<=set(state['all_queue'])
            await _wait(queued,tasks)
            raw['prefill_gate']=await control(admit_prefill=True,admit_decode=False)
            async def prefills_complete():return all(r['seen_tokens']>=1 for r in raw['client_requests'])
            await _wait(prefills_complete,tasks)
            raw['decode_gate']=await control(accepting=False,admit_prefill=False,admit_decode=True)
            raw['settle_started_s']=time.time();await asyncio.sleep(2.)
            require_running(tasks,raw['client_requests'],point['output_tokens'])
            raw['start_s']=time.time();await asyncio.sleep(5.);raw['end_s']=time.time()
            require_running(tasks,raw['client_requests'],point['output_tokens'])
            for row in raw['client_requests']:row['observed_running_through_s']=raw['end_s']
            raw['service_end_state']=await runner.state(spec)
            raw['pause']=await control(admit_decode=False)
        else:
            await control(accepting=True,admit_prefill=True,admit_decode=False)
            raw['settle_started_s']=time.time();index=0
            while time.time()-raw['settle_started_s']<2.:
                await (await submit(index));index+=1
            raw['start_s']=time.time()
            while time.time()-raw['start_s']<5.:
                await (await submit(index));index+=1
            raw['end_s']=time.time()
        raw['sample']=await runner.request(spec,'measurement/samples')
        raw['status']='measured'
    except BaseException as exc:
        raw['error']=repr(exc)
        raw['error_kind']='domain_unavailable' if isinstance(exc,PowerDomainUnavailable) else 'operational_failure'
    finally:
        # Cancellation is expected only after a complete measurement, and every
        # native request must acknowledge real scheduler/worker KV release.
        if point['role']=='decode':
            for row in raw['client_requests']:
                try:
                    receipt=await runner.request(spec,'cancel',dict(request_id=row['request_id']))
                    raw['cancel_receipts'][row['request_id']]=receipt
                    row['cancel_expected']=raw['status']=='measured' and receipt.get('acknowledged') is True
                except BaseException as exc:raw['cleanup_errors'].append(repr(exc))
        for task in tasks:
            if not task.done():task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        for key,endpoint,payload in [('measurement_stop','measurement/stop',{}),('drain','drain',dict(timeout_s=30))]:
            try:raw[key]=await runner.request(spec,endpoint,payload)
            except BaseException as exc:raw['cleanup_errors'].append(repr(exc))
        if raw['cleanup_errors']:raw.update(status='failed',error_kind='operational_failure')
        if 'start_s' in raw:
            end=raw.get('end_s',time.time());deadline=time.monotonic()+5
            while runner.sampler.error is None and time.monotonic()<deadline:
                if (runner.sampler.power_metadata
                        and min(runner.sampler.power_metadata[-1]['read_finished_s'])>end):break
                await asyncio.sleep(.05)
            raw['power']=power_slice(snapshot_sampler(runner.sampler,runner.uuids),raw['start_s'],end)
        write_new(path,raw)
    return raw


async def collect_power_pilot(specs,fleet,meter,sampler,out,*,gpu_uuids,plan):
    """Return immutable observations plus safe restoration; never qualify a fit.

    Serial target windows use one replica; all eight loaded boards are sampled.
    Native timing's concurrency qualification is not reused for pure power.
    Caller retains ownership of engine processes, the sampler, and their lease.
    """
    validate_plan(plan)
    need(len(specs)==8//plan['tp'] and all(s.tp==plan['tp'] and s.pp==1 and WORKER in s.extra_args and Path(s.model).name==plan['model_id']
             for s in specs),'pilot needs the same PD native timing worker and model')
    out=Path(out);runner=NativeRuntimeCollector(specs,fleet,meter,sampler,out,gpu_uuids=gpu_uuids)
    runner.lease=validate_inventory(specs,fleet,meter,sampler,gpu_uuids)
    out.mkdir(parents=True,exist_ok=False);(out/'journal.jsonl').touch(exist_ok=False)
    report=dict(schema='pdblend-native-power-pilot/v1',status='failed',complete=False,hardware_executed=True,
        formal_eligible=False,power_component_qualified=False,full_profile_qualified=False,
        ready_for_timing=False,safe_restore_passed=False,domain_unavailable=[],operational_failure=False,
        measurement_qualification_gaps=[],
        plan=plan,windows=[],cleanup_errors=[],lease=runner.lease,
        actual_launch=[dict(spec=asdict(s),argv=s.command()) for s in specs])
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        runner.session=session
        try:
            for spec in specs:
                runner.capabilities[spec.instance_id]=await runner.capability(spec)
                await runner.drain(spec)
            for i,point in enumerate(plan['points']):
                for repeat in range(point['repeats']):
                    path=out/'windows'/f'{i:02d}-{repeat}.json'
                    raw=await power_window(runner,specs[0],dict(point,repeat=repeat),path)
                    row=dict(point_index=i,repeat=repeat,raw=dict(path=str(path.resolve()),
                        sha256=__import__('hashlib').sha256(path.read_bytes()).hexdigest()))
                    if raw.get('error_kind')=='domain_unavailable' and not raw['cleanup_errors']:
                        row['audit']=dict(passed=False,error=raw['error'],error_kind='domain_unavailable')
                        report['windows'].append(row)
                        report['domain_unavailable'].append(dict(point_index=i,repeat=repeat,
                            omitted_identical_repeats=point['repeats']-repeat-1,reason=raw['error']))
                        break
                    try:row['audit']=audit_power_window(raw,
                        expected_capability=runner.capabilities[specs[0].instance_id])
                    except PowerFrequencyQualificationError as exc:
                        # A typed result from the final audit stage means all
                        # other raw/protocol/power gates passed and frequency
                        # acquisition is complete. The window remains invalid;
                        # only the independent timing stage may later continue.
                        row['audit']=exc.audit
                        report['windows'].append(row)
                        report['measurement_qualification_gaps'].append(dict(point_index=i,repeat=repeat,
                            raw=row['raw'],kind=exc.audit['qualification_gap'],
                            invalid_window_preserved=True,frequency_evidence=exc.audit['frequency_evidence']))
                        continue
                    except Exception as exc:row['audit']=dict(passed=False,error=repr(exc))
                    report['windows'].append(row)
                    if not row['audit']['passed']:
                        raise ValueError('power pilot observation unqualified: '+row['audit']['error'])
            report.update(status=('partial_measurement_qualification' if report['measurement_qualification_gaps']
                                  else 'partial_domain_coverage' if report['domain_unavailable'] else 'passed'),
                          collection_complete=True,
                          complete=not report['domain_unavailable'] and not report['measurement_qualification_gaps'])
        except BaseException as exc:report.update(error=repr(exc),operational_failure=True)
        finally:
            # No off/wake, no replacement worker, and no implicit process load.
            restored=[]
            for spec in specs:
                try:
                    need(fleet[spec.instance_id].alive(),'resident process died during pilot')
                    stopped=await runner.stop_measurement(spec)
                    state=await runner.state(spec)
                    for rid in state['all_queue']:await runner.request(spec,'cancel',dict(request_id=rid))
                    clock=await runner.clock(spec,2520);drain=await runner.drain(spec)
                    restored.append(dict(instance_id=spec.instance_id,clock=clock,drain=drain,measurement_stop=stopped,
                                         capability=await runner.capability(spec)))
                except BaseException as exc:report['cleanup_errors'].append(spec.instance_id+': '+repr(exc))
            report['restoration']=dict(passed=not report['cleanup_errors'],instances=restored)
            try:report['restoration']['lease']=validate_inventory(specs,fleet,meter,sampler,gpu_uuids)
            except BaseException as exc:
                report['cleanup_errors'].append('restoration sampler/fleet: '+repr(exc))
                report['restoration']['passed']=False
            report['safe_restore_passed']=report['restoration']['passed']
            if not report['safe_restore_passed']:
                report['operational_failure']=True
            report['ready_for_timing']=report['safe_restore_passed'] and not report['operational_failure']
            if not report['restoration']['passed']:report['status']='failed'
            report['remaining_gates']=plan['remaining_gates']
            write_new(out/'completion.json',report)
    return report
