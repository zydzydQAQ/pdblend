"""Private same-Fleet online observation; scripted actions, never formal PD.

The caller owns engines, sampler and lease. This collector owns only a local
proxy and temporary measurement windows; it never stops/reloads an engine.
"""
from __future__ import annotations
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import time

import aiohttp
from pdblend.bench.client import LoadClient,Request,SLOS
from pdblend.bench.run import _serve_proxy
from pdblend.online.controller import Controller
from pdblend.online.native_control import NativeControl
from pdblend.online.observations import backlog_snapshot
from pdblend.online.router import Router
from pdblend.online.server import Proxy
from pdblend.engine.client import PDTransfer
from pdblend.planner.pool import Plan,PlannerConfig,SLO
from .native_online_causal import CausalForecaster,forecast_value
from .native_online_plan import (MODEL,REVISION,online_trace,validate_online_plan,freeze_prior,
                                 validate_prior,validate_discovery_candidate)
from .native_frequency_domain import identity_frequencies
from .native_runtime_collect import NativeRuntimeCollector,validate_inventory,snapshot_sampler,write_new
from .native_runtime_audit import power_slice
from .native_timing_plan import binding,read_bound,digest
from .native_timing_audit import need


def plain(value):
    return json.loads(json.dumps(value,allow_nan=False,default=lambda v:sorted(v) if isinstance(v,set) else vars(v)))


def action_plan(frequency,*,generation,profile_key):
    # None is intentional: a measurement instruction has no invented energy
    # or SLO prediction. It is never sent through a qualified profile loader.
    return Plan({'M':4},frequency,frequency,frequency,0,None,None,None,
        dict(revision=REVISION,scripted_calibration_action=True,model_prediction_available=False),
        tp=2,pp=1,generation=generation,profile_key=profile_key)


async def restore_online_fleet(runner,specs):
    receipts=[];errors=[]
    for spec in specs:
        try:
            stop=await runner.stop_measurement(spec);drain=await runner.drain(spec)
            clock=await runner.clock(spec,runner.restore_frequency);resume=await runner.resume(spec)
            receipts.append(dict(instance_id=spec.instance_id,measurement_stop=stop,drain=drain,clock=clock,resume=resume))
        except BaseException as exc:errors.append(spec.instance_id+': '+repr(exc))
    return dict(passed=not errors,instances=receipts,errors=errors,finished_s=time.time())


async def collect_online_window(runner,specs,plan,point,path,*,proxy_port,candidate_ref=None):
    trace=online_trace(plan,point);path=Path(path);prior_ref=freeze_prior(point,path.with_suffix('.prior.json'))
    if point['purpose']=='holdout':
        value=read_bound(candidate_ref)
        need(value['plan_sha256']==digest(plan) and value['frozen_s']<=time.time(),'unbound online holdout candidate')
    else:need(candidate_ref is None,'online training cannot consume a candidate')
    urls={s.instance_id:s.base_url for s in specs};native=NativeControl({s.instance_id:s for s in specs})
    metadata={s.instance_id:dict(tp=s.tp,pp=s.pp,generation=s.generation,model_id=MODEL,
                                pool_id=s.pool_id,profile_key=s.profile_key) for s in specs}
    router=Router(urls,instance_metadata=metadata);controller=None;server=None;load_task=None;tasks=[];stop=asyncio.Event()
    raw=dict(schema='pdblend-native-online-discovery-window/v1',revision=REVISION,system='pdblend',
        plan_sha256=digest(plan),point=point,trace=trace,prior=prior_ref,candidate=candidate_ref,
        actual_launch=[dict(spec=asdict(s),argv=s.command()) for s in specs],lease=runner.lease,
        capabilities={},before={},resume={},initial_clocks={},measurement_start={},measurement_stop={},
        native_after={},samples={},state_observations=[],actions=[],cleanup_errors=[],
        status='failed',hardware_executed=True,formal_eligible=False,online_policy_qualified=False,
        query_scope='actual_forecaster_invocations_on_observation_ticks_not_planner_decisions',
        client_execution='same_event_loop_discovery_not_formal_load_isolation',
        energy_scope='observed_complete_window_and_real_tail_not_EWMA_prediction')
    async def states():
        while not stop.is_set():
            for spec in specs:
                started=time.time();state=await runner.state(spec)
                raw['state_observations'].append(dict(instance_id=spec.instance_id,requested_s=started,received_s=time.time(),state=state))
            try:await asyncio.wait_for(stop.wait(),plan['native_state_period_s'])
            except asyncio.TimeoutError:pass
    async def query_ticks():
        while not stop.is_set():
            observer.set_backlog(backlog_snapshot(router));observer.forecast()
            try:await asyncio.wait_for(stop.wait(),plan['query_period_s'])
            except asyncio.TimeoutError:pass
    async def scheduled_actions(origin):
        for action in point['actions'][1:]:
            await asyncio.sleep(max(0.,origin+action['offset_s']-time.time()))
            entry=dict(planned=action,requested_s=time.time(),status='failed');raw['actions'].append(entry)
            try:
                await controller.execute(action_plan(action['frequency_mhz'],generation=specs[0].generation,profile_key=digest(plan)))
                entry.update(status='passed',finished_s=time.time(),frequencies_mhz=dict(controller.freqs))
            except BaseException as exc:entry.update(error=repr(exc),finished_s=time.time());raise
    try:
        first=point['actions'][0]['frequency_mhz']
        for spec in specs:
            iid=spec.instance_id;raw['capabilities'][iid]=await runner.capability(spec)
            raw['before'][iid]=await runner.drain(spec);raw['resume'][iid]=await runner.resume(spec)
            raw['initial_clocks'][iid]=await runner.clock(spec,first)
            raw['measurement_start'][iid]=await runner.request(spec,'measurement/start',dict(system='pdblend',scope='runner'))
            need(raw['measurement_start'][iid].get('acknowledged') is True,'online measurement start lacks ACK')
        prior=validate_prior(prior_ref,point,time.time())
        observer=CausalForecaster(initial=forecast_value(prior['forecast']),prior_ref=prior_ref,router=router,
            identity=dict(model_id=MODEL,tp=2,pp=1,frequency_domain_sha256=plan['frequency_domain_sha256']),
            controller=lambda:controller)
        config=PlannerConfig(slots=4,slo=SLO(*SLOS[point['dataset']]),freqs=identity_frequencies(plan),max_num_seqs=32,min_m_instances=4)
        planner=SimpleNamespace(cfg=config,model=SimpleNamespace(profile_key={'discovery':digest(plan)}))
        controller=Controller(runner.fleet,router,runner.meter,planner,forecaster=observer,native_control=native,
            freeze=True,roles={s.instance_id:'M' for s in specs},freqs={s.instance_id:first for s in specs})
        await controller.execute(action_plan(first,generation=specs[0].generation,profile_key=digest(plan)))
        transfer=PDTransfer(specs[0].kv_connector,{s.instance_id:s.zmq_address for s in specs})
        proxy=Proxy(urls,router,transfer=transfer,native_cancel=native.cancel)
        server=await _serve_proxy(proxy,proxy_port)
        await asyncio.sleep(plan['clock_settle_s'])
        load=LoadClient(f'http://127.0.0.1:{proxy_port}',timeout_s=plan['tail_timeout_s'],sampling_seed=point['seed'],token_diagnostics=True)
        load_task=asyncio.create_task(load.replay([Request(**r) for r in trace['requests']],progress_every_s=1e9))
        while load.replay_started_s is None:
            if load_task.done():await load_task
            await asyncio.sleep(0)
        origin=raw['service_started_s']=load.replay_started_s;raw['service_end_s']=origin+point['duration_s']
        tasks=[asyncio.create_task(states()),asyncio.create_task(query_ticks()),asyncio.create_task(scheduled_actions(origin))]
        await asyncio.sleep(max(0.,raw['service_end_s']-time.time()))
        outcomes=await asyncio.wait_for(load_task,plan['tail_timeout_s']);raw['outcomes']=[asdict(r) for r in outcomes]
        await tasks[2]
        for spec in specs:
            iid=spec.instance_id;raw['native_after'][iid]=await runner.drain(spec)
            raw['samples'][iid]=await runner.request(spec,'measurement/samples')
            raw['measurement_stop'][iid]=await runner.stop_measurement(spec)
        raw['tail_end_s']=time.time();raw['status']='measured'
    except BaseException as exc:raw['error']=repr(exc)
    finally:
        stop.set()
        if load_task and not load_task.done():load_task.cancel()
        if load_task:await asyncio.gather(load_task,return_exceptions=True)
        for task in tasks:
            if task is tasks[-1] and not task.done():task.cancel()
        results=await asyncio.gather(*tasks,return_exceptions=True)
        raw['cleanup_errors'] += [repr(v) for v in results if isinstance(v,BaseException) and not isinstance(v,asyncio.CancelledError)]
        if server:
            try:await server.cleanup()
            except BaseException as exc:raw['cleanup_errors'].append('proxy cleanup: '+repr(exc))
        if controller is not None:
            raw['controller_events']=plain(controller._log);raw['transition_events']=plain(controller.transition_events)
            raw['causal_forecaster']=controller.forecaster.receipt()
            raw['routes']=plain([asdict(r) for r in router.records]);raw['router_inflight']=router.inflight()
        # Failure still requires native cancellation/drain. No process outside
        # the caller's named resident specs is touched.
        for spec in specs:
            try:
                if raw['status']!='measured':
                    state=await runner.state(spec)
                    for rid in state['all_queue']:
                        raw.setdefault('cancellations',[]).append(await runner.request(spec,'cancel',dict(request_id=rid)))
                if spec.instance_id not in raw['native_after']:raw['native_after'][spec.instance_id]=await runner.drain(spec)
                if spec.instance_id not in raw['measurement_stop']:raw['measurement_stop'][spec.instance_id]=await runner.stop_measurement(spec)
            except BaseException as exc:raw['cleanup_errors'].append(spec.instance_id+': '+repr(exc))
        if 'service_started_s' in raw:
            raw.setdefault('tail_end_s',time.time());deadline=time.monotonic()+5
            while runner.sampler.error is None and time.monotonic()<deadline:
                if runner.sampler.power_metadata and min(runner.sampler.power_metadata[-1]['read_finished_s'])>raw['tail_end_s']:break
                await asyncio.sleep(.05)
            raw['power']=power_slice(snapshot_sampler(runner.sampler,runner.uuids),raw['service_started_s'],raw['tail_end_s'])
        if raw['cleanup_errors']:raw['status']='failed'
        write_new(path,plain(raw))
    return raw


async def collect_online_phase(specs,fleet,meter,sampler,out,*,gpu_uuids,plan,proxy_port,phase='training',candidate_ref=None):
    validate_online_plan(plan);need(phase in ('training','holdout'),'unknown online discovery split')
    if phase=='holdout':validate_discovery_candidate(candidate_ref,plan,started_s=time.time())
    else:need(candidate_ref is None,'training cannot consume online candidate')
    need(len(specs)==4 and all(s.tp==2 and s.pp==1 and Path(s.model).name==MODEL for s in specs),
         'online discovery needs four native32 TP2 replicas')
    out=Path(out);need(not out.exists(),'new immutable online phase directory required')
    runner=NativeRuntimeCollector(specs,fleet,meter,sampler,out,gpu_uuids=gpu_uuids,frequency_domain_ref=plan['frequency_domain_ref'])
    runner.lease=validate_inventory(specs,fleet,meter,sampler,gpu_uuids)
    out.mkdir(parents=True);(out/'journal.jsonl').touch(exist_ok=False)
    report=dict(schema='pdblend-native-online-discovery-collection/v1',phase=phase,plan_sha256=digest(plan),
        started_s=time.time(),candidate=candidate_ref,windows=[],collection_complete=False,
        safe_restore_passed=False,operational_failure=False,formal_eligible=False,online_policy_qualified=False)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        runner.session=session
        try:
            for index,point in enumerate(plan['points']):
                if point['purpose']!=phase:continue
                path=out/'windows'/f'{index:03d}.json'
                raw=await collect_online_window(runner,specs,plan,point,path,proxy_port=proxy_port,candidate_ref=candidate_ref)
                from .native_online_audit import audit_online_window
                audited=audit_online_window(raw,plan)
                report['windows'].append(dict(raw=binding(path),audit=audited))
                need(audited['raw_complete'],'online discovery raw window failed: '+repr(audited.get('errors')))
                restoration=await restore_online_fleet(runner,specs)
                report['windows'][-1]['restoration']=restoration
                need(restoration['passed'],'online window failed safe restoration')
            report['collection_complete']=True
        except BaseException as exc:report.update(error=repr(exc),operational_failure=True)
        finally:
            report['restoration']=await restore_online_fleet(runner,specs)
            report['safe_restore_passed']=report['restoration']['passed']
            report['operational_failure'] |= not report['safe_restore_passed']
            report['ready_for_next']=report['safe_restore_passed'] and not report['operational_failure']
            report['finished_s']=time.time();write_new(out/'completion.json',report)
    return report


def replay_online_collection(reference,plan,*,phase):
    from .native_online_audit import replay_online_collection as replay
    return replay(reference,plan,phase=phase)
