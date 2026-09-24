"""Automatic four-member EcoServe macro qualification on native GPU services.

The original five-second scale loop and sixty-second history run unmodified.
The observer records evidence and never calls membership primitives.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import aclosing
import hashlib
import json
import os
from pathlib import Path
import time

from .auto_macro_replay import replay
from .mechanism_four import kv_blocks, observed_live_tokens, preserves, tokens
from .run_native import VerifiedTransport, automatic_actions, expected_identity, load_trace, sha, validate_profile, validate_state
from .runtime import EcoServeRuntime
from ..resident_campaign import drain_endpoints, eco_config, model_load_lock, verify_endpoints, warmup_endpoints
from pdblend_runtime.probe import NativeSpec
from pdblend_runtime.cleanup import cleanup_owned
from pdblend.engine.launcher import Fleet
from pdblend.bench.metering import Gpus
from pdblend.results.journal import CompactJournal, payload_receipt
from pdblend.results.power_archive import write_power_archive


def specs_and_config(model, tp, gpus, base_port, profile):
    required_tp = 2 if Path(model).name == 'Qwen2.5-32B-Instruct' else 1
    if (Path(model).name not in ('Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct', 'Qwen2.5-32B-Instruct')
            or tp != required_tp or len(gpus) != 4*tp or len(set(gpus)) != len(gpus)):
        raise ValueError('automatic macro requires four disjoint fixed-TP native engines')
    specs = [NativeSpec(f'eco{i}', tuple(gpus[i*tp:(i+1)*tp]), base_port+16*i, model, tp=tp,
                        max_num_seqs=32, extra_args=('--enforce-eager',)) for i in range(4)]
    config = eco_config(model, specs, profile)
    config.update(eco_initial_instances=3, eco_macro_lower=2, eco_macro_upper=3,
                  eco_scale_period_s=5.0, eco_history_window_s=60.0,
                  request_timeout_s=240.0, eco_drain_timeout_s=120.0)
    return specs, config


def verify_inputs(args):
    expected = json.loads(args.input_manifest.read_text())
    paths = dict(trace=args.trace, csv=args.eco_profile,
                 csv_manifest=Path(str(args.eco_profile)+'.manifest.json'),
                 source_manifest=Path(os.environ['PDBLEND_SOURCE_MANIFEST']),
                 model_verification=Path(os.environ['PDBLEND_MODEL_VERIFICATION_RECEIPT']))
    actual = {key:sha(path) for key, path in paths.items()}
    if actual != expected['exact_inputs_sha256']:
        raise ValueError('frozen automatic macro input checksum differs')
    source = json.loads(paths['source_manifest'].read_text())
    digest = hashlib.sha256(json.dumps(source['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (source['source_sha256'] != digest or digest != expected['source_sha256']
            or digest != os.environ['PDBLEND_SOURCE_SHA256'] or expected['image_digest'] != os.environ['PDBLEND_IMAGE_ID']):
        raise ValueError('frozen automatic macro source/image identity differs')
    root = Path(__file__).resolve().parents[2]
    if any(sha(root/name) != checksum for name, checksum in source['files'].items()):
        raise ValueError('frozen execution source bytes differ')
    return expected


class EvidenceObserver:
    def __init__(self, out):
        self.raw = CompactJournal(out/'events.jsonl.gz')
        self.rows, self.held, self.continuity, self.observation_tasks = [], {}, [], []
        self.runtime = self.transport = None

    def emit(self, kind, **fields):
        fields.setdefault('at_s', time.time())
        row = dict(kind=kind, **fields)
        self.raw.write(row)
        # Qualification needs token IDs and separate client/native clocks,
        # not a copy of cumulative decoded text for every observation.
        if isinstance(row.get('payload'), dict):
            row = dict(row, payload={key:value for key,value in row['payload'].items()
                                    if key not in ('text', 'choices')})
        self.rows.append(row)
        if self.runtime and kind in ('eco_engine_output', 'eco_admission'):
            for iid, buffer in self.runtime.controller.buffers.items():
                for rid, event in buffer.pending:
                    key = (rid, event.get('token_index'))
                    if key not in self.held:
                        packet = dict(instance_id=iid, request_id=rid, token_index=event.get('token_index'),
                                      token_ids=list(event.get('token_ids', [])), observed_s=time.time())
                        self.held[key] = packet
                        self.emit('eco_observed_output_hold', **packet)
        if kind == 'eco_membership_commit' and row.get('trigger') in ('mean_ttft', 'saved_tpot'):
            start = next((index for index in range(len(self.rows)-2, -1, -1)
                          if self.rows[index]['kind']=='eco_membership_prepare'
                          and all(self.rows[index].get(k)==row.get(k)
                                  for k in ('operation','instance_id','trigger','before'))), None)
            if start is not None:
                before = commit_baselines(self.rows[start:-1])
                self.observation_tasks.append(asyncio.create_task(self.observe_commit(row, before)))

    async def observe_commit(self, commit, before_states):
        """Observe continuing native workers after the automatic commit only."""
        candidates = {}
        after_members = {iid for group in commit['after'] for iid in group}
        for iid, state in before_states.items():
            if iid not in after_members:
                continue
            for rid in state.get('kv_allocations', {}):
                events = self.transport.engine_outputs.get(rid, [])
                count = observed_live_tokens(state, rid, events)
                if count:
                    candidates[rid] = dict(instance_id=iid, generation=state['generation'], kv_blocks=kv_blocks(state,rid),
                                           native_tokens=count, observed_s=state['native_at_s'],
                                           ranks=state['ranks'], acknowledged_generation=state['acknowledged_generation'])
        try:
            deadline = time.monotonic()+5
            matched = {}
            while candidates and time.monotonic()<deadline:
                for iid in {row['instance_id'] for row in candidates.values()}:
                    state = await self.transport.state(iid)
                    for rid, before in candidates.items():
                        if before['instance_id'] != iid:
                            continue
                        count = observed_live_tokens(state,rid,self.transport.engine_outputs.get(rid,[]))
                        if not count:
                            continue
                        after = dict(instance_id=iid,generation=state['generation'],kv_blocks=kv_blocks(state,rid),
                                     native_tokens=count,observed_s=state['native_at_s'],ranks=state['ranks'],
                                     acknowledged_generation=state['acknowledged_generation'])
                        if preserves(before,after) and count>before['native_tokens']:
                            matched[rid] = dict(before=before,after=after)
                if matched:
                    break
                await asyncio.sleep(.05)
            receipt = dict(operation=commit['operation'], trigger=commit['trigger'], version=commit['version'],
                           split=commit.get('split',False), merge=commit.get('merge',False),
                           native_request_continuity=matched, observed=bool(matched))
        except Exception as exc:
            receipt = dict(operation=commit['operation'], version=commit['version'], observed=False,error=repr(exc))
        self.continuity.append(receipt)
        self.emit('eco_automatic_live_kv_receipt', **receipt)


def commit_baselines(transition_rows):
    """Use the latest actual worker observation before the membership commit.

    Removing a member waits for its requests to drain. Other members continue
    admitting work throughout that wait, so their prepare-time requests may
    have finished before the layout changes. The controller already observes
    each worker again just before its unchanged control is acknowledged. Those
    fresh all-rank states provide the correct boundary, without issuing a
    control operation or changing the automatic decision.
    """
    before = dict(transition_rows[0].get('observed_engine_states', {}))
    for row in transition_rows[1:]:
        if row['kind'] != 'eco_control_confirmed' or row.get('noop') is not True:
            continue
        state, iid = row.get('observed_engine_state'), row.get('instance_id')
        if state and state['native_at_s'] >= before.get(iid, {}).get('native_at_s', -1):
            before[iid] = state
    return before


def qualification_checks(journal, continuity, held, outcomes):
    actions = automatic_actions(journal)
    commits = [journal[action['commit_index']] for action in actions]
    acknowledged_versions = {row['version'] for row in continuity if row.get('observed')}
    clients = {(row['request_id'],row['payload'].get('token_index'),tuple(row['payload'].get('token_ids',[]))):row['at_s']
               for row in journal if row['kind']=='eco_client_sse'}
    hold_flush = any(clients.get((packet['request_id'],packet['token_index'],tuple(packet['token_ids'])),-1)
                     >= packet['observed_s'] for packet in held.values())
    split = [row for row in commits if row.get('operation')=='add' and row.get('split') and row.get('trigger')=='mean_ttft']
    merge = [row for row in commits if row.get('operation')=='remove' and row.get('merge') and row.get('trigger')=='saved_tpot']
    return dict(automatic_split=bool(split), automatic_merge=bool(merge),
                split_live_kv_ack=any(row['version'] in acknowledged_versions for row in split),
                merge_live_kv_ack=any(row['version'] in acknowledged_versions for row in merge),
                automatic_park=any(row.get('operation')=='remove' for row in commits),
                policy_rotation=any(row['kind']=='eco_admission' and row.get('controls') for row in journal),
                actual_hold=bool(held), held_output_flush=hold_flush,
                continuous_complete_output=bool(outcomes) and all(row.get('ok') for row in outcomes),
                no_manual_membership=not any(row['kind']=='eco_membership_commit' and row.get('trigger') not in
                                              ('mean_ttft','saved_tpot') for row in journal)), actions


async def execute_window(config, endpoints, trace, out, duration=300):
    out.mkdir(parents=True, exist_ok=False)
    observer = EvidenceObserver(out)
    specs = {row['id']:row for row in config['instances']}
    physical = os.environ['PDBLEND_GPU_UUIDS'].split(',')
    transport = VerifiedTransport(endpoints,observer.emit,specs,{gpu:physical[gpu] for spec in specs.values() for gpu in spec['gpus']})
    runtime = EcoServeRuntime(config,transport,observer.emit)
    observer.runtime, observer.transport = runtime, transport
    result = dict(status='inconclusive', complete=False, hardware_executed=True, formal_eligible=False,
                  energy_comparable=False, automatic_policy_status='inconclusive', outcomes=[], cleanup_errors=[],
                  config=config, period_s=5, history_window_s=60, duration_s=duration, manual_layout_action=False)
    tasks = []
    try:
        _, rows = load_trace(trace,duration)
        await runtime.start()
        result['initial_states'] = {iid:await transport.state(iid) for iid in endpoints}
        started = time.monotonic()
        result['service_started_s'] = time.time()
        observer.emit('eco_service_window_start',duration_s=duration)
        async def request(index,arrival,prompt,count):
            await asyncio.sleep(max(0,started+arrival-time.monotonic()))
            rid = f'eco-auto-701-{index}'
            row = dict(request_id=rid,arrival_s=arrival,input_tokens=len(prompt),output_tokens=count,
                       submitted_s=time.time(),events=[],ok=False)
            try:
                async with aclosing(runtime.handle(dict(prompt=prompt,max_tokens=count,seed=701,
                                                        temperature=0,ignore_eos=True),rid)) as stream:
                    async for event in stream:
                        row['events'].append({key:value for key,value in event.items()
                                              if key not in ('text', 'choices')})
                        observer.emit('eco_client_sse',request_id=rid,payload=event)
                native = transport.engine_outputs.get(rid,[])
                row['token_ids'], row['native_token_ids'] = tokens(row['events']), tokens(native)
                row['ok'] = (len(row['token_ids'])==count and row['token_ids']==row['native_token_ids']
                             and bool(native and native[-1].get('finished'))
                             and bool(row['events'] and row['events'][-1].get('finished')))
            except Exception as exc:
                row['error'] = repr(exc)
            finally:
                row['finished_s'] = time.time()
                row.update(payload_receipt(row.pop('events'), journal_path='events.jsonl.gz', request_id=rid))
                row.pop('token_ids', None)
                row.pop('native_token_ids', None)
                result['outcomes'].append(row)
                observer.emit('eco_request_outcome',**row)
        tasks = [asyncio.create_task(request(i,*row)) for i,row in enumerate(rows)]
        await asyncio.sleep(duration)
        await asyncio.wait_for(asyncio.gather(*tasks),config['request_timeout_s'])
        result['service_finished_s'] = time.time()
        if runtime.controller.failure or runtime.controller.quarantined:
            raise RuntimeError('automatic native controller failed or quarantined worker')
        if len(result['outcomes']) != len(rows):
            raise RuntimeError('automatic workload outcomes are incomplete')
    except BaseException as exc:
        result['error'] = repr(exc)
    finally:
        for task in tasks:
            if not task.done():task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        try:
            await runtime.close()
            await asyncio.gather(*observer.observation_tasks)
        except BaseException as exc:
            result['cleanup_errors'].append(dict(component='runtime_close',error=repr(exc)))
        result['drain_receipts'] = {}
        for iid in endpoints:
            try:
                receipt = await transport.json(iid,'/drain',dict(timeout_s=25))
                if receipt.get('acknowledged') is not True or receipt.get('drained') is not True:
                    raise RuntimeError('native final drain ACK missing')
                validate_state(receipt,specs[iid]['tp'],drained=True)
                result['drain_receipts'][iid] = receipt
                await transport.park(specs[iid]['gpus'])
            except BaseException as exc:
                result['cleanup_errors'].append(dict(component='native_drain',instance_id=iid,error=repr(exc)))
        checks, actions = qualification_checks(observer.rows,observer.continuity,observer.held,result['outcomes'])
        checks['all_native_drains_acknowledged'] = set(result['drain_receipts'])==set(endpoints)
        checks['controller_healthy'] = not runtime.controller.failure and not runtime.controller.quarantined
        checks['no_execution_or_cleanup_error'] = not result.get('error') and not result['cleanup_errors']
        result.update(checks=checks,automatic_actions=actions,continuity=observer.continuity,
                      held_packets=list(observer.held.values()),journal_path='events.jsonl.gz',journal_rows=len(observer.rows),
                      automatic_commits=[observer.rows[row['commit_index']] for row in actions])
        qualified = all(checks.values())
        result.update(status='passed' if qualified else 'inconclusive',complete=qualified,
                      automatic_policy_status='passed' if qualified else 'inconclusive')
        observer.raw.close()
        result['events_sha256'] = sha(out/'events.jsonl.gz')
        (out/'completion.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


async def execute_owned(args,specs,config,identity):
    fleet = Fleet(specs,args.out/'logs')
    meter = Gpus(args.gpus,power_mode='instant')
    sampler = meter.sampler(interval_s=.1)
    result = dict(status='inconclusive',complete=False,hardware_executed=False,formal_eligible=False,
                  energy_comparable=False,source=identity,cleanup_errors=[],engine_loads=0)
    try:
        sampler.start()
        result['startup'] = {}
        with model_load_lock():
            for spec in specs:
                instance=fleet[spec.instance_id]
                instance.start();result['engine_loads']+=1
                result['startup'][spec.instance_id]=instance.wait_ready(timeout_s=600)
        result['hardware_executed']=True
        result['capabilities']=await verify_endpoints(specs)
        result['warmup']=await warmup_endpoints(specs,'ecoserve-auto')
        window=await execute_window(config,{s.instance_id:s.base_url for s in specs},args.trace,args.out/'automatic',args.duration)
        result.update(status=window['status'],complete=window['complete'],checks=window['checks'],
                      automatic_policy_status=window['automatic_policy_status'],
                      automatic_actions=window['automatic_actions'])
        result['final_drain_states']=await drain_endpoints(specs)
        result['automatic_receipt_sha256']=sha(args.out/'automatic/completion.json')
    except BaseException as exc:
        result.update(status='inconclusive',complete=False,error=repr(exc))
    finally:
        result['cleanup_errors']=cleanup_owned(fleet,meter,sampler)
        power=dict(samples=sampler.samples,frequency_samples=sampler.frequency_samples,
                   utilization_samples=sampler.utilization_samples,power_metadata=sampler.power_metadata,
                   error=sampler.error,gpu_ids=args.gpus,gpu_uuids=os.environ.get('PDBLEND_GPU_UUIDS'),
                   formal_eligible=False,energy_comparable=False)
        write_power_archive(args.out/'power.json',power)
        result['power_sha256']=sha(args.out/'power.json')
        if result['cleanup_errors'] or sampler.error or len(sampler.samples)<2 or not sampler.frequency_samples:
            result.update(status='inconclusive',complete=False)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True);parser.add_argument('--tp',type=int,required=True)
    parser.add_argument('--gpus',required=True);parser.add_argument('--base-port',type=int,required=True)
    parser.add_argument('--eco-profile',type=Path,required=True);parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--input-manifest',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--duration',type=float,default=300);parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args(argv);args.gpus=[int(v) for v in args.gpus.split(',')]
    args.out.mkdir(parents=True,exist_ok=True)
    if args.duration!=300:raise ValueError('automatic macro workload is frozen to its 300-second service window')
    identity=verify_inputs(args)
    specs,config=specs_and_config(args.model,args.tp,args.gpus,args.base_port,args.eco_profile)
    expected=expected_identity(config);profile=validate_profile(config,expected,args.tp)
    _,rows=load_trace(args.trace,args.duration)
    (args.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    if args.preflight_only:
        async def candidates():
            return [await replay(config,rows,args.duration,factor) for factor in (.5,1.,1.5)]
        scenarios=asyncio.run(candidates())
        passed=all(r['split_candidate'] and r['merge_candidate'] for r in scenarios)
        result=dict(status='cpu_preflight_passed' if passed else 'cpu_candidate_inconclusive',complete=False,
                    hardware_executed=False,source=identity,profile=profile,identity=expected,
                    config=config,cpu_replays=scenarios,requests=len(rows),
                    specs=[dict(instance_id=s.instance_id,tp=s.tp,gpus=s.gpus,command=s.command()) for s in specs])
    else:
        result=asyncio.run(execute_owned(args,specs,config,identity))
    receipt=args.out/('preflight.json' if args.preflight_only else 'completion.json')
    receipt.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],receipt=str(receipt),complete=result['complete'])))
    return 0 if result['status'] in ('passed','cpu_preflight_passed') else 2


if __name__=='__main__':raise SystemExit(main())
