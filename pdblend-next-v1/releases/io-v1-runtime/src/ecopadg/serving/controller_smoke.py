"""Prepare/run six sequential controller correctness cells, never a comparison.

CPU: ``generate --manifest FILE --out DIR``. The generated campaign contains
one GPU stage of at most 900 seconds. Queue it only after profile certification.
Its ``run --setup DIR --out DIR`` command shares the existing node lease.
The manifest explicitly names profiles, transfers, frequency_costs (files),
role_costs, image, four mixed TP1 instances, engine_template, runtime_dir,
ownership_root and the already-started campaign_root. Initial instances may
also include explicitly authorized instances needed by physical restoration.
"""
import argparse
import asyncio
import csv
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import aiohttp
from benchmarks.scripts.bench_vllm import send_request
from ecopadg.measure.backends import PynvmlBackend
from ecopadg.measure.power import instant_power_verified
from .budget import read_budget
from .calibration import implementation_sources
from .calibration_setup import read, write, spec, verify_artifacts, restore_layout, current_engine_sources
from .campaign import node_lease
from .cell import run_cell
from .evidence import sha256, freeze_files, validate_freeze
from .frequency import verify_frozen_costs
from .interconnect import InterconnectTopology
from .planner import TransferCost
from .profiling import HardwareProfiler
from .profiles import ProfileStore
from .reconfigure import RoleCost
from .topology import validate_layout


STRATEGIES=('mixed','mixed_dvfs','distserve','pdblend-greedy','pdblend-joint','pdblend-dynamic')
INPUTS=(128,2048,7168,128,7168,2048,128,2048)
PURPOSE='controller correctness smoke only; not energy comparison, formal evidence or capacity certification'


def workload():
    return dict(schema=1,dataset='controller_smoke',load='diagnostic',split='development',seed=77,
        purpose=PURPOSE,rate=.2,duration_s=35,
        requests=[dict(arrival_s=i*5.,prompt_len=n,output_len=64) for i,n in enumerate(INPUTS)],
        prompts=[([9707,1879,13]*(n//3+1))[:n] for n in INPUTS])


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def generate(manifest,out):
    """Validate explicit evidence and write configurations without hardware access."""
    out=Path(out).resolve()
    if out.exists(): raise ValueError('refusing to overwrite controller smoke preparation')
    instances=[spec(i) for i in manifest['instances']]
    initial=[spec(i) for i in manifest.get('initial_instances',manifest['instances'])]
    validate_layout(instances,range(8));validate_layout(initial,range(8))
    if len(instances)!=4 or any(i.tp!=1 or i.role!='mixed' for i in instances):
        raise ValueError('exactly four initial mixed TP1 instances required')
    image=manifest['image']
    if len(image)!=71 or not image.startswith('sha256:'):
        raise ValueError('immutable image digest required')
    int(image[7:],16)
    owner=Path(manifest['ownership_root']).resolve()
    runtime=Path(manifest['runtime_dir']).resolve()
    if not out.is_relative_to(owner) or not runtime.is_relative_to(owner):
        raise ValueError('output and runtime directory must be inside explicit ownership root')
    limit=manifest.get('limit_s',900)
    if not isinstance(limit,(int,float)) or not 120<limit<=900:
        raise ValueError('complete smoke deadline must be finite, above 120 and at most 900 seconds')
    root=Path(manifest['campaign_root']).resolve();budget=read_budget(root)
    if not budget.get('started_s') or limit+60>budget['remaining_s']:
        raise ValueError('smoke does not fit the original campaign deadline')
    profiles=read(manifest['profiles']);transfers=read(manifest['transfers'])
    if (profiles.get('status')!='validated_envelope' or profiles.get('engine_image')!=image
            or profiles.get('model')!='Qwen2.5-14B-Instruct'
            or any(profiles.get(k) is not True for k in ('frequency_commands_verified',
                'heldout_calibration_complete','mixed_interference_measured','resident_idle_measured',
                'instant_prefill_calibration_complete','instant_heldout_calibration_complete'))):
        raise ValueError('certified instant profiles-v2 required before smoke generation')
    if (not transfers.get('certified') or transfers.get('instant_power_costs_verified') is not True
            or transfers.get('receiver_transfer_energy_included') is not True or transfers.get('engine_image')!=image
            or not any(t['source_tp']==t['target_tp']==1 for t in transfers.get('links',[]))):
        raise ValueError('certified TP1 transfer evidence required')
    artifacts={}
    for value in (profiles,transfers):
        verify_artifacts(value['certification_artifacts']);artifacts.update(value['certification_artifacts'])
    paths=[Path(manifest[k]).resolve() for k in ('profiles','transfers','role_costs','engine_template')]
    topology=None
    if manifest.get('interconnect'):
        path=Path(manifest['interconnect']).resolve();paths.append(path)
        topology=InterconnectTopology.parse(path.read_text())
    links=[TransferCost(**link) for link in transfers['links']]
    for source,target in itertools.permutations(instances,2):
        if not any(link.validated and link.source_sha256 and link.max_input_tokens>=max(INPUTS)
                and link.profile_batch>=1 and link.matches_placement(source.gpus,target.gpus,topology) for link in links):
            raise ValueError('unmeasured smoke transfer placement: '+source.instance_id+' to '+target.instance_id)
    template=read(manifest['engine_template'])
    if template.get('model')!='/models/Qwen2.5-14B-Instruct' or template.get('max_model_len')!=8192:
        raise ValueError('fixed 14B / 8192-token engine template required')
    frequency_costs=[];frequency_evidence=[]
    for path in manifest['frequency_costs']:
        path=Path(path).resolve();frequency_costs.extend(read(path))
        raw=path.parent/'raw.json';frequency_evidence.append(str(raw));paths.extend((path,raw))
    role_path=Path(manifest['role_costs']).resolve();role_raw_path=role_path.parent/'raw.json'
    from .campaign_followup_setup import role_evidence
    roles,_=role_evidence([role_path],image,{1})
    paths.append(role_raw_path);artifacts.update(freeze_files(paths))
    store=ProfileStore.load(manifest['profiles'])
    for role in ('mixed','prefill','decode'):
        for n in set(INPUTS):
            context=n+1 if role=='prefill' else n+64
            if store.lookup(role,1,2520,n,context,1) is None:
                raise ValueError('smoke request outside certified profile coverage')
    common=dict(port=manifest.get('port',18080),model_name='Qwen2.5-14B-Instruct',
        profiles=str(Path(manifest['profiles']).resolve()),slo_ttft_s=5.,slo_tpot_s=.1,
        output_prior=64,manage_clocks=True,node_gpus=list(range(8)),power_mode='instant',
        allow_unprofiled_fallback=False,prepare_peers=True,max_pending=32,slow_topology=False,
        transfers=transfers['links'],transfer_evidence=str(Path(manifest['transfers']).resolve()),
        frequency_costs=frequency_costs,frequency_evidence=frequency_evidence,role_costs=roles,
        dynamic_pools=False,park_idle=True,distserve_prefill_batch=1,distserve_decode_batch=8)
    if manifest.get('interconnect'): common['interconnect']=str(Path(manifest['interconnect']).resolve())
    freeze=dict(files=artifacts,groups=dict(profiles=list(artifacts)),identities=dict(engine_image=image))
    verify_frozen_costs(dict(common,instances=[i.endpoint() for i in instances]),profiles,freeze)
    source=freeze_files(implementation_sources());artifacts.update(source)
    generated={out/'trace.json':workload()};configs={}
    for strategy in STRATEGIES:
        layout=[i.endpoint() for i in instances]
        if strategy=='distserve': roles_now=('prefill','decode','decode','decode')
        elif strategy.startswith('pdblend'): roles_now=('prefill','decode','decode','mixed')
        else: roles_now=('mixed',)*4
        for instance,role in zip(layout,roles_now): instance['role']=role
        config=dict(common,strategy=strategy,instances=layout,dynamic_pools=strategy=='pdblend-dynamic',
            dvfs=strategy not in ('mixed','distserve'),park_idle=strategy!='mixed')
        path=out/(strategy+'.json');generated[path]=config;configs[strategy]=str(path)
    restoration=dict(instances=[asdict(i) for i in instances],initial_instances=[asdict(i) for i in initial],
        image=image,engine_template=str(Path(manifest['engine_template']).resolve()),
        retained_weights=manifest.get('retained_weights'),ownership_root=str(owner))
    generated[out/'restoration.json']=restoration
    out.mkdir(parents=True)
    for path,value in generated.items(): write(path,value)
    artifacts.update(freeze_files(generated))
    setup=dict(status='prepared_not_executed',purpose=PURPOSE,configs=configs,trace=str(out/'trace.json'),
        instances=[i.endpoint() for i in instances],image=image,runtime_dir=str(runtime),
        restoration=str(out/'restoration.json'),limit_s=limit,source_files=source,artifacts=artifacts,
        campaign_root=str(root), original_deadline_s=budget['original_deadline_s'],
        effective_deadline_s=budget['deadline_s'], budget_revision_seq=budget['revision_seq'],
        authorization_sha256=budget['authorization_sha256'],
        limitations=['resident dynamic roles only; no claim that ROI causes a switch in this mild trace',
            'reference and preparation outside measured cells; all included in the smoke deadline',
            'generations advance and are acknowledged; restoration never rewinds a live generation'])
    write(out/'setup.json',setup)
    write(out/'campaign.json',dict(output=str(root),budget_s=budget['limit_s'],stages=[
        dict(name='controller-smoke-'+hashlib.sha256(str(out).encode()).hexdigest()[:8],gpu=True,limit_s=limit,
            argv=[sys.executable,'-m','ecopadg.serving.controller_smoke','run','--setup',str(out),'--out',str(out/'run')])]))
    return setup


async def reset_roles(profiler,instances):
    """Drain, issue next generation, and require the scheduler's actual ack."""
    async def reset(instance):
        deadline=time.monotonic()+10
        while True:
            before=await profiler.call(instance,'/runtime')
            if not any(before.get(k) for k in ('active','running','waiting','kv_allocations','transfer_allocations')): break
            if time.monotonic()>deadline: raise RuntimeError('cannot reset non-drained instance '+instance['id'])
            await asyncio.sleep(.05)
        await profiler.control(instance,role=instance['role'],mode='continuous',admit_prefill=True,admit_decode=True)
        after=await profiler.call(instance,'/runtime')
        if (after.get('generation')!=before['generation']+1
                or after.get('acknowledged_generation')!=after['generation']
                or after.get('role')!=instance['role'] or after.get('mode')!='continuous'
                or not after.get('admit_prefill') or not after.get('admit_decode') or not after.get('accepting')
                or after.get('runtime_error') or after.get('error')):
            raise RuntimeError('role restoration unconfirmed: '+instance['id'])
        return dict(instance_id=instance['id'],before=before,after=after)
    return await asyncio.gather(*(reset(i) for i in instances))


async def engine_sources(profiler,setup):
    records=await profiler.provenance();expected=current_engine_sources()
    by_id={i['id']:i for i in setup['instances']}
    if len(records)!=4 or {r.get('instance_id') for r in records}!=set(by_id):
        raise RuntimeError('missing engine provenance')
    for record in records:
        instance=by_id[record['instance_id']]
        if (record.get('image_id')!=setup['image'] or record.get('source_files_at_import')!=expected
                or record.get('tp')!=1 or record.get('engine_version')!='0.9.2'
                or record.get('dtype')!='bfloat16' or record.get('max_model_len')!=8192
                or record.get('model')!='/models/Qwen2.5-14B-Instruct'
                or record.get('cuda_visible_devices')!=','.join(map(str,instance['gpus']))):
            raise RuntimeError('engine source, image or physical configuration changed')
    return records


async def references(session,instances):
    async def per_instance(instance):
        results={}
        for n in sorted(set(INPUTS)):
            prompt=([9707,1879,13]*(n//3+1))[:n]
            output=await send_request(session,instance['url'],'Qwen2.5-14B-Instruct',prompt,64,
                request_id=f'smoke-reference-{instance["id"]}-{n}')
            if (not output['success'] or output['input_tokens']!=n or output['generated_tokens']!=64
                    or len(output['token_ids'])!=64 or not output['token_events_exact']):
                raise RuntimeError('ordinary engine reference failed')
            results[str(n)]=output['token_ids']
        return results
    values=await asyncio.gather(*(per_instance(i) for i in instances))
    if any(value!=values[0] for value in values[1:]):
        raise RuntimeError('ordinary TP1 engine references differ')
    return dict(tokens=values[0],per_instance={i['id']:v for i,v in zip(instances,values)})


async def clock_observer(stop,rows):
    """Read-only actual clocks; the cell remains the only clock writer."""
    backend=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
    while not stop.is_set():
        values=await asyncio.to_thread(lambda:[backend.current_freq(g) for g in range(8)])
        rows.append([time.time(),values])
        try: await asyncio.wait_for(stop.wait(),.05)
        except asyncio.TimeoutError: pass


def audit_cell(summary,rows,reference,events,clock_samples):
    by_id={str(r['request_id']):r for r in rows}
    matches=[];exact=[]
    for index,n in enumerate(INPUTS):
        row=by_id.get(str(index),{})
        matches.append(bool(row and int(row['success'])==1 and int(row['input_tokens'])==n
            and int(row['generated_tokens'])==64 and int(row['token_ids_verified'])==1
            and row['output_token_sha256']==token_hash(reference['tokens'][str(n)])))
        exact.append(bool(row and int(row['token_itl_exact'])==1 and len(json.loads(row['token_itl_s']))==63))
    denominator=(len(rows)==len(by_id)==8 and summary.get('n_expected')==8
        and summary.get('completed')==sum(int(r['success']) for r in rows)
        and isinstance(summary.get('slo_attainment'),(int,float))
        and math.isclose(summary['slo_attainment'],sum(int(r['slo_ok']) for r in rows)/8,abs_tol=1e-12))
    observed=[dict(at_s=b[0],before=a[1],after=b[1]) for a,b in zip(clock_samples,clock_samples[1:]) if a[1]!=b[1]]
    clocks=dict(source='NVML current SM clock; changes include idle clock drops',samples=len(clock_samples),
        observed_changes=observed,admission_outcomes=[c for e in events for c in e.get('clock_outcomes',[])],
        confirmed_periodic_plans=[e for e in events if e.get('kind')=='frequency_epoch'])
    passed=(all(matches) and all(exact) and denominator and summary.get('validity')=='ok'
        and summary.get('gpu_count')==8 and summary.get('generated_tokens')==512
        and summary.get('split')=='development' and summary.get('formal_eligible') is False
        and instant_power_verified(summary) and bool(clock_samples)
        and all(len(values)==8 and all(isinstance(f,(int,float)) and math.isfinite(f) and f>0 for f in values)
                for _,values in clock_samples) and not summary.get('runtime_error'))
    return dict(passed=passed,output_matches=matches,exact_token_intervals=exact,
        failure_denominator_verified=denominator,power_source_verified=instant_power_verified(summary),
        power_source={key:summary.get(key) for key in ('power_mode','power_source_id','power_field_id')},
        actual_clock_actions=clocks,role_commits=[e for e in events if e.get('kind')=='role_commit'],
        joint_slo_attainment=summary.get('slo_attainment'),slo_is_diagnostic_only=True)


async def run(setup_dir,out):
    setup_dir=Path(setup_dir);out=Path(out);setup=read(setup_dir/'setup.json')
    verify_artifacts(setup['artifacts'])
    if freeze_files(implementation_sources())!=setup['source_files']:
        raise ValueError('implementation inventory changed after smoke preparation')
    campaign_root=setup.get('campaign_root') or read(setup_dir/'campaign.json')['output']
    remaining=min(setup['limit_s'],read_budget(campaign_root)['remaining_s']-60)
    if remaining<=120: raise TimeoutError('insufficient existing campaign budget for smoke and restoration')
    out.mkdir(parents=True,exist_ok=False)
    report=dict(purpose=PURPOSE,status='invalid_or_incomplete',passed=False,formal_eligible=False,
        capacity_certified=False,cells={},errors=[],started_s=time.time(),limit_s=remaining)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60),trust_env=False) as session:
        profiler=HardwareProfiler(session,{i['id']:i for i in setup['instances']},setup['runtime_dir'])
        async def work():
            for strategy in STRATEGIES:
                config=read(setup['configs'][strategy])
                prepared=out/(strategy+'.preparation')
                await restore_layout(read(setup['restoration']),prepared)
                reset=await reset_roles(profiler,config['instances'])
                before=await engine_sources(profiler,setup)
                if strategy=='mixed':
                    report['ordinary_reference']=await references(session,setup['instances'])
                    write(out/'references.json',report['ordinary_reference'])
                cell_out=out/strategy;clock_samples=[];stop=asyncio.Event()
                observer=asyncio.create_task(clock_observer(stop,clock_samples))
                try:
                    summary=await run_cell(SimpleNamespace(config=Path(setup['configs'][strategy]),trace=Path(setup['trace']),
                        out=cell_out,strategy=None,split='development',dataset='controller_smoke',load='diagnostic',seed=77,
                        freeze=None,mechanisms=None,timeout=60,slo_ttft_s=None,slo_tpot_s=None))
                finally:
                    stop.set();await observer
                    write(out/(strategy+'.clocks.json'),clock_samples)
                with (cell_out/'bench.csv').open() as handle: rows=list(csv.DictReader(handle))
                events=[json.loads(line) for line in (cell_out/'control.jsonl').read_text().splitlines() if line]
                result=audit_cell(summary,rows,report['ordinary_reference'],events,clock_samples)
                result.update(provenance_before=before,provenance_after=await engine_sources(profiler,setup),
                    reset_confirmations=reset,summary_path=str(cell_out/'summary.json'))
                report['cells'][strategy]=result;write(out/'summary.json',report)
                verify_artifacts(setup['artifacts'])
                if freeze_files(implementation_sources())!=setup['source_files'] or not result['passed']:
                    raise RuntimeError('controller smoke failed: '+strategy)
        try:
            await asyncio.wait_for(work(),remaining-60)
        except BaseException as exc:
            report['errors'].append(type(exc).__name__+': '+str(exc))
        finally:
            write(out/'summary.json',report)
            try:
                report['final_mixed_restoration']=await asyncio.wait_for(reset_roles(profiler,setup['instances']),45)
                verify_artifacts(setup['artifacts'])
                if freeze_files(implementation_sources())!=setup['source_files']:
                    raise RuntimeError('source inventory changed before final restoration')
            except BaseException as exc:
                report['errors'].append('final mixed restoration failed: '+repr(exc))
            report['finished_s']=time.time()
            report['passed']=not report['errors'] and set(report['cells'])==set(STRATEGIES) and all(c['passed'] for c in report['cells'].values())
            if report['passed']: report['status']='controller_smoke_passed'
            write(out/'summary.json',report)
    if not report['passed']: raise RuntimeError('controller smoke incomplete; inspect '+str(out/'summary.json'))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('generate');prepare.add_argument('--manifest',type=Path,required=True);prepare.add_argument('--out',type=Path,required=True)
    execute=sub.add_parser('run');execute.add_argument('--setup',type=Path,required=True);execute.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='generate': result=generate(read(args.manifest),args.out)
    else:
        async def cancellable():
            loop=asyncio.get_running_loop();loop.add_signal_handler(signal.SIGTERM,asyncio.current_task().cancel)
            try:return await run(args.setup,args.out)
            finally:loop.remove_signal_handler(signal.SIGTERM)
        with node_lease(): result=asyncio.run(cancellable())
    print(json.dumps(dict(status=result['status'],passed=result.get('passed',False))),flush=True)


if __name__=='__main__': main()
