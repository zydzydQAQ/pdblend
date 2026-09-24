#!/usr/bin/env python3
"""Prepare an immutable PDblend-only native timing job. Never enqueue."""
from __future__ import annotations
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pdblend.profile.collection.native_timing_plan import binding,build_plan,digest,read_bound

BASE=ROOT/'results/2026-09-24/resident-comparison-eco-v3/sources/d7d9372cbfb57f21830a342448db9c14a3b5b2ca094665bb41298ddb2e21f5c3'
VERIFY=ROOT/'results/2026-09-22/three-model/profile-receipts/model-verification-99fabb0721f21aa50eb2a8518877acdf05cc76df32f0f769900be7b4d4471fc8.json'
IMAGE='sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc'
OVERLAYS=tuple('pdblend/profile/collection/native_timing_'+name+'.py' for name in ('plan','audit','collect','worker','single_pass'))
CYCLE_OVERLAYS=(
    'pdblend/profile/collection/native_serving_cycles.py',
    'pdblend/profile/collection/native_serving_cycles_audit.py',
    'pdblend/profile/query/native_cycle_model.py',
    'pdblend/profile/query/native_query_replay.py',
    'pdblend/profile/query/native_power_components.py',
    'pdblend/profile/collection/native_power_audit.py',
    'pdblend/profile/collection/native_timing_plan_v2.py',
    'pdblend/profile/collection/native_timing_replay.py',
)
LAYOUT_OVERLAYS=(
    'pdblend/profile/collection/native_layout_energy.py',
    'pdblend/profile/collection/native_layout_stage.py',
    'pdblend/profile/query/native_layout_model.py',
    'pdblend/profile/query/native_layout_profile.py',
    'pdblend/profile/query/native_query_replay.py',
    'pdblend/planner/native_layout.py',
    'pdblend/profile/collection/native_serving_cycles.py',
    'pdblend/profile/collection/native_serving_cycles_audit.py',
)
STAGE_OVERLAYS=(
    'pdblend/profile/collection/native_timing_stage.py',
    'pdblend/profile/collection/native_layout_stage.py',
)
FREQUENCY_OVERLAYS=('pdblend/profile/collection/native_frequency_domain.py',)
FREQUENCY_QUERY_OVERLAYS=('pdblend/profile/query/native_timing.py','pdblend/profile/query/native_composition.py')
FREQUENCY_RUNTIME_OVERLAYS=('pdblend/profile/collection/native_runtime_topology.py',)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:stream.write(json.dumps(value,sort_keys=True,indent=2)+'\n')


def _copy_module(name,staging):
    destination=staging/name;destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/'src'/name,destination)


def _close_imports(staging,seeds):
    """Close project imports, preserving every already-frozen base module.

    Parse imports without importing workers or CUDA. Missing dependencies are
    copied from the workspace; existing base/runtime bytes are never replaced
    unless they were named explicitly in the reviewed overlay list.
    """
    added=set();seen=set();pending=list(seeds)
    def locate(module):
        if module.split('.')[0] not in ('pdblend','pdblend_runtime','pdblend_baselines'):return None
        for name in (module.replace('.','/')+'.py',module.replace('.','/')+'/__init__.py'):
            if (staging/name).is_file() or (ROOT/'src'/name).is_file():return name
        return None
    def include(module):
        name=locate(module)
        if name is None:return
        if not (staging/name).exists():_copy_module(name,staging);added.add(name)
        if name not in seen:pending.append(name)
        parts=module.split('.')
        for index in range(1,len(parts)):
            parent=locate('.'.join(parts[:index]))
            if parent and parent.endswith('/__init__.py'):
                if not (staging/parent).exists():_copy_module(parent,staging);added.add(parent)
                if parent not in seen:pending.append(parent)
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name);module=name.removesuffix('.py').replace('/','.')
        package=module.removesuffix('.__init__') if module.endswith('.__init__') else module.rsplit('.',1)[0]
        for node in ast.walk(ast.parse((staging/name).read_text(),filename=name)):
            if isinstance(node,ast.Import):
                for alias in node.names:include(alias.name)
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    parts=package.split('.');prefix='.'.join(parts[:len(parts)-node.level+1])
                    owner='.'.join(x for x in (prefix,node.module) if x)
                else:owner=node.module or ''
                include(owner)
                for alias in node.names:
                    if alias.name!='*':include(owner+'.'+alias.name)
    return tuple(sorted(added)),tuple(sorted(seen))


def cycle_bindings(path,timing_plan):
    """Bind exactly the raw non-evaluation inputs used by cycle-plan replay."""
    from pdblend.profile.collection.native_serving_cycles import validate_cycle_plan
    return _supplement_bindings(path,timing_plan,validate_cycle_plan,'request-cycle')


def layout_bindings(path,timing_plan):
    from pdblend.profile.collection.native_layout_energy import validate_layout_plan
    return _supplement_bindings(path,timing_plan,validate_layout_plan,'layout-energy')


def _supplement_bindings(path,timing_plan,validate,label):
    from pdblend.profile.collection.native_frequency_domain import require_same_domain
    original=binding(path);plan=validate(read_bound(original))
    require_same_domain(timing_plan,plan)
    provenance=timing_plan.get('query_provenance',timing_plan.get('query_bindings'))
    if (any(plan[k]!=timing_plan[k] for k in ('model_id','tp','pp'))
            or plan['query_ledger']!=timing_plan['query_ledger'] or plan['query_provenance']!=provenance):
        raise ValueError(label+' model/topology/ledger/provenance differs from timing plan')
    references={}
    def add(ref):
        value=read_bound(ref);key=str(Path(ref['path']).resolve())
        if key in references and references[key]['sha256']!=ref['sha256']:
            raise ValueError(label+' raw path has inconsistent checksums')
        references[key]=dict(path=key,sha256=ref['sha256']);return value
    add(original);add(plan['query_ledger']);provenance=add(plan['query_provenance'])
    owned=provenance['inputs'][plan['model_id'].split('-')[1].lower()]
    add(owned['rate_anchor'])
    for refs in owned['datasets'].values():
        add(refs['confirmation']);add(refs['tuning_trace'])
    if owned.get('longbench_recovery'):
        recovery=owned['longbench_recovery'];value=add(recovery['completion']);add(recovery['preflight'])
        root=Path(recovery['completion']['path']).resolve().parent
        for row in value['candidates']:
            for phase in ('calibration','tuning'):
                if phase not in row:continue
                receipt=row[phase];relative=Path(receipt['path'])
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('recovery raw escapes its owner')
                measured=add(dict(path=str(root/relative),sha256=receipt['sha256']))
                add(dict(path=str((root/relative).parent/'requests.json'),sha256=measured['trace_sha256']))
    for point in plan['points']:add(point['parent_trace'])
    return plan,original,[references[k] for k in sorted(references)]


def prepare(out,ledger,bindings,*,collect_runtime=False,collect_power_pilot=False,point_plan=None,
            request_cycle_plan=None,source_base=None,layout_energy_plan=None,timing_first=False):
    out=Path(out).resolve();source_base=Path(source_base).resolve() if source_base is not None else BASE
    if out.exists():raise FileExistsError('new immutable preparation directory required')
    if point_plan is not None:
        from pdblend.profile.collection.native_timing_plan_v2 import validate_plan
        from pdblend.profile.collection import native_timing_single_pass as single_pass
        candidate=json.loads(Path(point_plan).read_text())
        plan=(single_pass.validate_plan(candidate) if single_pass.is_single_pass(candidate) else validate_plan(candidate))
    else:plan=build_plan(binding(ledger),binding(bindings))
    development=plan.get('schema')=='pdblend-native-timing-development-plan/v1'
    if development and any((collect_runtime,collect_power_pilot,request_cycle_plan,layout_energy_plan,timing_first)):
        raise ValueError('single-pass development only collects timing; no legacy stage or supplements')
    v2=point_plan is not None
    if type(timing_first) is not bool or (timing_first and not v2):
        raise ValueError('independent timing stage requires explicit v2 timing_first')
    from pdblend.profile.collection.native_frequency_domain import validate_collection_inputs
    frequency_inputs={key:plan[key] for key in ('frequency_domain_ref','frequency_domain','frequency_domain_sha256') if key in plan}
    runtime_inputs={}
    if collect_runtime:
        from pdblend.profile.collection.native_runtime_collect import build_runtime_plan
        runtime_plan=build_runtime_plan(frequency_inputs.get('frequency_domain_ref'))
        runtime_inputs.update(collect_runtime=True,runtime_plan=runtime_plan)
        if frequency_inputs:
            runtime_inputs.update(runtime_include_transfer=False,runtime_scope=runtime_plan['scope'])
    validate_collection_inputs(plan,dict(model_id=plan['model_id'],tp=plan.get('tp',1),pp=plan.get('pp',1),
        timing_first=timing_first,**frequency_inputs,**runtime_inputs),collect_runtime=collect_runtime,
        power_pilot=collect_power_pilot,request_cycles=bool(request_cycle_plan),layout_energy=bool(layout_energy_plan))
    cycle=cycle_bindings(request_cycle_plan,plan) if request_cycle_plan is not None else None
    layout=layout_bindings(layout_energy_plan,plan) if layout_energy_plan is not None else None
    if layout and (not collect_runtime or not v2 or cycle or collect_power_pilot):
        raise ValueError('layout-energy requires runtime and v2 timing, without legacy pilot/cycles')
    module_spec=importlib.util.spec_from_file_location('pd_timing_freezer',ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    helper=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(helper)
    helper.verify_snapshot(source_base,json.loads((source_base/'manifest.json').read_text())['files'])
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='pd-native-timing-source-') as tmp:
        staging=Path(tmp)/'source';shutil.copytree(source_base,staging)
        overlays = OVERLAYS + (tuple('pdblend/profile/collection/native_runtime_'+name+'.py'
                                     for name in ('collect','audit','topology')) if collect_runtime or collect_power_pilot or cycle or layout else ())
        if v2:
            overlays += tuple('pdblend/profile/collection/native_timing_'+name+'.py'
                              for name in ('plan_v2','capacity','replay','replay_v2'))
        if development:overlays+=('pdblend/profile/collection/native_timing_single_pass.py',)
        if collect_power_pilot:
            overlays += tuple('pdblend/profile/collection/native_power_'+name+'.py'
                              for name in ('plan','collect','audit'))
        if cycle:overlays+=CYCLE_OVERLAYS
        if layout:overlays+=LAYOUT_OVERLAYS
        if timing_first:overlays+=STAGE_OVERLAYS
        # Collector now imports the explicit revision contract even on its
        # unchanged legacy path; bind the exact module in every new snapshot.
        overlays+=FREQUENCY_OVERLAYS
        if frequency_inputs:overlays+=FREQUENCY_QUERY_OVERLAYS
        overlays=tuple(dict.fromkeys(overlays))
        for name in overlays:_copy_module(name,staging)
        dependencies,import_closure=_close_imports(staging,overlays)
        source,source_sha=helper.freeze_source(staging,out/'sources')
    write(out/'point-plan.json',plan)
    inputs=dict(schema=single_pass.INPUT_SCHEMA if development else ('pdblend-native-timing-inputs/v2' if v2 else 'pdblend-native-timing-inputs-v1'),point_plan=binding(out/'point-plan.json'),
        source_manifest=binding(source/'manifest.json'),model_verification=binding(VERIFY),image_digest=IMAGE,
        model_id=plan['model_id'],system='pdblend',source_sha256=source_sha)
    if development:inputs.update(qualification_level=single_pass.LEVEL, original_design_qualified=False)
    from pdblend.profile.collection.native_timing_collect import resident_phase_order
    inputs.update(timing_first=timing_first, phase_order=resident_phase_order(
        collect_runtime=collect_runtime, power_pilot=collect_power_pilot,
        request_cycles=bool(cycle), layout_energy=bool(layout), timing_first=timing_first))
    if frequency_inputs:inputs.update(tp=plan['tp'],pp=plan['pp'],**frequency_inputs)
    inputs.update(runtime_inputs)
    if collect_power_pilot:
        from pdblend.profile.collection.native_power_plan import build_plan as build_power_plan
        pilot=build_power_plan(plan['query_ledger'],plan.get('query_provenance',plan.get('query_bindings')),model_id=plan['model_id'])
        write(out/'power-pilot-plan.json',pilot)
        inputs['power_pilot_plan']=binding(out/'power-pilot-plan.json')
    if cycle:
        cycle_plan,original,raw_refs=cycle
        shutil.copyfile(original['path'],out/'request-cycle-plan.json')
        frozen=binding(out/'request-cycle-plan.json')
        if frozen['sha256']!=original['sha256']:raise ValueError('request-cycle input changed during freezing')
        inputs.update(request_cycle_plan=frozen,request_cycle_original_plan=original,request_cycle_raw_inputs=raw_refs)
    if layout:
        layout_plan,original,raw_refs=layout
        shutil.copyfile(original['path'],out/'layout-energy-plan.json')
        frozen=binding(out/'layout-energy-plan.json')
        if frozen['sha256']!=original['sha256']:raise ValueError('layout-energy input changed during freezing')
        inputs.update(layout_energy_plan=frozen,layout_energy_original_plan=original,layout_energy_raw_inputs=raw_refs)
    write(out/'inputs.json',inputs)
    size=plan['model_id'].split('-')[1].lower()
    job_id='pdblend-native-timing-'+size+'-'+digest(inputs)[:16]
    argv=['docker','run','--rm','--name',job_id,'--gpus','all','--cap-add','SYS_ADMIN','--ipc=host',
          '--network=host','--shm-size=16g','--ulimit','nofile=65536:65536','--entrypoint','/opt/venv/bin/python']
    mount_rows=[(str(source),'/opt/pdblend-src','ro'),(str(ROOT),str(ROOT),'ro'),
        ('/home/models','/models','ro'),('/tmp/pdblend-physical-clock-owners','/tmp/pdblend-physical-clock-owners','rw'),
        ('{attempt_dir}','{attempt_dir}','rw')]
    if not out.is_relative_to(ROOT):mount_rows.append((str(out),str(out),'ro'))
    if frequency_inputs:
        domain_path=Path(frequency_inputs['frequency_domain_ref']['path']).resolve()
        if not domain_path.is_relative_to(ROOT) and not domain_path.is_relative_to(out):
            mount_rows.append((str(domain_path),str(domain_path),'ro'))
    for supplement in (cycle,layout):
        for ref in supplement[2] if supplement else ():
            path=Path(ref['path'])
            if not path.is_relative_to(ROOT) and not path.is_relative_to(out):mount_rows.append((str(path),str(path),'ro'))
    for host,target,mode in dict.fromkeys(mount_rows):argv+=['-v',f'{host}:{target}:{mode}']
    env=dict(PYTHONPATH='/opt/pdblend-src',PYTHONDONTWRITEBYTECODE='1',PDBLEND_MODELS_DIR='/models',
        PDBLEND_MODEL_VERIFICATION_RECEIPT=str(VERIFY),PDBLEND_SOURCE_SHA256=source_sha,
        PDBLEND_SOURCE_MANIFEST=str(source/'manifest.json'),PDBLEND_IMAGE_ID=IMAGE,
        PDBLEND_GPU_UUIDS='{lease_gpu_uuids}',PDBLEND_CLOCK_LOCK_DIR='/tmp/pdblend-physical-clock-owners',
        PDBLEND_CONCURRENCY_ENVIRONMENT='{attempt_dir}/concurrency-environment.json',
        VLLM_USE_V1='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',NCCL_CUMEM_ENABLE='0',NCCL_IB_DISABLE='1',
        NCCL_P2P_DISABLE='0',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    for key,value in env.items():argv+=['-e',key+'='+value]
    argv += [IMAGE,'-B','-m','pdblend.profile.collection.native_timing_collect','--model','/models/'+plan['model_id'],
        '--gpus','{lease_local_indices}','--base-port','{lease_port}','--point-plan',str(out/'point-plan.json'),
        '--input-manifest',str(out/'inputs.json'),'--out','{attempt_dir}/native-timing']
    if collect_runtime:argv+=['--collect-runtime']
    if timing_first:argv+=['--timing-first']
    if collect_power_pilot:argv+=['--power-pilot-plan',str(out/'power-pilot-plan.json')]
    if cycle:argv+=['--request-cycle-plan',str(out/'request-cycle-plan.json')]
    if layout:argv+=['--layout-energy-plan',str(out/'layout-energy-plan.json')]
    job=dict(job_id=job_id,priority=800 if collect_runtime or collect_power_pilot or cycle or layout else 690,max_attempts=1,payload=dict(argv=argv,cwd=str(ROOT),container_name=job_id,
        gpu_count=8,exclusive=True,reserve_host=True,timeout_s=14400,required_receipts=['native-timing/completion.json'],
        system='pdblend',scope=plan['scope'],source_sha256=source_sha,image_digest=IMAGE,execution_ready=True,
        formal_eligible=False,energy_comparable=False,model_id=plan['model_id'],input_manifest=binding(out/'inputs.json')))
    write(out/'jobs.json',[job])
    report=dict(schema='pdblend-native-timing-prepared-v1',enqueued=False,hardware_executed=False,formal_eligible=False,
        source_manifest=binding(source/'manifest.json'),base_manifest=binding(source_base/'manifest.json'),
        overlays={name:binding(ROOT/'src'/name) for name in overlays},point_plan=binding(out/'point-plan.json'),
        dependency_overlays={name:binding(ROOT/'src'/name) for name in dependencies},source_import_closure=list(import_closure),
        jobs=binding(out/'jobs.json'),builder=binding(__file__),training_points=sum(p['purpose']=='training' for p in plan['points']),
        holdout_points=sum(p['purpose']=='holdout' for p in plan['points']),measurement_windows=sum(p.get('repeats',3) for p in plan['points']),
        interference_windows=0 if development else 12*(8//plan.get('tp',1)),
        minimum_sampling_wall_s=(0 if development else 6*(8//plan.get('tp',1))*7+6*7)+sum(p.get('repeats',3) for p in plan['points'])/(8//plan.get('tp',1))*7,
        timing_only=not (collect_runtime or collect_power_pilot or cycle or layout),collect_runtime=collect_runtime,
        collect_power_pilot=collect_power_pilot,
        collect_request_cycles=bool(cycle),
        collect_layout_energy=bool(layout),
        timing_first=timing_first,phase_order=inputs['phase_order'],
        power_expansion_qualified=False,remaining_gates=plan['required_remaining'])
    if development:
        report.update(qualification_level=single_pass.LEVEL, original_design_qualified=False,
                      parallel_qualified=False, component_qualified=False)
    if frequency_inputs:report.update(**frequency_inputs)
    if cycle:
        report.update(request_cycle_plan=inputs['request_cycle_plan'],request_cycle_original_plan=inputs['request_cycle_original_plan'],
            request_cycle_raw_inputs=inputs['request_cycle_raw_inputs'],request_cycle_windows=len(cycle[0]['points']),
            request_cycle_service_wall_s=sum(p['duration_s'] for p in cycle[0]['points']),
            request_cycle_scope='bounded_all_M_60s_request_cycles_not_formal_or_pure_kernel_power')
        report['minimum_sampling_wall_s']+=report['request_cycle_service_wall_s']
    if layout:
        report.update(layout_energy_plan=inputs['layout_energy_plan'],
            layout_energy_original_plan=inputs['layout_energy_original_plan'],
            layout_energy_raw_inputs=inputs['layout_energy_raw_inputs'],layout_energy_windows=len(layout[0]['points']),
            layout_energy_service_wall_s=sum(p['duration_s'] for p in layout[0]['points']),
            layout_energy_scope='bounded_Poisson_M4_TP2_component_not_online_PD_qualification')
        report['minimum_sampling_wall_s']+=report['layout_energy_service_wall_s']
    write(out/'manifest.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    base=ROOT/'results/2026-09-24/pdblend-offline-readiness/ledger-v2'
    parser.add_argument('--ledger',type=Path,default=base/'query-ledger.json')
    parser.add_argument('--bindings',type=Path,default=base/'bindings.json')
    parser.add_argument('--collect-runtime',action='store_true')
    parser.add_argument('--timing-first',action='store_true',help='Freeze a v2 timing substage before optional energy supplements')
    parser.add_argument('--collect-power-pilot',action='store_true')
    parser.add_argument('--point-plan',type=Path,help='Explicit predeclared model-owned v2 plan')
    parser.add_argument('--request-cycle-plan',type=Path,help='Immutable same-model/ledger 60s train/freeze/holdout cycle plan')
    parser.add_argument('--layout-energy-plan',type=Path,help='Opt-in 32B Poisson M4/TP2 whole-layout component plan')
    parser.add_argument('--source-base',type=Path,help='Explicit verified source snapshot; defaults to historical preparation base')
    args=parser.parse_args();print(json.dumps(prepare(args.out.resolve(),args.ledger,args.bindings,
        collect_runtime=args.collect_runtime,collect_power_pilot=args.collect_power_pilot,point_plan=args.point_plan,
        request_cycle_plan=args.request_cycle_plan,source_base=args.source_base,
        layout_energy_plan=args.layout_energy_plan,timing_first=args.timing_first),indent=2))
