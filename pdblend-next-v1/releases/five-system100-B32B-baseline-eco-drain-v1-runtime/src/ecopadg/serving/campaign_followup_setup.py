"""CPU preparation and explicit sequential continuation after profile validation.

Templates contain paths and experimental intentions, never synthetic costs or
capacities. Only ``execute`` runs prepared GPU stages; all other actions are
CPU-only. The continuation retains the original Campaign deadline.
"""
import argparse
from copy import deepcopy
import itertools
import json
import math
from pathlib import Path
import sys
import time


from .budget import read_budget
DATASETS=('alpaca','sharegpt','longbench')
BASELINES=('mixed','mixed_dvfs','distserve','ecoserve','dynamollm')
IMAGE='sha256:d11407cd827a43a0dec8ad7d4d7037c97c39bbe93c6f4b4fd951c94e67509a8b'


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def templates(root,out,*,calibration_budget_s=21660,method_budget_s=18000):
    root=Path(root).resolve();out=Path(out).resolve()
    if out.exists():raise ValueError('refusing to overwrite followup templates')
    ceiling=read_budget(root)['limit_s'] if (root/'budget.json').is_file() else 86400
    if (any(not math.isfinite(v) or v<=0 for v in (calibration_budget_s,method_budget_s))
            or calibration_budget_s+method_budget_s>ceiling):
        raise ValueError('positive followup allocations must fit the authorized campaign envelope')
    def p(name):return str(root/name)
    calibration=dict(campaign_root=str(root),image=IMAGE,profiles=p('profiles.certified-v2.json'),
        transfers=p('transfers.certified-v2.json'),search=p('configuration-search-v2.json'),
        interconnect=p('interconnect.txt'),corpus=p('corpus-v1'),engine_template=p('i3.json'),
        retained_weights=p('weights/aeabfaf47f4941e6ba56d5e20d27d055'),
        weights_evidence=p('retained-weights-full-validation-v2.json'),
        frequency_costs=[p(f'clock-profiling-tp{tp}-instant/frequency_costs.json') for tp in (1,2,4,8)],
        topology_costs=p('validate-physical-transitions/topology_costs.json'),
        initial_instances=[],initial_layout_evidence=p('validate-physical-transitions/raw.json'),
        calibration_budget_s=calibration_budget_s,preparation_stage_limit_s=600,calibration_stage_limit_s=1200,
        max_candidates_per_pair=1,max_trials=7,probe_requests=64,target=.99,slo_ttft_s=5,slo_tpot_s=.1,
        decision_budget_s=.02,
        entries=[dict(system=s,dataset=d,candidate_index=0) for d in DATASETS for s in BASELINES])
    methods=dict(campaign_root=str(root),corpus=p('corpus-v1'),method_budget_s=method_budget_s,
        calibration=str(out/'calibration/summary.json'),
        role_costs=[p(f'role-profiling-tp{tp}-instant/role_costs.json') for tp in (1,2,4)],
        capacity_measurements=[p(f'prepare-validation-tp{tp}/startup.json') for tp in (1,2,4,8)],
        layout=dict(tp=1,prefill=1,decode=4,mixed=3),slow_topology=True,seeds=[11,22],
        points=[dict(dataset=d,load='medium',fraction=.6) for d in DATASETS],
        requests=64,slo_min=.99,max_slo_drop=.01,
        request_timeout_s=60,restoration_allowance_s=60,control_source_dataset='longbench')
    write(out/'calibration.template.json',calibration);write(out/'methods.template.json',methods)
    continuation=dict(status='template_only_no_measurements_or_calibration',campaign_root=str(root),
        calibration_template=str(out/'calibration.template.json'),methods_template=str(out/'methods.template.json'),
        calibration_out=str(out/'calibration'),method_out=str(out/'methods'),
        purpose='explicit fifteen baseline candidates, then 78 fully paired development cells; no formal rule changes')
    write(out/'followup.json',continuation)
    return continuation


def initial_layout(manifest):
    """Only explicit prior experiment evidence can authorize initial engines."""
    from .calibration_setup import spec
    from .topology import validate_layout
    if manifest.get('initial_instances'):
        values=manifest['initial_instances']
    else:
        evidence=read(manifest['initial_layout_evidence'])
        if evidence.get('passed') is not True or evidence.get('complete') is not True or not evidence.get('live_instances'):
            raise ValueError('prior physical layout evidence is not complete and passed')
        values=evidence['live_instances']
    instances=[spec(v) for v in values];validate_layout(instances,range(8))
    return [s.endpoint() for s in instances]


def prepare_calibration(followup):
    from .calibration_setup import generate
    manifest=read(followup['calibration_template'])
    wanted={(s,d) for s in BASELINES for d in DATASETS}
    if {(e['system'],e['dataset']) for e in manifest['entries']}!=wanted:
        raise ValueError('followup requires all fifteen independent baseline/dataset pairs')
    manifest['initial_instances']=initial_layout(manifest)
    # Explicit entry.initial_rate survives the core generator's candidate
    # projection. Omitting it uses 60% of the measured-model predicted rate.
    return generate(manifest,Path(followup['calibration_out']))


def role_evidence(paths,image,reachable_tps):
    from .evidence import sha256
    from .role_profiling import costs as verified_costs
    costs=[];proofs=[]
    for path in paths:
        rows=read(path);raw_path=Path(path).parent/'raw.json';raw=read(raw_path);digest=sha256(raw_path)
        # Cost calibration and full failure/output correctness have separate
        # artifacts. Do not promote legacy average short-window joules, or
        # falsely require a no-generation calibration to prove token equality.
        provenance=raw.get('provenance_before',[])
        if not provenance or any(p.get('image_id')!=image for p in provenance):
            raise ValueError('resident-role costs lack same-image instant hardware evidence')
        if rows!=verified_costs(raw,digest):
            raise ValueError('resident-role cost differs from its measured conservative bound')
        costs.extend(rows);proofs.extend((str(Path(path).resolve()),str(raw_path.resolve())))
    pairs={(r['tp'],r['source_role'],r['target_role']) for r in costs}
    required={(tp,a,b) for tp in reachable_tps for a,b in itertools.permutations(('mixed','prefill','decode'),2)}
    if not required<=pairs:raise ValueError('reachable resident role change has no measured cost')
    return costs,proofs


def measured_capacities(paths,reachable_tps):
    from .evidence import sha256
    from .pd_topology import MeasuredCapacity
    grouped={};proofs=[]
    for path in paths:
        raw=read(path)
        if not raw.get('complete') or raw.get('errors') or raw.get('sampling_error'):
            raise ValueError('incomplete hardware KV/staging capacity measurement')
        for row in raw['instances']:
            value=MeasuredCapacity(**row['measured_runtime_capacity'],source_sha256=sha256(path))
            grouped.setdefault(value.tp,[]).append(value)
        proofs.append(str(Path(path).resolve()))
    from dataclasses import asdict
    result=[]
    for tp in sorted(reachable_tps):
        candidates=grouped.get(tp,[])
        # Choose an actually measured conservative point, never TP scaling or
        # a synthetic combination whose source artifact lacks those values.
        valid=[c for c in candidates if all(c.kv_tokens<=v.kv_tokens and
            c.transfer_buffer_bytes<=v.transfer_buffer_bytes and c.transfer_bytes_per_token>=v.transfer_bytes_per_token
            for v in candidates)]
        if not valid:raise ValueError('missing or inconsistent conservative capacity at TP '+str(tp))
        result.append(asdict(valid[0]))
    return result,proofs


def choose_control(calibration,dataset):
    choices=[r for r in calibration['results'] if r.get('passed') and r['system']=='mixed_dvfs' and r['dataset']==dataset]
    if not choices:raise ValueError('no independently calibrated mixed_dvfs control at '+dataset)
    best=max(choices,key=lambda r:r['capacity_rps'])
    config=read(best['config'])
    if config.get('strategy')!='mixed_dvfs':raise ValueError('calibrated control configuration changed')
    return config,dict(dataset=dataset,config=best['config'],capacity_rps=best['capacity_rps'],
        scope='best tested mixed_dvfs layout for the declared calibration dataset, reused for development; not a global optimum')


def prepare_methods(followup):
    from . import calibration_setup,method_selection
    from .calibration import implementation_sources
    from .evidence import common_capacity,freeze_files,sha256
    from .profiles import ProfileStore
    from .planner import TransferCost
    from .interconnect import InterconnectTopology
    from .topology import InstanceSpec,validate_layout
    from .datasets import make_trace
    import random
    manifest=read(followup['methods_template']);calibration=read(manifest['calibration'])
    if not calibration.get('passed') or not calibration.get('source_unchanged'):
        raise ValueError('all independent baseline capacities must pass before method setup')
    calibration_setup.verify_artifacts(read(Path(followup['calibration_out'])/'input-evidence.json')['artifacts'])
    capacity=common_capacity(calibration['results'])
    current=freeze_files(implementation_sources())
    for result in calibration['results']:
        confirmation=result.get('confirmation') or {}
        if result.get('passed'):
            source=Path(confirmation['artifact']).parent.parent/'source.before.json'
            if read(source)!=current:raise ValueError('calibration execution sources changed before method setup')
    cal_input=read(followup['calibration_template'])
    profiles,transfers,search,template,frequency,frequency_proofs,topology_costs,_=calibration_setup.validated_inputs(cal_input)
    for result in calibration['results']:
        config=read(result['config'])
        if Path(config['profiles']).resolve()!=Path(cal_input['profiles']).resolve():
            raise ValueError('method profiles differ from independently calibrated baseline profiles')
        if config['strategy'] in ('mixed_dvfs','dynamollm') and config.get('frequency_costs')!=frequency:
            raise ValueError('method frequency costs differ from independently calibrated baseline costs')
    layout=manifest['layout']
    if (layout.get('tp')!=1 or any(type(layout.get(k)) is not int or layout[k]<1 for k in ('prefill','decode','mixed'))
            or sum(layout[k] for k in ('prefill','decode','mixed'))!=8):
        raise ValueError('first method layout requires eight real TP1 replicas across three nonempty pools')
    roles=[role for role in ('prefill','decode','mixed') for _ in range(layout[role])]
    instances=[InstanceSpec('method'+str(gpu),1,(gpu,),30000+gpu,32000+gpu*8,role).endpoint()
               for gpu,role in enumerate(roles)]
    validate_layout([calibration_setup.spec(i) for i in instances],range(8))
    topology=InterconnectTopology.parse(Path(cal_input['interconnect']).read_text())
    links=[TransferCost(**row) for row in transfers['links']]
    store=ProfileStore.load(cal_input['profiles'])
    # Shape/prior setup reads calibration records only. Development records
    # are consulted below solely to materialize the already-declared workload.
    histories=[r for dataset in DATASETS for r in read(Path(manifest['corpus'])/(dataset+'.json'))['calibration']]
    max_input=max(r['input_tokens'] for r in histories)
    max_context=max(r['input_tokens']+r['output_tokens'] for r in histories)
    for role in ('prefill','decode','mixed'):
        frequencies=[2520] if role=='prefill' else (900,1500,2100,2520)
        context=max_input+1 if role=='prefill' else max_context
        if any(store.lookup(role,1,f,max_input,context,1) is None for f in frequencies):
            raise ValueError('predeclared TP1 three-pool layout exceeds measured operator coverage')
    for source,target in itertools.product([i for i in instances if i['role']=='prefill'],
                                           [i for i in instances if i['role']=='decode']):
        if not any(t.validated and t.source_sha256 and t.max_input_tokens>=max_input and
                   t.matches_placement(source['gpus'],target['gpus'],topology) for t in links):
            raise ValueError('three-pool physical P-to-D route has no certified transfer measurement')
    slow=manifest.get('slow_topology',False)
    if type(slow) is not bool:raise ValueError('slow_topology must be explicitly boolean')
    reachable={1}|({tp for c in topology_costs for tp in c['target_tps']} if slow else set())
    roles_cost,role_proofs=role_evidence(manifest['role_costs'],cal_input['image'],reachable)
    memory,memory_proofs=measured_capacities(manifest['capacity_measurements'],reachable)
    root=Path(followup['method_out']).resolve()
    if root.exists():raise ValueError('refusing to overwrite method preparation')
    control,control_choice=choose_control(calibration,manifest['control_source_dataset'])
    priors,prior_sources=method_selection.calibration_priors(manifest['corpus'],DATASETS,calibration['results'])
    prior=priors[manifest['control_source_dataset']]
    # All variants/control receive the same per-dataset calibration history.
    # config_for_point selects that dataset's scalar before every cell.
    template=dict(template,operation_timeout_s=45,transfer_buffer_bytes=4*1024**3,verify_transport=False,
        validated_tp_pairs=sorted({(t.source_tp,t.target_tp) for t in links}))
    inputs=root/'inputs';engine_path=inputs/'engine-template.json'
    base=dict(strategy='pdblend-dynamic',port=18080,model_name='Qwen2.5-14B-Instruct',
        instances=instances,profiles=cal_input['profiles'],transfers=transfers['links'],
        transfer_evidence=cal_input['transfers'],interconnect=cal_input['interconnect'],
        slo_ttft_s=cal_input.get('slo_ttft_s',5),slo_tpot_s=cal_input.get('slo_tpot_s',.1),
        output_prior=prior,output_priors=priors,output_prior_sources=prior_sources,
        manage_clocks=True,node_gpus=list(range(8)),prepare_peers=True,park_idle=True,
        decision_budget_s=cal_input.get('decision_budget_s',.01),
        power_mode='instant',
        frequency_costs=frequency,frequency_evidence=frequency_proofs,role_costs=roles_cost,role_evidence=role_proofs,
        topology_costs=topology_costs,topology_evidence=[str(Path(cal_input['topology_costs']).parent/'raw.json')],
        measured_capacities=memory,capacity_evidence=memory_proofs,retained_weights=cal_input['retained_weights'],
        retained_weight_evidence=[cal_input['weights_evidence'],str(Path(cal_input['retained_weights'])/'manifest.json')],
        slow_topology=slow,dynamic_pools=True,allow_pd=True,dvfs=True,max_pending=256,
        allow_unprofiled_fallback=False,journal=str(root/'unused-journal.jsonl'),
        topology=dict(runtime_dir=str(root/'dynamic-runtime'),image=cal_input['image'],engine_template=str(engine_path)))
    control=deepcopy(control)
    control.update(output_prior=prior,output_priors=priors,output_prior_sources=prior_sources,
                   frequency_costs=frequency,frequency_evidence=frequency_proofs,
                   profiles=base['profiles'],node_gpus=list(range(8)),manage_clocks=True,power_mode='instant')
    if any(control.get(k)!=base[k] for k in ('slo_ttft_s','slo_tpot_s')):
        raise ValueError('control and candidate SLO definitions differ')
    points=manifest['points'];seeds=manifest.get('seeds',[11,22])
    if (len(points)!=3 or {p['dataset'] for p in points}!=set(DATASETS) or len(seeds)!=2 or len(set(seeds))!=2
            or any(type(seed) is not int or seed in (101,202,303) for seed in seeds)
            or any(not math.isfinite(p['fraction']) or not 0<p['fraction']<=1 for p in points)):
        raise ValueError('all three development datasets and two independent seeds are required')
    records={d:read(Path(manifest['corpus'])/(d+'.json'))['development'] for d in DATASETS}
    count=manifest.get('requests',64)
    if type(count) is not int or not 1<=count<=256 or any(len(values)<count for values in records.values()):
        raise ValueError('explicit development request count must be 1..256 and fit every development split')
    # This is a coverage check for the predeclared experiment, not data-driven
    # layout selection. An uncovered development shape stops preparation.
    visible=[r for values in records.values() for r in values]
    development_input=max(r['input_tokens'] for r in visible)
    development_context=max(r['input_tokens']+r['output_tokens'] for r in visible)
    for tp in reachable:
        for role in ('prefill','decode','mixed'):
            frequencies=[2520] if role=='prefill' else (900,1500,2100,2520)
            context=development_input+1 if role=='prefill' else development_context
            if any(store.lookup(role,tp,f,development_input,context,1) is None for f in frequencies):
                raise ValueError('development shape exceeds a reachable measured TP/role profile')
    if development_input>max_input:
        for source,target in itertools.product([i for i in instances if i['role']=='prefill'],
                                               [i for i in instances if i['role']=='decode']):
            if not any(t.validated and t.source_sha256 and t.max_input_tokens>=development_input and
                       t.matches_placement(source['gpus'],target['gpus'],topology) for t in links):
                raise ValueError('development shape exceeds certified physical P-to-D coverage')
    spans={}
    for point in points:
        values=[]
        for seed in seeds:
            selected=random.Random(seed).sample(records[point['dataset']],count)
            trace=make_trace(selected,capacity[point['dataset']]*point['fraction'],seed,
                dataset=point['dataset'],load=point['load'],split='development')
            values.append(trace['duration_s'])
        spans[point['dataset']]=max(values)
    restore_s=manifest.get('restoration_allowance_s',60);timeout=manifest.get('request_timeout_s',60)
    if any(not math.isfinite(v) or v<=0 for v in (restore_s,timeout,manifest['method_budget_s'])):
        raise ValueError('finite positive restoration, timeout and complete-group allocation required')
    bounded_points=[]
    for point in points:
        minimum_limit=math.ceil(spans[point['dataset']]+restore_s+timeout)
        point_limit=point.get('cell_limit_s',manifest.get('cell_limit_s',minimum_limit))
        if not math.isfinite(point_limit) or point_limit<minimum_limit:
            raise ValueError('cell budget cannot cover its fixed trace, request timeout, and declared restoration allowance')
        bounded_points.append(dict(point,cell_limit_s=point_limit))
    cell_limit=max(p['cell_limit_s'] for p in bounded_points)
    expected_cells=13*len(points)*len(seeds)
    upper=13*len(seeds)*sum(p['cell_limit_s'] for p in bounded_points)
    budget=read_budget(manifest['campaign_root'])
    remaining=budget['remaining_s']
    if upper>manifest['method_budget_s'] or manifest['method_budget_s']>remaining-60:
        raise ValueError(f'complete development paired groups require stage bounds of {upper}s; allocation/deadline insufficient')
    # Only now create concrete artifacts; insufficient time leaves templates
    # intact and does not quietly reduce examples, datasets, seeds or variants.
    write(engine_path,template);write(inputs/'pdblend.config.json',base);write(inputs/'mixed_dvfs.config.json',control)
    initial=initial_layout(cal_input)
    for result in calibration['results']:
        initial.extend(read(result['config'])['instances'])
    owned={i.get('id',i.get('instance_id')):i for i in initial}
    restore=dict(instances=instances,initial_instances=list(owned.values()),image=cal_input['image'],
        engine_template=str(engine_path),retained_weights=cal_input['retained_weights'],ownership_root=str(root.parent))
    method_manifest=dict(campaign_root=manifest['campaign_root'],calibration=manifest['calibration'],
        corpus=manifest['corpus'],base_config=str(inputs/'pdblend.config.json'),control_config=str(inputs/'mixed_dvfs.config.json'),
        points=bounded_points,seeds=seeds,requests=count,slo_min=manifest['slo_min'],max_slo_drop=manifest['max_slo_drop'],
        method_budget_s=manifest['method_budget_s'],cell_limit_s=cell_limit,request_timeout_s=timeout,restore=restore)
    path=inputs/'method.manifest.json';write(path,method_manifest)
    prepared=method_selection.prepare(path,root/'paired')
    write(root/'setup.json',dict(status='prepared_not_measured',layout=layout,expected_cells=expected_cells,
        stage_upper_bound_s=upper,requests_per_cell=count,control=control_choice,
        output_priors=priors,output_prior_sources=prior_sources,
        point_budgets=[dict(dataset=p['dataset'],trace_max_duration_s=spans[p['dataset']],
            cell_limit_s=p['cell_limit_s'],cells=13*len(seeds)) for p in bounded_points],
        layout_scope=f'predeclared development candidate P{layout["prefill"]}/D{layout["decode"]}/M{layout["mixed"]} at measured TP1; no optimality claim',
        campaign=prepared['campaign'],prepared=str(root/'paired/prepared.json'),source_sha256=sha256(__file__)))
    return prepared


def execute(followup,manifest_path):
    """Explicit one-command continuation; never called by template generation."""
    from .campaign import Campaign
    from .method_selection import summarize
    campaign=Campaign(followup['campaign_root'],read_budget(followup['campaign_root'])['limit_s'])
    status=dict(complete=False,formal_eligible=False,started_s=time.time())
    try:
        def run_stages(path):
            passed=set()
            for stage in read(path)['stages']:
                if not set(stage.get('requires',[]))<=passed:raise RuntimeError('missing sequential stage prerequisite')
                campaign.run(stage['name'],stage['argv'],stage['limit_s'],gpu=stage.get('gpu',True));passed.add(stage['name'])
        campaign.run('prepare-followup-calibration',[sys.executable,'-m','ecopadg.serving.campaign_followup_setup','prepare-calibration',
            '--manifest',str(manifest_path)],600,gpu=False)
        run_stages(Path(followup['calibration_out'])/'campaign.json')
        calibration=read(Path(followup['calibration_out'])/'summary.json')
        if not calibration.get('passed'):raise RuntimeError('independent baseline calibration incomplete; dependent method stage remains closed')
        campaign.run('prepare-followup-methods',[sys.executable,'-m','ecopadg.serving.campaign_followup_setup','prepare-methods',
            '--manifest',str(manifest_path)],600,gpu=False)
        run_stages(Path(followup['method_out'])/'paired/campaign.json')
        selection=summarize(Path(followup['method_out'])/'paired/prepared.json')
        if selection.get('selected_variant') is None:raise RuntimeError('no method passed development selection')
        status.update(complete=True,development_selection=selection['selected_variant'])
    except BaseException as exc:
        status['error']=repr(exc)
        raise
    finally:
        status['finished_s']=time.time();write(Path(manifest_path).parent/'execution.status.json',status)
        campaign.close()
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('templates','prepare-calibration','prepare-methods','execute'))
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--out',type=Path)
    # Fifteen cells, up to six physical layouts, and the 60-second summary.
    parser.add_argument('--calibration-budget-s',type=float,default=21660)
    parser.add_argument('--method-budget-s',type=float,default=18000)
    args=parser.parse_args()
    if args.action=='templates':
        if not args.root or not args.out:parser.error('templates requires --root and --out')
        result=templates(args.root,args.out,calibration_budget_s=args.calibration_budget_s,method_budget_s=args.method_budget_s)
    else:
        if not args.manifest:parser.error('continuation requires --manifest')
        followup=read(args.manifest)
        result=(prepare_calibration(followup) if args.action=='prepare-calibration' else
                prepare_methods(followup) if args.action=='prepare-methods' else execute(followup,args.manifest.resolve()))
    print(json.dumps({k:v for k,v in result.items() if k in ('status','complete','campaign','split')},allow_nan=False))


if __name__=='__main__':main()
