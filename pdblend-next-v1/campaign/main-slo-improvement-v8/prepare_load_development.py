"""Declare historical-shape development traces and bind immutable GPU packages.

This program never runs GPU work. Independent arrival seeds do not constitute
independent workload corpora, and empirical shape bounds are not hard guarantees.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
COMMON = REPO / 'campaign/main-slo-improvement-v1/common'
DEADLINE = 1788872770.0400891

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')

def checked(reference):
    assert sha(reference['path']) == reference['sha256'], reference['path']
    return read(reference['path'])

def make_inputs(model, dataset, out, rates):
    assert not out.exists(), 'new input declaration required'
    declaration = read(ROOT/'work-declaration.json')
    candidates = [c for c in declaration['cells'] if c['model']==model and c['dataset']==dataset]
    sources = {c['source_row']['trace_sha256']: c['source_row'] for c in candidates}
    out.mkdir(parents=True)
    (out/'trace-generator.py').write_bytes((COMMON/'capacity_load_calibrate.py').read_bytes())
    shapes = {}
    for digest, source in sources.items():
        assert sha(source['trace']) == digest
        trace = read(source['trace'])
        for index, (request, prompt) in enumerate(zip(trace['requests'], trace['prompts'])):
            key = (tuple(prompt), request['output_len'])
            shapes.setdefault(key, dict(prompt=prompt, output_len=request['output_len'],
                historical_source=dict(trace=ref(source['trace']), request_index=index)))
    values = list(shapes.values())
    # Declare extrema and body representatives without looking at new outcomes.
    axes = (lambda v: len(v['prompt']), lambda v: v['output_len'],
            lambda v: len(v['prompt'])+v['output_len'])
    selected = []
    for axis in axes:
        item = max(values, key=axis)
        if item not in selected:
            selected.append(item)
    ordered = sorted(values, key=lambda v: (len(v['prompt']), v['output_len']))
    for q in (.25, .5, .75):
        item = ordered[int((len(ordered)-1)*q)]
        if item not in selected:
            selected.append(item)
    slo = candidates[0]['source_row']['slo']
    domain = dict(slo_ttft_s=slo['ttft_s'], slo_tpot_s=slo['tpot_s'],
        max_input_tokens=max(len(v['prompt']) for v in selected),
        max_output_limit=max(v['output_len'] for v in selected),
        max_context_tokens=max(len(v['prompt'])+v['output_len'] for v in selected))
    domain_digest = hashlib.sha256(json.dumps(domain,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    domain['sha256'] = domain_digest
    write(out/'domain.json', domain)
    write(out/'templates.json', dict(schema='capacity-historical-templates-v1',templates=selected,
        selection='max input/output/context followed by input-length quartiles; exact historical tokens',
        no_new_performance_results_used=True, report_label=dataset, source_declaration=ref(ROOT/'work-declaration.json'),
        limits='empirical shape sample; not all combinations in the bounding box were measured'))
    sys.path.insert(0,str(COMMON))
    from capacity_load_calibrate import generate_trace, validate_trace
    cycles = []
    for repeat in range(1,4):
        refs = {}
        for offset, name in enumerate(('low','low40','high2','high3','under_load')):
            duration = 120 if name=='low' and model=='14b' else 60
            seed = 2026090800 + (100 if model=='32b' else 0) + repeat*10 + offset
            phases=[dict(name=name,duration_s=duration,rate_rps=rates[name])]
            trace=generate_trace(selected,phases,seed,domain_digest)
            validate_trace(trace,domain_digest)
            assert trace['n_requests'] >= len(selected), 'declared low trace does not cover all selected shapes'
            trace['templates']=ref(out/'templates.json')
            path=out/f'cycle-{repeat}-{name}.json';write(path,trace);refs[name]=ref(path)
        cycles.append(refs)
    phases=[dict(name=name,duration_s=300,rate_rps=rate) for name,rate in
            [('low',rates['low']),('high',rates['qualification_high']),('low',rates['low'])]]
    trace=generate_trace(selected,phases,2026090891+(100 if model=='32b' else 0),domain_digest)
    trace['templates']=ref(out/'templates.json')
    validate_trace(trace,domain_digest,duration=900)
    write(out/'qualification900-trace.json',trace)
    result=dict(schema='capacity-load-input-declaration-v1',model=model,dataset_for_report=dataset,
        created_s=time.time(),deadline_s=DEADLINE,development_only=True,formal_eligible=False,
        no_automatic_retries=True,domain=ref(out/'domain.json'),templates=ref(out/'templates.json'),
        rates_for_trace_generation_only=rates,cycles=cycles,qualification900_trace=ref(out/'qualification900-trace.json'),
        qualification900_arms=['fixed2','dynamic'],qualification900_repeats_per_arm=1,
        same_trace_required_per_pair=True,source_generator=ref(out/'trace-generator.py'),
        source_builder=ref(__file__))
    write(out/'inputs.json',result)
    return ref(out/'inputs.json')

def bind(args):
    assert not args.out.exists(), 'new package required'
    inputs=read(args.inputs);release=read(args.fixed_release);capacity=copy.deepcopy(read(args.capacity_binding))
    assert inputs['model']==release['model'] and release['approved'] is True
    assert 1024<=args.http_port_base<32000 and 1024<=args.kv_port_base<32000
    original=checked(release['binding'])
    occupied={g for instance in original['instances'] for g in instance['gpus']}
    assert args.gpus and len(args.gpus)==capacity['identity']['tp'] and len(set(args.gpus))==len(args.gpus)
    assert all(type(g) is int and 0<=g<8 for g in args.gpus) and not occupied.intersection(args.gpus), 'extra GPUs overlap original owners'
    assert capacity['deadline_s']==inputs['deadline_s']==release['deadline_s']==DEADLINE
    code=args.out/'code';code.mkdir(parents=True)
    for source in sorted(args.code.glob('*.py')):
        (code/source.name).write_bytes(source.read_bytes())
    for name in ('capacity_load_calibrate.py','test_capacity_load_calibrate.py'):
        (code/name).write_bytes((COMMON/name).read_bytes())
    files=dict(release['files']);files.update(capacity['files'])
    files.update({str(p.resolve()):sha(p) for p in code.glob('*.py')})
    for path in (args.inputs,args.fixed_release,args.capacity_binding,Path(__file__)):
        files[str(path.resolve())]=sha(path)
    for path in args.inputs.parent.iterdir():
        if not path.is_file():continue
        files[str(path.resolve())]=sha(path)
    capacity.update(owner_id=args.owner,calibration_only=True,production_ready=False,max_creations=3,
        runtime_dir=str((args.out/'runtime').resolve()),demand_domains=[checked(inputs['domain'])],
        http_port_base=args.http_port_base,kv_port_base=args.kv_port_base,
        planner_source=ref(code/'capacity_planner.py'))
    capacity['files']=dict(files)
    write(args.out/'capacity-binding.json',capacity)
    files[str((args.out/'capacity-binding.json').resolve())]=sha(args.out/'capacity-binding.json')
    dataset=inputs['dataset_for_report'];config=copy.deepcopy(checked(release['configs']['fixed2'][dataset]))
    original=checked(release['binding']);domain=checked(inputs['domain'])
    config.update(instances=original['instances'],slo_ttft_s=domain['slo_ttft_s'],slo_tpot_s=domain['slo_tpot_s'],
        port=args.port,capacity_integration_v1=False)
    config.pop('measurement_window_protocol',None)
    # PD is disabled; retain no unused transfer candidates in this independent mixed calibration.
    config.pop('transfers',None)
    write(args.out/'config.json',config)
    files[str((args.out/'config.json').resolve())]=sha(args.out/'config.json')
    cycles=copy.deepcopy(inputs['cycles'])
    for index,cycle in enumerate(cycles,1):
        for op,budget in [('restore',240),('remove',120)]:
            path=args.out/f'cycle-{index}-{op}.json'
            write(path,dict(schema='capacity-calibration-action-v1',authorized=True,automatic_retries=False,
                cycle=index,operation=op,gpus=args.gpus,deadline_s=DEADLINE,work_budget_s=budget,
                scope='declared development loaded 2-to3-to2; not production qualification'))
            cycle[op]=ref(path);files[str(path.resolve())]=sha(path)
    assert all(sha(p)==h for p,h in files.items()), 'package source changed'
    spec=dict(schema='capacity-load-calibration-spec-v1',authorized=True,automatic_retries=False,
        mode='layout_calibration',original_binding=release['binding'],capacity_binding=ref(args.out/'capacity-binding.json'),
        config=ref(args.out/'config.json'),host_release=release['host_release'],files=files,
        common_executor=ref(REPO/'campaign/five-system-execution-v3/run.py'),deadline_s=DEADLINE,
        demand_domain_sha256=domain['sha256'],gpus=args.gpus,cycles=cycles,matched_idle_duration_s=60,
        stop_path=str((ROOT/args.node/'STOP').resolve()),api_base=f'http://127.0.0.1:{args.port}',
        served_model=config.get('served_model',config.get('model','pdblend')),cold_start_after_s=10,
        input_declaration=ref(args.inputs),fixed_release=ref(args.fixed_release))
    write(args.out/'spec.json',spec)
    return ref(args.out/'spec.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    inp=sub.add_parser('inputs');inp.add_argument('--model',choices=['14b','32b'],required=True)
    inp.add_argument('--out',type=Path,required=True)
    b=sub.add_parser('bind')
    for name in ('inputs','fixed-release','capacity-binding','code','out'):
        b.add_argument('--'+name,type=Path,required=True)
    b.add_argument('--node',choices=['A','B'],required=True);b.add_argument('--owner',required=True)
    b.add_argument('--http-port-base',type=int,required=True);b.add_argument('--kv-port-base',type=int,required=True)
    b.add_argument('--port',type=int,required=True);b.add_argument('--gpus',type=int,nargs='+',required=True)
    args=parser.parse_args()
    if args.command=='inputs':
        rates=(dict(low=.15,low40=.6,high2=1.5,high3=2.,under_load=1.5,qualification_high=2.) if args.model=='14b'
            else dict(low=.25,low40=1.,high2=2.5,high3=3.,under_load=2.5,qualification_high=4.))
        result=make_inputs(args.model,'sharegpt' if args.model=='14b' else 'alpaca',args.out,rates)
    else:
        result=bind(args)
    print(json.dumps(result))
