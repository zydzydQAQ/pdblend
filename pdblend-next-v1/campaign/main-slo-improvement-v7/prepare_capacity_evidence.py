"""Assemble all three declared loaded repetitions; never select passing repeats."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
import protocol as p


def import_code(code, name):
    """Fail closed on Python's flat-module cache rather than validate another version."""
    code=code.resolve()
    for module_name in ('capacity_executor','capacity_backend','capacity_certificate',
                        'capacity_runtime','capacity_planner'):
        path=code/(module_name+'.py')
        p.need(path.is_file(),'complete frozen capacity implementation required: '+str(path))
        previous=sys.modules.get(module_name)
        p.need(previous is None or p.sha(previous.__file__)==p.sha(path),
               'conflicting imported capacity implementation: '+module_name)
    sys.path.insert(0,str(code))
    return __import__(name)


def measured_result(spec, measured, cycle, phase):
    path=measured/f'cycle-{cycle}-{phase}'/'result.json'
    result=p.read(path)
    expected=dict(original_binding=spec['original_binding'],capacity_binding=spec['capacity_binding'],
        config=spec['config'],host_manifest=p.ref(Path(spec['host_release'])/'manifest.json'))
    p.need(result.get('source')==expected,'measured source/config differs from declared phase: '+str(path))
    p.need(result.get('phase')==path.parent.name,'measurement phase differs from its declaration')
    if phase.startswith('idle-'):
        p.need(result.get('phase_kind')=='idle' and
               result.get('declared_idle_duration_s')==spec['matched_idle_duration_s'],
               'declared matched idle duration changed')
    else:
        key=phase.rsplit('-layout',1)[0]
        p.need(result.get('trace')==spec['cycles'][cycle-1][key]
               and result.get('demand_domain_sha256')==spec['demand_domain_sha256'],
               'measured trace/domain differs from the exact declared repetition')
    return p.ref(path)


def assemble(spec_path, measured, code, out):
    p.need(not out.exists(), 'new evidence output required')
    spec=p.read(spec_path);binding=p.checked(spec['capacity_binding'])
    status=p.read(measured/'status.json')
    p.need(spec['mode']=='layout_calibration' and len(spec['cycles'])==3,
           'exact declared three loaded repetitions required')
    p.need(p.read(measured/'spec-reference.json')==p.ref(spec_path),
           'actual invocation used another calibration specification')
    p.need(all(p.sha(path)==digest for path,digest in spec['files'].items()),
           'declared loaded implementation changed')
    p.need(status.get('complete') is True and status.get('cleanup_complete') is True
           and not status.get('cleanup_errors') and not status.get('error'),
           'loaded calibration has not completed successfully with actual cleanup')
    inventory=p.ref(measured/'inventory.json')
    final=p.checked(inventory)
    p.need(final['complete'] is True and final['transition_inflight'] is False
           and final['identity']==binding['identity']
           and {i['id'] for i in final['active_instances']}==set(final['initial_ids']),
           'physical final inventory is not the unchanged initial owners')
    identity=binding['identity'];domain=spec['demand_domain_sha256']
    certificate=import_code(code,'capacity_certificate')
    groups=[]
    def group(name,kind,members,**extra):
        value=dict(schema='capacity-evidence-group-v1',kind=kind,identity=identity,
            capacity_binding=spec['capacity_binding'],members=members,**extra)
        path=out/'groups'/(name+'.json');p.write(path,value,exclusive=True)
        return p.ref(path)
    def result(cycle,phase):
        reference=measured_result(spec,measured,cycle,phase)
        p.need(reference in status['completed'],'result was not retained by the actual invocation')
        return reference
    for layout in (2,3):
        groups.append(group(f'layout{layout}','layout',
            [result(c,f'high{layout}-layout{layout}') for c in (1,2,3)]))
    savings=[]
    savings.append(group('idle-savings','idle_savings',[
        dict(source=result(c,'idle-layout3'),target=result(c,'idle-layout2')) for c in (1,2,3)],inventory=inventory))
    for phase in ('low','low40'):
        savings.append(group(phase+'-savings','savings',[
            dict(source=result(c,phase+'-layout3'),target=result(c,phase+'-layout2')) for c in (1,2,3)]))
    groups.append(group('savings-grid','savings_grid',savings,demand_domain_sha256=domain))
    groups.append(group('loaded-cold','transition',[
        dict(result=result(c,'under_load-layout2to3'),inventory=inventory) for c in (1,2,3)],
        operation='restore_cold',gpus=spec['gpus']))
    groups.append(group('loaded-remove','transition',[
        dict(result=p.ref(measured/f'cycle-{c}-remove.json'),inventory=inventory) for c in (1,2,3)],
        operation='remove',gpus=spec['gpus']))
    declaration=dict(schema='capacity-evidence-declaration-v1',identity=identity,evidence_groups=groups,
        input_spec=p.ref(spec_path),status=p.ref(measured/'status.json'),inventory=inventory,
        all_three_repetitions_retained=True,source=p.ref(__file__))
    p.write(out/'declaration.json',declaration,exclusive=True)
    qualification=dict(schema='capacity-evidence-qualification-v1',created_s=time.time(),passed=False,
        development_only=True,formal_performance_passed=False,declaration=p.ref(out/'declaration.json'))
    try:
        qualified=certificate.build(identity,groups,out/'certificate.json')
        certificate.validate(p.checked(qualified),identity)
        qualification.update(passed=True,certificate=qualified)
    except Exception as exc:
        qualification['error']=repr(exc)
        raise
    finally:
        p.write(out/'qualification.json',qualification,exclusive=True)
    return p.ref(out/'qualification.json')


def runtime_binding(source_path, qualification_path, inputs_path, code, out, owner):
    p.need(not out.exists(), 'fresh uniform dynamic binding required')
    source=copy.deepcopy(p.read(source_path));qualification=p.read(qualification_path)
    p.need(qualification.get('passed') is True, 'actual complete empirical certificate required')
    declaration=p.checked(qualification['declaration'])
    measured_spec=p.checked(declaration['input_spec'])
    p.need(measured_spec['capacity_binding']==p.ref(source_path),
           'uniform physical binding differs from the actually qualified loaded source')
    inputs=p.read(inputs_path);domain=p.checked(inputs['domain'])
    p.need(domain['sha256']==measured_spec['demand_domain_sha256'],
           'uniform domain differs from the actually measured calibration')
    p.need(source['deadline_s']==inputs['deadline_s']==p.DEADLINE, 'unchanged absolute deadline required')
    source.update(calibration_only=False,production_ready=False,
        calibration=qualification['certificate'],demand_domains=[domain],
        planner_source=p.ref(code/'capacity_planner.py'),owner_id=owner,
        max_creations=32,runtime_dir=str(out.parent/'runtime'),
        rate_observation_window_s=60.,arrival_count_margin=2.,
        policy=dict(down_utilization=.60,up_utilization=.85,low_hold_s=60.,high_hold_s=5.,
            min_resident_s=120.,min_off_s=30.,cooldown_s=60.),
        unknown_domain_capacity_action='hold',certificate_frozen_for_entire_declaration=True,
        empirical_only=True,automatic_main_performance_pass=False)
    files=dict(source['files'])
    files.update({str(v.resolve()):p.sha(v) for v in code.glob('*.py')})
    for path in (source_path,qualification_path,inputs_path,Path(__file__)):
        files[str(path.resolve())]=p.sha(path)
    from prepare_dynamic_release import certificate_inputs
    files.update(certificate_inputs(source['calibration']))
    source['files']=files
    p.need(all(p.sha(path)==digest for path,digest in files.items()), 'physical source changed')
    capacity_runtime=import_code(code,'capacity_runtime')
    module=capacity_runtime.load_planner(source['planner_source'])
    capacity_runtime.calibration_model(module,source)
    p.write(out,source,exclusive=True)
    return p.ref(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='command',required=True)
    a=subs.add_parser('assemble')
    for name in ('spec','measured','code','out'):a.add_argument('--'+name,type=Path,required=True)
    b=subs.add_parser('binding')
    for name in ('source','qualification','inputs','code','out'):b.add_argument('--'+name,type=Path,required=True)
    b.add_argument('--owner',required=True)
    args=parser.parse_args()
    result=(assemble(args.spec,args.measured,args.code,args.out) if args.command=='assemble' else
        runtime_binding(args.source,args.qualification,args.inputs,args.code,args.out,args.owner))
    print(json.dumps(result))
