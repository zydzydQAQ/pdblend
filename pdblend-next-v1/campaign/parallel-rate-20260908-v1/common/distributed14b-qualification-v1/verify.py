"""Pure saved-evidence qualification for distributed 14B comparisons.

This module performs no HTTP, process lookup, clock write, or Docker operation.
All scalar qualification claims are reconstructed from their saved raw inputs.
"""
import ast, copy, csv, hashlib, importlib.util, json, math, sys
from pathlib import Path

def need(value, message):
    if not value: raise ValueError(message)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def ref(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def checked(reference):
    need(isinstance(reference,dict) and set(reference) >= {'path','sha256'}, 'explicit frozen reference required')
    need(sha(reference['path']) == reference['sha256'], 'frozen input changed: '+reference['path'])
    return read(reference['path'])

def frozen_files(files):
    need(isinstance(files,dict) and files, 'nonempty frozen source/evidence closure required')
    for path,digest in files.items(): need(sha(path)==digest, 'source or raw bytes changed: '+path)

def bound(reference, files):
    need(isinstance(reference,dict) and files.get(reference.get('path'))==reference.get('sha256'), 'reference outside frozen qualification closure')
    return checked(reference)

def bound_path(path, files):
    path=str(Path(path));need(path in files and sha(path)==files[path], 'raw input outside frozen qualification closure: '+path)
    return Path(path)

def manifest(reference):
    value=checked(reference); root=Path(reference['path']).parent
    files={str(Path(p) if Path(p).is_absolute() else root/p):h for p,h in value['files'].items()}
    frozen_files(files); return files

def module(reference, files):
    need(files.get(reference['path'])==reference['sha256'] and sha(reference['path'])==reference['sha256'], 'unbound audit implementation')
    name='distributed14b_saved_'+reference['sha256'][:20]
    spec=importlib.util.spec_from_file_location(name,reference['path']); result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result; spec.loader.exec_module(result); return result

def original_power_functions(host_manifest):
    """Execute the exact three pure original functions without ambient imports."""
    files=manifest(host_manifest); root=Path(host_manifest['path']).parent/'src/ecopadg'
    namespace=dict(math=math,INSTANT_POWER_SOURCE_ID='nvml:field:186:scope:0:mW',ENERGY_PAD_S=0)
    for path,name in ((root/'serving/measurement.py','power_evidence'),(root/'measure/power.py','trapezoid_energy'),(root/'metrics.py','clip_power_window')):
        need(files.get(str(path))==sha(path),'original arithmetic not frozen')
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
        exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),namespace)
    return namespace

def power_operation(directory,status,host_manifest,files=None):
    directory=Path(directory); p=directory/'power'; functions=original_power_functions(host_manifest)
    if files is not None:
        for name in ('power.csv','power_metadata.jsonl','power_source.json','clocks.csv'):bound_path(p/name,files)
    with (p/'power.csv').open() as f:
        reader=csv.DictReader(f); need(all('gpu'+str(i)+'_w' in reader.fieldnames for i in range(8)), 'all eight power columns required')
        rows=[(float(r['t_s']),[float(r[f'gpu{i}_w']) for i in range(8)]) for r in reader]
    metadata=[json.loads(x) for x in (p/'power_metadata.jsonl').read_text().splitlines()]
    evidence=functions['power_evidence'](rows,read(p/'power_source.json'),metadata)
    need(evidence['power_source_verified'], 'original eight-card power provenance failed')
    start=status.get('operation_start_s',status.get('measurement_start_s')); end=status.get('operation_end_s',status.get('measurement_end_s'))
    need(start is not None and end is not None and start<end and status.get('sampling_error') is None, 'invalid sampling window/error')
    energy=functions['trapezoid_energy'](functions['clip_power_window'](rows,start,end,pad_s=0))
    expected=status.get('all8_operation_energy_j',status.get('full_operation_energy_j'))
    need(expected is not None and math.isclose(energy,expected,rel_tol=1e-9,abs_tol=1e-5),'raw operation energy differs')
    with (p/'clocks.csv').open() as f:
        reader=csv.DictReader(f); need(reader.fieldnames==['t_s']+[f'gpu{i}_sm_mhz' for i in range(8)], 'all-eight clock columns required')
        clocks=[(float(r['t_s']),[float(r[f'gpu{i}_sm_mhz']) for i in range(8)]) for r in reader]
    need(clocks and clocks[0][0]<=start<end<=clocks[-1][0] and all(b[0]>a[0] for a,b in zip(clocks,clocks[1:])),'clock window not fully bracketed')
    need(all(math.isfinite(t) and len(v)==8 and all(math.isfinite(x) and x>0 for x in v) for t,v in clocks),'invalid clock samples')
    return dict(energy_j=energy,power_evidence=evidence,power_samples=len(rows),clock_samples=len(clocks),start_s=start,end_s=end),clocks

def identities(saved,instances,files):
    need(isinstance(saved,list) and len(saved)==len(instances),'saved instance count differs')
    by={r['provenance']['instance_id']:r for r in saved}; need(set(by)=={i['id'] for i in instances},'saved native membership differs')
    for i in instances:
        r=by[i['id']]; c=r['container']; p=r['provenance']
        need(c['Id']==i['container']['id'] and c['Image']==i['container']['image'] and c['State']['StartedAt']==i['container']['StartedAt'] and c['State']['Running'] is True,'saved container/model execution differs')
        need(c['Name'].lstrip('/')==i['container']['name'] and c['State']['Pid']==i.get('host_pid',c['State']['Pid']) and c['State']['Pid']>0,'saved native process identity differs')
        need(all(p.get(k)==v for k,v in i['provenance'].items()),'saved imported source/TP/PID differs')
        need(p['instance_id']==i['id'] and p['tp']==i['tp'] and p['cuda_visible_devices']==','.join(map(str,i['gpus'])) and p['model']=='/models/Qwen2.5-14B-Instruct','saved hardware/model mapping differs')
        need(p['source_files_at_import'] and all(files.get(path)==h and sha(path)==h for path,h in p['source_files_at_import'].items()),'native imported source outside closure')
        config=read(i['engine_config']); need(files.get(i['engine_config'])==sha(i['engine_config']),'native config outside closure')
        need(config['model']==p['model'] and config['tp']==i['tp'] and config['max_model_len']==8192 and config['max_num_seqs']==32,'native model/shape contract differs')
        need(str(i['engine_config']) in c['Args'] and 'CUDA_VISIBLE_DEVICES='+','.join(map(str,i['gpus'])) in c['Config']['Env'],'actual argv/GPU environment differs')
    return by

def ordinary(reference,binding,host_manifest,validator,files):
    status=bound(reference,files); directory=Path(reference['path']).parent
    native=status['ordinary']; need(native['owned']==[] and len(native['replies'])==4 and len(native['restoration'])==2,'native ordinary work incomplete')
    need(status['clock_restore_complete'] and not native.get('cleanup_errors') and not status['errors'],'ordinary clock or cleanup failure')
    before=identities(read(bound_path(directory/'identity.before.json',files)),binding['instances'],files)
    after=identities(read(bound_path(directory/'identity.after.json',files)),binding['instances'],files)
    for side in (before,after):
        for i in binding['instances']: validator.native_saved(side[i['id']]['runtime'],i['id'],8192)
    for length in (128,7168):
        rows=[r for r in native['replies'] if r['prompt_length']==length]
        need({r['instance_id'] for r in rows}=={i['id'] for i in binding['instances']},'ordinary replica/length missing')
        answers=[]
        for r in rows:
            reply=r['response']; tokens=reply.get('token_ids')
            need(isinstance(tokens,list) and len(tokens)==64 and all(type(t)is int for t in tokens),'ordinary exact tokens missing')
            need(reply['usage']['prompt_tokens']==length and reply['usage']['completion_tokens']==64,'ordinary token work changed')
            answers.append(tokens)
        need(answers[0]==answers[1],'ordinary deterministic replicas differ')
    for i,r in zip(binding['instances'],native['restoration']): validator.cleanup_saved(r,i['id'])
    energy,_=power_operation(directory,status,host_manifest,files)
    return dict(requests=4,independently_recomputed=True,raw_power=energy)

def frequency_case(case,binding,host_manifest,files):
    status=bound(case['status'],files); directory=Path(case['status']['path']).parent
    spec=bound(status['spec'],files); need(spec['binding']==case['binding'],'frequency source binding differs')
    source_binding=bound(case['binding'],files); need(source_binding['instances']==binding['instances'],'frequency native deployment differs')
    need(spec['files'] and all(files.get(p)==h and sha(p)==h for p,h in spec['files'].items()),'frequency source closure missing')
    need(status['clock_restore_complete'] and all(r['complete'] and not r['errors'] for r in status['native_cleanup']),'frequency final cleanup failed')
    validator=module(case['validator'],files)
    raw=bound(case['raw'],files); need(raw['point']['point_id']==case['point_id'] and raw['point'] in spec['points'],'frequency point not predeclared')
    _,clocks=power_operation(directory,status,host_manifest,files)
    event_ref=raw['events']; checked_bytes=bound_path(event_ref['path'],files).read_bytes(); need(sha(event_ref['path'])==event_ref['sha256'] and checked_bytes.endswith(b'\n'),'native event evidence incomplete')
    events=[json.loads(x) for x in checked_bytes.splitlines()]
    result=validator.validate_point(raw,events,clocks)
    for side in ('identity.before.json','identity.after.json'): identities(read(bound_path(directory/side,files)),binding['instances'],files)
    return dict(point=raw['point'],evidence=result,independently_recomputed=True)

def policy_identity(config,parent,system):
    """The hardware migration cannot silently tune the scientific strategy."""
    operational={'instances','profiles','frequency_costs','frequency_evidence',
        'max_service_frequency_mhz','controller_source_release','host_source_release',
        'engine_source_release','journal','port'}
    if system=='pdblend':operational|={'capacity_integration_v1','capacity_binding_path','capacity_binding_sha256'}
    need({k:v for k,v in config.items() if k not in operational}=={k:v for k,v in parent.items() if k not in operational},
         'original serving policy changed outside declared hardware/profile/identity migration')
    return dict(strategy=config['strategy'],unchanged_policy_fields=sorted(set(config)-operational))

def verify(reference,binding,cp=None):
    """Return root reporting identity fields only after full saved-raw validation."""
    q=checked(reference); b=checked(binding) if isinstance(binding,dict) and set(binding)>= {'path','sha256'} else copy.deepcopy(binding)
    need(q['schema']=='distributed14b-qualified-execution-v1' and q['model']=='14b','unknown qualification/model')
    need(q['system']=='pdblend' and q['native']['kind']=='pdb_ordinary','v1 is actual fixed2 qualification; legacy strategy qualification requires its own successor')
    files=q['files']; frozen_files(files)
    parent=bound(q['binding'],files); need(b['instances']==parent['instances'] and b['hostname']==q['hostname']==parent['hostname'],'foreign actual binding/node')
    need(b['system']==q['system'] and b['model']==q['model'],'system qualification mismatch')
    need(ref(Path(b['host_release'])/'manifest.json')==q['host_manifest'],'actual controller source differs')
    for reference_name in ('host_manifest','measurement_host_manifest','profile','hardware_identity','execution_group'):
        bound(q[reference_name],files)
    manifest(q['host_manifest']); need(set(b['configs'])=={q['dataset']},'dataset configuration differs')
    frozen_files(b['files']);bound_path(b['configs'][q['dataset']],b['files'])
    config=read(b['configs'][q['dataset']]); need(ref(config['profiles'])==q['profile'],'actual scientific profile differs')
    need(config['instances']==b['instances'],'controller instances differ from the qualified native binding')
    need(ref(b['configs'][q['dataset']])==q['actual_configuration'],'actual configuration is not frozen in qualification')
    bound(q['actual_configuration'],files)
    original_config=bound(q['policy_parent'],files)
    policy=policy_identity(config,original_config,q['system'])
    need(config.get('max_service_frequency_mhz',2520)==q['max_service_frequency_mhz'],'actual hardware maximum differs')
    need(q['qualified_frequencies_mhz'] and max(q['qualified_frequencies_mhz'])==q['max_service_frequency_mhz'],'qualified frequency domain lacks its maximum')
    actual_profile=checked(q['profile'])
    need({p['frequency_mhz'] for p in actual_profile['points']}<=set(q['qualified_frequencies_mhz']),'serving profile contains an unqualified frequency')
    if q['system']=='pdblend':
        need(config.get('capacity_integration_v1') is False and 'capacity_binding_path' not in config and 'capacity_binding_sha256' not in config,'non-Alpaca capacity is not explicitly off')
    group=checked(q['execution_group']); selected=group['systems'][q['system']]
    need(group['node']==q['node'] and group['dataset']==q['dataset'] and group['model']=='14b' and selected['host_manifest']==q['host_manifest'] and selected['profile']==q['profile'],'dataset version selection differs')
    need(group['max_service_frequency_mhz']==q['max_service_frequency_mhz'],'dataset hardware domain differs')
    hardware=checked(q['hardware_identity']); need(hardware['hostname']==q['hostname'] and hardware['node']==q['node'],'saved hardware host differs')
    gpus=hardware['gpus']; need(len(gpus)==8 and {r['index'] for r in gpus}==set(range(8)) and len({r['uuid'] for r in gpus})==8 and all(r['name']=='NVIDIA L20' and r['uuid'].startswith('GPU-') for r in gpus),'actual eight L20 identities missing')
    validator=module(q['native']['validator'],files)
    if q['native']['kind']=='pdb_ordinary':
        native=ordinary(q['native']['status'],parent,q['measurement_host_manifest'],validator,files)
    elif q['native']['kind']=='legacy55':
        bound(q['native']['status'],files)
        functions=original_power_functions(q['measurement_host_manifest'])
        native,raw_files=validator.audit(Path(q['native']['status']['path']).parent,parent['instances'],q['system'],functions['power_evidence'],hetero=q['native'].get('heterogeneous',False))
        need(all(files.get(p)==h for p,h in raw_files.items()),'native55 raw outside closure')
    else: raise ValueError('unsupported native qualification kind')
    frequencies=[frequency_case(case,parent,q['measurement_host_manifest'],files) for case in q['frequency_cases']]
    required={(i['id'],f) for i in parent['instances'] for f in q['qualified_frequencies_mhz']}
    actual={(v['point']['instance_id'],v['point']['frequency_mhz']) for v in frequencies}
    need(required<=actual,'one or more used instance/frequency pairs lack actual loaded proof')
    registration_declaration=bound(q['profile_registration']['registration'],files)
    need(registration_declaration['node']==q['node'] and registration_declaration['hostname']==q['hostname'] and registration_declaration['binding']==q['binding'],
         'registered profile belongs to another actual node/deployment')
    registration=module(q['profile_registration']['validator'],files).verify(q['profile_registration']['registration'],q['profile'])
    need(registration['derived_profile']==checked(q['profile']),'independent profile derivation differs from actual serving profile')
    key=lambda c:(c['tp'],c['source_mhz'],c['target_mhz'])
    costs={key(c):c for c in original_config['frequency_costs'] if max(c['source_mhz'],c['target_mhz'])<=q['max_service_frequency_mhz']}
    costs.update({key(c):c for c in registration['frequency_costs']})
    actual_costs={key(c):c for c in config['frequency_costs']}
    need(len(actual_costs)==len(config['frequency_costs']) and all(costs.get(k)==c for k,c in actual_costs.items()),'frequency cost not derived from frozen original or actual target evidence')
    required_costs={(i['tp'],a,z) for i in b['instances'] for a in q['qualified_frequencies_mhz'] for z in q['qualified_frequencies_mhz'] if a!=z}
    need(required_costs<=set(actual_costs),'a used frequency transition has no measured cost')
    if cp is not None:
        record=checked(cp) if isinstance(cp,dict) and set(cp)>= {'path','sha256'} else cp
        need(record['qualification']==reference and record['measurement_host']==q['node'],'checkpoint qualification/host mismatch')
        frozen_files(record['artifacts']); actual_binding=bound(record['binding'],record['artifacts'])
        need(actual_binding==b,'checkpoint actual binding differs')
        receipt=bound(record['receipt'],record['artifacts']); op=Path(record['receipt']['path']).parent
        for name in ('identity.before.json','identity.after.json'): identities(read(bound_path(op/name,record['artifacts'])),b['instances'],{**files,**b['files']})
        actual_hardware=bound(record['hardware_identity'],record['artifacts']); need(actual_hardware['hostname']==hardware['hostname'] and actual_hardware['gpus']==gpus,'checkpoint GPU UUID changed')
    return dict(passed=True,hostname=q['hostname'],measurement_host=q['node'],gpu_identity=gpus,host_manifest=q['host_manifest'],profile_sha256=q['profile']['sha256'],execution_group=q['execution_group'],independently_recomputed=True,policy=policy,native=native,frequency_cases=frequencies,profile_registration=registration)

def qualified_execution(cp,binding,entry=None):
    """Reporting adapter: saved qualification plus the exact assigned source chain."""
    record=checked(cp) if isinstance(cp,dict) and set(cp)>={'path','sha256'} else copy.deepcopy(cp)
    actual=checked(binding) if isinstance(binding,dict) and set(binding)>={'path','sha256'} else copy.deepcopy(binding)
    release=checked(record['release']);frozen_files(release['files'])
    need(release['schema']=='distributed14b-static-release-v2' and release['system']=='pdblend','v1 reporting adapter is fixed2 PDB only')
    need(record['qualification']==release['qualification'] and record['qualification_validator']==release['qualification_validator'] and release['qualification_validator']==ref(__file__),'checkpoint qualification implementation/source differs')
    base=checked(release['binding'])
    need({k:v for k,v in actual.items() if k!='output'}=={k:v for k,v in base.items() if k!='output'},'actual cell binding changed beyond its output directory')
    need(record['binding'] in [ref(record['binding']['path'])] and checked(record['binding'])==actual,'checkpoint binding reference differs')
    jobs=checked(release['jobs']);checked(jobs['parent'])
    need(record['declaration']==release['jobs'] and jobs['parent']['sha256']=='913a2d5834dbcc3466cff9d6a45e30cba574069ed50d36c89704ffff89b80438','checkpoint assignment lineage differs')
    rows=[]
    for cell in jobs['pdb_cells']:
        row=copy.deepcopy(cell['source_row']);row.update(cell_id=cell['cell_id'],repeat=cell['repeat']);rows.append(row)
    need(record['row'] in rows and record['row'] in release['rows'] and len(release['rows'])==len(rows) and sorted(release['rows'],key=lambda x:x['cell_id'])==sorted(rows,key=lambda x:x['cell_id']),'checkpoint or release is not the exact whole assigned PDB group')
    need(record['actual_arm']==release['actual_arm']=='fixed2' and record['capacity_integration_v1'] is False and release['capacity_integration_v1'] is False,'actual capacity-off arm differs')
    need(record['source_successor']==release['source_successor'],'checkpoint successor differs')
    successor=checked(record['source_successor']);q=checked(record['qualification']);group=checked(q['execution_group'])
    need(successor['jobs']==release['jobs'] and successor['parent_controller_manifest']==jobs['common_controller_manifest'] and successor['host_manifest']==q['host_manifest'] and successor['max_service_frequency_mhz']==q['max_service_frequency_mhz'],'qualified source/profile successor differs')
    cpu=checked(successor['cpu_validation']);need(cpu['passed'] and any(s['manifest']==q['host_manifest'] for s in cpu['sources']),'qualified source is not in the frozen common CPU proof')
    need(group['jobs']==release['jobs'] and group['systems']['pdblend']['configuration']==q['actual_configuration'],'dataset execution configuration mapping differs')
    need(release['node']==jobs['node']==q['node']==record['measurement_host'] and release['dataset']==jobs['dataset']==q['dataset'],'assigned actual host/dataset differs')
    if entry is not None:
        for key in ('node','dataset','model'):
            if key in entry:need(entry[key]==q[key],'report entry '+key+' differs from actual qualified execution')
    result=verify(record['qualification'],actual,record)
    result.update(release=record['release'],jobs=release['jobs'],source_successor=record['source_successor'],actual_arm='fixed2',capacity_integration_v1=False)
    return result
