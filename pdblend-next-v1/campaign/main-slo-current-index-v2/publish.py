"""Read actual improvement status; optionally publish metadata only. No hardware imports."""
import argparse, fcntl, hashlib, json, os, socket, time, uuid
from pathlib import Path
C = Path('/root/workspace/pdblend-next-v1/campaign')
SLO = {'alpaca': (1., .1), 'sharegpt': (5., .15), 'longbench': (15., .2)}
PROTOCOL = 'per-dataset-slo-five-system-fixed-window-v1'
def need(ok, message):
    if not ok: raise ValueError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def ref(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def checked(value):
    need(sha(value['path']) == value['sha256'], 'changed reference: '+value['path'])
    return read(value['path'])
def guard(value, index_dir):
    targets={(index_dir/n).resolve() for n in ('current-experiment.json','CURRENT_EXPERIMENT.md')}
    need(not any(Path(p).resolve() in targets for p in value.get('files', {})), 'index is a frozen active input')
def proc(pid):
    root=Path('/proc')/str(pid)
    try:
        stat=(root/'stat').read_text().rsplit(')',1)[1].split()
        argv=(root/'cmdline').read_bytes().decode().strip('\0').split('\0')
    except FileNotFoundError: return None
    return dict(pid=pid, start_ticks=int(stat[19]), state=stat[0], argv=argv)
def process_evidence(status, expected):
    actual=proc(status['pid'])
    if actual is None:
        need(status.get('finished_s') is not None and status.get('phase') not in ('running','starting','preflight','cold_restore','cold_remove'), 'absent process without terminal status')
        return dict(pid=status['pid'], running=False, start_ticks=None, terminal_gone=True)
    need(actual['state'] not in ('Z','X'), 'process terminal/reap race; retry read')
    argv=actual['argv']; need('--run' in argv, 'not an actual run')
    for option, value in expected.items():
        need(option in argv and argv.count(option)==1 and argv.index(option)+1<len(argv) and argv[argv.index(option)+1]==value, 'actual argv differs: '+option)
    btime=int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')))
    began=btime+actual['start_ticks']/os.sysconf('SC_CLK_TCK')
    need(-2 <= status['started_s']-began <= 300 and status.get('finished_s') is None, 'PID reuse or terminal race')
    need(proc(status['pid'])==actual, 'process changed while reading')
    return dict(actual, running=True, terminal_gone=False)
def build(args):
    source_path=(args.release or args.spec).resolve(); status_path=args.status.resolve()
    source=read(source_path); status=read(status_path); status_ref=ref(status_path)
    guard(source,args.index_dir); need(source['deadline_s']==1788872770.0400891, 'original deadline differs')
    fixed=args.kind=='fixed-screen'
    if fixed:
        need(args.release and not args.spec and source['schema']=='main-slo-improvement-release-v1' and source['approved'] is True, 'fixed release required')
        binding=checked(source['binding']); model=source['model']; guard(binding,args.index_dir)
        need(status['model']==model and status['stage']=='screen_fixed2', 'wrong fixed stage/model')
        need(checked(read(status_path.parent/'release-reference.json'))==source, 'status output belongs to another release')
        host=source['host_release']; checked(source['host_manifest'])
        configs={}; config_values={}
        for dataset, value in source['configs']['fixed2'].items():
            cfg=checked(value); configs[dataset]=value; config_values[dataset]=cfg
            need(cfg['arrival_window_s']==100 and cfg['measurement_window_protocol']==PROTOCOL and cfg['request_timeout_s']==120 and cfg['slo_attainment_target']==.9, 'fixed config protocol differs')
        need(set(configs)==set(SLO), 'three dataset configs required')
        declaration=checked(source['declaration']); rows=[c for c in declaration['cells'] if c['model']==model and c['stage']=='screen_fixed2']
        count={'7b':12,'14b':12,'32b':16}[model]
        need(len(rows)==count and len({c['cell_id'] for c in rows})==count, 'model fixed-cell declaration differs')
        for cell in rows:
            row=cell['source_row']
            need(row['seed']==701 and row['arrival_window_s']==100 and row['slo_scale']==1 and (row['slo_ttft_s'],row['slo_tpot_s'])==SLO[cell['dataset']], 'effective dataset SLO/100s/seed differs')
        ids={c['cell_id'] for c in rows}
        need(set(status['attempted'])<=ids and set(status['completed'])<=set(status['attempted']), 'status cells outside declaration')
        expected={'--release':str(source_path),'--out':str(status_path.parent),'--stage':'screen_fixed2'}
        script=C/'main-slo-improvement-v1/runner.py'
        detail=dict(arrival_window_s=100,arrival_seed=701,request_hard_timeout_s=120,drain_after_arrival_window_s=120,protocol_id=PROTOCOL,
                    dataset_slo={k:dict(ttft_s=v[0],tpot_s=v[1],attainment_target=.9) for k,v in SLO.items()},slo_source='declared source_row; overrides static controller defaults',configs=configs)
    else:
        need(args.spec and not args.release and source['schema']=='capacity-cold-calibration-spec-v1' and source['authorized'] is True, 'cold calibration spec required')
        binding=checked(source['original_binding']); guard(binding,args.index_dir); model=binding['model']; host=binding['host_release']
        capacity=checked(source['capacity_binding']); guard(capacity,args.index_dir)
        need(model=='32b' and status['schema']=='capacity-cold-calibration-status-v1' and status['original_binding']==source['original_binding'], 'cold status/binding mismatch')
        scripts=[p for p in source['files'] if Path(p).name=='capacity_calibrate.py']; need(len(scripts)==1,'one exact calibration source required'); script=Path(scripts[0])
        expected={'--spec':str(source_path),'--spec-sha256':sha(source_path),'--out':str(status_path.parent)}
        need(args.launch is not None,'cold phase requires actual launch evidence')
        launch=read(args.launch); need(launch['pid']==status['pid'] and launch['spec_sha256']==sha(source_path) and str(script) in launch['argv'],'cold launch/source differs')
        for option,value in expected.items():
            argv=launch['argv']; need(option in argv and argv[argv.index(option)+1]==value,'historical cold argv differs')
        detail=dict(serving_slo_point=False,arrival_window_s=None,arrival_seed=None,dataset_slo=None,cycles_declared=len(source['cycles']),capacity_binding=source['capacity_binding'],production_ready=status.get('production_ready',False),launch=ref(args.launch))
    if fixed and status.get('current_cell'):
        cell=status['current_cell']; job=status_path.parent/'results/operations'/cell/'job.json'
        point=status_path.parent/'bindings'/(cell+'.json')
        if job.exists():
            job_value=read(job); row=job_value['row']; dataset=row['dataset']
            need(row['cell_id']==cell and row['model']==model and row['seed']==701 and row['arrival_window_s']==100 and (row['slo_ttft_s'],row['slo_tpot_s'])==SLO[dataset], 'actual job protocol differs')
            need(job_value['config']==configs[dataset]['path'], 'actual job config differs')
            point_value=read(point); guard(point_value,args.index_dir)
            need(point_value['host_release']==host and point_value['configs'][dataset]==configs[dataset]['path'], 'actual point binding host/config differs')
            detail['active_job']=ref(job); detail['active_point_binding']=ref(point)
    need(binding['hostname']==socket.gethostname(), 'must read on actual bound host')
    need(str(script) in source['files'] and sha(script)==source['files'][str(script)], 'actual producer source differs')
    process=process_evidence(status,expected)
    if process['running']: need(str(script) in process['argv'],'wrong actual producer script')
    if args.launch and process['running']: need(read(args.launch).get('start_ticks')==process['start_ticks'],'launch start ticks differ')
    need(sha(status_path)==status_ref['sha256'], 'status changed during read; retry')
    return dict(schema=1,kind='main-slo-improvement-current-index',written_s=time.time(),hostname=socket.gethostname(),model=model,
        experiment_kind=args.kind,phase=status['phase'],process=process,source=ref(source_path),status=status_ref,host_release=host,
        base_binding=source['binding'] if fixed else source['original_binding'],
        complete=status['complete'],completed_count=len(status['completed']),failed=status.get('failed',[]),error=status.get('error'),current_cell=status.get('current_cell'),
        current_cycle=status.get('current_cycle'),global_deadline_s=source['deadline_s'],energy_scope='all eight GPU boards, including unused GPUs; failures retained',
        metadata_only=True,physical_controls=False,detail=detail)
def atomic(path, content):
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('x') as f: f.write(content); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',required=True,choices=['fixed-screen','cold-calibration'])
    p.add_argument('--status',required=True,type=Path);p.add_argument('--release',type=Path);p.add_argument('--spec',type=Path);p.add_argument('--launch',type=Path)
    p.add_argument('--index-dir',type=Path,default=C);p.add_argument('--previous-index-sha256');p.add_argument('--publish',action='store_true');a=p.parse_args()
    need(a.release or a.spec,'release/spec required'); value=build(a)
    if a.publish:
        with (a.index_dir/'scale-index.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            old=a.index_dir/'current-experiment.json'; need(a.previous_index_sha256 and sha(old)==a.previous_index_sha256,'prior index changed or not pinned')
            value=build(a); archive=a.index_dir/'current-experiment-history'/('main-slo-'+str(time.time_ns()));archive.mkdir(parents=True)
            for name in ('current-experiment.json','CURRENT_EXPERIMENT.md'):
                path=a.index_dir/name
                if path.exists(): (archive/name).write_bytes(path.read_bytes())
            value['previous_index_archive']=str(archive)
            data=json.dumps(value,indent=2,allow_nan=False)+'\n'
            md=f"{value['model']} — {value['experiment_kind']} — {value['phase']}.\n\nRunning: {value['process']['running']}; PID: {value['process']['pid']}; completed: {value['completed_count']}.\n\nHost: {value['host_release']}\nSource: {value['source']['path']}\nStatus: {value['status']['path']}\n\n"+('Seed 701; 100s arrivals; dataset SLOs; 120s request/drain.\n' if a.kind=='fixed-screen' else 'Cold capacity calibration, not a serving SLO point; no 100s workload.\n')+'All eight GPU boards included. This index does not certify results.\n'
            atomic(a.index_dir/'CURRENT_EXPERIMENT.md',md); atomic(old,data)
    print(json.dumps(value,indent=2))
if __name__=='__main__': main()
