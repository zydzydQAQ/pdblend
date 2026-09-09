"""C8 retained baseline restoration through unchanged frozen hardware primitives."""
import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

HERE=Path(__file__).resolve().parent
CAMPAIGN=HERE.parents[1]
ORIGINAL=CAMPAIGN/'AC-baseline-deployment-prepared-v1/C-resident'
PARENT=CAMPAIGN/'A14B-resident-restore-v3/restore.py'
PREVIOUS=HERE/'baseline-restore-p3/previous-binding.json'
RELEASE=HERE/'new-rate-release-p3/release.json'
OUT=HERE/'baseline-restore-p3'
HOST='iZwz9gfq11hx1sbob59yrgZ'
DEADLINE=1788872770.0400891
def require(ok,why):
    if not ok:raise RuntimeError(why)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def package_check():
    m=read(OUT/'adapter-manifest.json')
    for path,digest in m['files'].items():require(sha(path)==digest,'restore source changed: '+path)
    return m

def process_scan():
    others=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:argv=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
        except (OSError,UnicodeError):continue
        if 'python' in argv and '/campaign/' in argv and '-c ' not in argv and 'ecopadg.serving.engine' not in argv:
            others.append(dict(pid=proc.name,argv=argv))
    return dict(no_live_serving_child=not others,competing_processes=others)

def verify_release(path,digest):
    require(Path(path)==RELEASE and sha(path)==digest,'wrong actual new-rate release')
    release=read(path)
    for file,h in release['files'].items():require(sha(file)==h,'new-rate released file changed')
    result=[]
    for name,count in [('screen-p3',12),('new-rate-pdb-p3',4)]:
        root=HERE/name;status=read(root/'status.json')
        require(status.get('complete') is True and status.get('phase')=='complete'
                and status.get('node_lease_held') is False and not status.get('failed')
                and len(status.get('completed',[]))==count,'PDB predecessor not cleanly complete: '+name)
        require(not Path('/proc/'+str(status['pid'])).exists(),'PDB predecessor process still alive')
        require(read(root/'setup/status.json').get('passed') is True,'fresh ordinary gate missing')
        cps=list((root/'results/checkpoints').glob('*.json'))
        require({p.stem for p in cps}==set(status['completed']),'PDB predecessor checkpoints differ')
        require({p.name for p in (root/'results/operations').iterdir() if p.is_dir()}==set(status['completed']),
                'uncheckpointed PDB operation exists')
        points=[]
        for p in cps:
            c=read(p);r=read(c['receipt']);g=read(root/'results/engineering-gates'/p.name)
            require(sha(c['receipt'])==c['receipt_sha256'] and c.get('measurement_valid') is True
                and r.get('measurement_valid') is True and r.get('child_stopped') is True
                and r.get('clock_restore_complete') is True and not r.get('outer_cleanup_errors')
                and r['summary']['post_measurement_cleanup']['cleanup_complete'] is True,
                'PDB measured cleanup missing: '+p.name)
            require(g.get('passed') is True and not g.get('http503'),'PDB frequency/engineering failure blocks baseline restoration')
            for file,h in c['artifacts'].items():require(sha(file)==h,'PDB predecessor raw artifact changed')
            points.append(dict(checkpoint=str(p),sha256=sha(p),work_complete=c.get('work_complete')))
        result.append(dict(stage=name,status=str(root/'status.json'),status_sha256=sha(root/'status.json'),points=points))
    scan=process_scan();require(scan['no_live_serving_child'],'competing campaign process: '+str(scan))
    return dict(passed=True,captured_s=time.time(),predecessors=result,release=str(path),release_sha256=digest)

def expected():
    original=read(ORIGINAL/'deployment.json');receipt=read(ORIGINAL/'deployment-receipt.json')
    inventory={r['Name'].lstrip('/'):r for r in read(ORIGINAL/'containers.after.json')}
    require(original['model']=='7b' and original['hostname']==HOST and original['layout']=='resident','original C resident spec required')
    require(receipt.get('complete') is True and receipt.get('measurement_valid') is True,'original deployment incomplete')
    require([(i['tp'],i['gpus']) for i in original['instances']]==[(1,[j]) for j in range(8)],'original eight TP1 layout required')
    actual=read(HERE/'baseline-retained-inventory.json')
    actual={r['Name'].lstrip('/'):r for r in actual['containers']}
    created={r['name']:r['container_id'] for r in receipt['created']}
    expected_containers={};expected_provenance={}
    parent=load(PARENT,'c8_original_static_validation')
    for i in original['instances']:
        name=i['container_name'];old=inventory[name];current=actual[name]
        require(old['Id']==created[name]==current['Id'] and current['State']['Running'] is False,
                'same original retained stopped baseline ID required')
        parent.static_container(current,old)
        expected_containers[name]=old;expected_provenance[i['id']]=receipt['new_provenance'][i['id']]
    previous=read(PREVIOUS)
    require(previous['model']=='7b' and previous['system']=='pdblend' and previous['deadline_s']==DEADLINE,
            'exact current PDB source/identity/deadline required')
    files=dict(original['files']);files.update(previous['files']);files.update(package_check()['files'])
    for p in [PREVIOUS,RELEASE,OUT/'adapter-manifest.json',ORIGINAL/'deployment.json',ORIGINAL/'deployment-receipt.json',
              ORIGINAL/'containers.after.json',HERE/'baseline-retained-inventory.json']:
        files[str(p)]=sha(p)
    result=copy.deepcopy(original)
    result.update(schema=2,out=str(OUT),operation='restart-retained-residents',previous_binding=str(PREVIOUS),pdb_binding=str(PREVIOUS),
        model_main_release=str(RELEASE),model_main_release_sha256=sha(RELEASE),files=files,
        expected_containers=expected_containers,expected_provenance=expected_provenance,
        required_predecessors=[dict(stage=str(HERE/'screen-p3'),count=12),dict(stage=str(HERE/'new-rate-pdb-p3'),count=4)],
        new_container_creation_allowed=False,output_correctness_verified=False,fresh_correctness_gate_required=True,
        baseline_policies_and_engine_source_unchanged=True)
    return result

def validate_spec(spec):require(spec==expected(),'C8 exact restored spec differs')

def bind_parent(spec):
    parent=load(PARENT,'c8_frozen_hardware_primitives')
    parent.HOST=HOST;parent.package_check=package_check;parent.validate_spec=validate_spec
    parent.validate_previous=lambda value:require(value==read(PREVIOUS),'current PDB identity changed')
    old_load=parent.load
    def wrapped(path,name):
        if Path(path)==parent.BARRIER:return sys.modules[__name__]
        m=old_load(path,name)
        if Path(path)==parent.DEPLOY:
            def terminal(binding,manifest,**kwargs):
                require(Path(binding)==PREVIOUS and Path(manifest)==Path(spec['workloads']),'unexpected predecessor arguments')
                return verify_release(spec['model_main_release'],spec['model_main_release_sha256'])
            m.terminal_group=terminal
        return m
    parent.load=wrapped
    return parent

def main():
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--run',action='store_true');args=a.parse_args()
    require(not(args.prepare and args.run),'one mode only')
    if args.prepare:
        require(not OUT.exists(),'fresh restore package required')
        OUT.mkdir()
        previous=read(CAMPAIGN/'main-rate-rerun-v1/C-attempt-001/binding.json');previous['deadline_s']=DEADLINE
        write(PREVIOUS,previous)
        dependencies=read(PARENT.parent/'manifest.json')['dependencies']
        dependencies.update({str(PARENT.parent/n):h for n,h in read(PARENT.parent/'manifest.json')['files'].items()})
        dependencies[str(PARENT.parent/'manifest.json')]=sha(PARENT.parent/'manifest.json')
        dependencies[str(Path(__file__).resolve())]=sha(Path(__file__))
        write(OUT/'adapter-manifest.json',dict(files=dependencies,unchanged_hardware_primitives=str(PARENT)))
        write(OUT/'deployment.json',expected())
        print(json.dumps(dict(prepared=True,cpu_only=True,spec=str(OUT/'deployment.json'),sha256=sha(OUT/'deployment.json'))));return
    package_check();spec=read(OUT/'deployment.json');validate_spec(spec)
    if not args.run:print(json.dumps(dict(passed=True,cpu_only=True,restored=False)));return
    require('PDBLEND_NODE_LOCK_FD' not in os.environ,'fresh node lease required')
    host=Path(spec['host_release']);sys.path[:0]=[str(host/'src'),str(host),'/root/workspace/pdblend/.runtime-deps']
    from ecopadg.serving.campaign import node_lease
    async def run():
        task=asyncio.current_task();loop=asyncio.get_running_loop();stopped=False
        def stop():
            nonlocal stopped
            if not stopped:stopped=True;task.cancel()
        for sig in (signal.SIGINT,signal.SIGTERM):loop.add_signal_handler(sig,stop)
        return await bind_parent(spec).launch(OUT/'deployment.json')
    with node_lease():result=asyncio.run(run())
    print(json.dumps(dict(restored=result['complete'],measurement_valid=result['measurement_valid'],fresh_mechanism_gate_required=True)))

if __name__=='__main__':main()
