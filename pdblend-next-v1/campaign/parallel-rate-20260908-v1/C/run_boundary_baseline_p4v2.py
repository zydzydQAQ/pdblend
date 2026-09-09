"""Run only the declared four new-rate cells through the frozen measurement executor."""
import argparse
import asyncio
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

HERE=Path(__file__).resolve().parent
CAMPAIGN=HERE.parents[1]
DECLARATION=HERE/'boundary-p4v2/declaration.json'
PACKAGE=HERE/'boundary-baseline-package-p4v2.json'
DECLARATION_SHA=None
COMMON=CAMPAIGN/'parallel-rate-20260908-v1/common/execution-until-complete-v1/run.py'
COMMON_SHA='77bcbbb68e20419e5bc469a838c71e1abfa789dc167d901501715fda0ff4a8a9'
DEADLINE=None
def require(ok,why):
    if not ok:raise RuntimeError(why)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v,exclusive=False):
    p.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
    else:
        tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def load(p,name):
    spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def contract(args):
    global DECLARATION_SHA
    package=read(PACKAGE);DECLARATION_SHA=package['files'][str(DECLARATION)]
    for p,h in package['files'].items():require(sha(p)==h,'boundary package changed: '+p)
    require(sha(DECLARATION)==DECLARATION_SHA and sha(COMMON)==COMMON_SHA,'frozen declaration/executor changed')
    d=read(DECLARATION);binding=read(args.binding)
    require(binding['hostname']==socket.gethostname() and binding['model']=='7b'
        and binding['system']==args.system and binding['deadline_s'] is None
        and binding.get('campaign_lifecycle')=='until_declared_complete_v1','wrong fresh baseline binding')
    require(binding.get('output_correctness_verified') is True and binding.get('mechanism_proof'),
            'fresh baseline mechanism qualification required')
    require(Path(binding['deployment_receipt'])==HERE/'baseline-boundary-restore-p4v2/deployment-receipt.json'
        and Path(binding['correctness_evidence'])==HERE/'boundary-baseline-gate-p4v2','old gate/restoration cannot qualify this attempt')
    require(binding['host_release']==d['baseline_controller_hosts'][args.system],'historical per-system baseline runtime changed')
    datasets={c['dataset'] for c in d['cells'] if c['system']==args.system}
    require(set(binding['configs'])==datasets and len(binding['instances'])==8,'C original baseline scope required')
    require(Path(binding['output'])==args.out/'results','binding output differs')
    cells=[copy.deepcopy(c) for c in d['cells'] if c['system']==args.system]
    require(cells and len({c['cell_id'] for c in cells})==len(cells),'declared unique boundary endpoints required')
    for c in cells:
        require(sha(c['trace_path'])==c['trace_sha256'],'new trace changed')
        require(c['baseline_controller_host']==binding['host_release'],'row controller source differs from binding')
    return binding,cells

async def execute(args,state):
    binding,cells=contract(args);host=Path(binding['host_release'])
    paths=[str(host/'src'),str(host),'/root/workspace/pdblend/.runtime-deps'];sys.path[:0]=paths
    os.environ['PYTHONPATH']=':'.join(paths);os.environ['PYTHONDONTWRITEBYTECODE']='1'
    common=load(COMMON,'c_newrate_original_measurement')
    from ecopadg.serving.campaign import node_lease
    from ecopadg.measure.backends import PynvmlBackend
    import aiohttp
    require('PDBLEND_NODE_LOCK_FD' not in os.environ,'fresh exclusive node lease required')
    require(not args.out.exists(),'fresh attempt directory required')
    args.out.mkdir(parents=True);output=args.out/'results';output.mkdir()
    write(args.out/'declaration-order.json',cells,True)
    write(args.out/'release-reference.json',dict(binding=dict(path=str(args.binding),sha256=sha(args.binding)),
        declaration=dict(path=str(DECLARATION),sha256=DECLARATION_SHA),source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__))),True)
    with node_lease():
        state['node_lease_held']=True
        common.validate_binding(binding);hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
        async with aiohttp.ClientSession(trust_env=False) as session:
            await common.identity(session,binding)
            for row in cells:
                if args.stop or (args.out/'STOP').exists():
                    state['phase']='stopped_at_boundary';break
                contract(args)
                state.update(phase='running',current_cell=row['cell_id']);state['attempted'].append(row['cell_id'])
                write(args.out/'status.json',state)
                b=copy.deepcopy(binding);b['files'][str(DECLARATION)]=DECLARATION_SHA
                b['files'][row['trace_path']]=row['trace_sha256'];b['files'][str(Path(__file__).resolve())]=sha(__file__)
                bp=args.out/'bindings'/(row['cell_id']+'.json');write(bp,b,True);common.validate_binding(b)
                receipt=await common.run_one(session,b,row,output,hardware)
                rp=output/'operations'/row['cell_id']/'receipt.json'
                artifacts={str(p):sha(p) for directory in (rp.parent,output/'cells'/row['cell_id']) for p in directory.rglob('*') if p.is_file()}
                write(output/'checkpoints'/(row['cell_id']+'.json'),dict(row=row,binding=str(bp),binding_sha256=sha(bp),receipt=str(rp),receipt_sha256=sha(rp),
                    artifacts=artifacts,measurement_valid=True,work_complete=receipt['summary'].get('work_complete'),completed_s=time.time(),new_rate=True,
                    same_trace_across_five_systems=True,poor_slo_does_not_trigger_retry=True),True)
                state['completed'].append(row['cell_id'])
                with (output/'cells'/row['cell_id']/'bench.csv').open() as f:
                    errors=[dict(request_id=r.get('request_id'),error=r.get('error'),http_status=r.get('http_status')) for r in csv.DictReader(f)
                        if r.get('http_status')=='503' or 'HTTP 503' in r.get('error','')]
                gate=dict(passed=not errors and receipt['summary'].get('work_complete') is True and receipt['summary'].get('failed_requests') == 0 and receipt['summary'].get('request_timeouts') == 0,http503=errors,work_complete=receipt['summary'].get('work_complete'),failed_requests=receipt['summary'].get('failed_requests'),request_timeouts=receipt['summary'].get('request_timeouts'),any_request_failure_stops_successors=True,complete_low_slo_negative_point_retained=True)
                write(output/'engineering-gates'/(row['cell_id']+'.json'),gate,True)
                if not gate['passed']:state.update(phase='engineering_gate_failed',engineering_gate_failed=True);break
                write(args.out/'status.json',state)
        state['complete']=len(state['completed'])==len(cells) and not state.get('engineering_gate_failed')
        if state['complete']:state['phase']='complete'
    state['node_lease_held']=False

def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--system',choices=['mixed','distserve','dynamollm','ecoserve'],required=True);p.add_argument('--run',action='store_true');a=p.parse_args();a.stop=False
    if not a.run:
        b,c=contract(a);print(json.dumps(dict(passed=True,cpu_only=True,cells=len(c))));return
    state=dict(pid=os.getpid(),phase='starting',started_s=time.time(),complete=False,attempted=[],completed=[],failed=[],automatic_retries=False)
    def stop(sig,frame):a.stop=True
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    try:asyncio.run(execute(a,state))
    except BaseException as exc:state.update(phase='failed',error=repr(exc));state['failed'].append(repr(exc));raise
    finally:
        if a.out.exists():
            state.update(finished_s=time.time(),node_lease_held=False);state.pop('current_cell',None);write(a.out/'status.json',state)

if __name__=='__main__':main()
