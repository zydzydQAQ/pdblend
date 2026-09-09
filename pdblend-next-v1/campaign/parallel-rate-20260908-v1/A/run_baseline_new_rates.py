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
DECLARATION=HERE/'added-rates-common-v2/declaration.json'
DECLARATION_SHA='4a5d372293afa7f399cf92f7936df0b7d96292ced8b7cb636d9df9cfaf79bcb6'
COMMON=CAMPAIGN/'five-system-execution-v3/run.py'
COMMON_SHA='7c7dbe217243b42a8f93b57476ed457a6111e8f90c71ac4269130c9b46420f92'
DEADLINE=1788872770.0400891
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
    require(sha(DECLARATION)==DECLARATION_SHA and sha(COMMON)==COMMON_SHA,'frozen declaration/executor changed')
    d=read(DECLARATION);binding=read(args.binding)
    require(binding['hostname']==socket.gethostname() and binding['model']=='14b'
        and binding['system']==args.system and binding['deadline_s']==DEADLINE,'wrong fresh baseline binding')
    require(binding.get('output_correctness_verified') is True and binding.get('mechanism_proof'),
            'fresh baseline mechanism qualification required')
    layout=args.layout
    require(layout in ('resident','distserve-longbench'),'unknown A layout')
    if layout=='distserve-longbench':
        require(args.system=='distserve' and args.dataset==['longbench'],'A heterogeneous scope differs')
    else:require(args.system!='distserve' or 'longbench' not in args.dataset,'A DistServe LongBench needs original TP2 route')
    require(Path(binding['deployment_receipt'])==args.restore/'deployment-receipt.json'
        and Path(binding['correctness_evidence'])==args.gate,'only declared fresh restore/gate can qualify')
    require(args.restore.is_relative_to(HERE) and args.gate.is_relative_to(HERE),'new A campaign evidence required')
    require(binding['host_release']==str(CAMPAIGN.parent/'releases/five-system100-A14B-baseline-v1-runtime'),
            'baseline runtime changed')
    expected=[(1,[j]) for j in range(8)] if layout=='resident' else [(1,[j]) for j in range(5)]+[(2,[6,7])]
    require(set(binding['configs'])==set(args.dataset) and [(i['tp'],i['gpus']) for i in binding['instances']]==expected,
            'A original baseline scope required')
    require(Path(binding['output'])==args.out/'results','binding output differs')
    cells=[copy.deepcopy(c) for c in d['cells'] if c['system']==args.system and c['dataset'] in args.dataset]
    require(len(cells)==2*len(args.dataset),'declared two repeats per dataset required')
    for dataset in args.dataset:
        rate={'sharegpt':1.75,'longbench':1.125}[dataset]
        require({(r['rate_rps'],r['repeat']) for r in cells if r['dataset']==dataset}=={(rate,1),(rate,2)},'exact declared rate/repeats required')
    for c in cells:require(sha(c['trace_path'])==c['trace_sha256'],'new trace changed')
    return binding,cells

async def execute(args,state):
    binding,cells=contract(args);host=Path(binding['host_release'])
    paths=[str(host/'src'),str(host),'/root/workspace/pdblend/.runtime-deps'];sys.path[:0]=paths
    os.environ['PYTHONPATH']=':'.join(paths);os.environ['PYTHONDONTWRITEBYTECODE']='1'
    common=load(COMMON,'a_newrate_original_measurement')
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
                if args.stop or (args.out/'STOP').exists() or time.time()+400>=DEADLINE:
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
                        if r.get('http_status')=='503' or 'HTTP 503' in r.get('error','') or 'frequency' in r.get('error','').lower()]
                gate=dict(passed=not errors,http503=errors,work_complete=receipt['summary'].get('work_complete'),capacity_failure_retained=True)
                write(output/'engineering-gates'/(row['cell_id']+'.json'),gate,True)
                if errors:state.update(phase='engineering_gate_failed',engineering_gate_failed=True);break
                write(args.out/'status.json',state)
        state['complete']=len(state['completed'])==len(cells) and not state.get('engineering_gate_failed')
        if state['complete']:state['phase']='complete'
    state['node_lease_held']=False

def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--system',choices=['mixed','distserve','dynamollm','ecoserve'],required=True);p.add_argument('--run',action='store_true')
    p.add_argument('--dataset',action='append',choices=['sharegpt','longbench'],required=True)
    p.add_argument('--layout',choices=['resident','distserve-longbench'],required=True)
    p.add_argument('--restore',type=Path,required=True);p.add_argument('--gate',type=Path,required=True)
    a=p.parse_args();a.stop=False
    for k in ('binding','out','restore','gate'):setattr(a,k,getattr(a,k).resolve())
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
