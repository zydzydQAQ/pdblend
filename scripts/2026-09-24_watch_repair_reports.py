#!/usr/bin/env python3
"""Read-only GPU campaign observer; each new report is an immutable snapshot.

The report program is supplied from a separately frozen CPU source package.
This watcher never writes queue state, receipts, campaign files or source code.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def binding(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load(path):
    return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class InputPending(ValueError):
    """A producer may still be writing this JSON; do not publish a report yet."""
    def __init__(self,path,error):
        self.problem=dict(path=str(Path(path).resolve()),error=str(error))
        super().__init__('input JSON is not ready: '+self.problem['path']+': '+str(error))


def json_input(path,refs):
    path=Path(path).resolve()
    try:
        raw=path.read_bytes()
        value=json.loads(raw)
    except (FileNotFoundError,json.JSONDecodeError,UnicodeDecodeError) as error:
        raise InputPending(path,error) from error
    refs.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
    return value


def inside(path,root,description):
    path=Path(path).resolve();root=Path(root).resolve()
    if not path.is_relative_to(root) or path==root:
        raise ValueError(description+' is outside the declared directory: '+str(path))
    return path


def job_directory(spec,attempt_root):
    identity=spec['job_id']
    if (not isinstance(identity,str) or not identity or '/' in identity or '\\' in identity
            or identity in ('.','..') or Path(identity).name!=identity):
        raise ValueError('invalid declared queue job ID')
    return inside(attempt_root/identity,attempt_root,'queue job')


def package_inputs(package,refs):
    preparation=inside(package/'preparation.json',package,'preparation')
    data=json_input(preparation,refs)
    documents={}
    for name in ('campaign','jobs'):
        ref=data[name]
        path=inside(ref['path'],package,'bound '+name)
        documents[name]=json_input(path,refs)
        if refs[-1]!=ref:raise ValueError('preparation '+name+' binding differs: '+str(path))
    return preparation,data,documents


def discover(repair_root,attempt_root,historical):
    repair_root=Path(repair_root).resolve();attempt_root=Path(attempt_root).resolve()
    packages=[repair_root/'paired-ab',repair_root/'paired-ab-v2']
    followup=repair_root/'followup/followup-status.json'
    if followup.exists():
        # The watcher status changes on every poll; bind its selected package,
        # not the status file itself, so polling cannot trigger new reports.
        matrix=json_input(inside(followup,repair_root,'followup status'),[]).get('matrix_path')
        if matrix:
            path=inside(matrix,repair_root/'followup','matrix package')
            if (path/'preparation.json').exists():packages.append(path)
    campaigns=[];job_dirs=[];refs=[];preparations=[]
    for package in packages:
        package=inside(package,repair_root,'repair package')
        preparation,data,documents=package_inputs(package,refs)
        preparations.append(str(preparation));campaigns.append(data['campaign']['path'])
        for spec in documents['jobs']:
            job_dirs.append(str(job_directory(spec,attempt_root)))
    supplement,_,supplement_documents=package_inputs(
        inside(repair_root/'baseline-supplements',repair_root,'supplement package'),refs)
    json_input(inside(Path(historical)/'points.json',historical,'historical points'),refs)
    supplement_dirs=[]
    for spec in supplement_documents['jobs']:
        supplement_dirs.append(job_directory(spec,attempt_root))
    receipts=set()
    for directory in [*map(Path,job_dirs),*supplement_dirs]:
        for path in directory.glob('attempt-*/session/windows/*/receipt.json'):
            receipts.add(inside(path,directory,'receipt'))
    for path in sorted(receipts):json_input(path,refs)
    inputs=dict(campaigns=campaigns,job_dirs=sorted(set(job_dirs)),repair_preparations=preparations,
                supplement_preparation=str(supplement),
                supplement_attempt_root=str(attempt_root),historical_dir=str(Path(historical).resolve()),
                refs=sorted({ref['path']:ref for ref in refs}.values(),key=lambda ref:ref['path']))
    return inputs


def command(report_script,inputs,output,python=sys.executable):
    argv=[str(python),'-B',str(report_script),'--historical-dir',inputs['historical_dir'],'--all-models',
          '--supplement-preparation',inputs['supplement_preparation'],
          '--attempt-root',inputs['supplement_attempt_root'],'--out',str(output)]
    for path in inputs['repair_preparations']:argv.extend(['--repair-preparation',path])
    return argv


def verify_program(manifest):
    data=load(manifest)
    declared={}
    for ref in data['files']:
        if set(ref)!={'path','sha256'} or not Path(ref['path']).is_absolute():
            raise ValueError('frozen report file must be an absolute binding')
        if ref['path'] in declared:raise ValueError('duplicate frozen report file: '+ref['path'])
        if binding(ref['path'])!=ref:raise ValueError('frozen report program changed: '+ref['path'])
        declared[ref['path']]=ref
    if str(Path(data['report_script']).resolve()) not in declared:
        raise ValueError('report_script is not bound by the frozen program manifest')
    if not Path(data['report_script']).is_absolute():raise ValueError('report_script must be absolute')
    return data


def text_output(value):
    return value.decode(errors='replace') if isinstance(value,bytes) else value or ''


def save_new(path,value):
    if Path(path).exists():raise ValueError('immutable invocation already exists: '+str(path))
    save(path,value)


def run_once(args):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    program=verify_program(args.program_manifest)
    identity=dict(program_manifest=binding(args.program_manifest),watcher=binding(__file__),
                  repair_root=str(Path(args.repair_root).resolve()),attempt_root=str(Path(args.attempt_root).resolve()),
                  historical_dir=str(Path(args.historical_dir).resolve()))
    state_path=out/'state.json';state=load(state_path) if state_path.exists() else dict(identity=identity,reports=[])
    if state['identity']!=identity:raise ValueError('report watcher input identity changed')
    try:
        inputs=discover(args.repair_root,args.attempt_root,args.historical_dir)
    except InputPending as error:
        state.update(status='waiting_for_input',input_problem=error.problem)
        save(state_path,state);return state
    key=digest(inputs)
    state.pop('input_problem',None)
    if state.get('latest_input_sha256')==key:
        latest=state['reports'][-1]
        if binding(latest['snapshot']['path'])!=latest['snapshot']:
            raise ValueError('published report snapshot changed')
        # Recover a crash between the state and latest-pointer commits without
        # rerunning the CPU program or touching its immutable output.
        latest_path=out/'latest.json'
        if not latest_path.exists() or load(latest_path)!=latest:save(latest_path,latest)
        state.update(status='watching');save(state_path,state);return state
    if state.get('processed_input_sha256')==key:return state
    target=out/'snapshots'/key
    invocation=out/'invocations'/(key+'.json')
    if target.exists() or invocation.exists():
        # A crash can leave an output before state.json was committed. Never
        # rerun into it or replace its evidence; require an explicit review.
        state.update(status='report_needs_diagnosis',processed_input_sha256=key,
                     failed_invocation=str(invocation),error='prior immutable report output exists')
        save(state_path,state);return state
    target.parent.mkdir(parents=True,exist_ok=True)
    argv=command(program['report_script'],inputs,target)
    receipt=dict(input_sha256=key,inputs=inputs,program_manifest=identity['program_manifest'],argv=argv,
                 returncode=None,stdout='',stderr='',status='failed',hardware_executed=False)
    try:
        process=subprocess.run(argv,text=True,capture_output=True,timeout=1800,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
        receipt.update(returncode=process.returncode,stdout=process.stdout,stderr=process.stderr)
        if process.returncode:raise RuntimeError('report process exited '+str(process.returncode))
        verify_program(args.program_manifest)
        if binding(args.program_manifest)!=identity['program_manifest'] or binding(__file__)!=identity['watcher']:
            raise ValueError('frozen report program identity changed during execution')
        post_inputs=discover(args.repair_root,args.attempt_root,args.historical_dir)
        receipt['post_input_sha256']=digest(post_inputs)
        if receipt['post_input_sha256']!=key:
            receipt.update(status='superseded',error='report inputs changed during execution')
        else:
            snapshot=inside(target/'snapshot.json',target,'report snapshot')
            report=inside(target/'report.md',target,'report markdown')
            load(snapshot)
            receipt.update(status='succeeded',snapshot=binding(snapshot),report=binding(report))
    except InputPending as error:
        receipt.update(status='superseded',error=str(error),input_problem=error.problem)
    except Exception as error:
        receipt['error']=repr(error)
        if isinstance(error,subprocess.TimeoutExpired):
            receipt.update(stdout=text_output(error.stdout),stderr=text_output(error.stderr))
    receipt['finished_s']=time.time()
    save_new(invocation,receipt)
    state.update(processed_input_sha256=key)
    if receipt['status']!='succeeded':
        state.update(status='waiting_for_input' if receipt['status']=='superseded' else 'report_needs_diagnosis',
                     failed_invocation=str(invocation),error=receipt['error'])
        save(state_path,state);return state
    for name in ('failed_invocation','error'):state.pop(name,None)
    state['reports'].append(dict(input_sha256=key,path=str(target),snapshot=receipt['snapshot'],
                                finished_s=receipt['finished_s']))
    state.update(status='watching',latest_input_sha256=key,latest_report=str(target/'report.md'))
    save(state_path,state);save(out/'latest.json',state['reports'][-1])
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for option in ('repair-root','attempt-root','historical-dir','program-manifest','out'):
        p.add_argument('--'+option,type=Path,required=True)
    p.add_argument('--watch',action='store_true');args=p.parse_args()
    import fcntl
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'watcher.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=None
        while True:
            try:state=run_once(args)
            except Exception as error:
                save(args.out/'watcher-error.json',dict(error=repr(error),failed_s=time.time()))
                raise
            update={key:state.get(key) for key in ('status','latest_report','processed_input_sha256',
                                                 'input_problem','failed_invocation','error')}
            if update!=prior:
                print(json.dumps(update),flush=True);prior=update
            if not args.watch:return
            time.sleep(15.)


if __name__=='__main__':main()
