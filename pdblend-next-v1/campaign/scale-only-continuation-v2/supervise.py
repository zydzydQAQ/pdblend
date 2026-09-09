"""Explicit scale-only continuation. No deployment, phase waiter or fake release."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import contract as c
import reference_map

ROOT=Path(__file__).resolve().parent

def package_check():
    require=c.require
    require('PDBLEND_NODE_LOCK_FD' not in os.environ,'supervisor may not inherit an active lease')
    manifest=c.read(ROOT/'manifest.json')
    for name,h in manifest['files'].items():require(c.sha(ROOT/name)==h,'scale supervisor source changed')
    for path,h in manifest['dependencies'].items():require(c.sha(path)==h,'frozen dependency changed')

def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def make_argv(source,binding,system,datasets,map_path,map_sha):
    return [sys.executable,'-u',str(c.DRIVER),'--manifest',source,'--binding',binding,'--system',system,
        '--phase','scale','--max-cells','1','--main-reference-map',str(map_path),'--main-reference-sha256',map_sha,
        '--run',*[v for ds in datasets for v in ('--dataset',ds)]]

class Supervisor:
    def __init__(self,args):
        self.a=args;self.stopped=False;self.state=None
    def boundary(self):
        c.require(not self.stopped and not (self.a.out/'STOP').exists() and not (ROOT/'STOP').exists(),'scale STOP: no successor')
        c.require(time.time()+520<c.DEADLINE,'original deadline cannot fit a bounded scale cell')
        c.require(c.sha(self.a.spec)==self.a.spec_sha256,'scale continuation spec changed')
        package_check()
    def command(self,group):
        self.boundary();binding=group['group']['scale_binding']['path'];b=group['current']
        c.require(not (Path(b['output'])/'STOP').exists(),'original queue STOP remains set; no implicit deletion')
        fixed=self.state['reference_maps'][group['group']['id']]
        reference_map.verify(fixed['path'],fixed['sha256'],self.state['source_manifest'],binding,b['system'],group['group']['datasets'])
        argv=make_argv(self.state['source_manifest'],binding,b['system'],group['group']['datasets'],fixed['path'],fixed['sha256'])
        step=dict(group=group['group']['id'],argv=argv,started_s=time.time(),complete=False)
        self.state['steps'].append(step);save(self.a.out/'status.json',self.state)
        h=Path(b['host_release']);environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',
            PYTHONPATH=f'{h}/src:{h}:/root/workspace/pdblend/.runtime-deps')
        # v3 owns its fresh node lease and all8 measurement/native cleanup.
        # Parent never inherits it and never sends HTTP or stops model containers.
        interrupt_at=min(c.DEADLINE-120,step['started_s']+400)
        with (self.a.out/f'cell-{len(self.state["steps"]):03d}.log').open('xb') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                env=environment,start_new_session=True,close_fds=True)
            step['pid']=child.pid;save(self.a.out/'status.json',self.state);sent=None
            while child.poll() is None:
                # Ordinary STOP does not interrupt the running measurement: each
                # subprocess can create at most one CP and exits at its boundary.
                if time.time()>=interrupt_at and sent is None:
                    child.send_signal(signal.SIGINT);sent=time.time();step['deadline_interrupt_s']=sent;save(self.a.out/'status.json',self.state)
                if time.time()>=c.DEADLINE or (sent is not None and time.time()-sent>=120):
                    step['unconfirmed_child']=True;save(self.a.out/'status.json',self.state)
                    raise RuntimeError('owned v3 child cleanup/exit unconfirmed; no successor or deployment; owner must recover')
                time.sleep(.5)
            step.update(finished_s=time.time(),complete=True,exitcode=child.returncode)
            save(self.a.out/'status.json',self.state)
        c.require(sent is None and child.returncode==0,'scale child failed/timed out; its raw/energy retained, no automatic retry')
    def run(self):
        package_check();c.require(c.sha(self.a.spec)==self.a.spec_sha256,'unfixed scale spec')
        spec=c.read(self.a.spec)
        # The very first gate precedes output creation, any process or hardware.
        checked=c.check_spec(spec,self.a.release,self.a.release_sha256,self.a.group)
        c.require(socket.gethostname()==checked['hostname'],'scale execution must run on actual bound host')
        c.require(not self.a.out.exists(),'new supervisor attempt directory required')
        self.a.out.mkdir(parents=True)
        self.state=dict(schema=2,pid=os.getpid(),started_s=time.time(),phase='scale_only',complete=False,
            model=spec['model'],protocol_id=c.PROTOCOL,deadline_s=c.DEADLINE,source_manifest=spec['source']['path'],
            source_sha256=spec['source']['sha256'],spec_sha256=self.a.spec_sha256,release_sha256=self.a.release_sha256,
            release_scope=checked['release']['verified_scope'],steps=[],reused=[],reference_maps={},parent_owns_node_lease=False)
        save(self.a.out/'status.json',self.state)
        try:
            for original in checked['groups']:
                name=original['group']['id'];remaining=original
                self.state['reused'].extend(original['reused']);save(self.a.out/'status.json',self.state)
                if remaining['pending']:
                    value=reference_map.build(self.a.spec,self.a.spec_sha256,self.a.release,self.a.release_sha256,name)
                    path=self.a.out/('main-references-'+str(len(self.state['reference_maps']))+'.json')
                    with path.open('x') as handle:json.dump(value,handle,indent=2,allow_nan=False)
                    self.state['reference_maps'][name]=dict(path=str(path.resolve()),sha256=c.sha(path))
                    save(self.a.out/'status.json',self.state)
                while remaining['pending']:
                    self.boundary()
                    # Revalidate pinned global release and original main refs at
                    # every cell boundary; an elapsed deadline is never a release.
                    current=c.check_spec(spec,self.a.release,self.a.release_sha256,[name])['groups'][0]
                    before={r['cell_id'] for r in current['pending']}
                    if not before:break
                    self.command(current)
                    after=c.check_spec(spec,self.a.release,self.a.release_sha256,[name])['groups'][0]
                    newly=before-{r['cell_id'] for r in after['pending']}
                    c.require(len(newly)==1,'v3 did not produce exactly one valid declared scale CP')
                    row=next(r for r in after['rows'] if r['cell_id'] in newly)
                    proof=c.verify_existing(row,[current['group']['scale_binding']],spec['source']['sha256'],
                        required_binding=current['group']['scale_binding'])
                    proof['reused']=False
                    self.state['steps'][-1]['verified_new_checkpoint']=proof;save(self.a.out/'status.json',self.state)
                    remaining=after
            self.state.update(complete=True,phase='selected_scale_groups_finished')
        except BaseException as exc:
            self.state.update(phase='stopped' if self.stopped else 'failed',error=repr(exc));raise
        finally:
            self.state['finished_s']=time.time();save(self.a.out/'status.json',self.state)
        return self.state

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--spec-sha256',required=True)
    p.add_argument('--release',type=Path,default=c.barrier.DEFAULT_RELEASE);p.add_argument('--release-sha256',required=True)
    p.add_argument('--group',action='append');p.add_argument('--out',type=Path);p.add_argument('--run',action='store_true')
    a=p.parse_args();package_check();c.require(c.sha(a.spec)==a.spec_sha256,'spec SHA differs')
    if not a.run:
        result=c.check_spec(c.read(a.spec),a.release,a.release_sha256,a.group)
        print(json.dumps({k:v for k,v in result.items() if k!='groups'},indent=2));return
    c.require(a.out is not None,'--run requires a new --out')
    s=Supervisor(a)
    def stop(signum,frame):s.stopped=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    s.run()

if __name__=='__main__':main()
