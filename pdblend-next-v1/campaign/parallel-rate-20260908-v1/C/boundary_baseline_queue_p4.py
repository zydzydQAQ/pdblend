"""Fresh baseline bootstrap/gate, then all four systems once before repeats."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
CAMPAIGN=HERE.parents[1]
OUT=HERE/'boundary-baseline-queue-p4'
SPEC=HERE/'baseline-boundary-restore-p4/deployment.json'
CAMPAIGN_LIFECYCLE='until_declared_complete_v1'
def require(ok,why):
    if not ok:raise RuntimeError(why)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)

class Queue:
    def __init__(self):
        require(not OUT.exists(),'new queue attempt required');OUT.mkdir()
        self.stop=False;self.state=dict(pid=os.getpid(),started_s=time.time(),complete=False,phase='starting',steps=[],completed_groups=[],node_lease_held=False)
    def save(self):write(OUT/'status.json',self.state)
    def boundary(self,reserve):
        require(not self.stop and not (OUT/'STOP').exists(),'stop requested; no successor')
        manifest=read(HERE/'boundary-baseline-package-p4.json')
        for p,h in manifest['files'].items():require(sha(p)==h,'baseline queue input changed: '+p)
    def command(self,name,argv,reserve=400,max_work_s=None,performance_out=None):
        self.boundary(reserve)
        spec=read(SPEC);host=Path(spec['host_release'])
        env=dict(os.environ,PYTHONPATH=f'{host}/src:{host}:/root/workspace/pdblend/.runtime-deps',PYTHONDONTWRITEBYTECODE='1')
        step=dict(name=name,argv=argv,started_s=time.time(),complete=False);self.state['steps'].append(step);self.state['phase']=name;self.save()
        with (OUT/(name+'.log')).open('xb') as stream:
            child=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
            step['pid']=child.pid;self.save();stopped=False;stop_cleanup_deadline=None
            while child.poll() is None:
                exceeded=max_work_s is not None and time.time()>step['started_s']+max_work_s
                if (self.stop or (OUT/'STOP').exists() or exceeded) and not stopped:
                    if performance_out is not None:
                        stopfile=performance_out/'STOP';stopfile.parent.mkdir(parents=True,exist_ok=True)
                        with stopfile.open('x') as f:f.write('Owned queue boundary stop; retain current full cell and cleanup.\n')
                        step['boundary_stop_s']=time.time()
                    else:
                        child.send_signal(signal.SIGINT);step['owned_interrupt_s']=time.time()
                    stopped=True;stop_cleanup_deadline=time.time()+400;self.save()
                if stop_cleanup_deadline is not None and time.time()>stop_cleanup_deadline:
                    step['unconfirmed_child']=True;self.save();raise RuntimeError('owned child did not confirm terminal state within bounded cleanup; no successor')
                time.sleep(1)
            step.update(exitcode=child.returncode,complete=True,finished_s=time.time());self.save()
        require(not stopped and child.returncode==0,'owned stage failed/stopped; no successor or retry')
    def run(self):
        try:
            self.boundary(1000)
            receipt=SPEC.parent/'deployment-receipt.json';r=read(receipt)
            require(r.get('complete') is True and r.get('measurement_valid') is True and not r.get('errors'),
                    'fresh measured C8 restoration missing')
            bootstrap=HERE/'boundary-baseline-bootstrap-p4';binder=HERE/'baseline-until-complete-v1/bind.py'
            self.command('bootstrap',[sys.executable,'-B',str(binder),'--spec',str(SPEC),'--receipt',str(receipt),'--out',str(bootstrap)],650,120)
            require(read(bootstrap/'binding.json')['configs']=={},'bootstrap cannot be performance qualified')
            gate=HERE/'boundary-baseline-gate-p4';entry=HERE/'baseline-until-complete-v1/validate.py'
            runtime=read(read(SPEC)['instances'][0]['config'])['runtime_dir']
            self.command('ordinary-pd-temporal',[sys.executable,'-B',str(entry),'--binding',str(bootstrap/'binding.json'),'--runtime-dir',runtime,'--out',str(gate),'--run'],650,520)
            evidence=read(gate/'status.json')
            require(evidence.get('complete') is True and evidence.get('measurement_valid') is True
                    and evidence.get('mechanism_gate')==dict(ordinary=True,pd=True,temporal=True),
                    'all original ordinary/PD/temporal mechanisms must pass before any baseline performance')
            declaration=read(HERE/'boundary-p4/declaration.json')
            for system in ('mixed','distserve','dynamollm','ecoserve'):
                rows=[c for c in declaration['cells'] if c['system']==system]
                if not rows:continue
                datasets=sorted({c['dataset'] for c in rows})
                name=system
                out=HERE/'boundary-baselines-p4'/system
                binding=HERE/'boundary-baseline-bindings-p4'/name
                strategy='dynamollm-resident' if system=='dynamollm' else system
                da=[v for dataset in datasets for v in ['--dataset',dataset]]
                self.command(name+'-bind',[sys.executable,'-B',str(binder),'--spec',str(SPEC),'--receipt',str(receipt),'--out',str(binding),
                    '--output',str(out/'results'),'--strategy',strategy,'--gate',str(gate),*da],500,120)
                self.command(name+'-measure',[sys.executable,'-B',str(HERE/'run_boundary_baseline_p4.py'),
                    '--binding',str(binding/'binding.json'),'--out',str(out),'--system',system,'--run'],400,max_work_s=400*len(rows)+90,performance_out=out)
                status=read(out/'status.json')
                require(status.get('complete') is True and status.get('phase')=='complete' and len(status.get('completed',[]))==len(rows)
                        and not status.get('failed') and not status.get('engineering_gate_failed') and status.get('node_lease_held') is False,
                        'baseline endpoint group did not cleanly complete')
                self.state['completed_groups'].append(name);self.save()
            self.state.update(complete=True,phase='complete')
        except BaseException as exc:self.state.update(phase='stopped' if self.stop else 'failed',error=repr(exc));raise
        finally:self.state['finished_s']=time.time();self.save()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args()
    require('PDBLEND_NODE_LOCK_FD' not in os.environ,'supervisor must not inherit a node lease')
    if not a.run:
        m=read(HERE/'boundary-baseline-package-p4.json')
        for f,h in m['files'].items():require(sha(f)==h,'queue source changed')
        print(json.dumps(dict(passed=True,cpu_only=True,baseline_cells=len(read(HERE/'boundary-p4/declaration.json')['cells']),all_newly_explored_rates_paired=True)));return
    q=Queue()
    def stop(sig,frame):q.stop=True;q.state['stop_requested_s']=time.time();q.save()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    q.run()

if __name__=='__main__':main()
