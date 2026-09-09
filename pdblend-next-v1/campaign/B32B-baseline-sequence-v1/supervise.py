"""Wait for PDB100, then run only terminally verified original baseline mechanisms."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
import signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
R=ROOT.parents[1];P=R/'campaign/B32B-five-system100-v1'
SPEC=R/'campaign/B32B-five-system100-baseline-deployment-v1/deployment.json'
SPEC_SHA='c3caa8ea6efcf46c08ec8d703891148e8c4e09c12249034d382bf94bf728f382'
DEADLINE=1788872770.0400891
SYSTEMS=('mixed','dynamollm-resident','ecoserve','distserve')

def require(ok,why):
    if not ok:raise RuntimeError(why)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def decide_system(gate,checks,system):
    require(gate.get('complete') is True and gate.get('measurement_valid') is True and gate.get('native_cleanup_complete') is True and gate.get('clock_restore_complete') is True and not gate.get('cleanup_errors'),'gate incomplete/invalid: no baseline may proceed')
    require(checks.get('complete') is True and checks.get('cleanup',{}).get('complete') is True,'raw gate not terminal/clean')
    gates=gate.get('mechanism_gate',{});requirements={'mixed':('ordinary',),'distserve':('ordinary','pd'),'dynamollm-resident':('ordinary',),'ecoserve':('ordinary','temporal')}[system]
    return all(gates.get(k) is True for k in requirements),list(requirements)

def check_files():
    require('PDBLEND_NODE_LOCK_FD' not in os.environ,'supervisor must not inherit or own a node lease')
    require(sha(SPEC)==SPEC_SHA,'declared baseline spec changed')
    m=read(ROOT/'manifest.json')
    for path,digest in m['files'].items():require(sha(path)==digest,'supervision input changed: '+path)

def command_deadlines(started_s,max_work_s,has_queue_stop):
    # A queue gets enough advance notice to finish 100 s arrivals, 120 s drain,
    # and its native cleanup; the later signal is only a fail-stop fallback.
    boundary_stop_s=DEADLINE-400 if has_queue_stop else None
    interrupt_s=min(DEADLINE-125,started_s+max_work_s) if max_work_s else DEADLINE-125
    return boundary_stop_s,interrupt_s

class Supervisor:
    def __init__(self,out):
        require(not out.exists(),'fresh supervision attempt required');out.mkdir(parents=True)
        self.out=out;self.active=None;self.active_stop=None;self.stop_requested=False
        self.state=dict(pid=os.getpid(),started_s=time.time(),complete=False,phase='waiting_for_pdb48',steps=[],skipped_systems={},deadline_s=DEADLINE,parent_owns_lease=False)
        self.save()
    def save(self):write(self.out/'status.json',self.state)
    def stopping(self):return self.stop_requested or (ROOT/'STOP').exists() or (self.out/'STOP').exists()
    def boundary(self,reserve=400):
        require(not self.stopping(),'supervision STOP requested; do not start another stage')
        require(time.time()+reserve<DEADLINE,'original global deadline leaves insufficient stage/cleanup allowance')
        check_files()
    def command(self,label,args,*,reserve=400,stop_path=None,max_work_s=None,allowed_exitcodes=(0,)):
        self.boundary(reserve);step=dict(label=label,argv=args,started_s=time.time(),complete=False);self.state['steps'].append(step);self.state['phase']=label;self.save()
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');host=Path(read(SPEC)['host_release']);env['PYTHONPATH']=f'{host}/src:{host}:/root/workspace/pdblend/.runtime-deps'
        boundary_stop_s,deadline=command_deadlines(step['started_s'],max_work_s,stop_path is not None)
        with (self.out/(label+'.log')).open('xb') as log:
            child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
            self.active=child;self.active_stop=stop_path;step['pid']=child.pid;self.save();sent=None
            while child.poll() is None:
                if boundary_stop_s is not None and time.time()>=boundary_stop_s:
                    self.stop_requested=True;step.setdefault('global_deadline_boundary_stop_s',time.time())
                if self.stopping() and stop_path is not None and not Path(stop_path).exists():
                    Path(stop_path).parent.mkdir(parents=True,exist_ok=True);Path(stop_path).write_text('B supervision boundary STOP; finish current cell and native cleanup\n');step['boundary_stop_requested_s']=time.time();self.save()
                if time.time()>=deadline and sent is None:
                    # SIGINT unwinds asyncio deployment; queue/gate handlers own their real cleanup.
                    child.send_signal(signal.SIGINT);sent=time.time();step['deadline_interrupt_s']=sent;self.save()
                if sent is not None and time.time()-sent>125:
                    # Never run a successor or issue engine controls while an owned host child is unconfirmed.
                    step['unconfirmed_child_after_cleanup_allowance']=True;self.save();raise RuntimeError('owned stage did not exit after its bounded cleanup; fail stop, no successor')
                time.sleep(1)
            step.update(exitcode=child.returncode,finished_s=time.time(),complete=True);self.active=None;self.active_stop=None;self.save()
        require(sent is None,'owned stage exceeded declared execution allowance; preserve partial evidence')
        require(not self.stopping(),'boundary STOP reached; no next stage')
        require(child.returncode in allowed_exitcodes,label+' failed; no automatic retry')
        return child.returncode
    def run(self):
        spec=read(SPEC);deploy=load('b_sequence_deploy',spec['deployment_implementation']);binder=P/'bind_baseline_v2.py';workloads=Path(spec['workloads']);execution=Path(spec['executor_release'])/'run.py'
        try:
            while True:
                self.boundary(840)
                predecessor=read(P/'bridge-pdblend-r2/status.json')
                require(predecessor.get('phase')!='failed','PDB predecessor bridge failed')
                if predecessor.get('complete') is True and predecessor.get('phase')=='finished':break
                self.state['predecessor_phase']=predecessor.get('phase');self.save();time.sleep(15)
            proof=deploy.terminal_group(spec['pdb_binding'],workloads);write(self.out/'pdb-predecessor-proof.json',proof)
            self.command('deploy',[sys.executable,str(deploy.__file__),'deploy','--spec',str(SPEC),'--run'],reserve=960,max_work_s=900)
            receipt=Path(spec['out'])/'deployment-receipt.json';r=read(receipt);require(r.get('complete') is True and r.get('measurement_valid') is True,'deployment did not prove readiness and energy')
            bootstrap=self.out/'correctness-binding'
            self.command('bind-correctness',[sys.executable,str(binder),'--spec',str(SPEC),'--receipt',str(receipt),'--out',str(bootstrap)],reserve=650,max_work_s=120)
            binding=bootstrap/'binding.json';require(read(binding)['configs']=={},'bootstrap must not bind performance configs')
            gate_out=self.out/'legacy-correctness';gate=R/'campaign/B32B-legacy-baseline-correctness-v1/validate.py'
            code=self.command('correctness',[sys.executable,str(gate),'--binding',str(binding),'--runtime-dir',str(Path(spec['out'])/'runtime'),'--out',str(gate_out),'--run'],reserve=650,max_work_s=520,allowed_exitcodes=(0,1))
            gs=read(gate_out/'status.json');checks=read(gate_out/'checks/checks.json')
            write(self.out/'correctness-invocation.json',dict(binding=str(binding),binding_sha256=sha(binding),entry=str(gate),entry_sha256=sha(gate),exitcode=code,status_sha256=sha(gate_out/'status.json'),checks_sha256=sha(gate_out/'checks/checks.json')))
            for system in SYSTEMS:
                allowed,needed=decide_system(gs,checks,system)
                if not allowed:
                    self.state['skipped_systems'][system]=dict(reason='required exact mechanism failed',required=needed,gate=str(gate_out/'status.json'));self.save();continue
                dest=self.out/'bindings'/system
                self.command('bind-'+system,[sys.executable,str(binder),'--spec',str(SPEC),'--receipt',str(receipt),'--out',str(dest),'--strategy',system,'--gate',str(gate_out)],max_work_s=120)
                b=dest/'binding.json';bound=read(b);canonical=bound['system'];output=Path(bound['output'])
                for phase,count in (('main',30),('scale',18)):
                    if phase=='scale':
                        bridge=load('b_sequence_bridge',R/'campaign/five-system-orchestration-v1/bridge.py');require(bridge.verify_main_references(read(workloads),bound,canonical)==30,'scale needs all exact main references')
                    self.command(system+'-'+phase,[sys.executable,str(execution),'--manifest',str(workloads),'--binding',str(b),'--system',canonical,'--phase',phase,'--max-cells',str(count),'--run'],stop_path=output/'STOP')
                write(self.out/(system+'-terminal-proof.json'),deploy.terminal_group(b,workloads,system=canonical))
            self.state.update(complete=True,phase='finished',all_four_baselines_measured=not bool(self.state['skipped_systems']))
        except BaseException as exc:
            self.state.update(phase='stopped' if self.stopping() else 'failed',error=repr(exc));raise
        finally:self.state['finished_s']=time.time();self.save()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path);p.add_argument('--run',action='store_true');a=p.parse_args();check_files()
    if not a.run:print(json.dumps(dict(cpu_only=True,hardware_actions=False,spec_sha256=SPEC_SHA)));return
    require(a.out is not None,'new supervision output required');s=Supervisor(a.out.resolve())
    def stop(signum,frame):s.stop_requested=True;s.state['signal_stop_requested_s']=time.time();s.save()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    s.run()
if __name__=='__main__':main()
