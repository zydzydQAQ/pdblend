"""Observe actual B baseline queue processes and update only the host index."""
import argparse,hashlib,importlib.util,json,os,socket,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
R=ROOT.parents[1]
SOURCE=R/'campaign/B32B-baseline-main-first-sequence-v1/attempt-001/status.json'
PUBLISH_SHA='46564c454c0a62b7da0b86c218f25789253ccc9a91500aacc07b0644ca6fe41f'
RUNNER=R/'campaign/five-system-execution-v3/run.py'
HOSTNAME='iZwz9i5bte3xkpmcoes3t2Z'
DEADLINE=1788872770.0400891

def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def check():
    require(sha(ROOT/'publish.frozen.py')==PUBLISH_SHA,'frozen publisher changed')
    for p,h in read(ROOT/'manifest.json')['files'].items():require(sha(p)==h,'observer dependency changed: '+p)

def queue_step(step):
    argv=step.get('argv',[])
    if str(RUNNER) not in argv:return None
    require('--run' in argv,'only actual serving queue invocations may publish')
    def arg(k):
        require(argv.count(k)==1,'missing or ambiguous queue field '+k)
        return argv[argv.index(k)+1]
    binding=Path(arg('--binding'));manifest=Path(arg('--manifest'));phase=arg('--phase')
    require(phase in ('main','scale'),'unexpected execution phase')
    b=read(binding);require(b['model']=='32b' and b['hostname']==HOSTNAME,'wrong physical scope')
    require(b['system']==arg('--system'),'actual system/binding mismatch')
    datasets=[argv[i+1] for i,v in enumerate(argv) if v=='--dataset'] or list(b['configs'])
    require(set(datasets)==set(b['configs']) and len(datasets)==len(set(datasets)),'selected configuration mismatch')
    require(len(datasets)==3,'B queue must bind the original three datasets')
    return dict(binding=binding,manifest=manifest,phase=phase,datasets=datasets)

def live_identity(step,proc_root=Path('/proc')):
    pid=step['pid'];require(type(pid) is int and pid>0,'invalid observed PID')
    p=proc_root/str(pid);argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
    require(argv==step['argv'],'actual queue argv differs from supervisor declaration')
    # starttime distinguishes a recycled PID. A process name may contain spaces.
    stat=(p/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
    require(tail[0]!='Z','queue is already a zombie')
    return dict(pid=pid,argv=argv,proc_start_ticks=int(tail[19]))

class Observer:
    def __init__(self,out,publisher):
        require(not out.exists(),'fresh observer output required');out.mkdir(parents=True)
        self.out=out;self.publisher=publisher
        self.state=dict(pid=os.getpid(),started_s=time.time(),complete=False,source=str(SOURCE),observed={},publications=[],pending_process_observations={},errors=[],hardware_actions=False,service_configuration_writes=False)
        self.save()
    def save(self):write(self.out/'status.json',self.state)
    def tick(self,source):
        self.state['source_phase']=source.get('phase');self.state['last_checked_s']=time.time()
        for step in source.get('steps',[]):
            q=queue_step(step)
            if q is None:continue
            # The supervisor journals creation intent before Popen returns a PID.
            if 'pid' not in step:
                require(step.get('complete') is not True,'completed queue step lacks its PID')
                continue
            key=str(step['pid'])+':'+step['label'];prior=self.state['observed'].get(key)
            terminal=step.get('complete') is True
            if terminal and prior is None:continue # Never invent a historical actual-PID observation.
            if prior is not None and prior.get('terminal_published'):continue
            if not terminal:
                try:identity=live_identity(step)
                except FileNotFoundError:
                    # Exit can precede the supervisor's terminal status write.
                    self.state['pending_process_observations'].setdefault(key,time.time())
                    continue
                self.state['pending_process_observations'].pop(key,None)
                if prior is not None:
                    require(identity==prior['identity'],'PID recycled or argv changed')
                    require(sha(q['binding'])==prior['binding_sha256'] and sha(q['manifest'])==prior['manifest_sha256'],'observed inputs changed')
                    continue
                prior=dict(identity=identity,binding_sha256=sha(q['binding']),manifest_sha256=sha(q['manifest']),observed_s=time.time())
                self.state['observed'][key]=prior;self.save()
            require(sha(q['binding'])==prior['binding_sha256'] and sha(q['manifest'])==prior['manifest_sha256'],'observed inputs changed')
            if terminal:require(type(step.get('exitcode')) is int,'terminal step lacks a real exit status')
            result=self.publisher.update(q['binding'],q['manifest'],step['pid'],q['phase'],q['datasets'],SOURCE,step['exitcode'] if terminal else None)
            prior['terminal_published']=terminal
            self.state['publications'].append(dict(key=key,terminal=terminal,published_s=time.time(),receipt=result));self.save()
        if source.get('phase') in ('finished','failed','stopped','main_complete','main_incomplete_correctness'):
            self.state.update(complete=True,finished_s=time.time())
        self.save()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path);p.add_argument('--run',action='store_true');a=p.parse_args();check()
    if not a.run:print(json.dumps(dict(cpu_only=True,index_writes=False,hardware_actions=False,publisher_sha256=PUBLISH_SHA)));return
    require(socket.gethostname()==HOSTNAME,'run only on the actual B host');require(a.out is not None,'fresh output required')
    s=importlib.util.spec_from_file_location('b_actual_host_index',ROOT/'publish.frozen.py');publisher=importlib.util.module_from_spec(s);s.loader.exec_module(publisher)
    obs=Observer(a.out.resolve(),publisher)
    try:
        while time.time()<DEADLINE+180 and not (ROOT/'STOP').exists():
            try:
                check();obs.tick(read(SOURCE))
            except Exception as exc:
                obs.state['errors'].append(dict(at_s=time.time(),error=repr(exc)));obs.save()
                raise # Never silently advertise an unverified process or retry an index mutation.
            if obs.state['complete']:break
            time.sleep(2)
    finally:obs.state['observer_finished_s']=time.time();obs.save()
if __name__=='__main__':main()
