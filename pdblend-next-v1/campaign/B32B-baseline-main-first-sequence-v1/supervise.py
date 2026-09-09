"""B main-first continuation; reuse frozen executor and native cleanup unchanged."""
import argparse,hashlib,importlib.util,json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
R=ROOT.parents[1]
OLD_ROOT=R/'campaign/B32B-baseline-sequence-v1'
OLD_OUT=OLD_ROOT/'attempt-001'
SPEC=R/'campaign/B32B-five-system100-baseline-deployment-v1/deployment.json'
DEADLINE=1788872770.0400891
EXPECTED_STOP='B supervision boundary STOP; finish current cell and native cleanup\n'
SYSTEMS=('mixed','dynamollm-resident','ecoserve','distserve')

def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
old=load('b_frozen_sequence',OLD_ROOT/'supervise.py')
read,write,sha,require=old.read,old.write,old.sha,old.require
binder=load('b_frozen_baseline_binder',R/'campaign/B32B-five-system100-v1/bind_baseline_v2.py')

def check_files():
    old.check_files()
    for p,h in read(ROOT/'manifest.json')['files'].items():require(sha(p)==h,'main-first source changed: '+p)

def pid_live(pid,proc=Path('/proc')):
    p=proc/str(pid)
    try:
        cmd=(p/'cmdline').read_bytes();s=(p/'stat').read_text();state=s[s.rfind(')')+2:].split()[0]
        return bool(cmd) and state!='Z'
    except FileNotFoundError:return False

def old_terminal(status,live=pid_live):
    require(status.get('phase')=='stopped' and status.get('finished_s'), 'old supervisor must be observed stopped')
    require(not live(status['pid']), 'old supervisor still live')
    steps=status.get('steps',[]);require(steps, 'old steps absent')
    for step in steps:
        require(step.get('complete') is True and type(step.get('exitcode')) is int, 'unfinished old stage')
        if step.get('pid'):require(not live(step['pid']), 'old stage PID still live')
    require(steps[-1]['exitcode']==0 and steps[-1]['label'] in ('mixed-main','mixed-scale'),
            'old current queue did not exit cleanly at requested boundary')
    return True

def verify_phase(binding,manifest,phase,require_all=True):
    b=read(binding) if not isinstance(binding,dict) else binding
    rows=[r for r in manifest['cells'] if r['system']==b['system'] and r['phase']==phase]
    require(len(rows)==(30 if phase=='main' else 18),'exact declared phase count')
    records=[];gap=False
    for row in rows:
        cp=Path(b['output'])/'checkpoints'/(row['cell_id']+'.json')
        if not cp.exists():gap=True;continue
        require(not gap, 'completed checkpoints must be an exact original-order prefix')
        v=read(cp);require(v['row']==row and v.get('measurement_valid') is True,'changed checkpoint row/validity')
        receipt=Path(v['receipt']);require(sha(receipt)==v['receipt_sha256'],'receipt changed')
        r=read(receipt);require(r.get('measurement_valid') is True and r.get('finished_s'),'invalid/nonterminal receipt')
        require(v.get('artifacts') and str(receipt) in v['artifacts'],'receipt absent from artifact hashes')
        for path,h in v['artifacts'].items():require(sha(path)==h,'changed raw artifact: '+path)
        records.append(dict(cell_id=row['cell_id'],checkpoint=str(cp),checkpoint_sha256=sha(cp),
            receipt_sha256=v['receipt_sha256'],measurement_valid=True,work_complete=v.get('work_complete')))
    if require_all:require(len(records)==len(rows), 'declared phase incomplete')
    return dict(system=b['system'],phase=phase,declared=len(rows),completed=len(records),
                complete=len(records)==len(rows),binding=str(binding) if not isinstance(binding,dict) else None,
                records=records)


def archive_owned_queue_stop(binding,dest):
    stop=Path(read(binding)['output'])/'STOP'
    if not stop.exists():return None
    require(stop.read_text()==EXPECTED_STOP,'unrecognized queue STOP: preserve it')
    receipt=dict(source=str(stop),sha256=sha(stop),text=stop.read_text(),archived=str(dest))
    require(not dest.exists(),'STOP archive already exists')
    # Atomic move preserves the owned STOP bytes. Old supervisor STOP remains untouched.
    stop.rename(dest)
    require(sha(dest)==receipt['sha256'],'STOP archive changed')
    return receipt


class Supervisor(old.Supervisor):
    def stopping(self):
        return self.stop_requested or (ROOT/'STOP').exists() or (self.out/'STOP').exists()
    def boundary(self,reserve=400):
        require(not self.stopping(),'main-first STOP requested')
        require(time.time()+reserve<DEADLINE,'original global deadline leaves insufficient cleanup margin')
        check_files()
    def run(self):
        spec=read(SPEC);manifest=read(spec['workloads']);receipt=Path(spec['out'])/'deployment-receipt.json'
        gate_out=OLD_OUT/'legacy-correctness';gs=read(gate_out/'status.json');checks=read(gate_out/'checks/checks.json')
        bindings={};proofs={}
        try:
            self.state['phase']='waiting_for_old_boundary';self.save()
            while True:
                self.boundary()
                previous=read(OLD_OUT/'status.json')
                if previous.get('phase')=='stopped' and not pid_live(previous['pid']):break
                require(previous.get('phase') not in ('failed','finished'),'unexpected old terminal result')
                time.sleep(2)
            old_terminal(previous);write(self.out/'old-supervisor-terminal.json',previous)
            old_binding=OLD_OUT/'bindings/mixed/binding.json'
            old_mixed=read(old_binding)
            invocations=list((Path(old_mixed['output'])/'invocations').glob('mixed-*.json'))
            require(invocations,'old queue invocation missing')
            for p in invocations:
                inv=read(p);require(inv.get('complete') is True and inv.get('finished_s') and not inv.get('error'),
                                    'old invocation not cleanly terminal')
            verify_phase(old_binding,manifest,'main',False)
            record=archive_owned_queue_stop(old_binding,self.out/'old-mixed-queue.STOP')
            write(self.out/'queue-stop-archive.json',record)
            self.state.update(original_temporal_gate_passed=gs.get('passed'),
                              original_gate_preserved=str(gate_out),bindings={},main_proofs={})
            for system in SYSTEMS:
                self.boundary()
                try:mechanism=binder.mechanism_ready(gs,checks,system)
                except RuntimeError as exc:
                    # Only failed temporal mechanism is eligible to be declared blocked.
                    require(system=='ecoserve' and gs.get('complete') is True and
                            gs.get('measurement_valid') is True and gs.get('mechanism_gate',{}).get('temporal') is False and
                            str(exc).startswith('required original baseline mechanism has not passed:'),
                            'nonterminal or unexpected mechanism failure')
                    self.state['skipped_systems'][system]=dict(reason=str(exc),main_missing=30,
                        scale_missing=18,gate_status=str(gate_out/'status.json'),gate_sha256=sha(gate_out/'status.json'))
                    self.save();continue
                if system=='mixed':b=old_binding
                else:
                    dest=self.out/'bindings'/system
                    self.command('bind-'+system,[sys.executable,str(binder.__file__),'--spec',str(SPEC),
                        '--receipt',str(receipt),'--out',str(dest),'--strategy',system,'--gate',str(gate_out)],
                        max_work_s=120)
                    b=dest/'binding.json'
                bound=read(b);bindings[system]=b;self.state['bindings'][system]=str(b);self.save()
                before=verify_phase(b,manifest,'main',False)
                if not before['complete']:
                    self.command(system+'-main',[sys.executable,str(Path(spec['executor_release'])/'run.py'),
                        '--manifest',spec['workloads'],'--binding',str(b),'--system',bound['system'],
                        '--phase','main','--max-cells','30','--run'],stop_path=Path(bound['output'])/'STOP')
                proof=verify_phase(b,manifest,'main');proofs[system]=proof
                write(self.out/(system+'-main-proof.json'),proof);self.state['main_proofs'][system]=proof;self.save()
            self.state.update(phase='waiting_for_global_main_barrier',
                all_four_baseline_main_complete=len(proofs)==4,
                baseline_main_completed=sum(p['completed'] for p in proofs.values()),
                baseline_main_declared=120)
            self.save()
            # No scale launch path in this first revision: explicit global proof is mandatory.
            # A reviewed barrier adapter can continue these immutable bindings later.
            pd_proof=verify_phase(spec['pdb_binding'],manifest,'main')
            main_proof=dict(schema=1,model='32b',hostname=spec['hostname'],protocol_id=spec['protocol_id'],
                deadline_s=DEADLINE,source_manifest=spec['workloads'],source_sha256=sha(spec['workloads']),
                pdblend=pd_proof,baseline_systems={p['system']:p for p in proofs.values()},
                missing_systems=self.state['skipped_systems'],all_baseline_main_complete=len(proofs)==4,
                baseline_main_completed=sum(p['completed'] for p in proofs.values()),baseline_main_declared=120,
                temporal_failure_preserved=True,finished_s=time.time())
            write(self.out/'main-proof.json',main_proof)
            self.state.update(complete=True,phase='main_complete' if len(proofs)==4 else 'main_incomplete_correctness',
                main_proof=str(self.out/'main-proof.json'),main_proof_sha256=sha(self.out/'main-proof.json'),
                scale_launched=False,global_main_barrier_verified=False)
        except BaseException as exc:
            self.state.update(phase='stopped' if self.stopping() else 'failed',error=repr(exc));raise
        finally:self.state['finished_s']=time.time();self.save()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path);p.add_argument('--run',action='store_true')
    a=p.parse_args();check_files()
    if not a.run:print(json.dumps(dict(cpu_only=True,hardware_actions=False,main_first=True,scale_requires_global_barrier=True)));return
    require(a.out is not None,'fresh attempt required')
    s=Supervisor(a.out.resolve())
    def stop(signum,frame):s.stop_requested=True;s.state['signal_stop_requested_s']=time.time();s.save()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    s.run()
if __name__=='__main__':main()
