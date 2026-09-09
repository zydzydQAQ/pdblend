"""Persistent B owner handoff. Waiting never holds the GPU lease or retries work."""
import argparse,hashlib,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent.parent

def need(x,m):
    if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(r):
    need(sha(r['path'])==r['sha256'],'immutable reference changed: '+r['path']);return read(r['path'])
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def alive(owner):
    p=Path('/proc')/str(owner['pid'])/'stat'
    if not p.exists():return False
    fields=p.read_text().split()
    return fields[2]!='Z' and int(fields[21])==owner['start_ticks']
def full_status(s):
    need(s.get('finished_s') is not None and s.get('complete') is True and s.get('passed') is True and s.get('measurement_valid') is True and not s.get('errors'),'profile did not fully qualify')
    need(len(s['points'])==28 and len(s['switches'])==2 and s['clock_restore_complete'] and all(x['complete'] and not x.get('errors') for x in s['native_cleanup']),'profile point/transition/cleanup group incomplete')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    plan=read(args.plan)
    need(plan['schema']=='B-distributed14b-P12-persistent-handoff-v3' and plan['authorized'] is True and plan['node']=='B' and plan['dataset']=='sharegpt','exact B assignment required')
    need(socket.gethostname()==plan['hostname']=='iZwz9i5bte3xkpmcoes3t2Z' and 'PDBLEND_NODE_LOCK_FD' not in os.environ,'waiter requires unleased actual B')
    need(not args.out.exists(),'one-shot immutable continuation required');args.out.mkdir(parents=True)
    state=dict(pid=os.getpid(),started_s=time.time(),complete=False,node_lease_held=False,automatic_retries=False,phase='waiting_profile',plan=ref(args.plan),stages=[])
    stop=False;child=None
    def update(**values):state.update(values,updated_s=time.time());save(args.out/'status.json',state)
    def stopped(*_):
        nonlocal stop
        stop=True;update(stop_requested=True)
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stopped)
    def check():
        need(not stop and not (args.out/'STOP').exists(),'explicit safe boundary stop')
        for path,h in plan['files'].items():need(sha(path)==h,'handoff input changed: '+path)
        need(checked(plan['jobs'])['parent']['sha256']=='913a2d5834dbcc3466cff9d6a45e30cba574069ed50d36c89704ffff89b80438','assignment parent changed')
    def invoke(name,command):
        nonlocal child
        check();need(command[0]==sys.executable and command[1]=='-B' and plan['files'].get(command[2])==sha(command[2]),'stage code is not frozen in the continuation')
        with (args.out/(name+'.log')).open('x') as log:
            env=dict(os.environ);env['PYTHONPATH']='/root/workspace/pdblend/.runtime-deps';env['PYTHONDONTWRITEBYTECODE']='1'
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            update(phase=name,child_pid=child.pid,argv=command)
            code=child.wait();state['stages'].append(dict(name=name,exitcode=code,finished_s=time.time()));update(last_child_exitcode=code)
            need(code==0,'stage failed without automatic retry: '+name)
    update()
    try:
        while True:
            check();path=Path(plan['profile_status'])
            if path.exists():
                s=read(path);need(s['pid']==plan['profile_owner']['pid'],'profile owner differs')
                if s.get('finished_s') is not None and not alive(plan['profile_owner']):full_status(s);break
            need(alive(plan['profile_owner']),'profile owner exited without complete terminal evidence')
            time.sleep(2)
        update(profile_terminal=ref(path),phase='profile_qualified')
        prior=checked(plan['prior_engineering_diagnosis'])
        need(prior['node']=='B' and prior['classification']=='postpark_owned_command_actual_observation_after_original_settling_bound' and prior['scientific_comparison_eligible'] is False and prior['valid_capacity_negative'] is False, 'exact preserved P11 engineering failure required')
        need(prior['native_cleanup_complete'] is True and prior['clock_restore_complete'] is True, 'prior engineering attempt not clean')
        for key in ('prior_runner_status','prior_pipeline_status'):
            previous=checked(plan[key]);need(previous.get('finished_s') is not None and previous.get('node_lease_held') is False,'prior owner not terminal')
            need(not Path('/proc',str(previous['pid'])).exists(), 'prior owner PID still exists; inspect identity before a new declaration')
        probe=checked(plan['prior_probe_status'])
        need(probe.get('finished_s') is not None and probe.get('passed') is True and probe.get('clock_restore_complete') is True and all(x.get('complete') and not x.get('errors') for x in probe['native_cleanup']),'prior six-transition diagnostic not complete and clean')
        need(not Path('/proc',str(probe['pid'])).exists(),'prior diagnostic owner still exists')
        checked(prior['checkpoint']);checked(prior['receipt'])
        for name in ('prepare_pdb','run_pdb'):invoke(name,plan['commands'][name])
        pdb=read(plan['pdb_status'])
        need(pdb.get('node_lease_held') is False and pdb.get('finished_s') is not None and not pdb.get('failed'),'PDB did not leave a clean terminal state')
        if pdb.get('phase')=='needs_new_rate_declaration':
            update(phase='needs_new_rate_declaration',pdb_status=ref(plan['pdb_status']));return
        need(pdb.get('complete') is True and pdb.get('first_complete_slo_miss_rate') is not None,'PDB first complete SLO miss not established')
        update(phase='waiting_baseline_declaration',pdb_status=ref(plan['pdb_status']))
        target=Path(plan['baseline_continuation_declaration'])
        while not target.exists():check();time.sleep(2)
        baseline=read(target)
        need(baseline['schema']=='B-distributed14b-baseline-continuation-v1' and baseline['jobs']==plan['jobs'] and baseline['pdb_status']==ref(plan['pdb_status']) and baseline['pdb_release']==ref(plan['pdb_release']),'baseline continuation is not the exact completed PDB handoff')
        need(baseline['node']=='B' and baseline['dataset']=='sharegpt' and baseline['system_set']==['mixed','distserve','dynamollm','ecoserve'],'baseline complete system scope differs')
        for path,h in baseline['files'].items():need(sha(path)==h,'baseline source/qualification changed')
        command=baseline['command'];need(command[0]==sys.executable and command[1]=='-B' and baseline['files'].get(command[2])==sha(command[2]) and str(Path(command[2]).resolve()).startswith(str(R/'B')+'/'),'new immutable B baseline runner required')
        # The immutable suffix pins its source and the newly observed PDB boundary.
        # It must enforce fresh physical identity/native/profile gates itself.
        with (args.out/'baselines.log').open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            update(phase='baselines',child_pid=child.pid,baseline_declaration=ref(target));code=child.wait()
            need(code==0,'baseline suffix stopped; preserved result requires diagnosis')
        terminal=read(baseline['terminal_status'])
        need(terminal['complete'] is True and not terminal.get('failed') and terminal.get('node_lease_held') is False,'baseline group not complete and clean')
        update(phase='complete',complete=True,baseline_terminal=ref(baseline['terminal_status']))
    except BaseException as exc:update(phase='stopped_failure',error=repr(exc));raise
    finally:update(finished_s=time.time(),node_lease_held=False)

if __name__=='__main__':main()
