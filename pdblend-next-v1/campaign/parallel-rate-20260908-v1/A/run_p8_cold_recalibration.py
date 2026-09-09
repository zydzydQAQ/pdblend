"""One serial declared P8 physical queue; every failure stops after owned cleanup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import hashlib

A = Path(__file__).resolve().parent
R = A.parent
CODE = A / 'load-p8-code-001'
OUT = A / 'p8-cold-recalibration-queue-001'
DECL = A / 'p8-cold-recalibration-inputs-001/declaration.json'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p): return dict(path=str(Path(p).resolve()), sha256=sha(p))
def need(v, message):
    if not v: raise ValueError(message)
def read(p): return json.loads(Path(p).read_text())
def write(p, value):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(p)


def audit(run):
    out = Path(run['output']); spec = read(run['spec']['path'])
    need(sha(run['spec']['path']) == run['spec']['sha256'] and all(sha(p) == h for p,h in spec['files'].items()),
         'actual calibration source evidence changed')
    sys.path[:0] = [str(A), str(CODE), str(R)]
    import continue_p6_qualification_v3 as prior
    status = prior.audit(out, 1, run['spec'])
    result = prior.fixed(status['completed'][0])
    need(result['trace'] == spec['cycles'][0]['under_load'] and result['n_expected'] == 752
         and result['measured_arrival_duration_s'] == 60 and result['work_complete']
         and result['failed_requests'] == 0 and result['request_timeouts'] == 0,
         'exact full752 work required before any next physical repetition')
    restoration = prior.fixed(status['retained_restoration'])
    need(status['retained_restoration_complete'] and restoration['passed'], 'independent original-budget restoration required')
    removal = prior.fixed(ref(out/'remove.json'))
    prior.raw_measurement(removal['measurement']['receipt'])
    need(removal['execution_verified'] and removal['operation'] == 'remove', 'actual owned removal missing')
    return dict(passed=True, spec=run['spec'], status=ref(out/'status.json'), result=status['completed'][0],
                remove=ref(out/'remove.json'), inventory=ref(out/'inventory.json'),
                n_expected=result['n_expected'], slo_attainment=result['slo_attainment'], energy_j=result['energy_j'])


def main():
    need(not OUT.exists() and 'PDBLEND_NODE_LOCK_FD' not in os.environ, 'fresh CPU-only coordinator required')
    need(sha(DECL) == 'f7cc60d1f5f4b4ca77c7486090c762436542a87cf9b5970c727a44eb3002c94e', 'frozen three-run declaration changed')
    declaration = read(DECL)
    OUT.mkdir()
    state = dict(schema='P8-physical-three-run-queue-v1', pid=os.getpid(), started_s=time.time(),
                 complete=False, node_lease_held=False, steps=[], automatic_retries=False, declaration=ref(DECL))
    pins = {str(p):sha(p) for p in [Path(__file__), DECL, CODE/'manifest.json', CODE/'cpu-validation.json']}
    write(OUT/'source-manifest.json',pins)
    def update(**values):
        state.update(values,updated_s=time.time());write(OUT/'status.json',state)
    update(phase='ready')
    try:
        for run in declaration['runs']:
            need(not (A/'STOP_P8_RECALIBRATION').exists(), 'STOP prevents next declared repetition')
            need(all(sha(p)==h for p,h in pins.items()), 'queue source changed')
            need(not Path(run['output']).exists(), 'no retry or overwrite of any physical observation')
            args=[sys.executable,'-u',str(CODE/'capacity_load_calibrate.py'), '--spec',run['spec']['path'],
                  '--spec-sha256',run['spec']['sha256'],'--out',run['output']]
            dry=subprocess.run(args,text=True,capture_output=True)
            write(OUT/f"repeat-{run['repeat']}-dryrun.json",dict(exitcode=dry.returncode,stdout=dry.stdout,stderr=dry.stderr))
            need(dry.returncode==0,'source-only preflight failed before GPU')
            with (OUT/f"repeat-{run['repeat']}.log").open('xb') as log:
                child=subprocess.Popen([*args,'--run'],stdout=log,stderr=subprocess.STDOUT)
                update(phase='running',repeat=run['repeat'],child_pid=child.pid)
                code=child.wait()
            state.pop('child_pid',None)
            state['steps'].append(dict(repeat=run['repeat'],exitcode=code,output=run['output']))
            update()
            need(code==0,'physical repetition failed; source/raw/cleanup retained and no successor')
            evidence=audit(run)
            write(OUT/f"repeat-{run['repeat']}-audit.json",evidence)
            state['steps'][-1]['audit']=ref(OUT/f"repeat-{run['repeat']}-audit.json")
            update()
        update(phase='complete',complete=True)
    except BaseException as exc:
        update(phase='stopped_failure',error=repr(exc),complete=False)
        raise
    finally:
        update(finished_s=time.time())


if __name__ == '__main__': main()
