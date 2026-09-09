"""Read-only remote mirror of terminal C 006 points and their audit chain."""
import json,subprocess,time,hashlib,sys
from pathlib import Path
import operate as op
C=Path(__file__).resolve().parent
QUEUE=C/'boundary-continuation-p4v2-006'
SPEC=json.loads((QUEUE/'continuation.json').read_text())
SSH="ssh -oBatchMode=yes -oConnectTimeout=8 -oStrictHostKeyChecking=yes -oHostKeyAlias=47.106.163.29"
H='172.16.50.105'
SNAPS=C/'mirror-continuation006-observations'
def copy_dir(p):
    p.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--ignore-existing','-e',SSH,H+':'+str(p)+'/',str(p)+'/'],check=True,timeout=180,capture_output=True)
def snapshot():
    raw=op.remote("from pathlib import Path;print(Path("+repr(str(QUEUE/'execution/status.json'))+").read_text(),end='')")
    d=json.loads(raw);SNAPS.mkdir(exist_ok=True)
    op.write_new(SNAPS/(str(time.time_ns())+'.json'),raw)
    dest=QUEUE/'execution/status.json';dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix('.mirror-tmp');tmp.write_bytes(raw);tmp.replace(dest)
    return d

def once():
    state=snapshot()
    ready=json.loads(op.remote('import pathlib,json;o=pathlib.Path('+repr(SPEC['output_root'])+');print(json.dumps([json.loads(p.read_text())["row"]["cell_id"] for p in o.rglob("checkpoints/*.json") if json.loads((p.parents[2]/"status.json").read_text()).get("finished_s")]))'))
    for cid in ready:
        cpdir=Path(SPEC['output_root'])/('dynamollm' if '-dynamollm-' in cid else 'ecoserve')/cid
        localcp=cpdir/'results/checkpoints'/(cid+'.json')
        if not localcp.exists():op.sync(cpdir)
        copy_dir(cpdir)
        copy_dir(Path(SPEC['binding_root'])/cid)
    for name in ('verified-cells','capacity-diagnoses'):
        folder=QUEUE/'execution'/name
        if json.loads(op.remote('import pathlib,json;print(json.dumps(pathlib.Path('+repr(str(folder))+').is_dir()))')):copy_dir(folder)
    print(json.dumps(dict(captured_s=time.time(),completed=len(state['completed']),phase=state['phase'],current=state.get('current_cell'),error=state.get('error'))),flush=True)
    return state

def main():
    for rel in ['baseline-after-gamma-restore-003','restore003-closure-001','boundary-baseline-bootstrap-after-gamma-002','boundary-baseline-gate-after-gamma-002','baseline-strategy-specs-after-gamma-002']:
        copy_dir(C/rel)
    while True:
        state=once()
        if '--watch' not in sys.argv or state['phase'] in ('complete','failed'):break
        time.sleep(25)
if __name__=='__main__':main()
