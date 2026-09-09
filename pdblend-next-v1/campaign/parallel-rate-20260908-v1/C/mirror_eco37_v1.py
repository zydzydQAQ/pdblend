"""Read-only mirror of every natural CP, including a failed terminal point."""
import json,subprocess,time,sys
from pathlib import Path
import operate as op
C=Path(__file__).resolve().parent;OUT=C/'eco-drain37-v1';P=OUT/'performance'
SSH='ssh -oBatchMode=yes -oConnectTimeout=8 -oStrictHostKeyChecking=yes -oHostKeyAlias=47.106.163.29';HOST='172.16.50.105'
def copy_dir(path):
    path.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--ignore-existing','-e',SSH,HOST+':'+str(path)+'/',str(path)+'/'],check=True,capture_output=True,timeout=180)
def once():
    op.sync(P)
    raw=op.remote('from pathlib import Path;print(Path('+repr(str(P/'status.json'))+').read_text(),end="")')
    state=json.loads(raw);snap=OUT/'mirror-observations'/(str(time.time_ns())+'.json');op.write_new(snap,raw)
    p=P/'status.json';tmp=p.with_suffix('.mirror-tmp');tmp.write_bytes(raw);tmp.replace(p)
    folder=P/'results/independent-audits'
    if json.loads(op.remote('import pathlib,json;print(json.dumps(pathlib.Path('+repr(str(folder))+').is_dir()))')):copy_dir(folder)
    print(json.dumps(dict(at_s=time.time(),phase=state['phase'],completed=len(state['completed']),attempted=len(state['attempted']),current=state.get('current_cell'),error=state.get('error'))),flush=True)
    return state
def main():
    for name in ['bootstrap','gate','qualified']:copy_dir(OUT/name)
    for name in ['gate-invocation.json','performance-invocation.json']:
        op.write_new(OUT/name,op.remote('from pathlib import Path;print(Path('+repr(str(OUT/name))+').read_text(),end="")'))
    while True:
        state=once()
        if state.get('finished_s') or '--watch' not in sys.argv:break
        time.sleep(25)
if __name__=='__main__':main()
