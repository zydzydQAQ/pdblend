"""Read-only independent verification of either newly qualified physical node."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parent.parent
HOSTS={'A':'iZwz9274emxme9019d2sjgZ','C':'iZwz9gfq11hx1sbob59yrgZ'}
def checked(r):
 p=Path(r['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];return json.loads(p.read_text())
def verify(reference):
 qualification=checked(reference);binding=checked(qualification['binding']);node=binding['node']
 assert node in HOSTS and binding['hostname']==HOSTS[node] and binding['model']=='14b' and binding['system']=='pdblend'
 assert Path(reference['path']).resolve().is_relative_to(ROOT/node)
 path=ROOT/node/'tooling-v2';code='import json,sys;from pathlib import Path;t=Path(sys.argv[1]);e=t.parents[1]/"env";sys.path[:0]=[str(t),str(e/"runtime/src"),str(e/"runtime"),str(e/"meter"),"/root/workspace/pdblend/.runtime-deps"];import power_selftest as p;import verify_idle;print(json.dumps(verify_idle.verify(p.ref(sys.argv[2]))))'
 r=subprocess.run([sys.executable,'-c',code,str(path),reference['path']],capture_output=True,text=True)
 if r.returncode:raise RuntimeError(r.stderr[-8000:])
 value=json.loads(r.stdout);assert value['passed'] and value['independently_recomputed'] and value['binding']==qualification['binding'];return value
if __name__=='__main__':
 p=Path(sys.argv[1]);print(json.dumps(verify({'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})))
