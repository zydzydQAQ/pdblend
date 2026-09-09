"""Original-protocol loader only; no source, SLO, timing or policy transforms."""
import hashlib,importlib.util,json,sys
from pathlib import Path
REPO=Path('/root/workspace/pdblend-next-v1')
PROTOCOL='per-dataset-slo-five-system-fixed-window-v1'
LIFECYCLE='until_declared_complete_v1'
def require(ok,why):
 if not ok:raise ValueError(why)
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stat_identity(path):
 s=Path(path).stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
def checked_manifest(directory):
 directory=Path(directory).resolve();m=read(directory/'manifest.json')
 for name,h in m['files'].items():
  p=Path(name) if Path(name).is_absolute() else directory/name
  require(sha(p)==h,'frozen runtime changed '+str(p))
 return m
def load_runtime(host_release,common_dir):
 host=Path(host_release);common=Path(common_dir)
 require(common==REPO/'campaign/parallel-rate-20260908-v1/common/execution-until-complete-v1','original common executor required')
 require(sha(common/'run.py')=='77bcbbb68e20419e5bc469a838c71e1abfa789dc167d901501715fda0ff4a8a9','common source changed')
 checked_manifest(host);checked_manifest(common)
 sys.path[:0]=[str(host/'src'),str(host),'/root/workspace/pdblend/.runtime-deps']
 spec=importlib.util.spec_from_file_location('distributed14b_original_common',common/'run.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module
