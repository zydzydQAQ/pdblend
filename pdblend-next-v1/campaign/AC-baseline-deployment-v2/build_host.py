"""Copy a frozen 100s host; opt-in lifecycle startup uses a hash-bound entry."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build_baseline_host(parent,out):
    parent=Path(parent).resolve();out=Path(out).resolve()
    if out.exists():raise ValueError('new immutable host directory required')
    manifest=json.loads((parent/'manifest.json').read_text())
    for name,h in manifest['files'].items():
        if sha(parent/name)!=h:raise ValueError('parent host changed: '+name)
    old=(parent/'src/ecopadg/serving/topology.py').read_text()
    new=old.replace('import asyncio\n','import asyncio\nimport hashlib\n',1)
    anchor="        path=self.root/(spec.instance_id+'.json')\n"
    addition="""        entry=self.template.get('observation_engine_entry')
        entry_sha=self.template.get('observation_engine_sha256')
        launch=['python3','-m','ecopadg.serving.engine']
        if entry is not None:
            if (not isinstance(entry,str) or not Path(entry).is_absolute()
                    or not isinstance(entry_sha,str) or len(entry_sha)!=64
                    or hashlib.sha256(Path(entry).read_bytes()).hexdigest()!=entry_sha):
                raise ValueError('observation engine entry is not absolute and hash-bound')
            launch=['python3',entry]
"""
    if new.count(anchor)!=1:raise ValueError('unexpected parent lifecycle source')
    new=new.replace(anchor,addition+anchor,1)
    source="            self.image,'python3','-m','ecopadg.serving.engine','--config',str(path))"
    if new.count(source)!=1:raise ValueError('unexpected parent engine command')
    new=new.replace(source,"            self.image,*launch,'--config',str(path))",1)
    out.mkdir(parents=True)
    for name in manifest['files']:
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent/name,target)
    (out/'src/ecopadg/serving/topology.py').write_text(new)
    (out/'lifecycle-observation-entry.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='parent/topology.py',tofile='baseline/topology.py')))
    result=dict(schema=1,parent_release=str(parent),parent_manifest_sha256=sha(parent/'manifest.json'),
        changed_runtime_files=['src/ecopadg/serving/topology.py'],
        scope='Explicit observation-only engine entry; scheduling policy, topology periods, costs, default entry, environment and mounts unchanged',
        files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    value=build_baseline_host(a.parent,a.out)
    print(json.dumps(dict(host=str(a.out.resolve()),manifest_sha256=sha(a.out/'manifest.json'),files=len(value['files']),hardware_actions=False)))
