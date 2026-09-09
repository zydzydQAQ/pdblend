"""Copy one explicitly frozen host, replacing only its shared Cell source."""
import argparse
import difflib
import json
from pathlib import Path
import shutil

from generate import HERE, PROTOCOL, encode, frozen, read, ref, require, sha

CELL='src/ecopadg/serving/cell.py'
PARENT_CELL_SHA='5230e4f9078d673320fd68eea839bbb6700c41b9a3c02b2d143ffef68205f5ee'


def build(parent,parent_sha256,out):
    parent=Path(parent).resolve();out=Path(out).resolve()
    reference=dict(path=str(parent/'manifest.json'),sha256=parent_sha256)
    original=read(frozen(reference));files=original['files']
    require(files.get(CELL)==PARENT_CELL_SHA,'this overlay requires the reviewed parent Cell bytes')
    for relative,expected in files.items():
        path=Path(relative)
        require(not path.is_absolute() and '..' not in path.parts,'unsafe parent member')
        frozen(dict(path=str(parent/path),sha256=expected))
    require(not out.exists(),'immutable host destination must be new')
    out.mkdir(parents=True)
    for relative in files:
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent/relative,target)
    overlay=HERE/'host-overlay/ecopadg/serving/cell.py'
    shutil.copyfile(overlay,out/CELL)
    for relative,expected in files.items():frozen(dict(path=str(parent/relative),sha256=expected))
    frozen(reference)
    (out/'WINDOW-README.md').write_text(
        'Shared five-system development measurement: 100 seconds, seed 701.\n'
        'Only serving/cell.py changes relative to the parent host. Baseline and PDB planner, '
        'transport, benchmark, source identity and measurement algorithms retain parent bytes.\n'
        'Arrival idle suffix is included; request hard timeout and post-window drain each remain 120 seconds.\n'
        'All eight GPU-board energy includes unsuccessful work. Actual legacy/v3 native proof and '
        'hardware configuration remain each wrapper\'s responsibility. Resident DynamoLLM is labeled separately.\n')
    result=dict(release=str(out),source_release=str(parent),source_manifest_sha256=parent_sha256,
        protocol_id=PROTOCOL,changed_files=[CELL],added_files=['WINDOW-README.md'],
        overlay=ref(overlay),builder=ref(__file__),
        scope='CPU-built immutable host; same parent runtime/bench/measurement, only shared100s Cell overlay; no GPU validation claimed',
        files={relative:sha(out/relative) for relative in sorted([*files,'WINDOW-README.md'])})
    require([k for k in files if files[k]!=result['files'][k]]==[CELL],'host source change exceeds the Cell overlay')
    (out/'manifest.json').write_bytes(encode(result))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--parent-sha256',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=build(a.parent,a.parent_sha256,a.out)
    print(json.dumps(dict(out=str(a.out),manifest_sha256=sha(a.out/'manifest.json'),
        files=len(result['files']),changed_source=result['changed_files'],
        pythonpath=f'{a.out}/src:{a.out}:/root/workspace/pdblend/.runtime-deps'),indent=2))


if __name__=='__main__':main()
