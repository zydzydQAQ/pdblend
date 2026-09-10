"""Freeze and enqueue the complete B14B successor behind the current B32B scope."""
import argparse
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'common/uniform-rate-20260909-v2'))
import support as p
import pipeline


def prepare(predecessor,out):
    p.need(not out.exists(),'fresh queue declaration required')
    p.need(predecessor.is_file(),'explicit current predecessor status required')
    old=p.read(predecessor)
    p.need(old['node']=='B' and old['model']=='32b' and old['scope']=='five_systems','predecessor scope differs')
    files={str(f):p.sha(f) for f in HERE.glob('*.py')}
    for f in (ROOT/'common/uniform-rate-20260909-v2/support.py',ROOT/'common/uniform-rate-20260909-v2/contract.py'):
        files[str(f)]=p.sha(f)
    value=dict(schema='migration-B14B-sharegpt-five-system-queued-plan-v1',node='B',model='14b',datasets=['sharegpt'],scope='five_systems',
        expected_hostname='iZwz9i5bte3xkpmcoes3t2Z',declaration=p.ref(ROOT/'common/uniform-rate-20260909-v2/release-002/declaration.json'),
        predecessor_terminal_path=str(predecessor),predecessor_supervisor_lock=str(HERE.parent/'pipeline.lock'),
        stage_release_path=str(HERE/'stage-release-001.json'),supervisor_lock=str(HERE/'migration.lock'),
        pdb_parent=p.ref(ROOT/'B/distributed-14b-v1/pdb-p12-release-001/binding.json'),
        stop_paths=[str(HERE.parent/'STOP'),str(HERE/'STOP')],files=files,
        stage_release_required_before_any_hardware_change=True,old_node_qualification_inherited=False)
    p.save(out,value);pipeline.check_files(value);return p.ref(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--predecessor-terminal',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    print(prepare(a.predecessor_terminal,a.out))
