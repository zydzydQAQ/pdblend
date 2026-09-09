"""Remove only the user-cancelled global deadline; keep bounded cell cleanup."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'five-system-execution-v3'
OUT=ROOT/'common/execution-until-complete-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def once(s,a,b):
    if s.count(a)!=1:raise ValueError((a,s.count(a)))
    return s.replace(a,b)

def transform(s):
    s=once(s,'GLOBAL_DEADLINE = 1788872770.0400891',
        "GLOBAL_DEADLINE = None\nCAMPAIGN_LIFECYCLE = 'until_declared_complete_v1'")
    s=once(s,"    require(binding['deadline_s'] == GLOBAL_DEADLINE, 'global deadline changed')",
        "    require(binding.get('deadline_s') is None and binding.get('campaign_lifecycle') == CAMPAIGN_LIFECYCLE,\n            'explicit until-complete campaign lifecycle required')")
    s=once(s,'    latest = min(now + 90, GLOBAL_DEADLINE - 100 - 120 - 90)',
        '    latest = now + 90  # Bounded setup; no task-wide wall-clock cutoff.')
    s=once(s,'        cleanup_end = min(GLOBAL_DEADLINE, time.time() + 90, job[\'execution_deadline_s\'] + 90)',
        '        cleanup_end = min(time.time() + 90, job[\'execution_deadline_s\'] + 90)')
    s=once(s,"                if (output / 'STOP').exists() or time.time() > GLOBAL_DEADLINE - 400:",
        "                if (output / 'STOP').exists():")
    ast.parse(s);return s

def build():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    (OUT/'run.py').write_text(transform((PARENT/'run.py').read_text()))
    shutil.copyfile(PARENT/'child.py',OUT/'child.py')
    result=dict(schema=1,created_s=time.time(),campaign_lifecycle='until_declared_complete_v1',deadline_s=None,
        files={str(p):sha(p) for p in (OUT/'run.py',OUT/'child.py')},
        frozen_references={str(p):sha(p) for p in (PARENT/'run.py',PARENT/'child.py',Path(__file__))},
        original_child_byte_identical=True,arrival_window_s=100,request_timeout_s=120,
        setup_budget_s=90,cell_execution_after_latest_arrival_s=220,cleanup_budget_s=90,
        node_energy_interfaces_and_native_identity_checks_unchanged=True,
        authorization='User explicitly cancelled the task-wide deadline and requested completion of all remaining measurements.')
    with (OUT/'manifest.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(sha(OUT/'manifest.json'),sha(OUT/'run.py'))

if __name__=='__main__':build()
