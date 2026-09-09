"""Freeze the declared new-rate PDB portion, conditional on the entire p2 screen."""
import copy
import hashlib
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
OUT=HERE/'workflow-new-rates-p2'
assert not OUT.exists()
OUT.mkdir()
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
source=HERE/'new-rates-p1/declaration.json'
d=read(source);cells=[]
for row in d['cells']:
    if row['system']!='pdblend':continue
    row=copy.deepcopy(row);row['cell_id']=row['cell_id'].replace('parallel-rate-p1-new-','parallel-rate-p2-new-')
    cells.append(dict(schema=1,model='7b',dataset='longbench',arm='fixed2',repeat=row['repeat'],stage='screen_fixed2',
        source_row=row,trace=dict(path=row['trace_path'],sha256=row['trace_sha256']),
        original_cell_id=row['workload_id']+'-pdblend-slo1',cell_id=row['cell_id'],
        new_rate=True,original_snapshot_point_exists=False,baseline_pairing_pending=True))
write(OUT/'work-declaration.json',dict(schema='parallel-rate-C-new-rates-p2',created_s=time.time(),
    deadline_s=d['deadline_s'],seed=701,arrival_window_s=100,request_hard_timeout_s=120,
    drain_after_arrival_window_s=120,independent_arrival_seeds=False,automatic_retries=False,
    full_five_system_declaration=dict(path=str(source),sha256=sha(source)),
    predecessor=str(HERE/'screen-p2'),cells=cells))
for name in ['protocol.py','prepare_release.py','operate.py']:
    (OUT/name).write_bytes((HERE/'workflow-p2'/name).read_bytes())
runner=(HERE/'workflow-p2/runner.py').read_text()
old=sha(HERE/'workflow-p2/work-declaration.json')
runner=runner.replace(old,sha(OUT/'work-declaration.json'))
before="    with node_lease():\n        state['node_lease_held'] = True"
after="""    with node_lease():
        predecessor=p.read(p.ROOT.parent / 'screen-p2/status.json')
        p.need(predecessor.get('complete') is True and predecessor.get('phase') == 'complete'
               and len(predecessor.get('completed', [])) == 12 and not predecessor.get('failed')
               and predecessor.get('node_lease_held') is False,
               'entire p2 fixed screen and predecessor cleanup required')
        p.need(not Path('/proc/' + str(predecessor['pid'])).exists(), 'predecessor process still alive')
        predecessor_cps=p.ROOT.parent / 'screen-p2/results/checkpoints'
        p.need({x.stem for x in predecessor_cps.glob('*.json')} == set(predecessor['completed']),
               'predecessor checkpoints incomplete')
        for checkpoint in predecessor_cps.glob('*.json'):
            cp=p.read(checkpoint);receipt=p.read(cp['receipt'])
            p.need(p.sha(cp['receipt']) == cp['receipt_sha256'] and receipt.get('measurement_valid') is True
                   and receipt.get('child_stopped') is True and receipt.get('clock_restore_complete') is True
                   and not receipt.get('outer_cleanup_errors'), 'predecessor measurement/cleanup missing')
            for file,digest in cp['artifacts'].items():p.need(p.sha(file) == digest, 'predecessor artifact changed')
        state['node_lease_held'] = True"""
assert runner.count(before)==1
runner=runner.replace(before,after).replace("len(state['completed']) == 1)","False)")
(OUT/'runner.py').write_text(runner)
write(OUT/'lineage.json',dict(parent_workflow=str(HERE/'workflow-p2'),source_declaration=str(source),
    actual_baseline_restoration_required_after_pdb=True,new_rates_are_capacity_observations=True,
    capacity_incomplete_work_retained=True,any_http503_stops_successors=True,
    source_files={str(OUT/n):sha(OUT/n) for n in ['protocol.py','runner.py','prepare_release.py','operate.py','work-declaration.json']}))
print(json.dumps(dict(cells=len(cells),declaration_sha256=sha(OUT/'work-declaration.json'),workflow=str(OUT))))
