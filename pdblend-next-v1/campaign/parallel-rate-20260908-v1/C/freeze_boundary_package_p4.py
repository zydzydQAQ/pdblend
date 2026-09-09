"""Hash freeze the predeclared baseline queue and original recovery dependencies."""
from pathlib import Path
import ast,hashlib,json,time
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=read(HERE/'baseline-boundary-restore-p4/deployment.json');decl=read(HERE/'boundary-p4/declaration.json')
files=dict(spec['files']);files.update(decl['files'])
for name in ['boundary-p4/declaration.json','baseline-boundary-restore-p4/deployment.json','baseline-boundary-restore-p4/adapter-manifest.json',
 'boundary_baseline_queue_p4.py','run_boundary_baseline_p4.py','prepare_boundary_baselines_p4.py',Path(__file__).name]:
 p=HERE/name;files[str(p)]=sha(p)
assert all(sha(p)==h for p,h in files.items())
runner=(HERE/'run_boundary_baseline_p4.py').read_text();queue=(HERE/'boundary_baseline_queue_p4.py').read_text()
assert 'time.time()+400>=DEADLINE' not in runner and 'time.time()+reserve<DEADLINE' not in queue
assert "receipt['summary'].get('work_complete') is True" in runner and "receipt['summary'].get('failed_requests') == 0" in runner and "receipt['summary'].get('request_timeouts') == 0" in runner
assert "and not state.get('engineering_gate_failed')" in runner
assert "max_work_s=400*len(rows)+90" in queue and 'stop_cleanup_deadline=time.time()+400' in queue
assert spec['deadline_s'] is None and spec['campaign_lifecycle']=='until_declared_complete_v1'
assert read(spec['previous_binding'])['deadline_s'] is None
for p in [HERE/'run_boundary_baseline_p4.py',HERE/'boundary_baseline_queue_p4.py']:compile(p.read_text(),str(p),'exec')
value=dict(schema=1,created_s=time.time(),passed=True,files=files,cells=len(decl['cells']),deadline_s=None,campaign_lifecycle='until_declared_complete_v1',
 original_hardware_primitives_unchanged=True,local_restore_work_s=720,local_restore_cleanup_s=120,local_gate_work_s=390,local_gate_cleanup_s=90,
 any_request_failure_stops_successors=True,complete_low_slo_is_valid_capacity_negative=True,all_new_rates_each_baseline_twice=True)
p=HERE/'boundary-baseline-package-p4.json'
with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(passed=True,package=str(p),sha256=sha(p),files=len(files),cells=value['cells'])))
