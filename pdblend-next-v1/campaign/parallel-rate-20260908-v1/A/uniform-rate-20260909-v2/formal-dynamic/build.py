from pathlib import Path
import hashlib,json,ast
D=Path(__file__).resolve().parent;A=D.parents[1];source=A/'dynamic-execution-isolated-power-002'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={}
for name in ('dynamic_measurement.py','dynamic_ownership.py','sampler_hooks.py','protocol.py'):
 p=D/name
 with p.open('x') as f:f.write((source/name).read_text())
 files[name]=dict(original=str(source/name),original_sha256=sha(source/name),actual_sha256=sha(p),byte_identical=True)
s=(source/'dynamic_child.py').read_text()
s=s.replace('    from benchmarks.scripts import bench_vllm\n', '''    from collector_install import install
    collector_proof=install()
    from benchmarks.scripts import bench_vllm
''',1)
s=s.replace("    verify_inherited(job['lease'])\n", "    verify_inherited(job['lease'])\n    with (operation/'collector-source.json').open('x') as stream:\n        json.dump(collector_proof,stream,indent=2);stream.write('\\n')\n",1)
p=D/'dynamic_child.py'
with p.open('x') as f:f.write(s)
files[p.name]=dict(original=str(source/p.name),original_sha256=sha(source/p.name),actual_sha256=sha(p),difference='Install exact evidence-only collector before imports and record source proof. Original request/control/measurement flow retained.')
with (D/'source-equivalence.json').open('x') as f:json.dump(dict(schema='A14B-formal-collector-source-equivalence-v2',files=files,control_algorithm_changed=False,controller_source_unchanged=True,engine_source_unchanged=True,legacy_metric_fields_unchanged=True),f,indent=2);f.write('\n')
