from pathlib import Path
import ast, hashlib, json
H=Path(__file__).resolve().parent
OLD=H.parents[1]/'uniform-rate-20260909-v1/dynamic-producer/driver/capacity_load_calibrate.py'
s=OLD.read_text()
s=s.replace("def validate_spec(spec):\n", "def validate_spec(spec):\n    require(spec.get('supplement_kind') == 'matched-low-1.5-three-pairs-plus-one-idle-pair-v1', 'explicit supplement required')\n    require(spec.get('native_idle_poll_interval_s') == .1, 'declared native observation cadence required')\n    for cycle in spec['cycles']:\n        trace=fixed(cycle['low'])\n        require(trace['duration_s']==60 and len(trace['phases'])==1 and trace['phases'][0]['rate_rps']==1.5, 'exact independent sixty-second low1.5 required')\n",1)
a=s.index("                if 'matched_idle_duration_s' in spec:\n",s.index("elif spec['mode'] == 'layout_calibration':"))
b=s.index("                boundary(f'cycle-{index}-remove',360)",a)
s=s[:a]+'''                if index == 1:
                    name=f'cycle-{index}-idle-layout2';boundary(name,60)
                    require(len(controller.backend.instances)==2,'idle source layout differs')
                    await measure_idle(controller,out/name,spec,60)
                    state['completed'].append(ref(out/name/'result.json'));update()
                for layout in (2,3):
                    if layout == 3:
                        boundary(f'cycle-{index}-restore',360)
                        physical=await service.executor.calibrate('restore',tuple(spec['gpus']),declaration=cycle['restore'])
                        durable(out/f'cycle-{index}-restore.json',physical)
                    name=f'cycle-{index}-low-layout{layout}';boundary(name,420)
                    require(len(controller.backend.instances)==layout,'measured layout differs')
                    result=await measure_phase(controller,service,cycle['low'],out/name,spec)
                    state['completed'].append(ref(out/name/'result.json'));update()
                    require(result['work_complete'],'incomplete measured work prevents successor calibration')
                if index == 1:
                    name=f'cycle-{index}-idle-layout3';boundary(name,60)
                    require(len(controller.backend.instances)==3,'idle target layout differs')
                    await measure_idle(controller,out/name,spec,60)
                    state['completed'].append(ref(out/name/'result.json'));update()
''' +s[b:]
s=s.replace("    async def observe():\n", "    previous_native_at = None\n    async def observe():\n        nonlocal previous_native_at\n",1)
s=s.replace("        native_log.write(json.dumps(dict(at_s=now,raw=rows))", "        require(previous_native_at is None or 0 < now-previous_native_at <= 1., 'native idle observation gap exceeds unchanged one-second gate')\n        previous_native_at=now\n        native_log.write(json.dumps(dict(at_s=now,raw=rows))",1)
s=s.replace("await asyncio.sleep(min(.5,max(0.,end-time.time())))", "await asyncio.sleep(min(spec['native_idle_poll_interval_s'],max(0.,end-time.time())))",1)
path=H/'supplement_driver.py'
with path.open('x') as stream:stream.write(s)
oldtree={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(OLD.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
newtree={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
changed=[k for k in oldtree if oldtree[k]!=newtree[k]]
assert changed==['validate_spec','measure_idle','execute'],changed
report=dict(original=str(OLD),original_sha256=hashlib.sha256(OLD.read_bytes()).hexdigest(),generated=str(path),generated_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),changed_functions=changed,all_other_functions_ast_identical=True,control_algorithm_changed=False,native_gap_limit_s=1,native_poll_interval_s=.1)
with (H/'driver-source-audit.json').open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps(report))
