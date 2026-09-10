"""Resume only the remaining idle gate after a terminal successful fixed gate."""
from pathlib import Path
import json, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parent.parent
node = sys.argv[1]
assert node in ('A', 'C')
out = ROOT / node
paths = [str(out / 'tooling-v2'), str(ROOT / 'env/runtime/src'),
         str(ROOT / 'env/runtime'), str(ROOT / 'env/meter'),
         '/root/workspace/pdblend/.runtime-deps']
sys.path[:0] = paths
import bootstrap as b
import power_selftest as p
from inputs import idle
import verify_idle

previous = json.loads((out / 'environment-status.json').read_text())
assert previous['phase'] == 'failed' and 'aiohttp' in previous['error']
assert previous['steps'][-1]['name'] == 'fixed' and previous['steps'][-1]['exitcode'] == 0
assert not (out / 'environment-failed-002.json').exists()
b.save(out / 'environment-failed-002.json', previous)
state = dict(schema='slo14-environment-pipeline-v1', node=node, pid=os.getpid(),
             started_s=time.time(), complete=False, phase='idle_inputs', steps=[],
             fixed_qualification=p.ref(out / 'fixed-qualification/qualified.json'),
             prior_pipeline=p.ref(out / 'environment-failed-002.json'))
b.save(out / 'environment-status.json', state)
try:
    reference = idle(state['fixed_qualification'], out / 'idle-inputs')
    argv = [sys.executable, str(p.HERE / 'qualify_idle.py'), '--spec', reference['path'],
            '--out', str(out / 'pdb-qualification'), '--run']
    state.update(phase='idle')
    state['steps'].append(dict(name='idle', argv=argv, started_s=time.time()))
    b.save(out / 'environment-status.json', state)
    with (out / 'idle.log').open('x') as log:
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, 'PYTHONPATH': ':'.join(paths)})
    state['steps'][-1].update(exitcode=result.returncode, finished_s=time.time())
    assert result.returncode == 0, 'idle failed'
    verified = verify_idle.verify(p.ref(out / 'pdb-qualification/qualified.json'))
    b.save(out / 'pdb-ready.json', dict(node=node, qualification=p.ref(out / 'pdb-qualification/qualified.json'),
           qualification_validator=p.ref(p.HERE / 'verify_idle.py'), binding=verified['binding'],
           runtime_pythonpath=paths, measurement_executor=p.ref(ROOT / 'env/run.py'), complete=True))
    state.update(complete=True, phase='pdb_ready')
except BaseException as error:
    state.update(error=repr(error), phase='failed')
    raise
finally:
    state['finished_s'] = time.time()
    b.save(out / 'environment-status.json', state)
