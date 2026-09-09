"""Final selection must recompute A's frozen actual qualification, not a flag."""
import json
from pathlib import Path
import subprocess
import sys


def validate(p, specification, release):
    expected = specification.get('qualification_contract')
    p.need(expected is not None, 'A final selection has no reviewed qualification contract')
    p.need(p.sha(expected['path']) == expected['sha256']
           and release['qualification_auditor'] == expected
           and release['files'][expected['path']] == expected['sha256'],
           'A release changed the reviewed independent qualification contract')
    qref = release['qualification900']
    recorded = p.checked(qref)
    p.need(recorded['qualification_auditor'] == expected
           and recorded['source'] == specification['actual_manifest']
           and recorded['profile'] == specification['profile'],
           'A actual qualification uses another controller or profile')
    program = '''import importlib.util,json,sys
from pathlib import Path
path,reference=sys.argv[1:]
sys.path.insert(0,str(Path(path).parent))
spec=importlib.util.spec_from_file_location('selected_A_actual_qualification',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
result=module.validate_qualification(json.loads(reference))
print(json.dumps(dict(passed=result['passed'],source=result['source'],profile=result['profile'],auditor=result['qualification_auditor'])))
'''
    result = subprocess.run([sys.executable, '-I', '-c', program, expected['path'], json.dumps(qref)],
                            text=True, capture_output=True)
    p.need(result.returncode == 0, 'A original source/work/energy/cleanup qualification failed: '
           + result.stderr[-1800:])
    actual = json.loads(result.stdout)
    p.need(actual == dict(passed=True, source=specification['actual_manifest'],
           profile=specification['profile'], auditor=expected), 'A actual qualification reconstruction differs')
    return dict(qualification=qref, auditor=expected, independently_recomputed=True,
                source=actual['source'], profile=actual['profile'])
