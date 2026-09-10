"""Native fixed B14B handoff requires both complete fresh shapes and idle recovery."""
from pathlib import Path
import bootstrap as b
import power_selftest as p
import verify_idle


def verify(reference):
    saved=b.checked(reference)
    qualification=reference if saved.get('schema')=='migration-B-fixed14B-idle-qualification-v1' else p.ref(Path(reference['path']).parent/'qualified.json')
    result=verify_idle.verify(qualification)
    if qualification!=reference:assert result['binding']==reference
    assert result['node']=='B' and result['model']=='14b' and result['native_shape_cases']==82
    assert result['idle_probe_requests']==6 and result['idle_cycles']==4 and result['v3_cancellations']==2
    result['files'][str(Path(__file__))]=p.sha(__file__)
    return result
