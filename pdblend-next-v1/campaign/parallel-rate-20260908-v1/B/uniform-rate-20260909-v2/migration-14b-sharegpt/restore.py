"""Measured retained B14B restart and independent immutable bootstrap audit."""
import argparse
import asyncio
import copy
import csv
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import bootstrap as b
import power_selftest as p
ROOT=p.ROOT
HELPER=ROOT/'B/baseline-return-after-external-source-v1/execution.py'
POWER=ROOT.parents[1]/'campaign/B32B-temporal-qualification-v2/power_source.py'
ENERGY=ROOT.parents[1]/'campaign/AC-baseline-binding-v2/gate_evidence.py'


def audit(reference):
    boot=b.checked(reference)
    assert boot['schema']=='migration-B-fresh14B-bootstrap-v1'
    assert all(p.sha(f)==h for f,h in boot['files'].items())
    raw=b.checked(boot['restored_binding']);old=b.checked(boot['parent_binding'])
    status=b.checked(boot['restore_status']);directory=Path(boot['restore_status']['path']).parent
    assert boot['hostname']==raw['hostname']==old['hostname']=='iZwz9i5bte3xkpmcoes3t2Z'
    assert boot['instances']==raw['instances'] and len(raw['instances'])==2
    assert raw['model']==old['model']=='14b' and raw['system']==old['system']=='pdblend'
    assert raw['configs']==old['configs'] and raw['host_release']==old['host_release']==str(p.HOST)
    before={r['Id']:r for r in p.read(directory/'containers.before.json')}
    after={r['Id']:r for r in p.read(directory/'containers.after.json')}
    assert set(before)==set(after)=={i['container']['id'] for i in old['instances']}
    for prior,new in zip(old['instances'],raw['instances']):
        a,c=copy.deepcopy(prior),copy.deepcopy(new)
        assert a['container']['StartedAt']!=c['container']['StartedAt']
        for v in (a,c):
            v.pop('host_pid',None);v['container'].pop('StartedAt');v['provenance'].pop('pid')
        assert a==c and new['tp']==1 and new['gpus'] in ([6],[7])
        x,y=before[new['container']['id']],after[new['container']['id']]
        assert x['State']['Running'] is False and x['State']['Pid']==0
        assert y['State']['Running'] and y['State']['Pid']==new['host_pid']>0
        assert y['State']['StartedAt']==new['container']['StartedAt']
        for k in ('Id','Name','Image','Path','Args','Config','HostConfig','Mounts'):
            assert x[k]==y[k], 'retained Docker policy changed: '+k
    assert status['complete'] and status['clock_restore_complete'] and not status.get('error')
    assert not status.get('errors') and not status.get('sampling_error') and status['power_evidence']['power_source_verified']
    ordinary=status['correctness']
    assert ordinary['passed'] and not ordinary.get('error') and not ordinary.get('owned') and not ordinary.get('cleanup_errors')
    assert len(ordinary['restoration'])==2 and all(x['complete'] for x in ordinary['restoration'])
    replies=ordinary['replies'];assert replies==b.checked(boot['ordinary'])
    assert len(replies)==4 and len({r['request_id'] for r in replies})==4
    assert {(r['instance_id'],r['prompt_length']) for r in replies}=={(i['id'],n) for i in raw['instances'] for n in (128,7168)}
    for n in (128,7168):
        answers=[r['response'] for r in replies if r['prompt_length']==n]
        for a in answers:
            assert len(a['token_ids'])==64 and all(type(t) is int and t>=0 for t in a['token_ids'])
            assert a['usage']['prompt_tokens']==n and a['usage']['completion_tokens']==64
        assert all(a['token_ids']==answers[0]['token_ids'] for a in answers)
    assert status['started_s']<=status['measurement_start_s']<ordinary['started_s']<=ordinary['finished_s']<=status['measurement_end_s']<=status['finished_s']
    power=b.load(POWER,'migration_B_restore_power_auditor')
    assert power.audit_raw(directory/'power',boot['files'])['power_source_verified']
    reader=b.load(ENERGY,'migration_B_restore_energy_reader')
    with (directory/'power/power.csv').open() as f:rows=list(csv.DictReader(f))
    samples=[(float(r['t_s']),[float(r[f'gpu{i}_w']) for i in range(8)]) for r in rows]
    assert all(math.isfinite(t) and all(math.isfinite(w) and w>=0 for w in v) for t,v in samples)
    energy=reader.integrate(samples,status['measurement_start_s'],status['measurement_end_s'])
    assert math.isclose(energy,status['setup_and_correctness_energy_j'],rel_tol=1e-8,abs_tol=1e-6)
    assert boot['complete'] and boot['ordinary_passed'] and boot['setup_measurement']==dict(measurement_valid=True,energy_j=energy)
    assert not boot['node_lease_held'] and boot['old_node_qualification_inherited'] is False
    return dict(passed=True,independently_recomputed=True,binding=boot['restored_binding'],energy_j=energy)


async def run(parent_ref,previous_ref,out,state):
    helper=b.load(HELPER,'migration_B_restore_original_executor')
    parent,previous=b.checked(parent_ref),b.checked(previous_ref)
    common=helper.load_common(parent['host_release'])
    from ecopadg.serving.campaign import node_lease
    assert 'PDBLEND_NODE_LOCK_FD' not in os.environ
    with node_lease():
        p.actual_identity();state['node_lease_held']=True;b.save(out/'status.json',state)
        raw=await helper.restore_core(common,parent,previous,out/'raw')
    state['node_lease_held']=False
    status=p.read(out/'raw/status.json')
    b.save(out/'ordinary.json',status['correctness']['replies'])
    frozen={str(x):p.sha(x) for x in (out/'raw').rglob('*') if x.is_file()}
    for ref in (parent_ref,previous_ref,p.ref(__file__),p.ref(HELPER),p.ref(POWER),p.ref(ENERGY)):
        frozen[ref['path']]=ref['sha256']
    frozen.update(raw['files'])
    frozen.update(p.source_check())
    boot=dict(schema='migration-B-fresh14B-bootstrap-v1',node='B',model='14b',hostname=raw['hostname'],
        instances=raw['instances'],complete=True,ordinary_passed=True,node_lease_held=False,
        setup_measurement=dict(measurement_valid=True,energy_j=status['setup_and_correctness_energy_j']),
        restored_binding=p.ref(out/'raw/binding.json'),parent_binding=parent_ref,previous_binding=previous_ref,
        restore_status=p.ref(out/'raw/status.json'),ordinary=p.ref(out/'ordinary.json'),
        files=frozen,old_node_qualification_inherited=False)
    b.save(out/'bootstrap.json',boot)
    audit(p.ref(out/'bootstrap.json'))
    state.update(complete=True,binding=boot['restored_binding'],bootstrap=p.ref(out/'bootstrap.json'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--previous',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    assert a.run and not a.out.exists();a.out.mkdir(parents=True)
    state=dict(schema='uniform-v2-B14B-retained-restoration-status',node='B',model='14b',pid=os.getpid(),
        started_s=time.time(),complete=False,node_lease_held=False)
    async def controlled():
        task=asyncio.current_task()
        for sig in (signal.SIGINT,signal.SIGTERM):asyncio.get_running_loop().add_signal_handler(sig,task.cancel)
        await run(p.ref(a.parent),p.ref(a.previous),a.out,state)
    try:asyncio.run(controlled())
    except BaseException as e:state['error']=repr(e);raise
    finally:state.update(finished_s=time.time(),node_lease_held=False);b.save(a.out/'status.json',state)

if __name__=='__main__':main()
