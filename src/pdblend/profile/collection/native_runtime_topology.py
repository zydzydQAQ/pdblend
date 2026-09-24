"""Physical TP-group proof for same-fleet runtime calibration.

All-eight-board meter observations and a target replica's group watts are
different quantities. No target row may silently omit one rank's physical GPU.
"""
from __future__ import annotations
from pathlib import Path

from pdblend.online.native_control import validate_state
from .native_timing_audit import need,finite
from .native_timing_plan_v2 import MODEL_TP


def validate_topology(specs,lease,*,power=None,capabilities=None):
    ids=[s['instance_id'] for s in specs];gpus=lease['gpu_ids'];uuids=lease['gpu_uuids']
    need(len(gpus)==len(set(gpus))==len(uuids)==len(set(uuids))==8
         and all(type(g)is int for g in gpus) and all(isinstance(u,str) and u.startswith('GPU-') for u in uuids),
         'runtime topology requires eight distinct physical GPU identities')
    need(ids and len(ids)==len(set(ids)),'runtime replica inventory missing/duplicate')
    model=Path(specs[0]['model']).name;tp=MODEL_TP.get(model)
    need(tp in (1,2) and len(specs)==8//tp and all(Path(s['model']).name==model and s['tp']==tp and s['pp']==1
         and len(s['gpus'])==tp and len(set(s['gpus']))==tp and s['max_num_seqs']==32 for s in specs),
         'runtime model-owned TP1/TP2 native32 replica topology differs')
    need(sorted(g for s in specs for g in s['gpus'])==sorted(gpus),'runtime replicas do not partition all eight physical GPUs')
    physical=dict(zip(gpus,uuids))
    if power is not None:
        need(power['gpus']==gpus and power['gpu_uuids']==uuids,'runtime sampler physical ordering differs from lease')
    if capabilities is not None:
        need(set(capabilities)==set(ids),'runtime capability inventory differs')
        for spec in specs:
            cap=capabilities[spec['instance_id']]
            need(cap.get('supported') is True and cap.get('model_id')==model
                 and cap.get('tp')==tp and cap.get('pp')==1
                 and cap.get('gpu_uuids')==[physical[g] for g in spec['gpus']],
                 'runtime capability model/rank/physical identity differs')
    return physical


def _state(value,spec,*,drained=True,after=None):
    return validate_state(value,generation=spec['generation'],tp=spec['tp'],pp=spec['pp'],
                          drained=drained,observed_after_s=after)


def validate_off(receipt,spec,physical,*,before=None,after=None):
    observations=receipt['observations']
    need(receipt.get('compute_processes_gone') is True and receipt['owned_stop'].get('kind')=='stop'
         and observations and finite(receipt.get('started_s')) and finite(receipt.get('finished_s')),
         'off observation lacks owned process stop or timing')
    stamps=[]
    for observation in observations:
        stamp=observation['at_s'];devices=observation['devices'];stamps.append(stamp)
        need(finite(stamp) and receipt['started_s']<=stamp<=receipt['finished_s']
             and len(devices)==spec['tp'] and {d['gpu'] for d in devices}==set(spec['gpus'])
             and all(d.get('uuid')==physical[d['gpu']] and isinstance(d.get('compute_pids'),list)
                     and all(type(pid)is int and pid>0 for pid in d['compute_pids']) for d in devices),
             'off physical observation is missing a TP rank or has a wrong UUID')
    need(all(a<=b for a,b in zip(stamps,stamps[1:])) and all(not d['compute_pids'] for d in observations[-1]['devices']),
         'off target retains compute processes')
    if before is not None:need(receipt['finished_s']<=before,'off process-empty proof is after the start boundary')
    if after is not None:need(receipt['started_s']>=after,'off process-empty proof is before the end boundary')


def validate_clock(receipt,spec,physical,target):
    ack=receipt['ack'];observations=receipt['observations']
    need(ack.get('acknowledged') is True and ack.get('success') is True and ack.get('requested_frequency_mhz')==target
         and len(ack['gpus'])==spec['tp'] and {g['gpu_uuid'] for g in ack['gpus']}=={physical[g] for g in spec['gpus']}
         and observations and all(len(r['frequencies_mhz'])==spec['tp'] and finite(r.get('at_s')) for r in observations)
         and all(finite(f) and abs(f-target)<=15 for f in observations[-1]['frequencies_mhz']),
         'runtime clock receipt lacks every physical TP rank at target frequency')


def validate_phase(row,specs,physical,*,restore_frequency=2520):
    kind=row['kind']
    if kind=='transfer':
        p,d=row['prefill_instance'],row['decode_instance']
        need(p!=d and row['gpus']==list(specs[p]['gpus'])+list(specs[d]['gpus'])
             and set(row['before'])==set(row['after'])=={p,d},'transfer does not bind the complete two-peer TP GPU groups')
        for iid in (p,d):_state(row['before'][iid],specs[iid]);_state(row['after'][iid],specs[iid],after=row['finished_s'])
        return
    spec=specs[row['instance_id']]
    need(row['gpus']==list(spec['gpus']),'runtime phase dropped or substituted a physical TP-group GPU')
    if kind=='static':
        need(len(row['memory_before_mhz'])==len(row['memory_after_mhz'])==spec['tp'],
             'static memory clock boundary omitted a TP rank')
        if row['state']=='off':
            need(row.get('before') is None and row.get('after') is None,'off instance must not invent native live states')
            validate_off(row['off_before'],spec,physical,before=row['settle_started_s'])
            validate_off(row['off_evidence'],spec,physical,after=row['finished_s'])
        else:
            _state(row['before'],spec);_state(row['after'],spec,after=row['finished_s'])
            need(row['before']['native_at_s']<=row['settle_started_s'],'static pre-window native state is from the future')
    elif kind=='operation':
        name,receipt=row['operation'],row['receipt']
        if name=='off':
            _state(receipt['drain'],spec)
            validate_off(receipt['off'],spec,physical,before=row['finished_s'])
        elif name=='wake':
            _state(receipt['drain'],spec);_state(receipt['resume']['state'],spec)
            cap=receipt['capability']
            need(cap.get('tp')==spec['tp'] and cap.get('pp')==1
                 and cap.get('gpu_uuids')==[physical[g] for g in spec['gpus']],
                 'wake capability omitted a TP rank or changed physical boards')
        elif name.startswith('clock_') and name!='clock_reset':
            validate_clock(receipt,spec,physical,int(name.rsplit('_',1)[1]))
        elif name=='clock_reset':
            need(len(receipt['observed_mhz'])==spec['tp'],'clock reset omitted a TP rank')
        elif name=='park':
            _state(receipt['drain'],spec)
            need(len(receipt['memory_mhz'])==len(receipt['observed_mhz'])==spec['tp'],'park omitted a physical TP rank')
        elif name=='unpark':
            validate_clock(receipt['clock'],spec,physical,restore_frequency)
            _state(receipt['resume']['state'],spec);_state(receipt['drain'],spec)


def with_off_start_proof(row,measurements):
    """Reuse a prior actual stop receipt for old raw formats, never invent one."""
    if row.get('kind')!='static' or row.get('state')!='off' or row.get('off_before') is not None:return row
    prior=[r for r in measurements if r.get('kind')=='operation' and r.get('instance_id')==row['instance_id']
           and r['finished_s']<=row['settle_started_s']]
    need(prior,'off window lacks a preceding native owned-stop observation')
    previous=max(prior,key=lambda r:r['finished_s'])
    need(previous.get('operation')=='off','off window followed a different operation without a fresh stop proof')
    return dict(row,off_before=previous['receipt']['off'])


def validate_frequencies(power,gpus,start,end,target):
    clocks=[r for r in power['frequency_samples'] if start<=r[0]<end]
    indexes=[power['gpus'].index(g) for g in gpus]
    need(clocks and clocks[0][0]<=start+1 and clocks[-1][0]>=end-1
         and all(0<b[0]-a[0]<=1 for a,b in zip(clocks,clocks[1:]))
         and all(len(values)==8 and all(finite(values[i]) and abs(values[i]-target)<=15 for i in indexes)
                 for _,values in clocks),'actual static/clock frequencies differ or omit a physical sampling interval')
